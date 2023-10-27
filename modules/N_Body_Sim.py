import warnings
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from modules.engine.Enc_Proc_Dec import Enc_Proc_Dec


class N_Body_Simulator(nn.Module):


    def __init__(self, num_layers, im_dim=2, emb_dim=16, nonlinearity=nn.ReLU(), dynamic_channels=None,
                 processor='gnn', decoder='vae_decoder', num_latent_dim=0, **kwargs):

        super().__init__()

        all_channels = (i for i in range(im_dim))
        self.dynamic_channels = dynamic_channels  # channels that vary over time
        self.static_channels = tuple(set(all_channels) - set(dynamic_channels))  # fixed channels (e.g. mass)

        self.processor = processor.lower()
        self.use_05_std = False  # whether to use a fixed std so that the reconstruction loss is equivalent to MSE

        if self.processor == 'gnn':
            self.emb_to_mu = nn.Sequential(  # map the output embedding of Enc_Proc_Dec to mu in the data space
                nonlinearity,
                nn.Linear(emb_dim, emb_dim),  # we overwrite the static channels later
                nonlinearity,
                nn.Linear(emb_dim, len(self.static_channels) + len(self.dynamic_channels))
            )
            self.emb_to_std = nn.Sequential( # map the output embedding of Enc_Proc_Dec to Softplus^{-1}(sigma) in the data space
                nonlinearity,
                nn.Linear(emb_dim, emb_dim),  # we overwrite the static channels later
                nonlinearity,
                nn.Linear(emb_dim, len(self.static_channels) + len(self.dynamic_channels))
            )

        elif self.processor == 'eqgnn':
            self.final_layer = None  # we already have a mu in data space a output of the eqgnn
            self.node_std_layer = nn.Linear(emb_dim, 2)  # scalar std for isotropic gaussian over both pos and vel
            self.scale_vel = nn.Linear(0,1)
            # vel and pos are handled separately in a tuple, so no dynamic channels in e(n) invariant node feat input:
            dynamic_channels = tuple()
        elif self.processor == 'fagnn':
            self.node_std_layer = nn.Linear(emb_dim, 2)  # scalar std for isotropic gaussian over both pos and vel
            # vel and pos are handled separately in a tuple, so no dynamic channels in e(n) invariant node feat input:
            dynamic_channels = tuple()
        elif self.processor == 'mlp':
            self.emb_to_mu = nn.Sequential(  # map the output embedding of Enc_Proc_Dec to mu in the data space
                nonlinearity,
                nn.Linear(emb_dim, emb_dim),  # we overwrite the static channels later
                nonlinearity,
                nn.Linear(emb_dim, im_dim)
            )
            self.emb_to_std = nn.Sequential(  # map the output embedding of Enc_Proc_Dec to Softplus^{-1}(sigma) in the data space
                nonlinearity,
                nn.Linear(emb_dim, emb_dim),  # we overwrite the static channels later
                nonlinearity,
                nn.Linear(emb_dim, im_dim)
            )
            # logistics to handle that we process the unflattened output in this wrapper class
            self.dynamic_channels = tuple(set((i % 7 for i in self.dynamic_channels)))
            self.im_dim = 7
            self.static_channels = tuple(set((i % 7 for i in self.static_channels)))
        else:
            raise NotImplementedError('you probably want a gnn processor for the n-body simulations...')

        self.engine = Enc_Proc_Dec(num_layers, im_dim, emb_dim, kernel_size=0, nonlinearity=nonlinearity,
                                   dynamic_channels=dynamic_channels, processor=processor,
                              decoder=decoder, decoder_kernel_size=0, num_latent_dim=num_latent_dim, **kwargs)
        self.kwargs = kwargs

    def forward(self, x_orig: torch.Tensor, x_true_orig: torch.Tensor,):
        # x_orig is x^t, x_true_orig is x^{t+1} -- only available when calculating LL, e.g. during training
        # returns distribution, additional (KL) loss, sample
        DOWNSCALING_FACTOR = 10.
        EPS = 1e-5
        if 'EPS' in self.kwargs.keys():
            if self.kwargs['EPS'] is not None:
                EPS = self.kwargs['EPS']

        idx_pos = self.dynamic_channels[:int(len(self.dynamic_channels) / 2)]
        idx_vel = self.dynamic_channels[int(len(self.dynamic_channels) / 2):]

        # we simply scale all features down by a factor of 10 to try to avoid having large absolute value
        # inputs in a potentially sensitive architecture (e.g. analogous to going from hectometer to kilometer units)
        # in the end we will re-multiply with this factor to get back to the original magnitude

        x = x_orig / DOWNSCALING_FACTOR
        x_true = x_true_orig / DOWNSCALING_FACTOR if x_true_orig is not None else None

        # NOTE: we assume a fully connected graph here!
        # there will need to be some extensions in the GNN modules to support arbitrary graph topologies.
        # edge feat should have shape (bs, num_nodes, num_nodes, num_feat)
        edge_feat = self.construct_edge_features(x)
        edge_feat_true = self.construct_edge_features(x_true)

        if self.processor == 'gnn':
            input_to_model = (x, edge_feat)
            ground_truth_input_to_model = (x_true, edge_feat_true) if x_true is not None else None
        elif self.processor == 'eqgnn' or self.processor == 'fagnn':
            # split the input into inv. featrues, edges, pos, and vel
            input_to_model = (x[..., self.static_channels], edge_feat, x[..., idx_pos], x[..., idx_vel])
            ground_truth_input_to_model = (x_true[..., self.static_channels], edge_feat_true, x_true[..., idx_pos], x_true[..., idx_vel]) if x_true is not None else None
        elif self.processor == 'mlp':
            input_to_model = torch.flatten(x, start_dim=1).unsqueeze(1)
            ground_truth_input_to_model = torch.flatten(x_true, start_dim=1).unsqueeze(1) if x_true is not None else None
        else:
            raise NotImplementedError('processor string not recognized')

        engine_output, additional_loss = self.engine(input_to_model, ground_truth_input_to_model)
        assert isinstance(engine_output, tuple) or self.processor == 'mlp',\
            'expected to see a tuple of (nodes, edges, ...) here, except for an MLP processor!'

        mu = torch.zeros_like(x)
        log_sigma = torch.zeros_like(x)


        if self.processor == 'gnn':  # we need to  map from the final embedding dim to the data space
            out_dyn_mu = self.emb_to_mu(engine_output[0])  #[0] to select nodes
            out_dyn_std = self.emb_to_std(engine_output[0])


            # NOTE: the channel at idx zero of out_dyn is just a dummy channel to make the indexing match with x
            # (we don't predict mass but manually set it as it remains constant)
            mu[..., self.static_channels] += x[..., self.static_channels]  # factors that stay the same throughout

            # residual connection for the mu: for pos, do x + vel_{new} * dt + resid, for vel, do vel+resid

            # vel -- first change velocity
            mu[..., idx_vel] += out_dyn_mu[..., idx_vel] + x[..., idx_vel]
            # pos -- use updated velocity + residual term accounting for inaccuracies with big stepsize.
            # note: data is saved at 0.01s granularity, but the gt simulator uses dt=10^{-4}
            mu[..., idx_pos] += out_dyn_mu[..., idx_pos] + x[..., idx_pos] + mu[..., idx_vel] * self.kwargs['pred_stepsize'] / 100

            log_sigma[..., self.static_channels] += EPS  # factors that stay the same throughout
            log_sigma[..., self.dynamic_channels] += out_dyn_std[..., self.dynamic_channels]

        elif self.processor == 'eqgnn':
            # we already get the (node_emb, edge_feat, x, v) tuple from the engine
            final_node_emb, final_edge_emb, x_pred, v_pred = engine_output

            # tried: (velocity is updated and used internally multiple times, so we need to learn the right coefficient to map
            # it back to the magnitude corresponding to the data stepsize)
            # this layer takes an empty tensor as input, use the below as quick hack to ensure same device and precision
            mu[..., idx_vel] += v_pred #* (1+nn.LeakyReLU()(self.scale_vel(v_pred[0:0, 0, 0].squeeze())))
            mu[..., idx_pos] += x_pred
            mu[..., self.static_channels] += x[..., self.static_channels]

            log_std_pos_and_vel = self.node_std_layer(final_node_emb) # invariant to rotations, translations: isotropic gaussian
            log_std_pos = log_std_pos_and_vel[..., 0].unsqueeze(-1).repeat(1,1, len(idx_pos))  # shape (bs,nodes,3)
            log_std_vel = log_std_pos_and_vel[..., 1].unsqueeze(-1).repeat(1,1, len(idx_vel))  # shape (bs,nodes,3)

            log_sigma[..., idx_pos] += log_std_pos  # assume constant std in all directions
            log_sigma[..., idx_vel] += log_std_vel  # assume constant std in all directions
            log_sigma[..., self.static_channels] += EPS  # factors that stay the same throughout
        elif self.processor == 'fagnn':
            h, edge_feat, x_pred, v_pred = engine_output

            mu[..., idx_vel] += v_pred
            mu[..., idx_pos] += x_pred
            mu[..., self.static_channels] += x[..., self.static_channels]

            log_std_pos_and_vel = self.node_std_layer(h)  # invariant to rotations, translations: isotropic gaussian
            log_std_pos = log_std_pos_and_vel[..., 0].unsqueeze(-1).repeat(1,1, len(idx_pos))  # shape (bs,nodes,3)
            log_std_vel = log_std_pos_and_vel[..., 1].unsqueeze(-1).repeat(1,1, len(idx_vel))  # shape (bs,nodes,3)

            log_sigma[..., idx_pos] += log_std_pos  # assume constant std in all directions
            log_sigma[..., idx_vel] += log_std_vel  # assume constant std in all directions
            log_sigma[..., self.static_channels] += EPS  # factors that stay the same throughout
        elif self.processor == 'mlp':
            out_dyn_mu = self.emb_to_mu(engine_output).reshape(-1, x_orig.shape[1], x_orig.shape[2])  # bs, bodies, feat
            out_dyn_std = self.emb_to_std(engine_output).reshape(-1, x_orig.shape[1], x_orig.shape[2])

            mu[..., self.static_channels] += x[..., self.static_channels]  # factors that stay the same throughout

            # vel -- first change velocity
            mu[..., idx_vel] += out_dyn_mu[..., idx_vel] + x[..., idx_vel]
            # pos -- use updated velocity + residual term accounting for inaccuracies with big stepsize.
            # note: data is saved at 0.01s granularity, but the gt simulator uses dt=10^{-4}
            mu[..., idx_pos] += out_dyn_mu[..., idx_pos] + x[..., idx_pos] + mu[..., idx_vel] * self.kwargs[
                'pred_stepsize'] / 100

            # postprocess so that expected system momentum and com remain constant
            # mu = self.postprocess_constant_expected_comand_momentum(mu, x)

            log_sigma[..., self.static_channels] += EPS  # factors that stay the same throughout
            log_sigma[..., self.dynamic_channels] += out_dyn_std[..., self.dynamic_channels]

        else:
            raise NotImplementedError('processor str not recognized in fw pass of n body sim module!')

        sigma = nn.Softplus()(log_sigma) + EPS

        # now, we scale everyting up again to the original data magnitude
        mu = DOWNSCALING_FACTOR * mu

        pred_dist = torch.distributions.Normal(mu, sigma)
        if self.use_05_std:
            # pred_dist = torch.distributions.Normal(mu, sigma + 0.1)
            pred_dist = torch.distributions.Normal(mu, torch.ones_like(sigma) * (0.5)**0.5)

        pred_sample = pred_dist.mean
        if 'sample_from_pred' in self.kwargs:   # we might want to sample from p(x^t | x^{t-1}, z)
            if self.kwargs['sample_from_pred']:
                pred_sample = pred_dist.rsample()

        pred_sample[..., self.static_channels] = x_orig[..., self.static_channels]  # mass always remains constant

        return pred_dist, additional_loss, pred_sample

    @torch.no_grad()
    def get_additional_val_stats(self, pred_sample, x_true):

        return nn.MSELoss(reduction='sum')(pred_sample, x_true)

    def construct_edge_features(self, x):
        if x is None:
            return None
        # let's have relative velocity magnitude, relative distance as input for the edges
        edge_feat = torch.zeros(x.size(0), x.size(1), x.size(1), 2)
        coors_rep = x[..., 1:4].unsqueeze(1).repeat(1, x.size(1), 1, 1)
        dist_squared = torch.sum((coors_rep - coors_rep.transpose(1, 2)) ** 2, dim=3)

        diagonal_mask = torch.eye(dist_squared.shape[1], dist_squared.shape[2]).repeat(
            dist_squared.shape[0], 1, 1).to(dist_squared.device)
        dist = torch.sqrt(dist_squared + diagonal_mask * 1e-6)  # the 1e-6 here prevents NaN gradient error for 0 under sqrt
        dist = dist * (1 - diagonal_mask)

        edge_feat[..., 0] = dist

        # repeat velocity and calculate the magnitudes of all pairwise velocity differences
        vel_rep = x[..., 5:8].unsqueeze(1).repeat(1, x.size(1), 1, 1)
        rel_vel_mag = torch.sum((vel_rep - vel_rep.transpose(1, 2)) ** 2, dim=3)
        rel_vel_mag = torch.sqrt(rel_vel_mag + diagonal_mask * 1e-6)  # the 1e-6 here prevents NaN gradient error for 0 under sqrt
        rel_vel_mag = rel_vel_mag * (1 - diagonal_mask)
        edge_feat[..., 1] = rel_vel_mag

        return edge_feat.to(x)

    def postprocess_constant_expected_com_and_momentum(self, mu, x_input):
        # mu has shape [bs, bodies, 7] -> [..., 1:] are the elements that matter
        mean_in_times_mass = torch.mean(x_input[..., 0:1] * x_input[..., 1:], 1, keepdim=True)
        mean_mu_times_mass = torch.mean(x_input[..., 0:1] * mu[..., 1:], 1, keepdim=True)
        diff_times_mass = mean_in_times_mass - mean_mu_times_mass
        mu[..., 1:] = mu[..., 1:] - diff_times_mass / x_input[..., :1]
        return mu

    def calc_energy(self, state, idx_pos, idx_vel, return_split=False):
        '''
        Gets the Kinetic and Potential energy of the system

        RETURNS
        [KE, PE]: list containing:
            KE: ndarray
            PE: ndarray

        adapted from code via Philip Mocz
        https://github.com/pmocz/nbody-python/blob/master/nbody.py
        '''
        # get Kinetic Energy .5*m*v**2
        KE = torch.sum(state[..., 0:1] * state[..., idx_vel] ** 2, dim=(1, 2)) / 2.

        # get Potential Energy G*m1*m2/r**2
        r = self.construct_edge_features(state)[..., 0]  # (bs, n_bodies, n_bodies)
        r = torch.sqrt(r**2 + 1e-2)  # apply some softening to prevent instabilities, softening for force calc in ground truth simulator = 1e-2
        # r = torch.sqrt(torch.sum(delta_coors**2, dim=(1,2)))
        r_inv = torch.zeros_like(r)
        r_inv = r_inv + torch.triu((1 / r), 1)
        m = state[..., 0:1]
        m_matrix = m * m.transpose(-2, -1)  # (bs, n_bodies, n_bodies)

        PE = torch.sum(torch.triu(-m_matrix * r_inv, 1), dim=(1,2))

        if not return_split:
            return KE + PE
        else:
            if (PE > 0).any():
                print('help! found potential energy greater than 0!')
            return [KE, PE]




