import warnings

import numpy as np
import torch
import torch.nn as nn
from modules.engine.GNN_modules import GNN, LazyGNN, AggregateNodes, LazyEqGNN, FAGNN, LazyMLP
from modules.engine.Grid_modules import UNet, UNetModern, GridGNN, AggregateGridNodes, GridDeepSet
EPS = 1e-5


class Enc_Proc_Dec(nn.Module):

    # the main class that does the heavy lifting
    # takes as input x^t and produces an embedding which can be decoded to x^{t+1} in the
    # 'application specific' classes (.e.g Cellsort_Sim or N_body_Sim).
    # the Encoder maps to an embedding space, the processor corresponds to the forward model defined in the paper,
    # and the decoder corresponds to the conditional VAE part of the model.

    def __init__(self, num_layers, im_dim=2, emb_dim=16, kernel_size=3, nonlinearity=nn.ReLU(),
                 dynamic_channels=None, processor='unet', decoder='vae_decoder',
                 decoder_kernel_size=25, num_latent_dim=0, global_downsampling_factor=1, free_bits=0,
                 prior_and_encoder_arch=None, num_prior_and_encoder_layers=None, do_node_aggregation=None, **kwargs):

        if not (num_latent_dim == 0 or decoder.lower() == 'vae_decoder'):
            warnings.warn('can only have latent space when using vae decoder! defaulting to latent space = 0')
            num_latent_dim = 0

        super().__init__()

        #######
        # store some general information:
        #######
        if dynamic_channels is None:  # all channels are assumed to be predicted
            self.dynamic_channels = (i for i in range(im_dim))
        else:
            self.dynamic_channels = dynamic_channels

        all_channels = set([i for i in range(im_dim)])
        self.static_channels = tuple(all_channels - set(self.dynamic_channels))
        self.num_latent_dim = num_latent_dim
        self.processor_str = processor.lower()
        self.decoder_str = decoder.lower()
        if num_prior_and_encoder_layers is None:
            num_prior_and_encoder_layers = num_layers

        #####
        # PROCESSOR (forward model)
        #####

        self.proc = Processor(num_layers, emb_dim, kernel_size, nonlinearity, self.processor_str, self.static_channels,
                              **kwargs)
        self.domain_str = self.proc.domain_str

        #####
        # ENCODER  (simply maps to embedding dim)
        #####

        self.enc = Encoder(emb_dim, domain_str=self.domain_str)  # downsampling factor only has effect on grids

        #####
        # DECODER (produces an output embedding which is decoded into data space in the application specific wrappers)
        #####

        if self.decoder_str == 'vae_decoder':
            print('using conditional VAE decoder')

            decoder = self._get_decoder_object(emb_dim=emb_dim, num_layers=num_layers, num_latent_dim=num_latent_dim,
                                               kernel_size=kernel_size, decoder_kernel_size=decoder_kernel_size,
                                               nonlinearity=nonlinearity, do_node_aggregation=do_node_aggregation,
                                               prior_and_encoder_arch=prior_and_encoder_arch,
                                               num_prior_and_encoder_layers=num_prior_and_encoder_layers,
                                               dynamic_channels=self.dynamic_channels, free_bits=free_bits,
                                               **kwargs)
            self.dec = decoder
            if self.domain_str == 'grid' and global_downsampling_factor > 1:
                self.global_downsampling = nn.LazyConv2d(emb_dim, global_downsampling_factor,
                                                         global_downsampling_factor)
                self.global_upsampling = nn.LazyConvTranspose2d(emb_dim, global_downsampling_factor,
                                                                global_downsampling_factor)
                self.global_downsampling_ground_truth = nn.LazyConv2d(emb_dim, global_downsampling_factor,
                                                                      global_downsampling_factor)
            else:
                self.global_downsampling = nn.Identity()
                self.global_upsampling = nn.Identity()
                self.global_downsampling_ground_truth = nn.Identity()

        else:
            raise NotImplementedError(
                "the decoder string is not recognized! Use 'vae_decoder' (if you want to have no vae, set num_latent_dim=0)")

    def forward(self, x, x_true=None):

        if isinstance(x, tuple):
            x = self.global_downsampling(x[0]), *x[1:]  # NOTE: nn.Identity() for graph, only works for grid!
        else:
            x = self.global_downsampling(x)

        if x_true is not None:
            x_true_dynamic = index_channels_for_domain(x_true, self.dynamic_channels, self.domain_str)
            if isinstance(x_true_dynamic, tuple):
                x_true_dynamic = self.global_downsampling_ground_truth(x_true_dynamic[0]), *x_true_dynamic[1:]
            else:
                x_true_dynamic = self.global_downsampling_ground_truth(x_true_dynamic)
        else:
            x_true_dynamic = None

        # initialize additional loss terms at 0:
        additional_loss = torch.sum(torch.zeros(size=(1,)))

        # encode:
        e = self.enc(x)

        # process (apply forward model to get h^t)
        input_to_decoder = self.proc(e, x)

        # decode (apply cVAE to get embedding that can be decoded to x^{t+1}):
        if self.decoder_str == 'vae_decoder':  # probabilistic decoder
            d, add_loss = self.dec((input_to_decoder, x_true_dynamic))
            additional_loss = additional_loss + add_loss
        else:
            raise NotImplementedError('decoder string not recognized!')

        if isinstance(d, tuple):
            d = e[0] + d[0], *d[
                              1:]  # another residual connection for the embedding. in case of E(n) graph, this only affects the node embedding, not the spatial attributes
            d = self.global_upsampling(d[0]), *d[1:]  # only has effect if d is a grid, otherwise applies nn.Identity()
        else:
            d = e + d
            d = self.global_upsampling(d)

        return d, additional_loss

    def get_emb_no_dec(self, x):
        e = self.enc(x)
        out = self.proc(e, x)
        return out

    def _get_decoder_object(self, emb_dim, num_layers, num_latent_dim, kernel_size, decoder_kernel_size, nonlinearity,
                            do_node_aggregation, prior_and_encoder_arch, num_prior_and_encoder_layers,
                            dynamic_channels, free_bits, **kwargs):
        # get the right VAE decoder object
        # same as in the processor, we take care of some logistics here.
        # The below code is a bit ugly and again we hardcoded some design choices to make our lives a bit easier,
        # sorry for that.
        ps = self.processor_str
        if ps == 'unet' or ps == 'convnet' or ps == 'unet_modern':
            prior_and_encoder_arch = 'conv' if prior_and_encoder_arch is None else prior_and_encoder_arch
            # note: kernel size is ignored if the processor is a gnn
            # the below nets return a tensor of shape (bs, emb_dim) regardless of grid or graph domain!
            vae_encoder_net, vae_prior_net = get_vae_enc_and_prior_nets(prior_and_encoder_arch, emb_dim,
                                                                        num_prior_and_encoder_layers, kernel_size,
                                                                        nonlinearity,
                                                                        do_node_aggregation=do_node_aggregation,
                                                                        **kwargs)
            decoder_end_arch = nn.Sequential(
                nn.Conv2d(emb_dim + num_latent_dim + len(self.static_channels), emb_dim,
                          (decoder_kernel_size, decoder_kernel_size), padding='same', **kwargs),
                nn.ReLU(),
                *[nn.Sequential(nn.Conv2d(emb_dim, emb_dim, (kernel_size, kernel_size), padding='same', **kwargs),
                                nn.ReLU()) for _ in range(num_layers - 1)])
        elif ps == 'gridgnn':
            prior_and_encoder_arch = 'gridgnn' if prior_and_encoder_arch is None else prior_and_encoder_arch

            vae_encoder_net, vae_prior_net = get_vae_enc_and_prior_nets(prior_and_encoder_arch, emb_dim,
                                                                        num_prior_and_encoder_layers, kernel_size,
                                                                        nonlinearity,
                                                                        agg_grid_map_to_dim=2 * num_latent_dim,
                                                                        do_node_aggregation=do_node_aggregation,
                                                                        **kwargs)
            decoder_end_arch = GridGNN(num_layers=3, emb_dim=emb_dim, activation=nonlinearity,
                                       kernel_size=decoder_kernel_size, **kwargs)
        elif ps == 'griddeepset':
            prior_and_encoder_arch = 'griddeepset' if prior_and_encoder_arch is None else prior_and_encoder_arch

            vae_encoder_net, vae_prior_net = get_vae_enc_and_prior_nets(prior_and_encoder_arch, emb_dim,
                                                                        num_prior_and_encoder_layers, kernel_size,
                                                                        nonlinearity,
                                                                        agg_grid_map_to_dim=2 * num_latent_dim,
                                                                        do_node_aggregation=do_node_aggregation,
                                                                        **kwargs)
            nl = 3
            if 'use_unet' in kwargs.keys():
                if kwargs['use_unet']:
                    nl = 1  # if we use the unet within the griddeepset module, just apply one UNet module
            decoder_end_arch = GridDeepSet(num_layers=nl, emb_dim=emb_dim, activation=nonlinearity,
                                           kernel_size=decoder_kernel_size, **kwargs)
        elif ps == 'gnn':
            prior_and_encoder_arch = 'gnn' if prior_and_encoder_arch is None else prior_and_encoder_arch
            vae_encoder_net, vae_prior_net = get_vae_enc_and_prior_nets(prior_and_encoder_arch, emb_dim,
                                                                        num_prior_and_encoder_layers, kernel_size,
                                                                        nonlinearity,
                                                                        do_node_aggregation=do_node_aggregation,
                                                                        **kwargs)
            decoder_end_arch = LazyGNN(num_layers=3, emb_size=emb_dim, **kwargs)
        elif ps == 'eqgnn':
            prior_and_encoder_arch = 'gnn' if prior_and_encoder_arch is None else prior_and_encoder_arch
            vae_encoder_net, vae_prior_net = get_vae_enc_and_prior_nets(prior_and_encoder_arch, emb_dim,
                                                                        num_prior_and_encoder_layers, kernel_size,
                                                                        nonlinearity,
                                                                        do_node_aggregation=do_node_aggregation,
                                                                        **kwargs)
            decoder_end_arch = LazyEqGNN(num_layers=3, emb_size=emb_dim, update_coors=True, **kwargs)
        elif ps == 'fagnn':
            prior_and_encoder_arch = 'gnn' if prior_and_encoder_arch is None else prior_and_encoder_arch
            vae_encoder_net, vae_prior_net = get_vae_enc_and_prior_nets(prior_and_encoder_arch, emb_dim,
                                                                        num_prior_and_encoder_layers, kernel_size,
                                                                        nonlinearity,
                                                                        do_node_aggregation=do_node_aggregation,
                                                                        **kwargs)
            decoder_end_arch = FAGNN(num_layers=3, emb_size=emb_dim, activation=nonlinearity, **kwargs)
        elif ps == 'mlp':
            prior_and_encoder_arch = 'mlp' if prior_and_encoder_arch is None else prior_and_encoder_arch
            vae_encoder_net, vae_prior_net = get_vae_enc_and_prior_nets(prior_and_encoder_arch, emb_dim,
                                                                        num_prior_and_encoder_layers, kernel_size,
                                                                        nonlinearity,
                                                                        do_node_aggregation=do_node_aggregation,
                                                                        **kwargs)
            decoder_end_arch = LazyMLP(num_layers=3, emb_size=emb_dim, activation=nonlinearity, **kwargs)
        else:
            raise NotImplementedError('processor string not recognized in determining the right VAE decoder!')

        return Probabilistic_Decoder(num_layers, emb_dim, dynamic_channels, self.static_channels,
                                     num_latent_dim, decoder_kernel_size, vae_encoder_net,
                                     vae_prior_net, end_arch=decoder_end_arch, domain_str=self.proc.domain_str,
                                     free_bits=free_bits)

class Encoder(nn.Module):

    def __init__(self, emb_dim=16, domain_str='grid'):
        super().__init__()
        self.domain_str = domain_str
        if domain_str == 'grid':
            self.arch = nn.Sequential(nn.LazyConv2d(emb_dim, (1, 1)))
        elif domain_str == 'graph':
            self.arch = nn.Sequential(nn.LazyLinear(emb_dim))
        else:
            raise NotImplementedError('domain string not recognized!')

    def forward(self, x):
        if isinstance(x, tuple):
            # x can be a tuple, because we could also pass edge feats or edge index
            return self.arch(x[0]), *x[1:]
        return self.arch(x)


class Probabilistic_Decoder(nn.Module):

    def __init__(self, num_layers, emb_dim, dynamic_channels, static_channels, num_latent_dims, kernel_size,
                 encoder_net: nn.Module, prior_net: nn.Module, end_arch:nn.Module, domain_str, free_bits):
        super().__init__()
        self.num_layers = num_layers
        self.num_latent_dims = num_latent_dims
        self.encoder_net = encoder_net
        self.prior_net = prior_net
        self.end_arch = end_arch

        self.dynamic_channels = dynamic_channels

        self.emb_dim = emb_dim

        self.enc_mu = nn.LazyLinear(num_latent_dims)
        self.enc_log_sigma = nn.LazyLinear(num_latent_dims)

        self.prior_mu = nn.LazyLinear(num_latent_dims)
        self.prior_log_sigma = nn.LazyLinear(num_latent_dims)
        self.domain_str = domain_str
        self.free_bits = free_bits

        # Note: we move one final layer to go from the emb dim to the data dim to the domain-specific wrappers that
        # call Enc-Proc-Dec for more flexibility

    def forward(self, tup_of_e_and_x_true):

        e, x_true = tup_of_e_and_x_true

        mu_prior, log_sigma_prior = self.vae_prior(e)
        prior_dist = torch.distributions.Normal(mu_prior, nn.Softplus()(log_sigma_prior) + EPS)
        kl_loss = torch.tensor(0)

        if x_true is not None:
            # we are in training mode and have the ground-truth available -> sample from approx. posterior
            mu_enc, log_sigma_enc = self.vae_encoder(e, x_true)  # (bs, num_latent_dim)
            enc_dist = torch.distributions.Normal(mu_enc, nn.Softplus()(log_sigma_enc) + EPS)
            kl_loss = self.kl_loss(enc_dist, prior_dist)
        else:
            # we are in sampling mode and don't have the approx posterior -> sample from prior
            enc_dist = prior_dist

        z = self.reparameterization(enc_dist)

        bs = z.shape[0]

        # case distinction between domains in {grid, graph} and data as {single tensor, tuple}
        if self.domain_str == 'grid':
            if isinstance(e, tuple):  # griddeepset model
                # concat the node embeddings, leave edge and batch idx untouched - note z is already broadcasted
                # back to the nodes here in case of agg. z, ony spatial broadcasting remains. so we have (bs*nodes, c)
                bs = (torch.max(e[2]) + 1)
                z = z.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, e[0].shape[-2],
                                                         e[0].shape[-1])  # unsqueeze twice for two spatial dims
                final_emb = (torch.cat([e[0], z], dim=1), *e[1:])
                avg_num_cells_per_batch_el = e[0].shape[0] / bs  #(total cells in batch) / max(batch_idx) == bs
                kl_loss = kl_loss / avg_num_cells_per_batch_el  # rescale to make the KL loss invariant to the number of cells
            else:  # just a single grid as data, e.g. modern unet applied to cell dynamics. no edges or other fancy stuff
                z = z.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, e.shape[-2],
                                                         e.shape[-1])
                final_emb = torch.cat([e, z], dim=1)
            out = self.end_arch(final_emb)

        elif self.domain_str == 'graph':  # graph domain
            if isinstance(e, tuple):  # we have a graph with edge attr and possibly other info like x and v separate
                if len(z.shape) != len(e[0].shape):
                    # we aggregated the z node and should broadcast it back to the entire graph
                    z = z.unsqueeze(1).repeat(1, e[0].shape[1], 1)  # (bs, num_nodes, latent_dim)
                node_emb = e[0]  # discard edges
                final_emb = torch.cat([node_emb, z], dim=2)
                final_input = (final_emb, *e[1:])  # concat edges again (and x and v or other remainder when applicable)
                out = self.end_arch(final_input)
            else:
                final_input = torch.cat([e, z], dim=-1)  # no edges or edge attr, only nodes. for example in case of MLP
                out = self.end_arch(final_input)
        else:
            raise NotImplementedError(f'domain string not recognized, got {self.domain_str}')

        # apply the free bits objective -- note, we don't clamp but rather detach the KL loss for dimensions where the avg KL is already low
        # this is exactly equivalent as clamping for the gradient flow but preserves the KL values
        kl_avgd_over_batch = torch.sum(kl_loss, dim=0, keepdim=True) / bs
        low_kl_idx = kl_avgd_over_batch < self.free_bits

        # Don't penalize the KL loss for dimensions where the avg KL for that dim is already very low (implemented by detaching from the compute graph at those dimensions)
        kl_loss_with_detach = torch.where(low_kl_idx, kl_loss.detach(), kl_loss)
        additional_loss = torch.sum(kl_loss_with_detach)

        return out, additional_loss

    def vae_encoder(self, x, x_true):
        if self.domain_str == 'grid':  # (bs, c, h, w) or (bs*cells, c, h, w) for gridgnn/griddeepset
            if isinstance(x, tuple):  # gridgnn/griddeepset
                # keep edge and batch idx of the input -- batch idx should match and we can only have one set of edges
                # note for griddeepset we dont have edges anyway, but for gridgnn we do
                x = (torch.cat([x[0], x_true[0]], dim=1), *x[1:])
            else:
                x = torch.cat([x, x_true], dim=1)  # only grid tensor, e.g. modern unet
        elif self.domain_str == 'graph':  # (bs, nodes, c)
            # x is a tuple of (nodes, ...), same for x_true
            if isinstance(x, tuple):
                x = (torch.cat([x[0], x_true[0]], dim=-1), torch.cat([x[1], x_true[1]], dim=-1)) # concat node and edge feat -- only the E(n) invariant parts
            else:
                x = torch.cat([x, x_true], dim=-1)
        enc = self.encoder_net(x)  # (bs, latent)
        mu = self.enc_mu(enc)
        log_sigma = self.enc_log_sigma(enc)
        return mu, log_sigma

    def vae_prior(self, x):
        # x is tup of (node, edge, ...) -> ... is possibly x and v in de eqgnn/fagnn case
        # we want z to be invariant to E(n) transformations
        if isinstance(x, tuple) and self.domain_str == 'graph':  #for equivariant gnn, we want z to be invariant to E(3), so drop x and v
            x = x[:2]
        p = self.prior_net(x)
        mu = self.prior_mu(p)
        log_sigma = self.prior_log_sigma(p)
        return mu, log_sigma

    def reparameterization(self, dist) -> torch.Tensor:
        z = dist.rsample()
        # z = dist.mean + torch.randn_like(dist.mean) * (dist.scale) * 1.5
        return z

    def kl_loss(self, enc_dist: torch.distributions.Distribution, prior_dist: torch.distributions.Distribution):
        return torch.distributions.kl_divergence(enc_dist, prior_dist)

class Processor(nn.Module):

    def __init__(self, num_layers, emb_dim, kernel_size, nonlinearity=nn.ReLU(), processor_type_str='unet',
                 channels_cat_to_output=None, **kwargs):
        super().__init__()

        self.channels_cat_to_output = channels_cat_to_output

        modules = self._get_module_list_for_processor(processor_type_str.lower(), num_layers, emb_dim, kernel_size, nonlinearity,
                                                     **kwargs)

        self.arch = nn.Sequential(*modules)

    def forward(self, e, x):
        resid = self.arch(e)
        if self.domain_str == 'graph':  # first element of tuple is node emb
            if isinstance(e, tuple):
                # e[0]/x[0] because we give as input both node and edge feat as a tuple!
                out = resid[0] + e[0]
                if self.channels_cat_to_output is not None:  # we have a conditioning signal that remains static over time
                    out = torch.cat([x[0][..., self.channels_cat_to_output], out], dim=2)
                out = (out, *resid[1:])  # again add the edge features (and possibly x and v, edge_index, batch_index ...) for input to the decoder
            else:
                # only node info, no edges or other stuff
                out = resid + e
                if self.channels_cat_to_output is not None:  # we have a conditioning signal that remains static over time
                    out = torch.cat([x[..., self.channels_cat_to_output], out], dim=2)
        else:
            if isinstance(e, tuple):  # griddeepset/gridgnn
                out = resid[0] + e[0]
                if self.channels_cat_to_output is not None:  # a static conditioning signal that we add again for the decoder
                    out = torch.cat([x[0][:, self.channels_cat_to_output], out], dim=1)
                out = (out, *resid[1:])
            else:  # only a single tensor
                out = resid + e
                if self.channels_cat_to_output is not None:  # a static conditioning signal that we add again for the decoder
                    out = torch.cat([x[:, self.channels_cat_to_output], out], dim=1)
        return out

    def _get_module_list_for_processor(self, processor_type_str, num_layers, emb_dim, kernel_size, nonlinearity, **kwargs):
        # take care of some logistics of defining the arch for the processor here.
        # Sorry, we sort of hardcoded some design choices here for some of the architectures, esp. unets,
        # which is not great but it made our lives a bit easier.
        modules = []
        if processor_type_str.lower() == 'unet':
            self.domain_str = 'grid'
            # define the #channels for the unet layers:
            enc_chs = [emb_dim, emb_dim]
            for _ in range(num_layers - 1):
                chs_to_append = enc_chs[-1] * 2 if enc_chs[-1] < 1024 else enc_chs[-1]  # hard cap on max enc chs
                enc_chs.append(chs_to_append)
            dec_chs = enc_chs[::-1]
            dec_chs = dec_chs[:-1]

            modules.append(UNet(enc_chs, dec_chs, kernel_size, **kwargs))

        elif processor_type_str.lower() == 'unet_modern':
            self.domain_str = 'grid'
            # note: hyperparams inspired by https://github.com/microsoft/pdearena/blob/db7664bb8ba1fe6ec3217e4079979a5e4f800151/pdearena/modules/conditioned/twod_unet.py
            # define the number of channels for the unet layers
            ch_mults = []
            for i in range(num_layers):
                mult_to_append = 2 if 2 ** (
                            i + 1) * emb_dim < 1024 else 1  # prevent blowing up of the unet hidden dim, might change later
                ch_mults.append(mult_to_append)

            unet_modern = UNetModern(num_spatial_dims=2, hidden_features=emb_dim, activation=nonlinearity, norm=True,
                                     ch_mults=ch_mults,
                                     is_attn=[False for _ in range(num_layers)],
                                     mid_attn=True, n_blocks=2, use1x1=True, **kwargs)
            modules.append(unet_modern)
        elif processor_type_str.lower() == 'convnet':
            self.domain_str = 'grid'
            print("using simple conv processor")
            for l in range(num_layers):
                modules.append(nn.Conv2d(emb_dim, emb_dim, (kernel_size, kernel_size), padding='same'))
                modules.append(nonlinearity)
        elif processor_type_str.lower() == 'gridgnn':
            # griddeepset variant with edges, did some preliminary experiments but did not test if this still works
            warnings.warn('!!! PLEASE NOTE: gridgnn has not been kept up to date, this might not work as expected! Consider using griddeepset instead')
            self.domain_str = 'grid'
            modules.append(
                GridGNN(num_layers=num_layers, emb_dim=emb_dim, activation=nonlinearity, kernel_size=kernel_size,
                        **kwargs))
        elif processor_type_str.lower() == 'griddeepset':
            self.domain_str = 'grid'
            modules.append(
                GridDeepSet(num_layers=num_layers, emb_dim=emb_dim, kernel_size=kernel_size, activation=nonlinearity,
                            **kwargs))
        elif processor_type_str.lower() == 'gnn':
            self.domain_str = 'graph'
            modules.append(LazyGNN(num_layers, emb_dim, nonlinearity, **kwargs))
        elif processor_type_str.lower() == 'eqgnn':
            # eqgnn variant with edge features, did some preliminary experiments but did not test if this still works
            warnings.warn('!!! PLEASE NOTE: eqgnn has not been kept up to date, this might not work as expected! Consider using gnn instead')
            self.domain_str = 'graph'
            modules.append(LazyEqGNN(num_layers, emb_dim, nonlinearity, update_coors=True, **kwargs))
        elif processor_type_str.lower() == 'fagnn':
            self.domain_str = 'graph'
            modules.append(FAGNN(num_layers, emb_dim, nonlinearity, update_coors=False, **kwargs))
        elif processor_type_str.lower() == 'mlp':
            self.domain_str = 'graph'  # falls under graph but no real graph structure
            modules.append(LazyMLP(num_layers=num_layers, emb_size=emb_dim, activation=nonlinearity))
        else:
            raise NotImplementedError('processor type str not recognized!')

        return modules


def get_vae_enc_and_prior_nets(arch, emb_dim, num_layers, kernel_size=3, nonlinearity=nn.ReLU(), agg_grid_map_to_dim=16,
                               do_node_aggregation=None, **kwargs):
    # method for getting the required prior and encoder nets
    # here we also hardcoded some design choices and the code below could use some software engineering improvements...

    # agg_grid_map_to_dim is the dimension to which we map the grid nodes before aggregating them
    # in case of griddeepset/gridgnn.

    if arch == 'conv':
        if do_node_aggregation is None:
            do_node_aggregation = False
        # apparently, we are working on a grid domain, or something went terribly wrong!
        encoder_net_layerlist = []
        prior_net_layerlist = []

        # # if we are dealing with tuples (cellsort with gridgnn) select only the first (=node) elements
        encoder_net_layerlist.append(Lambda(lambda x: x[0] if isinstance(x, tuple) else x))
        prior_net_layerlist.append(Lambda(lambda x: x[0] if isinstance(x, tuple) else x))

        #  first layer has to have the right input emb_dim. for encoder, emb and ground-truth dynamic channels
        encoder_net_layerlist.append(nn.LazyConv2d(emb_dim, kernel_size))
        # sum of the static channels (which remain constant and are given as input to the decoder)
        # and the dynamic channels (at time t+1, also given to the decoder)

        encoder_net_layerlist.append(nonlinearity)
        # for prior, only the emb of the enc-proc-dec encoder
        prior_net_layerlist.append(nn.LazyConv2d(emb_dim, kernel_size))

        prior_net_layerlist.append(nonlinearity)

        for _ in range(num_layers - 1):  # hidden layers
            encoder_net_layerlist.append(nn.Conv2d(emb_dim, emb_dim, kernel_size))
            encoder_net_layerlist.append(nonlinearity)
            encoder_net_layerlist.append(nn.MaxPool2d(2))

            prior_net_layerlist.append(nn.Conv2d(emb_dim, emb_dim, kernel_size))
            prior_net_layerlist.append((nonlinearity))
            prior_net_layerlist.append(nn.MaxPool2d(2))

        # perform node aggregation if specified:
        if do_node_aggregation:
            enc_and_agg = nn.Sequential(
                PassAlongWithLayer(nn.Sequential(*encoder_net_layerlist)),
                AggregateGridNodes(map_to_dim=agg_grid_map_to_dim, activation=nonlinearity)
            )
            prior_and_agg = nn.Sequential(
                PassAlongWithLayer(nn.Sequential(*prior_net_layerlist)),
                AggregateGridNodes(map_to_dim=agg_grid_map_to_dim, activation=nonlinearity)
            )
            encoder_net_layerlist = [enc_and_agg]
            prior_net_layerlist = [prior_and_agg]
        else:  # only cell-wise spatial aggregation
            encoder_net_layerlist.append(Lambda(
                lambda x: torch.mean(x, dim=(2, 3)) if not isinstance(x, tuple) else torch.mean(x[0], dim=(2, 3))
            ))  # global mean-pooling over spatial dims
            prior_net_layerlist.append(Lambda(
                lambda x: torch.mean(x, dim=(2, 3)) if not isinstance(x, tuple) else torch.mean(x[0], dim=(2, 3))
            ))

        # finally
        encoder_net = nn.Sequential(*encoder_net_layerlist,
                                    nn.LazyLinear(emb_dim), nonlinearity)
        prior_net = nn.Sequential(*prior_net_layerlist,
                                  nn.LazyLinear(emb_dim), nonlinearity)
    elif arch == 'gnn':
        if do_node_aggregation is None:
            do_node_aggregation = True
        encoder_net = nn.Sequential(LazyGNN(num_layers, emb_dim, nonlinearity, **kwargs),
                                    AggregateNodes() if do_node_aggregation else Lambda(lambda x: x[0]) # drop edges if no aggregation
                                    )
        prior_net = nn.Sequential(LazyGNN(num_layers, emb_dim, nonlinearity, **kwargs),
                                  AggregateNodes() if do_node_aggregation else Lambda(lambda x: x[0]) # drop edges if no aggregation
                                  )
    elif arch == 'gridgnn':
        encoder_net = nn.Sequential(GridGNN(num_layers, emb_dim, kernel_size, nonlinearity, **kwargs),
                                    AggregateGridNodes(map_to_dim=agg_grid_map_to_dim, activation=nonlinearity))
        prior_net = nn.Sequential(GridGNN(num_layers, emb_dim, kernel_size, nonlinearity, **kwargs),
                                  AggregateGridNodes(map_to_dim=agg_grid_map_to_dim, activation=nonlinearity))
    elif arch == 'griddeepset':
        encoder_net = nn.Sequential(GridDeepSet(num_layers, emb_dim, kernel_size, nonlinearity, **kwargs),
                                    AggregateGridNodes(map_to_dim=agg_grid_map_to_dim, activation=nonlinearity))
        prior_net = nn.Sequential(GridDeepSet(num_layers, emb_dim, kernel_size, nonlinearity, **kwargs),
                                  AggregateGridNodes(map_to_dim=agg_grid_map_to_dim, activation=nonlinearity))
    elif arch == 'mlp':
        encoder_net = LazyMLP(num_layers=num_layers, emb_size=emb_dim, activation=nonlinearity)
        prior_net = LazyMLP(num_layers=num_layers, emb_size=emb_dim, activation=nonlinearity)
    else:
        raise NotImplementedError('VAE prior/encoder net arch string not recognized!')

    return encoder_net, prior_net

def index_channels_for_domain(x: torch.Tensor, channel_index, domain_str: str):
    if domain_str == 'grid':
        if isinstance(x, tuple):
            return x[0][:, channel_index], *x[1:]
        else:
            return x[:, channel_index]
    elif domain_str == 'graph':
        if isinstance(x, tuple):
            return x[0][..., channel_index], *x[1:]
        else:
            return x[..., channel_index]
    else:
        raise NotImplementedError('domain string not recognized!')


class Lambda(nn.Module):

    def __init__(self, lambd: callable):
        super().__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)



class PassAlongWithLayer(nn.Module):

    # apply a layer to the first element of a tuple and pass along the remaining elements

    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    def forward(self, t):
        if isinstance(t, tuple):
            return (self.layer(t[0]), *t[1:]) if len(t) > 1 else (self.layer(t[0]), )
        else:
            return self.layer(t)



