import math

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import utils

class MLPLayer(nn.Module):

    def __init__(self, emb_size, activation, residual=False, lazy=False):
        super().__init__()
        self.emb_size = emb_size
        if not lazy:
            self.layer = nn.Sequential(nn.Linear(emb_size, emb_size), activation)
        else:
            self.layer = nn.Sequential(nn.LazyLinear(emb_size), activation)
        self.residual = residual

    def forward(self, x):
        out = self.layer(x)
        if self.residual:
            out = x + out
        return out

class LazyMLP(nn.Module):

    def __init__(self, num_layers, emb_size=32, activation=nn.ReLU(), residual=False, **kwargs):
        super().__init__()
        self.num_layers = num_layers
        self.emb_size = emb_size
        self.activation = activation
        self.residual = residual
        self.net = nn.Sequential(MLPLayer(emb_size, activation, residual, lazy=True),
                                          *[MLPLayer(emb_size, activation, residual, lazy=False) for _ in range(num_layers-1)])



    def forward(self, x):
        # x is expected to have shape (bs, 1, feat)
        assert len(x.shape) == 3, 'expected x to have shape (bs,1, feat)'
        x_sq = x[:,0,:]  #(bs, feat)
        out_sq = self.net(x_sq)
        out = out_sq.unsqueeze(1) # (bs, 1, feat)
        return out


class GNNLayer(nn.Module):  # one graph mp layer
    # -- NOTE: THIS ASSUMES A FULLY CONNECTED GRAPH WITH NODE INPUT SHAPE (bs, num_nodes, features)
    # extension to varying topologies can be done but here we assumed a fully connected graph for n-body

    def __init__(self, emb_dim=32, hidden_dim=32, n_edge_features=1, activation=nn.ReLU(), residual=False):
        super().__init__()

        self.residual = residual

        # edge network: takes in 2 node embeddings (dimension of each node is emb_dim) and edge attribute,
        # and outputs an edge message (dimension hidden_dim)
        self.edge_network = nn.Sequential(nn.Linear(2*emb_dim + n_edge_features, hidden_dim),
                                          activation,
                                          nn.Linear(hidden_dim, hidden_dim),
                                          activation
                                          )

        # node network: takes in the old node embedding (dim emb_dim) and messages (dim hidden_dim), returns emb_dim
        self.node_network = nn.Sequential(nn.Linear(emb_dim+hidden_dim, hidden_dim),
                                          activation,
                                          nn.Linear(hidden_dim, emb_dim),
                                          )

    def forward(self, node_emb_edge_feat_tup):
        # dim node embeddings tensor should be (batch_size, num_nodes, emb_dim)
        # dim edge_features tensor should be (batch_size, num_nodes, num_nodes, edge_features)

        # we want the following shape as input to our edge network:
        # (batch_size, num_nodes, num_nodes, edge_features +2*emb_dim). The mlps of the edge model will operate along
        # the last axis
        node_embeddings_in, edge_features = node_emb_edge_feat_tup
        num_nodes = node_embeddings_in.size(1)
        node_embeddings = torch.unsqueeze(node_embeddings_in, 1)
        node_embeddings = node_embeddings.repeat(1, num_nodes, 1, 1) # shape (batch_size, num_nodes, num_nodes, emb)

        # now concatenate the node embeddings with itself but with dim 1 and 2 transposed
        # (concatenation along the embedding dimension)
        node_embeddings_concatenated = torch.concat([node_embeddings, torch.transpose(node_embeddings, 1, 2)], dim=3)

        #finally, add the edge features:
        edge_message_input = torch.concat([edge_features, node_embeddings_concatenated], dim=3)

        edge_message_output = self.edge_network(edge_message_input)  # dim(bs, num_nodes, num_nodes, hidden_dim)
        aggregated_messages = torch.sum(edge_message_output, dim=2)  # dim(bs, num_nodes, hidden_dim)
        # aggregated_messages = aggregated_messages / aggregated_messages.shape[1] * 5  # renormalize to scale of 5 bodies
        new_node_emb = self.node_network(torch.cat([node_embeddings_in, aggregated_messages], dim=2))  # dim (bs, num_nodes, emb)
        if self.residual:
            new_node_emb = new_node_emb + node_embeddings_in
        return new_node_emb, edge_features

class LazyGNNLayer(GNNLayer):  # one lazy graph mp layer
    # -- NOTE: THIS ASSUMES A FULLY CONNECTED GRAPH WITH NODE INPUT SHAPE (bs, num_nodes, features)
    # extension to varying topologies can be done but here we assumed a fully connected graph for n-body
    def __init__(self, emb_dim=32, activation=nn.ReLU(), residual=False):
        super().__init__()

        self.residual = residual

        # edge network: takes in 2 node embeddings (dimension of each node is emb_dim) and edge attribute,
        # and outputs an edge message (dimension hidden_dim)
        self.edge_network = nn.Sequential(nn.LazyLinear(emb_dim),
                                          activation,
                                          nn.Linear(emb_dim, emb_dim),
                                          activation
                                          )

        # node network: takes in the old node embedding (dim emb_dim) and messages (dim hidden_dim), returns emb_dim
        self.node_network = nn.Sequential(nn.LazyLinear(emb_dim),
                                          activation,
                                          nn.Linear(emb_dim, emb_dim),
                                          )



class GNN(nn.Module):  #a message passing GNN, assumes fully connected graph, see GNNLayer class for details

    def __init__(self, num_layers, edge_feat_size=1, emb_size=32,
                 hidden_size=32, activation=nn.ReLU(), residual=False, ignore_x_and_v=True):
        super(GNN, self).__init__()
        self.ignore_x_and_v = ignore_x_and_v

        self.forward_network = nn.Sequential(*[   # stack a number of graph convolutional layers
            GNNLayer(emb_size, hidden_size, edge_feat_size, activation, residual) for _ in range(num_layers)
        ])


    def forward(self, tup_of_node_and_edge_feat):
        # nodes is expected to have shape (bs, num_nodes, node_feat_size)
        if self.ignore_x_and_v:
            tup_of_node_and_edge_feat = tup_of_node_and_edge_feat[2:]  # first two elements in tuple are x and v node atts
            assert len(tup_of_node_and_edge_feat) == 2, 'expected node feat tensor and edge feat tensor'
        emb, edge_feat = self.forward_network(tup_of_node_and_edge_feat)
        return emb

class LazyGNN(nn.Module):  #a lazy message passing GNN, assumes fully connected graph, see LazyGNNLayer class for details

    def __init__(self, num_layers, emb_size=32, activation=nn.ReLU(), residual=False, **kwargs):
        super(LazyGNN, self).__init__()


        self.forward_network = nn.Sequential(*[   # stack a number of graph convolutional layers
            LazyGNNLayer(emb_size, activation, residual) for _ in range(num_layers)
        ])



    def forward(self, tup_of_node_and_edge_feat):
        # nodes is expected to have shape (bs, num_nodes, node_feat_size)

        emb, edge_feat = self.forward_network(tup_of_node_and_edge_feat)

        return emb, edge_feat


class EqGNNLayer(nn.Module):  # one equivariant graph convolutional layer
    #NOTE: THIS ASSUMES A FULLY CONNECTED GRAPH!

    def __init__(self, emb_dim=32, hidden_dim=32, n_edge_features=1, activation=nn.ReLU(), residual=False,
                 normalize_intermediate_dists=False, clamp_spatial_updates_to=torch.inf, use_tanh_for_spatial=False):
        super().__init__()

        self.residual = residual
        self.normalize_intermediate_dists = normalize_intermediate_dists
        self.clamp_update_value = torch.abs(torch.Tensor([clamp_spatial_updates_to]))

        # edge network: takes in 2 node embeddings (dimension of each node is emb_dim) and edge attribute,
        # and outputs an edge message (dimension hidden_dim)
        self.edge_network = nn.Sequential(nn.Linear(2*emb_dim + n_edge_features + 1, hidden_dim), # plus 1 for dist
                                          activation,
                                          nn.Linear(hidden_dim, hidden_dim),
                                          activation
                                          )

        # node network: takes in the old node embedding (dim emb_dim) and messages (dim hidden_dim), returns emb_dim
        self.node_network = nn.Sequential(nn.Linear(emb_dim+hidden_dim, hidden_dim),
                                          activation,
                                          nn.Linear(hidden_dim, emb_dim),
                                          )

        self.vel_coef_network = nn.Sequential(nn.Linear(emb_dim, hidden_dim),
                                         activation,
                                         nn.Linear(hidden_dim, 1),
                                              #nn.Softplus()
                                              )  # velocity scaling coefficient

        self.vel_network = nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
                                         activation,
                                         nn.Linear(hidden_dim, 1),
                                         nn.Tanh() if use_tanh_for_spatial else nn.Identity(),
                                         )  # forcing weighting coefficient

        self.pos_network = nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
                                         activation,
                                         nn.Linear(hidden_dim, 1),
                                         nn.Tanh() if use_tanh_for_spatial else nn.Identity()
                                         )  # residual pos update weighting coefficient



    def forward(self, feat_tup):
        EPS = 1e-6
        normalize_diff_lower_bound = 1e-3
        # dim node embeddings tensor should be (batch_size, num_nodes, emb_dim)
        # dim edge_features tensor should be (batch_size, num_nodes, num_nodes, edge_features)

        # we want the following shape as input to our edge network:
        # (batch_size, num_nodes, num_nodes, edge_features +2*emb_dim). The mlps of the edge model will operate along
        # the last axis
        node_embeddings_in, edge_features,x_old, v_old = feat_tup
        num_nodes = node_embeddings_in.size(1)

        x_old_unsq = torch.unsqueeze(x_old, 1)
        x_old_unsq = x_old_unsq.repeat(1, num_nodes, 1, 1)  # shape (bs, num_nodes, num_nodes, n_dim)
        x_old_diffs = x_old_unsq - torch.transpose(x_old_unsq,1,2)
        x_old_dists = torch.sum(x_old_diffs**2, 3, keepdim=True)  # actually squared dists (bs, num_nodes, num_nodes, 1)

        diagonal_mask = torch.eye(x_old_dists.shape[1], x_old_dists.shape[2]).repeat(
            x_old_dists.shape[0], 1, 1).unsqueeze(-1).to(x_old_dists.device)
        x_old_dists = torch.sqrt(
            x_old_dists + diagonal_mask * EPS)  # the EPS here prevents NaN gradient error for 0 under sqrt
        x_old_dists = x_old_dists * (1 - diagonal_mask)  # set diagonal back to 0


        #ordinary message passing layer:
        node_embeddings = torch.unsqueeze(node_embeddings_in, 1)
        node_embeddings = node_embeddings.repeat(1, num_nodes, 1, 1)  # shape (batch_size, num_nodes, num_nodes, emb)

        # now concatenate the node embeddings with itself but with dim 1 and 2 transposed
        # (concatenation along the embedding dimension)
        node_embeddings_concatenated = torch.concat([node_embeddings, torch.transpose(node_embeddings, 1, 2)], dim=3)

        # finally, add the edge features:
        edge_message_input = torch.concat([edge_features, x_old_dists, node_embeddings_concatenated], dim=3)

        # edge transfer function:
        edge_message_output = self.edge_network(edge_message_input)  # dim(bs, num_nodes, num_nodes, hidden_dim)

        # node aggregation and update
        aggregated_messages = torch.sum(edge_message_output, dim=2)  # dim(bs, num_nodes, hidden_dim)
        new_node_emb = self.node_network(torch.cat([node_embeddings_in, aggregated_messages], dim=2))  # dim (bs, num_nodes, emb)
        if self.residual:
            new_node_emb = new_node_emb + node_embeddings_in

        # now, update vel and pos vectors...
        # shape of x_old and v_old: (bs, num_nodes, n_dim)
        v_coef = self.vel_coef_network(new_node_emb)

        if self.normalize_intermediate_dists:
            x_old_diffs = x_old_diffs / torch.clamp(x_old_dists, min=normalize_diff_lower_bound)

        vel_update_coefs = torch.clamp(self.vel_network(edge_message_output), min=-self.clamp_update_value, max=self.clamp_update_value)
        vel_update_all_neighbours = vel_update_coefs * x_old_diffs
        vel_update = torch.mean(vel_update_all_neighbours, dim=2)  # (bs, num_nodes, n_dim)

        x_resid_update_coefs = torch.clamp(self.pos_network(edge_message_output), min=-self.clamp_update_value, max=self.clamp_update_value)
        x_resid_update_all_neighbours = x_resid_update_coefs * x_old_diffs
        x_resid_update = torch.mean(x_resid_update_all_neighbours, dim=2)

        v_new = v_old + vel_update  # update the velocity
        # update x in two steps: first, add v_new multiplied by a learned constant. then, add a residual term to account
        # for the fact that v_new is also the input velocity for the next timestep
        x_new = x_old + v_coef * v_new + x_resid_update

        return new_node_emb, edge_features, x_new, v_new


class LazyEqGNNLayer(EqGNNLayer):
    # NOTE: THIS ASSUMES A FULLY CONNECTED GRAPH!
    def __init__(self, emb_dim=32, activation=nn.ReLU(), residual=False, normalize_intermediate_dists=False,
                 clamp_spatial_updates_to=torch.inf, use_tanh_for_spatial=False):
        super().__init__(residual=residual, normalize_intermediate_dists=normalize_intermediate_dists,
                 clamp_spatial_updates_to=clamp_spatial_updates_to, use_tanh_for_spatial=use_tanh_for_spatial)

        self.normalize_intermediate_dists = normalize_intermediate_dists
        self.clamp_update_value = clamp_spatial_updates_to

        self.residual = residual

        # edge network: takes in 2 node embeddings (dimension of each node is emb_dim) and edge attribute,
        # and outputs an edge message (dimension hidden_dim)
        self.edge_network = nn.Sequential(nn.LazyLinear(emb_dim),  # plus 1 for dist
                                          activation,
                                          nn.Linear(emb_dim, emb_dim),
                                          activation
                                          )

        # node network: takes in the old node embedding (dim emb_dim) and messages (dim hidden_dim), returns emb_dim
        self.node_network = nn.Sequential(nn.LazyLinear(emb_dim),
                                          activation,
                                          nn.Linear(emb_dim, emb_dim),
                                          )

        self.vel_coef_network = nn.Sequential(nn.LazyLinear(emb_dim),
                                              activation,
                                              nn.Linear(emb_dim, 1),
                                              #nn.Softplus()
                                              )  # velocity scaling coefficient

        self.vel_network = nn.Sequential(nn.LazyLinear(emb_dim),
                                         activation,
                                         nn.Linear(emb_dim, 1),
                                         nn.Tanh() if use_tanh_for_spatial else nn.Identity()
                                         )  # forcing weighting coefficient

        self.pos_network = nn.Sequential(nn.LazyLinear(emb_dim),
                                         activation,
                                         nn.Linear(emb_dim, 1),
                                         nn.Tanh() if use_tanh_for_spatial else nn.Identity()
                                         )  # residual pos update weighting coefficient


class LazyEqGNN(nn.Module):  # an equivariant message passing GNN
    #NOTE: THIS ASSUMES A FULLY CONNECTED GRAPH!

    def __init__(self, num_layers, emb_size=32, activation=nn.ReLU(), residual=False, normalize_intermediate_dists=False,
                 clamp_spatial_updates_to=torch.inf, weight_init_func=None, update_coors=True, update_only_last_v=False,
                 use_tanh_for_spatial=False, **kwargs):
        super(LazyEqGNN, self).__init__()


        self.forward_network = nn.Sequential(*[  # stack a number of graph convolutional layers
            LazyEqGNNLayer(emb_size, activation, residual, normalize_intermediate_dists, clamp_spatial_updates_to,
                           use_tanh_for_spatial)
            for _ in range(num_layers)
        ])

        self.update_coors = update_coors
        self.update_only_last_v = update_only_last_v

        self.weight_init_func = weight_init_func

        if weight_init_func is not None:
            def weight_init_(m):
                if isinstance(m, nn.Linear):
                    weight_init_func(m.weight.data)
                    weight_init_func(m.bias.data)
            self.apply(weight_init_)


    def forward(self, tup_of_node_and_edge_feat):
        # nodes is expected to have shape (bs, num_nodes, node_feat_size)

        node_features, edge_features, x, v = tup_of_node_and_edge_feat
        v_init = v
        for i, layer in enumerate(self.forward_network.children()):
            if self.update_coors:  # update x and v at each iteration
                if self.update_only_last_v:
                    node_features, edge_features, x, v = layer((node_features, edge_features, x, v_init))
                else:
                    node_features, edge_features, x, v = layer((node_features, edge_features, x, v))
            else:  # only update invariant features
                node_features, edge_features, _, _ = layer((node_features, edge_features, x, v))

        return node_features, edge_features, x, v


class FAGNN(nn.Module):
    #NOTE: THIS ASSUMES A FULLY CONNECTED GRAPH!

    # based on https://github.com/omri1348/Frame-Averaging/blob/master/nbody/n_body_system/model_FA.py

    def __init__(self, num_layers, emb_size=32, activation=nn.ReLU(), residual=False, update_coors=True, **kwargs):
        super(FAGNN, self).__init__()
        self.emb_size = emb_size
        self.num_layers = num_layers
        self.dimension_reduce = nn.ModuleList()

        self.backbone_gnn = LazyGNN(num_layers, emb_size, activation, residual, **kwargs)

        self.decoder = nn.Sequential(activation,  # map to x and v
                                     nn.LazyLinear(emb_size),
                                     activation,
                                     nn.LazyLinear(6))  if update_coors else None
        self.v_coef = nn.Sequential(activation,
                                    nn.LazyLinear(emb_size),
                                    activation,
                                    nn.LazyLinear(1)) if update_coors else None

        self.equivariance_tested = False
        self.update_coors = update_coors  # whether to update the coordinates or only the node embeddings

    def forward(self, tup_of_node_and_edge_feat):
        if not self.equivariance_tested:
            self.equivariance_tested = True
            self._check_equivariance(tup_of_node_and_edge_feat)  # small test to make sure we dont break something

        node_feat, edge_feat, x, v = tup_of_node_and_edge_feat  # NOTE: These edge_feat need to be invariant to E(3)
        n_frame = 8
        n_nodes = x.shape[1]
        spatial = torch.cat([x, v], dim=-1)


        h, F_ops, center = self.create_frame(spatial, n_nodes)
        h = torch.cat([h, node_feat.unsqueeze(0).repeat(h.shape[0], 1, 1, 1)], dim=-1)

        # initialize output object:
        h_out = torch.zeros(*h.shape[:-1], self.emb_size).to(h)
        for i in range(n_frame):
            # process each frame, bit of a mediocre solution to do this with a loop but I'd like to stick to the MPGNN implementation I had already :)
            h_out_i, edge_feat = self.backbone_gnn((h[i], edge_feat))
            h_out[i] = h_out[i] + h_out_i

        if self.update_coors:
            spatial_out = self.decoder(h_out)
            spatial_inverted = self.invert_frame(spatial_out, F_ops, center)
            delta_x_out = spatial_inverted[..., :3]
            delta_v_out = spatial_inverted[..., 3:]
            # E(3) invariant node embedding vector -> used in enc-proc-dec for inferring z and also for the log sigma of the node spatial
            h_out = h_out.mean(0)
            v_out = v + delta_v_out
            x_out = x + v_out * self.v_coef(h_out) + delta_x_out
        else:
            h_out = h_out.mean(0)
            x_out, v_out = x, v

        return h_out, edge_feat, x_out, v_out

    def create_frame(self, nodes, n_nodes):
        pos_idx = (0, 1, 2)
        vel_idx = (3, 4, 5)
        pnts = nodes[..., pos_idx]
        v = nodes[..., vel_idx]
        pnts = pnts.view(-1, n_nodes, 3).transpose(1, 2)
        v = v.view(-1, n_nodes, 3).transpose(1, 2)

        center = pnts.mean(2, True)
        pnts_centered = pnts - center
        lambdas, V_, R = self._get_pca_eig(pnts_centered)
        F = V_.to(R)

        ops = torch.tensor([[1, 1, 1],
                            [1, 1, -1],
                            [1, -1, 1],
                            [1, -1, -1],
                            [-1, 1, 1],
                            [-1, 1, -1],
                            [-1, -1, 1],
                            [-1, -1, -1]]).unsqueeze(1).to(F)
        F_ops = ops.unsqueeze(0) * F.unsqueeze(1)
        framed_input = torch.einsum('boij,bpj->bopi', F_ops.transpose(2, 3), (pnts - center).transpose(1, 2))
        framed_v = torch.einsum('boij,bpj->bopi', F_ops.transpose(2, 3), (v).transpose(1, 2))
        framed_input = framed_input.transpose(0, 1)
        # framed_input = torch.reshape(framed_input, (-1, 3))
        framed_v = framed_v.transpose(0, 1)
        # framed_v = torch.reshape(framed_v, (-1, 3))
        out = torch.cat([framed_input, framed_v], dim=-1)
        # self.highest_eig = max(self.highest_eig, lambdas.max())
        # self.lowest_diff = min(self.lowest_diff, torch.diff(lambdas).min())
        return out, F_ops.detach(), center.detach()

    def invert_frame(self, nodes, F_ops, center):
        # pnts = pnts.view(8, -1, n_nodes, 3)
        # pos idx and vel_idx are shifted by 1 because we don't output the mass as prediction
        pos_idx = (0, 1, 2)
        vel_idx = (3, 4, 5)
        pnts = nodes[..., pos_idx]
        v = nodes[..., vel_idx]
        pnts = pnts.transpose(0, 1)  #(frames, batch, ...) to (batch, frames, ...)
        v = v.transpose(0, 1)

        framed_input = torch.einsum('boij,bopj->bopi', F_ops, pnts)
        framed_input = framed_input.mean(1)
        # NOTE: we don't add the center back here because we are modeling in residual space
        # (so this is actually the \Delta x \approx v)!
        # since we are modeling as residual, the x+residual term already contains the center implicitly in x
        # if center is not None:
        #     framed_input = framed_input + center.transpose(1, 2)

        framed_v = torch.einsum('boij,bopj->bopi', F_ops, v)
        framed_v = framed_v.mean(1)

        return torch.cat([framed_input, framed_v], dim=-1)

    @torch.no_grad()
    def _check_equivariance(self, input_tup):
        return_to_float = False
        if list(self.parameters())[0].dtype is not torch.double:
            self.double()
            return_to_float = True
        rotation_matrices = utils.get_three_rotation_matrices()
        node_feat, edge_feat, x, v = input_tup
        shuffle_idx = utils.get_permutation_idx(num_elements=node_feat.shape[1])

        for mat in rotation_matrices:
            mat = mat.to(x)
            x = x @ mat
            v = v @ mat

        edge_feat_perm = torch.zeros_like(edge_feat)
        edge_feat_perm[:,:,:] = edge_feat[:,shuffle_idx, :]
        edge_feat_perm[:,:,:] = edge_feat_perm[:, :, shuffle_idx]
        input_tup_rot = node_feat[:, shuffle_idx], edge_feat_perm, x[:, shuffle_idx], v[:, shuffle_idx]
        output_tup = self.forward((e.double() for e in input_tup))
        output_tup_rot = self.forward((e.double() for e in input_tup_rot))
        node_feat, edge_feat, x_rot_after, v_rot_after = output_tup
        for mat in rotation_matrices:
            mat = mat.to(x_rot_after)
            x_rot_after = x_rot_after @ mat
            v_rot_after = v_rot_after @ mat

        eq_mse_x = torch.sum((x_rot_after[:, shuffle_idx] - output_tup_rot[2])**2)
        eq_mse_v = torch.sum((v_rot_after[:, shuffle_idx] - output_tup_rot[3])**2)
        inv_mse_nodes = torch.sum((node_feat[:, shuffle_idx] - output_tup_rot[0])**2)
        torch.testing.assert_close(x_rot_after[:, shuffle_idx], output_tup_rot[2], rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(v_rot_after[:, shuffle_idx], output_tup_rot[3], rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(node_feat[:, shuffle_idx], output_tup_rot[0], rtol=1e-6, atol=1e-6)
        print('equivariance check successful')
        if return_to_float:
            self.float()

    def _get_pca_eig(self, pnts_centered):

        try:  # try to get the eigenvectors of the PCA matrix of the points to project upon
            R = torch.bmm(pnts_centered, pnts_centered.transpose(1, 2))
            lambdas, V_ = torch.symeig(R.detach().cpu(), True)
            return lambdas, V_, R
        except Exception as e:  # torch._C._LinAlgError: symeig_cpu: (Batch element 8): The algorithm failed to converge because the input matrix is ill-conditioned or has too many repeated eigenvalues (error code: 2).
            # add some noise to prevent numerical stability issues and try again
            custom_exception = utils.PCAException('Found numerical instability in PCA!')
            custom_exception.pnts_centered = pnts_centered
            custom_exception.e_orig = e
            raise custom_exception



class AggregateNodes(nn.Module):

    def __init__(self, agg_fn=torch.mean):
        super().__init__()
        self.agg_fn = agg_fn

    def forward(self, tup_of_nodes_and_edges):
        node_feat = tup_of_nodes_and_edges[0]
        # node feat has shape (bs, nodes, node_feat_size)
        return self.agg_fn(node_feat, dim=1)


