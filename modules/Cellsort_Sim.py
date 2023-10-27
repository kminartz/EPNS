import random
import warnings
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.nn.functional import one_hot

import utils
from modules.engine.Enc_Proc_Dec import Enc_Proc_Dec

class Cellsort_Simulator(nn.Module):

    def __init__(self, num_layers, num_cell_types: int = 4, max_num_cells=50, im_dim=5, emb_dim=16, kernel_size=3, nonlinearity=nn.ReLU(),
                 dynamic_channels=None, processor='unet', decoder='vae_decoder',
                 decoder_kernel_size=25, num_latent_dim=0, edge_dist_thresh=0, pred_stepsize=1, global_downsampling_factor=1,
                 minimum_volume=None, EPS=1e-6, sample_from_pred=False, **kwargs):

        super(Cellsort_Simulator, self).__init__()

        self.num_cell_types = num_cell_types
        self.max_num_cells = max_num_cells  # max num cells in the system at any time
        self.dist_thresh = edge_dist_thresh  # only relevant for gridgnn, not for griddeepset (griddeepset = spatialconv-gnn)
        self.processor = processor.lower()
        self.engine = Enc_Proc_Dec(num_layers, im_dim, emb_dim, kernel_size, nonlinearity, dynamic_channels, processor,
                              decoder, decoder_kernel_size, num_latent_dim, global_downsampling_factor, **kwargs)

        all_channels = (i for i in range(im_dim))
        self.dynamic_channels = dynamic_channels
        self.static_channels = tuple(set(all_channels) - set(dynamic_channels))


        if self.processor == 'gridgnn' or self.processor == 'griddeepset':
            # process each node separately to preserve permutation symmetry
            self.final_layer = nn.Conv2d(emb_dim, 1, kernel_size=(1,1))
        else:
            self.final_layer = nn.Conv2d(emb_dim, max_num_cells, kernel_size=(1,1))  # process the entire grid at once

        self.EPS = EPS  # increase categorical probs by this amount before final normalization
        self.kwargs = kwargs
        self.sample_from_pred = sample_from_pred



    def forward(self, x, x_true, id_to_type_dict=None):

        # x is x^{t}, x_true is x^{t+1}
        # id_to_type_dict is a dict that maps the cell id to its corresponding type
        # it will be constructed in the first call of a rollout from the first datapoint, and then propagated along the
        # rollout by the rollout methods in the code.
        # If it is not given, it will simply be constructed from the current x, which might be suboptimal if a cell has
        # dissapeared and regains some volume, since we cannot infer the type then anymore

        if 'unet' in self.processor:
            cell_id_onehot = self.get_onehot_grid(x[:,0:1], num_classes=self.max_num_cells)
            cell_type_onehot = self.get_onehot_grid(x[:,1:2], num_classes=self.num_cell_types)
            cell_id_onehot_true = self.get_onehot_grid(x_true[:,0:1], num_classes=self.max_num_cells) if x_true is not None else None
            cell_type_onehot_true = self.get_onehot_grid(x_true[:,1:2], num_classes=self.num_cell_types) if x_true is not None else None

            model_input = torch.cat([cell_id_onehot, cell_type_onehot], dim=1)
            model_input_true = torch.cat([cell_id_onehot_true, cell_type_onehot_true], dim=1) if x_true is not None else None
            final_dec_emb, additional_loss = self.engine(model_input, model_input_true)
            out_dyn = self.final_layer(final_dec_emb)  #(bs, max_no_cells, h, w) -- logits along dim=1

        elif self.processor == 'gridgnn' or self.processor == 'griddeepset':

            if id_to_type_dict is not None:
                num_unique_cells_in_x_and_x_true = max(id_to_type_dict.keys()) + 1
            elif x_true is not None:
                num_unique_cells_in_x_and_x_true = int(torch.max(torch.cat([x[:, 0:1], x_true[:, 0:1]], dim=1)) + 1)
            else:
                num_unique_cells_in_x_and_x_true = -1  # let F.one_hot infer the number of cells
            num_cells = num_unique_cells_in_x_and_x_true
            cell_id_onehot = self.get_onehot_grid(x[:,0:1], num_classes=num_cells)  # (bs, max_num_cells, h, w)
            cell_type_onehot = self.get_onehot_grid(x[:,1:2], num_classes=self.num_cell_types)  # (bs, num_types, h, w)
            cell_id_onehot_true = self.get_onehot_grid(x_true[:,0:1], num_classes=num_cells) if x_true is not None else None
            cell_type_onehot_true = self.get_onehot_grid(x_true[:,1:2], num_classes=self.num_cell_types) if x_true is not None else None

            # convert the one_hot cell id tensor to nodes with grid features:
            cell_id_and_type_input, batch_idx = self.from_grid_to_nodes(cell_id_onehot, cell_type_onehot)  #(bs*num_cells, 1+num_types, h, w)
            cell_id_and_type_true, batch_idx_true = self.from_grid_to_nodes(cell_id_onehot_true, cell_type_onehot_true) if x_true is not None else (None, None)
            edge_index = self.get_edge_index(cell_id_and_type_input[:, 0], batch_idx, self.dist_thresh)  #cell id channel
            edge_index_true = self.get_edge_index(cell_id_and_type_true[:, 0], batch_idx_true, self.dist_thresh) if x_true is not None else None

            # give the nodes (cell grids), edge index and batch_index as input to the model:
            input_model = (cell_id_and_type_input, edge_index, batch_idx)
            ground_truth = (cell_id_and_type_true, edge_index_true, batch_idx_true) if x_true is not None else None
            final_emb, additional_loss = self.engine(input_model, ground_truth)  #(bs*num_cells, emb_dim, h, w)

            # for each cell, map the final embedding to the unnormalized log_probs of the cell being at that coordinate:
            logits_id_batched = self.final_layer(final_emb[0])  # (bs*num_cells, 1, h, w)

            # handle the logistics of putting everything back in the original batch-tensor format, and convert to probs:
            logits_id_unbatched = torch.zeros_like(cell_id_onehot)  # (bs, max_num_cells, h, w)
            for b in batch_idx.unique():
                logits_batch = logits_id_batched[batch_idx == b].squeeze()  #(num_cells, h, w)
                # probs_batch = self.final_activation(logits_batch.unsqueeze(0)).squeeze()  # take softmax over dim=1
                logits_id_unbatched[b, :logits_batch.shape[0]] = logits_id_unbatched[b, :logits_batch.shape[0]] + logits_batch

            out_dyn = logits_id_unbatched  # logits along dim=1
        else:
            raise NotImplementedError('this processor has not been implemented for the cellsort simulator!')

        pred, pred_disc, id_to_type_dict = self.postprocess_and_discretize(out_dyn, x, id_to_type_dict=id_to_type_dict)  # pred should have shape (bs, c, h, w) -- logits
        # shape of pred: (bs, 2, h, w, num_cells)
        # shape of pred_disc: (bs, 2, h, w)
        # assert pred.shape[-1] == 145, 'debugging'

        logits = pred
        # (bs, 2, h, w, num_cells) - for second channel, the last dim is the type logits.
        # We have types <= cells, so the last entries will likely be 0 in this channel

        # sometimes, we got some instability errors even when initializing the categorical distribution directly from
        # the unnormalized log probs. I think that taking the softmax to normalize to probs, and then adding a small
        # EPS should fix it as there is a lower bound on the prob for each cell ID (almost equal to EPS)
        EPS = self.EPS  # was: 1e-6 for old models!
        pred = torch.distributions.Categorical(probs=nn.Softmax(dim=-1)(logits) + EPS)

        if self.sample_from_pred:
            # replace pred disc of cell ids with sample from categorical dist:
            pred_disc = pred.sample()[:,0:1] # first channel is cell id channel
            # change discretized types to correspond to the actually sampled pixels
            type_disc = torch.zeros_like(pred_disc)
            for k, v in id_to_type_dict.items():
                type_disc += v.view(-1, 1, 1, 1) * (
                            pred_disc == k)  # wherever the cell equals a certain ID, the disc type is the type of that cell id
            pred_disc = torch.cat([pred_disc, type_disc], dim=1)  # shape (bs, 2, h, w)

        return pred, additional_loss, pred_disc, id_to_type_dict

    def get_edge_index(self, cell_id_input, batch_idx, dist_thresh=15):
        edge_index = []
        node_counter = 0
        for b in batch_idx.unique():
            one_batch = cell_id_input[batch_idx == b]
            edge_idx_within_batch = self.get_edge_index_for_one_batch(one_batch, dist_thresh=dist_thresh)
            edge_idx = edge_idx_within_batch + node_counter  # (2, n_edges)
            edge_index.append(edge_idx)
            node_counter += one_batch.shape[0]
        edge_index = torch.cat(edge_index, dim=-1)
        return edge_index

    def get_edge_index_for_one_batch(self, cell_id_onehot, dist_thresh=15):
        # shape (num_cells, h, w)
        h = cell_id_onehot.shape[-2]
        w = cell_id_onehot.shape[-1]
        h_range = torch.arange(0, h).view(1,-1,1).to(cell_id_onehot)
        w_range = torch.arange(0, w).view(1,1,-1).to(cell_id_onehot)
        # calc center of mass
        coms_h = torch.sum(cell_id_onehot * h_range, dim=(-2, -1)) / torch.sum(cell_id_onehot, dim=(-2, -1))  # (cells)
        coms_w = torch.sum(cell_id_onehot * w_range, dim=(-2, -1)) / torch.sum(cell_id_onehot, dim=(-2, -1))
        coms = torch.cat([coms_h.unsqueeze(-1), coms_w.unsqueeze(-1)], dim=-1)  #(cells, 2)
        coms_rep = coms.unsqueeze(0).repeat(cell_id_onehot.shape[0], 1, 1)  # (cells, cells, 2)
        # dist between cells
        dist_coms = torch.sqrt(torch.sum((coms_rep - coms_rep.transpose(0,1))**2, dim=-1)) # (cells, cells)
        adj_mat = dist_coms <= dist_thresh

        # do not build edges between cells and medium, set these values to 0
        medium_index = torch.argmax(torch.sum(cell_id_onehot, dim=(-2, -1)))  # medium is the 'cell' that occupies most space
        adj_mat[medium_index, :] = 0
        adj_mat[:, medium_index] = 0

        # do not build edges between cell channels that have zeros everywhere:
        allzero_idx = torch.all(torch.all(cell_id_onehot == 0, dim=2), dim=1)
        adj_mat[allzero_idx, :] = 0
        adj_mat[:, allzero_idx] = 0

        edge_index = torch.nonzero(adj_mat).transpose(0,1)  # (2, N)
        return edge_index

    def from_grid_to_nodes(self, cell_id_onehot, cell_type_onehot):
        num_cells = cell_id_onehot.shape[1]  # largest number of cells in this batch
        # assert num_cells == 145, 'debugging'
        batch_idx = torch.arange(cell_id_onehot.shape[0]).repeat_interleave(num_cells).to(cell_id_onehot.device)  # (bs*max_num_cells)
        cell_id_onehot = cell_id_onehot.flatten(start_dim=0, end_dim=1).unsqueeze(
            1)  # shape(bs * max_num_cells, 1, h, w)
        cell_type_onehot = cell_type_onehot.repeat_interleave(repeats=num_cells,
                                                              dim=0)  # (bs*max_num_cells, num_types, h, w)
        # remove possible nodes containing only zero grid values (i.e. these cells do not exist)

        # allzero = torch.all(cell_id_onehot.flatten(start_dim=1) == 0, dim=1)  # shape (bs * max_num_nodes)
        # batch_idx = batch_idx[~allzero]
        # cell_id_onehot = cell_id_onehot[~allzero]  # remove non-existing cells, add channel axis
        # cell_type_onehot = cell_type_onehot[~allzero]

        cell_id_type_input = torch.cat([cell_id_onehot, cell_type_onehot],
                                       dim=1)  # (bs * max_num_cells, 1+num_types, j, w)

        return cell_id_type_input, batch_idx

    @torch.no_grad()
    def get_id_to_type_dict(self, id_tensor, type_tensor):
        id_to_type_dict = {}
        for cell_id in torch.unique(id_tensor):
            cell_id = int(cell_id)
            type = torch.where(id_tensor == cell_id, type_tensor, torch.Tensor([0]).to(id_tensor))  # simply put type 0 for the type where this cell is not
            # get the type as a scalar
            unique_type = torch.max(type.flatten(start_dim=1), dim=1)[0]  # get max (is actual type or zero in case this cell id exists nowhere)
            assert torch.bitwise_or(type == unique_type.view(-1,1,1,1), type == 0).all(), 'should only find 0 and the type in type tensor!'
            id_to_type_dict[cell_id] = unique_type.long()
        return id_to_type_dict


    def get_type_tensor_from_id(self, id_tensor, id_to_type_dict):
        # id_tensor has shape (bs, 1, h, w, cell_id_onehot)
        EPS = 1e-6
        pred_type = torch.zeros(*id_tensor.shape).to(id_tensor)  # (bs,1,h,w,cell_type_onehot)
        probs_id_tensor = nn.Softmax(dim=-1)(id_tensor)
        for id in id_to_type_dict.keys():
            type = id_to_type_dict[id]
            pred_type[range(len(type)), ..., type] += probs_id_tensor[..., id] # add the probability of seeing this id here as this id is of this type
        logits_pred_type = torch.log(pred_type + EPS)
        return logits_pred_type  # contains logits

    def get_onehot_grid(self, tensor, num_classes=None):

        return utils.get_onehot_grid(tensor, num_classes)


    def get_additional_val_stats(self, pred_disc, x_true):
        correct = torch.sum(
            torch.mean(1.0 * (pred_disc == x_true), dim=[-2, -1])).item()
        return correct

    def postprocess_and_discretize(self, out, x, id_to_type_dict=None):
        # nothing fancy (yet), simply get the highest prob of the softmax and discretize
        if id_to_type_dict is None:
            id_to_type_dict = self.get_id_to_type_dict(x[:, 0:1], x[:, 1:2])
        # which cells were present in the model input?


        # cells_present should now have shape (bs, cell_ids_present_per_batch_element)
        # hardcode cells that are not present at the start to 0
        # we only add the one-hot probs for channels where cells were already present!
        pred = out

        # pred_disc = torch.max(pred, dim=1, keepdim=True)[1]  # highest logit for a cell
        pred_disc = self.discretize_cells(pred)
        # pred_disc has shape(bs, 1, h, w), where 1 is a single channel containing the cell ID. so this is not one-hot encoded anymore!
        # change pred shape to (bs, 1, h, w, one_hot_dim)
        pred_perm = torch.movedim(pred, source=(1, 2, 3), destination=(3, 1, 2)).unsqueeze(1)

        # decision: whether we only care about the cell id and infer the type post-hoc or optimize for the type as well
        # (for now we did the first, but we can do the second by removing the detach from the first arg of get_type_tensor_from_id)
        pred_type_logprobs = self.get_type_tensor_from_id(pred_perm.detach(),
                                                 id_to_type_dict)  # don't backpropagate through the type, optimize only on id

        pred = torch.cat([pred_perm, pred_type_logprobs], dim=1)  # shape(bs, 2, h, w, num_cells)
        type_disc = torch.zeros_like(pred_disc)
        for k, v in id_to_type_dict.items():
            type_disc += v.view(-1,1,1,1) * (pred_disc == k)  # wherever the cell equals a certain ID, the disc type is the type of that cell id
        pred_disc = torch.cat([pred_disc, type_disc], dim=1)  # shape (bs, 2, h, w)

        return pred, pred_disc, id_to_type_dict

    @torch.no_grad()
    def discretize_cells(self, pred,):
        return torch.max(pred, dim=1, keepdim=True)[1]  # simply take Maximmum likelihood sample per pixel



# class OneHotCategoricalImageDistribution(torch.distributions.OneHotCategorical):
#     """
#     wrapper around OneHotCategorical -- same functionality but probs are along 1st (channel) axis rather than last axis.
#     Probably, we will only use log_prob and sample (maybe rsample). I assume any other methods that have not been
#     implemented would raise an error somewhere due to incompatible shapes
#     """
#     def __init__(self, probs=None, logits=None, validate_args=None):
#         #permute the init agras to the right shape for onehotcategorical
#         if probs is not None:
#             assert len(probs.shape) == 4, 'expected shape (bs, c, h ,w)!'
#             probs = torch.permute(probs, dims=(0, 2, 3, 1))
#         if logits is not None:
#             assert len(logits.shape) == 4, 'expected shape (bs, c, h ,w)!'
#             logits = torch.permute(logits, dims=(0, 2, 3, 1))
#
#         super().__init__(probs, logits, validate_args)
#
#     def log_prob(self, value):
#         # expected shape for value: (bs, c, h, w) -> change to (bs, h, w, c) as OneHotCategorical expects probs along last dim
#         value_permuted = torch.permute(value, dims=(0, 2, 3, 1))
#         logprob = super().log_prob(value_permuted)  # (bs, h, w)
#
#         return logprob
#
#     def sample(self, *args, **kwargs):
#         return torch.permute(super().sample(*args, **kwargs), dims=(0, 3, 1, 2))
#
#     def rsample(self, *args, **kwargs):
#         return torch.permute(super().rsample(*args, **kwargs), dims=(0, 3, 1, 2))









