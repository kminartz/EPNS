import torch
import torch.nn as nn
import torchvision
import numpy as np
from torch.nn import functional as F
import warnings
# Unet code:
import utils

######################## BASIC UNET MODEL: ########################
class Block(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, **kwargs):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, (kernel_size, kernel_size), padding='same', **kwargs)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, (kernel_size, kernel_size), padding='same', **kwargs)

    def forward(self, x):
        return self.relu(self.conv2(self.relu(self.conv1(x))))


class UNetEncoder(nn.Module):
    def __init__(self, chs=(64, 64, 128, 256, 512, 1024), kernel_size=3, **kwargs):
        super().__init__()
        self.enc_blocks = nn.ModuleList([Block(chs[i], chs[i + 1], kernel_size, **kwargs) for i in range(len(chs) - 1)])
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        ftrs = []
        for block in self.enc_blocks:
            x = block(x)
            ftrs.append(x)
            x = self.pool(x)
        return ftrs


class UNetDecoder(nn.Module):
    def __init__(self, chs=(1024, 512, 256, 128, 64), kernel_size=3, **kwargs):
        super().__init__()
        self.chs = chs
        self.upconvs = nn.ModuleList(
            [nn.ConvTranspose2d(chs[i], chs[i + 1], (kernel_size, kernel_size), (2, 2)) for i in range(len(chs) - 1)])
        self.dec_blocks = nn.ModuleList([Block(2*chs[i+1], chs[i + 1], kernel_size, **kwargs) for i in range(len(chs) - 1)])

    def forward(self, x, encoder_features):
        for i in range(len(self.chs) - 1):
            x = self.upconvs[i](x)
            enc_ftrs = self.crop(encoder_features[i], x)
            x = torch.cat([x, enc_ftrs], dim=1)
            x = self.dec_blocks[i](x)
        return x

    def crop(self, enc_ftrs, x):
        _, _, H, W = x.shape  # TODO: swap order of x and enc_ftrs? ie crop x (coarse info) rather than enc_ftrs (fine info) (probably not?)
        enc_ftrs = torchvision.transforms.CenterCrop([H, W])(enc_ftrs)
        return enc_ftrs


class UNet(nn.Module):
    def __init__(self, enc_chs=(64, 64, 128, 256, 512, 1024), dec_chs=(1024, 512, 256, 128, 64), kernel_size=3, **kwargs):
        super().__init__()
        self.encoder = UNetEncoder(enc_chs, kernel_size, **kwargs)
        self.decoder = UNetDecoder(dec_chs, kernel_size, **kwargs)
        # self.head        = nn.Conv2d(dec_chs[-1], num_class, (1, 1))
        self.printed = False

    def forward(self, x):
        out_sz = x.shape[-2:]
        enc_ftrs = self.encoder(x)
        if not self.printed:
            print('lowest level shape in U-net processor:', enc_ftrs[-1].shape)
            self.printed = True
        out = self.decoder(enc_ftrs[::-1][0], enc_ftrs[::-1][1:])
        # out = F.interpolate(out, out_sz, mode='nearest')  #TODO: we might want to get rid of this later, but it is a decent quick fix fn
        # out      = self.head(out)
        out = torchvision.transforms.CenterCrop(out_sz)(out)
        return out


######################## permutation-equivariant cell grid GNN: ############################
# not used in the paper but only for preliminary experiments
# might be useful for having local connectivity


class GridGNNLayer(nn.Module):

    """GNN layer with node features living on the same grid"""

    def __init__(self, emb_dim, hidden_dim, num_edge_feat=0, activation=nn.ReLU(), residual=False, kernel_size=3,
                 edge_network_downsampling_factor=2, edge_emb_dim=4, num_edge_network_layers=2, num_node_network_layers=2):
        super().__init__()
        if not isinstance(kernel_size, tuple):
            kernel_size = (kernel_size, kernel_size)

        self.residual = residual
        self.edge_emb_dim = edge_emb_dim

        edge_network_list = [nn.LazyConv2d(edge_emb_dim, kernel_size, padding='same'),
            activation]
        for _ in range(num_edge_network_layers - 1):
            edge_network_list.append(nn.Conv2d(edge_emb_dim, edge_emb_dim, kernel_size, padding='same'))
            edge_network_list.append(activation)
        self.edge_network = nn.Sequential(*edge_network_list)

        node_network_list = [nn.LazyConv2d(hidden_dim, kernel_size, padding='same')]
        for _ in range(num_node_network_layers - 1):
            node_network_list.append(activation)
            node_network_list.append(nn.Conv2d(hidden_dim, emb_dim, kernel_size, padding='same'))
        self.node_network = nn.Sequential(*node_network_list)

        self.downsampling = nn.LazyConv2d(edge_emb_dim, edge_network_downsampling_factor, edge_network_downsampling_factor)
        self.upsampling = nn.LazyConvTranspose2d(edge_emb_dim, edge_network_downsampling_factor, edge_network_downsampling_factor)


    def forward(self, tup_of_nodes_and_edges_and_batch_idx):
        x = tup_of_nodes_and_edges_and_batch_idx[0]
        edge_index = tup_of_nodes_and_edges_and_batch_idx[1]
        batch_idx = tup_of_nodes_and_edges_and_batch_idx[2]

        # shape of x: (total cells over batch, c, h, w)
        # shape of edge_attr: (num_edges, num_feat)
        row, col = edge_index
        # a full 100x100 grid can be very memory intensive to be processed if there are a lot of edges, so we learn a downsampling
        x_downsampled = self.downsampling(x)
        edge_feat = self.edge_model(x_downsampled[row], x_downsampled[col], edge_attr=None)
        x = self.node_model(x, edge_index, edge_feat)
        return x, edge_index, batch_idx

    def edge_model(self, source, target, edge_attr):
        edge_in = torch.cat([source, target], dim=1)
        if edge_attr is not None:
            edge_attr = edge_attr.unsqueeze(-1).unsqueeze(-1)
            edge_in = torch.cat([edge_in, edge_attr], dim=1)
        out = self.edge_network(edge_in)
        return out

    def node_model(self, h, edge_index, edge_attr):
        row, col = edge_index
        agg = self.unsorted_segment_sum(edge_attr, row, num_segments=h.size(0))
        agg = self.upsampling(agg)  # learn an upsampling to the full grid now that we have aggregated all edge messages to n nodes
        out = torch.cat([h, agg], dim=1)
        out = self.node_network(out)
        if self.residual:
            out = out + h
            #out = self.gru(out, h)
        return out

    def unsorted_segment_sum(self, data, segment_ids, num_segments):
        """Custom PyTorch op to replicate TensorFlow's `unsorted_segment_sum`."""
        result_shape = (num_segments, *data.shape[1:])
        result = data.new_full(result_shape, 0)  # Init empty result tensor.
        segment_ids = segment_ids.view(-1,1,1,1).expand_as(data)
        result.scatter_add_(0, segment_ids, data)
        return result

class GridGNN(nn.Module):

    def __init__(self, num_layers, emb_dim, kernel_size, activation, edge_network_downsampling_factor, edge_emb_dim,
                 num_edge_network_layers=2, num_node_network_layers=2, **kwargs):
        super().__init__()
        self.forward_network = nn.Sequential(*[GridGNNLayer(emb_dim, emb_dim, num_edge_feat=0, activation=activation,
                                                             kernel_size=kernel_size,
                                                            edge_network_downsampling_factor=edge_network_downsampling_factor,
                                                            edge_emb_dim=edge_emb_dim,
                                                            num_edge_network_layers=num_edge_network_layers,
                                                            num_node_network_layers=num_node_network_layers,
                                                            **kwargs) for _ in range(num_layers)])

    def forward(self, x):
        return self.forward_network(x)


################################# GRIDDEEPSET MODULE ##########################################
#(this is the spatialconv-gnn used in the paper!)

class GridDeepSetLayer(nn.Module):
    """deep set model with grid embedding for the set elements"""
    def __init__(self, emb_dim, hidden_dim, activation=nn.ReLU(), residual=False, kernel_size=3,
                 edge_network_downsampling_factor=2, edge_emb_dim=4, num_edge_network_layers=2, num_node_network_layers=2,
                 use_unet=False, unet_kernel_size=3,
                 **kwargs):

        # 'edge network' is the function applied to all set elements before aggregation
        # 'node network' is the function taking the aggregated embedding and updating each set element
        super().__init__()
        if not isinstance(kernel_size, tuple):
            kernel_size = (kernel_size, kernel_size)

        self.residual = residual
        self.edge_emb_dim = edge_emb_dim

        edge_network_list = [nn.LazyConv2d(edge_emb_dim, kernel_size, padding='same', **kwargs),
            activation]
        for _ in range(num_edge_network_layers - 1):
            edge_network_list.append(nn.Conv2d(edge_emb_dim, edge_emb_dim, kernel_size, padding='same', **kwargs))
            edge_network_list.append(activation)
        self.edge_network = nn.Sequential(*edge_network_list)

        if not use_unet:
            node_network_list = [nn.LazyConv2d(hidden_dim, kernel_size, padding='same', **kwargs)]
            for _ in range(num_node_network_layers - 1):
                node_network_list.append(activation)
                node_network_list.append(nn.Conv2d(hidden_dim, emb_dim, kernel_size, padding='same', **kwargs))
            self.node_network = nn.Sequential(*node_network_list)
        else:
            to_emb_dim = nn.LazyConv2d(out_channels=emb_dim, kernel_size=1)
            enc_chs = [emb_dim, emb_dim]
            for _ in range(num_node_network_layers - 1):
                chs_to_append = enc_chs[-1] * 2 if enc_chs[-1] < 1024 else enc_chs[-1]  # hard cap on max enc chs
                enc_chs.append(chs_to_append)
            dec_chs = enc_chs[::-1]
            dec_chs = dec_chs[:-1]
            unet = UNet(enc_chs, dec_chs, kernel_size=unet_kernel_size, **kwargs)
            self.node_network = nn.Sequential(to_emb_dim, unet)

        self.downsampling = nn.LazyConv2d(edge_emb_dim, edge_network_downsampling_factor, edge_network_downsampling_factor)
        self.upsampling = nn.LazyConvTranspose2d(edge_emb_dim, edge_network_downsampling_factor, edge_network_downsampling_factor)



    def forward(self, tup_of_nodes_and_edges_and_batch_idx):

        x = tup_of_nodes_and_edges_and_batch_idx[0]
        edge_index = tup_of_nodes_and_edges_and_batch_idx[1]  # should be empty for this model, but kept for consistency with the gridgnn
        batch_idx = tup_of_nodes_and_edges_and_batch_idx[2]
        # shape of x: (total cells over batch, c, h, w)
        # a full 100x100 grid can be very memory intensive to be processed, so we optionally learn a downsampling
        x_downsampled = self.downsampling(x)
        node_feat_before_agg = self.node_update_before_agg(x_downsampled)
        x = self.node_update(x, node_feat_before_agg, batch_idx)
        return x, edge_index, batch_idx

    def node_update_before_agg(self, set_elements):   # node processing function before aggregation
        edge_in = set_elements
        out = self.edge_network(edge_in)
        return out

    def node_update(self, h, nodes_processed_before_agg, batch_idx):  # processes node embeddings and aggregate of the set
        agg = self.unsorted_segment_mean(nodes_processed_before_agg, batch_idx, num_segments=int(torch.max(batch_idx) +1))  #(bs, c, h, w)
        # learn an upsampling to the full grid now that we have aggregated all edge messages to n nodes
        agg = self.upsampling(agg)

        # cast the batch-wise agg back to the total number of cells:
        agg_broadcasted = agg[batch_idx]  #(bs*num_cells, c, h, w)
        out = torch.cat([h, agg_broadcasted], dim=1)
        out = self.node_network(out)
        if self.residual:
            out = out + h
            #out = self.gru(out, h)
        return out

    def unsorted_segment_mean(self, data, segment_ids, num_segments):
        """Custom PyTorch op to replicate TensorFlow's `unsorted_segment_sum`."""
        result_shape = (num_segments, *data.shape[1:])
        result = data.new_full(result_shape, 0)  # Init empty result tensor.
        counts = data.new_full(result.shape, 0)
        segment_ids = segment_ids.view(-1,1,1,1).expand_as(data)
        result.scatter_add_(0, segment_ids, data)
        counts.scatter_add_(0, segment_ids, torch.ones_like(data))
        result = result / counts
        return result



class GridDeepSet(nn.Module):

    def __init__(self, num_layers, emb_dim, kernel_size, activation, edge_network_downsampling_factor=1, edge_emb_dim=None,
                 num_edge_network_layers=2, num_node_network_layers=2, **kwargs):
        super().__init__()
        if edge_emb_dim is None:
            edge_emb_dim = emb_dim
        self.forward_network = nn.Sequential(*[GridDeepSetLayer(emb_dim, emb_dim, activation=activation,
                                                             kernel_size=kernel_size,
                                                            edge_network_downsampling_factor=edge_network_downsampling_factor,
                                                            edge_emb_dim=edge_emb_dim,
                                                            num_edge_network_layers=num_edge_network_layers,
                                                            num_node_network_layers=num_node_network_layers,
                                                            **kwargs) for _ in range(num_layers)])
        self.perm_equi_tested = False

    def forward(self, x):
        if not self.perm_equi_tested:
            self.perm_equi_tested = True
            self._test_permutation_equivariance(x)
        out = self.forward_network(x)
        return out

    @torch.no_grad()
    def _test_permutation_equivariance(self, tup_of_nodes_and_edges_and_batch_idx):

        x = tup_of_nodes_and_edges_and_batch_idx[0]
        batch_idx = tup_of_nodes_and_edges_and_batch_idx[2]
        x = x[batch_idx == 0]

        # return_to_float = False
        # if list(self.parameters())[0].dtype is not torch.double:
        #     self.double()
        #     return_to_float = True
        #     x = x.double()

        shuffle_idx = utils.get_permutation_idx(num_elements=x.shape[0])

        x_perm = x[shuffle_idx, ...]

        out, _, _ = self.forward((x, None, batch_idx[batch_idx==0]))
        out = out.cpu()
        out_perm, _, _ = self.forward((x_perm, None, batch_idx[batch_idx == 0]))
        out_perm = out_perm.cpu()

        out_perm_after = out[shuffle_idx, ...]
        diff = torch.mean((out_perm - out_perm_after)**2)
        if diff < 1e-5:
            print('griddeepset permutation equivariance check succesful!')
        else:
            msg = 'griddeepset permutation equivariance check failed! This is probably due to floating point arithmetic.\n'
            msg += 'consider checking this on the cpu and with double() types just to be sure! (see the code that is commented out in this function)'
            warnings.warn(msg)
        # if return_to_float:
        #     self.float()


class AggregateGridNodes(nn.Module):

    def __init__(self, agg_fn=torch.mean, map_to_dim=None, activation=nn.ReLU()):
        super().__init__()
        self.agg_fn = agg_fn

        if map_to_dim is None:
            self.to_emb = nn.Identity()
            self.embed_each_cell = nn.Identity()
        else:
            self.to_global_emb = nn.Sequential(*[nn.LazyConv2d(map_to_dim, kernel_size=1),
                                        activation,
                                        nn.Conv2d(map_to_dim, map_to_dim, kernel_size=1)
                                        ])
            self.embed_each_cell = nn.Sequential(*[nn.LazyConv2d(map_to_dim, kernel_size=1),
                                                 activation,
                                                 nn.Conv2d(map_to_dim, map_to_dim, kernel_size=1)])




    def forward(self, tup_of_nodes_and_edges_and_batch_idx):
        node_feat = self.embed_each_cell(tup_of_nodes_and_edges_and_batch_idx[0])
        batch_idx = tup_of_nodes_and_edges_and_batch_idx[-1]

        # node feat has shape (bs*cells, c, h, w)
        aggregate_batches = torch.zeros(len(batch_idx.unique()), node_feat.shape[1], node_feat.shape[2], node_feat.shape[3]).to(node_feat)
        batch_idx_exp = batch_idx.view(-1,1,1,1).expand_as(node_feat)
        aggregate_batches.scatter_add_(dim=0, src=node_feat, index=batch_idx_exp)  # shape(bs, c, h, w)
        counts = torch.zeros_like(aggregate_batches)
        counts.scatter_add_(dim=0, src=torch.ones_like(node_feat), index=batch_idx_exp)
        aggregate_batches = aggregate_batches / counts
        aggregate_batches = self.to_global_emb(aggregate_batches)

        # aggregate_batches is a grid embedding for all nodes in each batch (shape: bs, emb_dim, h, w)
        # now, we broadcast these to the (bs*num_cell) nodes in the batch
        aggregate_batches = self.agg_fn(aggregate_batches, dim=(-2, -1))  # sum over spatial dims to get a representative aggregate vector (translation invariant)
        aggregates = torch.zeros(node_feat.shape[0], aggregate_batches.shape[1]).to(node_feat)  # (bs*cells, c)
        aggregates = aggregates + aggregate_batches[batch_idx]

        return aggregates






########################################### modern UNet code: ##################################


"""
Modern UNet implementation
Largely based on / extended from
https://github.com/microsoft/pdearena/blob/db7664bb8ba1fe6ec3217e4079979a5e4f800151/pdearena/modules/conditioned/twod_unet.py
which is largely based on
https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/05321d644e4fed67d8b2856adc2f8585e79dfbee/labml_nn/diffusion/ddpm/unet.py
Implemented in 1, 2 and 3 dimensions. Conditioning implemented by broadcasting and concatenating the 0d signal along
hidden features in both the encoder and decoder
"""


class UNetModern(nn.Module):


    """Modern U-Net architecture
    This is a modern U-Net architecture with wide-residual blocks and spatial attention blocks
    Args:
        num_spatial_dims (int): Number of spatial dimensions
        n_cond (int): Dimensionality of conditioning signal
        hidden_features (int): Number of channels in the hidden layers
        cond_mode (str): Type of conditioning to apply
        activation (nn.Module): Activation function to use
        norm (bool): Whether to use normalization
        ch_mults (list): List of channel multipliers for each resolution
        is_attn (list): List of booleans indicating whether to use attention blocks
        mid_attn (bool): Whether to use attention block in the middle block
        n_blocks (int): Number of residual blocks in each resolution
        use1x1 (bool): Whether to use 1x1 convolutions in the initial and final layers
    """

    def __init__(
        self,
        num_spatial_dims: int = 2,
        hidden_features: int = 128,
        activation: nn.Module = nn.GELU(),
        norm: bool = False,
        ch_mults=(2, 2, 2, 2),
        is_attn=(False, False, False, False),
        mid_attn: bool = False,
        n_blocks: int = 2,
        use1x1: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        self.hidden_features = hidden_features
        self.num_spatial_dims = num_spatial_dims

        self.activation: nn.Module = activation

        # Number of resolutions
        n_resolutions = len(ch_mults)
        n_channels = hidden_features

        # #### First half of U-Net - decreasing resolution
        down = []
        # Number of channels
        out_channels = in_channels = n_channels
        # For each resolution
        for i in range(n_resolutions):
            # Number of output channels at this resolution
            out_channels = in_channels * ch_mults[i]
            # Add `n_blocks`
            for _ in range(n_blocks):
                down.append(
                    DownBlock(
                        in_channels,
                        out_channels,
                        has_attn=is_attn[i],
                        activation=activation,
                        norm=norm,
                        num_spatial_dims=num_spatial_dims,
                        **kwargs
                    )
                )
                in_channels = out_channels
            # Down sample at all resolutions except the last
            if i < n_resolutions - 1:
                down.append(Downsample(in_channels, num_spatial_dims=num_spatial_dims, **kwargs))

        # Combine the set of modules
        self.down = nn.ModuleList(down)

        # Middle block
        self.middle = MiddleBlock(in_channels=out_channels, out_channels=out_channels, has_attn=mid_attn,
                                  activation=activation, norm=norm, num_spatial_dims=num_spatial_dims, **kwargs)

        # #### Second half of U-Net - increasing resolution
        up = []
        # Number of channels
        in_channels = out_channels
        # For each resolution
        for i in reversed(range(n_resolutions)):
            # `n_blocks` at the same resolution
            out_channels = in_channels
            for _ in range(n_blocks):
                up.append(
                    UpBlock(
                        in_channels,
                        out_channels,
                        has_attn=is_attn[i],
                        activation=activation,
                        norm=norm,
                        num_spatial_dims=num_spatial_dims,
                        **kwargs
                    )
                )
            # Final block to reduce the number of channels
            out_channels = in_channels // ch_mults[i]
            up.append(UpBlock(in_channels, out_channels, has_attn=is_attn[i], activation=activation, norm=norm, num_spatial_dims=num_spatial_dims, **kwargs))
            in_channels = out_channels
            # Up sample at all resolutions except last
            if i > 0:
                up.append(Upsample(in_channels, num_spatial_dims=num_spatial_dims, **kwargs))

        # Combine the set of modules
        self.up = nn.ModuleList(up)

        if norm:
            self.norm = nn.GroupNorm(8, n_channels)
        else:
            self.norm = nn.Identity()

        if use1x1:
            self.final = get_conv_with_right_spatial_dim(num_spatial_dims, in_channels=hidden_features, out_channels=hidden_features, kernel_size=1, **kwargs)
        else:
            self.final = get_conv_with_right_spatial_dim(num_spatial_dims, in_channels=hidden_features, out_channels=hidden_features, kernel_size=3, padding=1, **kwargs)



    def _crop_Nd(self, enc_ftrs: torch.Tensor, shape: torch.Tensor):
        if isinstance(shape, torch.Tensor) or isinstance(shape, np.ndarray):
            shape = shape.shape
        s_des = shape[-self.num_spatial_dims:]
        s_current = enc_ftrs.shape[-self.num_spatial_dims:]
        # first, calculate preliminary paddings - may contain non-integers ending in .5):
        pad_temp = np.repeat(np.subtract(s_des, s_current) / 2, 2)
        # to break the .5 symmetry to round one padding up and one down, we add a small pos/neg number respectively
        # note this will not impact the case where pad_temp[i] is integer since it is still rounded to that integer
        breaking_arr = np.tile([1, -1], int(len(pad_temp) / 2)) / 1000
        pad = tuple(map(lambda p: int(round(p)), pad_temp + breaking_arr))
        enc_ftrs = F.pad(enc_ftrs, pad)
        return enc_ftrs

    def forward(self, h: torch.Tensor, variables: torch.Tensor = None, **kwargs):
        assert h.dim() == 2 + self.num_spatial_dims  # [b, c, *spatial_dims]
        h_shape = h.shape
        h_features = [h]
        for m in self.down:
            h = m(h)
            h_features.append(h)

        h = self.middle(h)

        for m in self.up:
            if isinstance(m, Upsample):
                h = m(h)
            else:
                s = self._crop_Nd(h_features.pop(), h)  # crop spatial dim to match features
                # Get the skip connection from first half of U-Net and concatenate
                h = torch.cat((h, s), dim=1)
                h = m(h)

        h = self.final(self.activation(self.norm(h)))
        h = self._crop_Nd(h, h_shape)  # crop spatial dim to match features
        return h


class ResidualBlock(nn.Module):
    """Wide Residual Blocks used in modern Unet architectures.
    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        activation (nn.Module): Activation function to use.
        norm (bool): Whether to use normalization.
        n_groups (int): Number of groups for group normalization.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation: nn.Module = torch.nn.GELU(),
        norm: bool = False,
        n_groups: int = 1,
        num_spatial_dims: int = 1,
            **kwargs
    ):
        super().__init__()
        self.activation: nn.Module = activation

        self.conv1 = get_conv_with_right_spatial_dim(num_spatial_dims, in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, **kwargs)
        self.conv2 = get_conv_with_right_spatial_dim(num_spatial_dims, in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1, **kwargs)
        # If the number of input channels is not equal to the number of output channels we have to
        # project the shortcut connection
        if in_channels != out_channels:
            self.shortcut = get_conv_with_right_spatial_dim(num_spatial_dims, in_channels=in_channels, out_channels=out_channels, kernel_size=1, **kwargs)
        else:
            self.shortcut = nn.Identity()

        if norm:
            self.norm1 = nn.GroupNorm(n_groups, in_channels)
            self.norm2 = nn.GroupNorm(n_groups, out_channels)
        else:
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()

    def forward(self, x: torch.Tensor):
        # First convolution layer
        h = self.conv1(self.activation(self.norm1(x)))
        # Second convolution layer
        h = self.conv2(self.activation(self.norm2(h)))
        # Add the shortcut connection and return
        return h + self.shortcut(x)


class AttentionBlock(nn.Module):
    """Attention block This is similar to [transformer multi-head
    attention]
    Args:
        in_channels (int): the number of channels in the input
        n_heads (int): the number of heads in multi-head attention
        d_k: the number of dimensions in each head
        n_groups (int): the number of groups for [group normalization][torch.nn.GroupNorm].
    """

    def __init__(self, in_channels: int, out_channels: int = None, n_heads: int = 1, d_k=None, n_groups: int = 1, num_spatial_dims: int = 1, **kwargs):
        super().__init__()

        # Default `d_k`
        if out_channels is None:
            out_channels = in_channels
        if d_k is None:
            d_k = in_channels

        self.in_channels = in_channels
        self.out_channels = out_channels

        # Normalization layer
        self.norm = nn.GroupNorm(n_groups, in_channels)
        # Projections for query, key and values
        self.projection = nn.Linear(in_channels, n_heads * d_k * 3)
        # Linear layer for final transformation
        self.output = nn.Linear(n_heads * d_k, out_channels)
        # Scale for dot-product attention
        self.scale = d_k**-0.5
        #
        self.n_heads = n_heads
        self.d_k = d_k

        if in_channels != out_channels:
            self.shortcut = get_conv_with_right_spatial_dim(num_spatial_dims, in_channels=in_channels, out_channels=out_channels, kernel_size=1, **kwargs)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor):
        # Get shape
        batch_size, _, *spatial_dims = x.shape
        # Change `x` to shape `[batch_size, seq, n_channels]`
        x = x.view(batch_size, self.in_channels, -1).permute(0, 2, 1)
        # Get query, key, and values (concatenated) and shape it to `[batch_size, seq, n_heads, 3 * d_k]`
        qkv = self.projection(x).view(batch_size, -1, self.n_heads, 3 * self.d_k)
        # Split query, key, and values. Each of them will have shape `[batch_size, seq, n_heads, d_k]`
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        # Calculate scaled dot-product $\frac{Q K^\top}{\sqrt{d_k}}$
        attn = torch.einsum("bihd,bjhd->bijh", q, k) * self.scale
        # Softmax along the sequence dimension $\underset{seq}{softmax}\Bigg(\frac{Q K^\top}{\sqrt{d_k}}\Bigg)$
        attn = attn.softmax(dim=1)
        # Multiply by values
        res = torch.einsum("bijh,bjhd->bihd", attn, v)
        # Reshape to `[batch_size, seq, n_heads * d_k]`
        res = res.view(batch_size, -1, self.n_heads * self.d_k)
        # Transform to `[batch_size, seq, out_channels]`
        res = self.output(res)

        # Add skip connection
        res += self.shortcut(x)

        # Change to shape `[batch_size, out_channels, *spatial_dims]`
        res = res.permute(0, 2, 1).view(batch_size, self.out_channels, *spatial_dims)
        return res


class DownBlock(nn.Module):
    """Down block. This combines ResidualBlock and AttentionBlock.
    These are used in the first half of U-Net at each resolution.
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        has_attn (bool): Whether to use attention block
        activation (nn.Module): Activation function to use.
        norm (bool): Whether to use normalization
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        has_attn: bool = False,
        activation: nn.Module = nn.GELU(),
        norm: bool = False,
        num_spatial_dims: int = 1,
            **kwargs
    ):
        super().__init__()
        self.res = ResidualBlock(in_channels, out_channels, activation=activation, norm=norm,
                                 num_spatial_dims=num_spatial_dims, **kwargs)
        if has_attn:
            self.attn = AttentionBlock(out_channels, num_spatial_dims=num_spatial_dims, **kwargs)
        else:
            self.attn = nn.Identity()

    def forward(self, x: torch.Tensor):
        x = self.res(x)
        x = self.attn(x)
        return x


class UpBlock(nn.Module):
    """Up block that combines ResidualBlock and AttentionBlock.
    These are used in the second half of U-Net at each resolution.
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        has_attn (bool): Whether to use attention block
        activation (nn.Module): Activation function to use.
        norm (bool): Whether to use normalization
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        has_attn: bool = False,
        activation: nn.Module = nn.GELU(),
        norm: bool = False,
        num_spatial_dims: int = 1,
            **kwargs
    ):
        super().__init__()
        # The input has `in_channels + out_channels` because we concatenate the output of the same resolution
        # from the first half of the U-Net
        self.res = ResidualBlock(in_channels + out_channels, out_channels, activation=activation, norm=norm, num_spatial_dims=num_spatial_dims, **kwargs)
        if has_attn:
            self.attn = AttentionBlock(out_channels, num_spatial_dims=num_spatial_dims, **kwargs)
        else:
            self.attn = nn.Identity()

    def forward(self, x: torch.Tensor):
        x = self.res(x)
        x = self.attn(x)
        return x


class MiddleBlock(nn.Module):
    """Middle block
    It combines a `ResidualBlock`, `AttentionBlock`, followed by another
    `ResidualBlock`.
    This block is applied at the lowest resolution of the U-Net.
    Args:
        n_channels (int): Number of channels in the input and output.
        has_attn (bool, optional): Whether to use attention block. Defaults to False.
        activation (nn.Module): Activation function to use.
        norm (bool, optional): Whether to use normalization. Defaults to False.
    """

    def __init__(self, in_channels, out_channels: int, has_attn: bool = False, activation: nn.Module = nn.GELU(), norm: bool = False,
                 num_spatial_dims: int = 1, **kwargs):
        super().__init__()
        self.res1 = ResidualBlock(in_channels, out_channels, activation=activation, norm=norm,
                                  num_spatial_dims=num_spatial_dims, **kwargs)
        self.attn = AttentionBlock(out_channels, num_spatial_dims=num_spatial_dims, **kwargs) if has_attn else nn.Identity()
        self.res2 = ResidualBlock(out_channels, out_channels, activation=activation, norm=norm,
                                  num_spatial_dims=num_spatial_dims, **kwargs)

    def forward(self, x: torch.Tensor):
        x = self.res1(x)
        x = self.attn(x)
        x = self.res2(x)
        return x


class Upsample(nn.Module):
    r"""Scale up the feature map by $2 \times$
    Args:
        n_channels (int): Number of channels in the input and output.
    """

    def __init__(self, n_channels: int, num_spatial_dims: int, **kwargs):
        super().__init__()
        self.conv = get_upconv_with_right_spatial_dim(num_spatial_dims, in_channels=n_channels, out_channels=n_channels, kernel_size=4, stride=2, padding=1, **kwargs)

    def forward(self, x: torch.Tensor):
        return self.conv(x)


class Downsample(nn.Module):
    r"""Scale down the feature map by $\frac{1}{2} \times$
    Args:
        n_channels (int): Number of channels in the input and output.
    """

    def __init__(self, n_channels: int, num_spatial_dims: int, **kwargs):
        super().__init__()
        self.conv = get_conv_with_right_spatial_dim(num_spatial_dims, in_channels=n_channels, out_channels=n_channels, kernel_size=3, stride=2, padding=1, **kwargs)

    def forward(self, x: torch.Tensor):
        return self.conv(x)






########################## some utility functions to get conv layers with the right spatial dims ###################
def get_conv_with_right_spatial_dim(spatial_dim, **kwargs):
    if spatial_dim == 1:
        conv = nn.Conv1d(**kwargs)
    elif spatial_dim == 2:
        conv = nn.Conv2d(**kwargs)
    elif spatial_dim == 3:
        conv = nn.Conv3d(**kwargs)
    else:
        raise NotImplementedError(f'only 0<x<=3d convs implemented so far, but found spatial dim {spatial_dim}!')

    return conv


def get_upconv_with_right_spatial_dim(spatial_dim, in_channels, out_channels, **kwargs):
    kwargs_copy = kwargs
    if 'padding_mode' in kwargs:
        kwargs_copy = {}
        for k, v in kwargs.items():
            if k == 'padding_mode':
                kwargs_copy[k] = 'zeros'
            else:
                kwargs_copy[k] = v

    if spatial_dim == 1:
        upconv = nn.ConvTranspose1d(in_channels, out_channels, **kwargs_copy)
    elif spatial_dim == 2:
        upconv = nn.ConvTranspose2d(in_channels, out_channels, **kwargs_copy)
    else:
        raise NotImplementedError(f'only 0<x<=2d convs implemented so far, but found spatial dim {spatial_dim}!')

    return upconv
