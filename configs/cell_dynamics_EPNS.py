import torch
import torch.nn as nn
import numpy as np
from modules.Cellsort_Sim import Cellsort_Simulator
import os
from datasets import Cell_Combined_Dataset

config_dict = dict(
    ###### general ######
    device='cuda' if torch.cuda.is_available() else 'cpu',
    # device='cpu',

    ###### data ######
    dataset=Cell_Combined_Dataset,  # dataset class to use (from datasets directory)
    data_directory=os.path.join('data', 'cellular_dynamics'),  # path to data -- must contain train, val, test subdirectories
    limit_num_data_points_to=np.inf,  # put an integer to limit the training set size to this many data points
    batch_size=8,
    im_dim=1 + 5,  # total amount of feautures/channels per cell -- 1 + num_cell_types for one-hot encodings of id and type
    dynamic_channels=tuple(i for i in range(1+5)),  # channels that actually change over time (all in this case)
    pred_stepsize=1,  # how many timesteps to predict ahead in one go

    ###### model ######
    model=Cellsort_Simulator,
    model_params=dict(
        num_layers=1,  # num griddeepset layers in the 'processor' (\approx forward model)
        emb_dim=32,  # hidden dimensionality throughout the model
        kernel_size=9,  # kernel size for processor
        nonlinearity=nn.ReLU(), # activation function
        processor='griddeepset', # processor type -- 'griddeepset' for spatialconv-gnn, 'unet_modern' for UNet as described in the appendix
        decoder='vae_decoder', # 'vae_decoder' -- do not change. If you want a deterministic model, set num_latent_dim to 0 and sample_from_pred to False
        decoder_kernel_size=11,  # kernel size for decoder
        num_latent_dim=64,  # number of latent dimensions -- set to 0 for deterministic model or model without latents
        num_cell_types=5, # number of different cell types in the data - medium is considered a separate type
        max_num_cells=65,  # (max) number of cells in the system + 1 ==> 65= max_num_cells+1, cell type medium = 0, we also count this
        edge_dist_thresh=0.,  # not relevant for griddeepset since all cells in the batch are aggregated. keep to 0.
        edge_network_downsampling_factor=2, # spatial downsampling factor for 'edge network' -- this network processes the node embeddings before aggregation
        edge_emb_dim=32,  # hidden dimensionality of 'edge network'
        num_edge_network_layers=3, # number of layers in 'edge network'
        num_node_network_layers=4,  # number of layers in 'node network' -- this network updates the node embeddings from the aggregated nodes and the old node embedding
        global_downsampling_factor=1,  # spatial downsampling by this factor before going into the forward model, and upsampling by this factor after the decoder
        free_bits=0.1175,  # do not penalize KL loss if the average minibatch KL divergence (per z dim) is lower than this value -- see paper appendix for details
        prior_and_encoder_arch='conv',  # architecture for inferring prior and posterior distribution -- 'conv' for node-wise convolutions, 'griddeepset' for griddeepset
        num_prior_and_encoder_layers=4, # number of layers in prior and encoder networks
        use_unet=True, # whether to use a unet in the node network rather than conv2D layers
        unet_kernel_size=5, # kernel size for the node network if unet is used
        EPS=1e-6,  # epsilon for numerical stability
        do_node_aggregation=False, # whether to aggregate node embeddings before inferring prior/posterior distribution -- set to True for permutation invariant latents
        sample_from_pred=False,  # whether to sample from the decoding distribution or take the most likely class for each pixel
    ),

    ###### training etc ######
    loss_func=lambda pred, true: nn.CrossEntropyLoss(reduction='sum')(pred, true),  # only used for calculating some validation statistics
    optimizer=torch.optim.Adam,  # optimizer
    opt_kwargs={'lr': 1e-4, 'weight_decay': 1e-4, 'betas': (0.9, 0.9), 'eps':1e-06},  # optimizer parameters
    num_epochs=180,  # number of epochs to train for
    training_strategy='multi-step',   # either 'one-step' or 'multi-step' -- see paper
    num_kl_annealing_cycles=1,  # number of KL annealing cycles -- defaults to 1 if not specified
    kl_increase_proportion_per_cycle=0.125,  # proportion of each cycle spent increasing beta (vs keeping it constant),
    beta=64.,  # beta value in beta-vae loss term

    ###### save state dict as ######
    try_use_wandb=True,
    experiment={'state_dict_fname': 'cell_dynamics_EPNS.pt'},
)
