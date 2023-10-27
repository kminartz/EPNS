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
    batch_size=32,
    im_dim=65 + 5,  # total amount of feautures/channels per cell -- 65 for cell ids, 5 for cell types
    dynamic_channels=tuple(i for i in range(65 + 5)),  # channels that actually change over time (all in this case)

    pred_stepsize=1,  # how many timesteps to predict ahead in one go

    ###### model ######
    model=Cellsort_Simulator,
    model_params=dict(
        num_layers=4,  # num unet layers in the 'processor' (\approx forward model)
        emb_dim=128,   # hidden dimensionality throughout the model
        kernel_size=3,  # kernel size for processor
        nonlinearity=nn.ReLU(),  # activation function
        processor='unet_modern',  # processor type -- 'griddeepset' for spatialconv-gnn, 'unet_modern' for UNet as described in the appendix
        decoder='vae_decoder',  # 'vae_decoder' -- do not change. If you want a deterministic model, set num_latent_dim to 0 and sample_from_pred to False
        decoder_kernel_size=11,  # kernel size for decoder
        num_latent_dim=64,  # number of latent dimensions -- set to 0 for deterministic model or model without latents
        num_cell_types=5,  # number of different cell types in the data - medium is considered a separate type
        max_num_cells=65,  # (max) number of cells in the system + 1 ==> 65= max_num_cells+1, cell type medium = 0, we also count this
        free_bits=0.1,  # do not penalize KL loss if the average minibatch KL divergence (per z dim) is lower than this value
    ),




    ###### training etc ######
    loss_func=lambda pred, true: nn.CrossEntropyLoss(reduction='sum')(pred, true),  # only used for calculating some validation statistics
    optimizer=torch.optim.Adam,  # optimizer
    opt_kwargs={'lr': 1e-4, 'weight_decay': 1e-4},  # optimizer parameters
    num_epochs=180,  # number of epochs to train for
    training_strategy='multi-step',  # either 'one-step' or 'multi-step' -- see paper
    num_kl_annealing_cycles=1,  # defaults to 1 if not specified
    kl_increase_proportion_per_cycle=0.125, # proportion of cycle time spent increasing beta (vs keeping it constant at 1),
    beta=1., # beta value in beta-vae loss term

    ###### save state dict as ######
    try_use_wandb=True,
    experiment={'state_dict_fname': 'cell_dynamics_PNS.pt'},
)
