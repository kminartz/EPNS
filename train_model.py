import argparse

import utils
from utils import load_config
import time
from Trainer import Trainer
import torch
import torch.nn as nn
import os
import pickle
import numpy as np
import datetime

# torch.autograd.set_detect_anomaly(True)
def train_model(config: dict):


    # check for some optional parameters in the config that we need to handle here:
    if 'state_dict_fname' in config['experiment'].keys():
        fname = config['experiment']['state_dict_fname']
    else:
        fname = f"state_dict_{config['experiment']['time']}.pt"

    if 'limit_num_data_points_to' in config.keys():
        num_data_points = config['limit_num_data_points_to']
    else:
        num_data_points = np.inf

    dataloader, val_dataloader, _ = config['dataset'].get_data_loaders(config, additional_loaders=[],
                                                                       limit_num_data_points_to=num_data_points)
    one_example_batch = next(iter(dataloader))  #(bs, c, t, h, w)

    model: nn.Module = config['model'](**config['model_params'], im_dim=config['im_dim'],
                            dynamic_channels=config['dynamic_channels'], pred_stepsize=config['pred_stepsize'])

    if 'starting_weight_state_dict' in config.keys():
        starting_state_dict = config['starting_weight_state_dict']
        if starting_state_dict is not None:
            print(f'initializing model from state dict {starting_state_dict}', flush=True)
            model.load_state_dict(torch.load(starting_state_dict))

    if 'start_from_epoch' in config.keys():
        start_from_epoch = config['start_from_epoch']
    else:
        start_from_epoch = 0

    ### initialize model

    device = config['device']
    model.to(device)
    with torch.no_grad():
        # initialize lazy layers by calling a fw pass:
        model(one_example_batch[:, :, 0].to(device), one_example_batch[:, :, 1].to(device))

    print(f'the model has {utils.count_parameters(model)} parameters.')

    # get ready for training and check for optional training parameters:
    opt = config['optimizer'](model.parameters(), **config['opt_kwargs'])

    if 'num_kl_annealing_cycles' in config.keys():
        num_kl_annealing_cycles = config['num_kl_annealing_cycles']
    else:
        num_kl_annealing_cycles = 1

    if 'kl_increase_proportion_per_cycle' in config.keys():
        kl_increase_proportion_per_cycle = config['kl_increase_proportion_per_cycle']
    else:
        kl_increase_proportion_per_cycle = 1

    if 'try_use_wandb' in config.keys():
        try_use_wandb = config['try_use_wandb']
    else:
        try_use_wandb = True

    if 'beta' in config.keys():
        beta = config['beta']
    else:
        beta = 1.0

    if 'clip_reconstr_loss_to' in config.keys():
        clip_reconstr_loss_to = config['clip_reconstr_loss_to']
    else:
        clip_reconstr_loss_to = torch.inf


    # initialize trainer object
    trainer = Trainer(model, config['loss_func'], opt, config['pred_stepsize'], num_kl_annealing_cycles,
                      kl_increase_proportion_per_cycle, config, try_use_wandb, beta, clip_reconstr_loss_to)

    epochs = config['num_epochs']
    training_strategy = config['training_strategy']

    print(f'will train for {epochs} epochs with {training_strategy} training.')
    if not os.path.exists('models/state_dicts'):
        os.makedirs(os.path.join('models', 'state_dicts'))


    save_path = os.path.join('models', 'state_dicts', fname)
    print(f'will save model state dict as {fname}')

    config['experiment']['state_dict_path'] = save_path

    # train the model:
    train_losses, train_acc, val_losses, val_acc, best_state_dict, last_state_dict = trainer.train(
        dataloader, val_dataloader, epochs, device, training_strategy, save_fname=fname, start_from_epoch=start_from_epoch)

    print(f'all done, saved at state dict at {save_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training config')
    parser.add_argument('config', type=str, help='config file path')
    args, remaining_args = parser.parse_known_args()
    config = load_config(args.config).config_dict

    def parse_to_int_or_float(str):
        try:
            return int(str)
        except ValueError:
            return float(str)

    for arg in remaining_args:  # any argument given as --kwarg=x after the config file will be parsed
        # and added to the config dict or overwrite the parameters in the config dict it they are already present
        arg: str
        arg = arg.strip('-')
        k, v = arg.split('=')
        try:
            v = parse_to_int_or_float(v)
        except:
            v = v
        if k in config.keys():
            config[k] = v
            print(f'set {k} to {v} in main config!', flush=True)
        elif k in config['model_params'].keys():
            config['model_params'][k] = v
            print(f'set {k} to {v} in model_params config!', flush=True)
        elif k in config['opt_kwargs'].keys():
            config['opt_kwargs'][k] = v
            print(f'set {k} to {v} in optimizer parameters config!', flush=True)
        else:
            config[k] = v
            print(f'did not find {k} in main or model param config keys -- set {k} to {v} in main config nevertheless', flush=True)

        if k != 'data_directory' and k != 'starting_weight_state_dict':
            # we update the state dict name with the command line params
            old_state_dict_fname = config['experiment']['state_dict_fname']
            config['experiment']['state_dict_fname'] = old_state_dict_fname[:-3] + f'--{k[:5]}{v}' + old_state_dict_fname[-3:]

    start = time.time()
    print(f'starting new model training run at {start}')

    now = datetime.datetime.now()
    now_str = now.strftime("%Y-%m-%d-%H-%M-%S")
    config['experiment']['time'] = now_str

    train_model(config)
    stop = time.time()

    now = datetime.datetime.now()
    now_str = now.strftime("%Y-%m-%d-%H-%M-%S")
    print(f'finished at {now_str}. Total time: {np.round((stop - start) / 60., 2)} minutes.')
