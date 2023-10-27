import itertools
import warnings

import torch
import torch.nn as nn
import numpy as np
import os
import matplotlib.pyplot as plt
from typing import Tuple
import utils
import time
import pickle
from utils import plot_cell_image, make_gif, count_parameters, plot_NS_image
from typing import *
from scipy import ndimage
from tueplots import axes, bundles, figsizes
from modules.engine.Enc_Proc_Dec import index_channels_for_domain
from modules.Cellsort_Sim import Cellsort_Simulator
EPS = 1e-6



@torch.no_grad()
def load_model_and_get_dataloaders(experiment_config: dict, return_test_loader=False,
                                   return_other_loaders=tuple()):

    timestamp = experiment_config['experiment']['timestamp'] if 'timestamp' in experiment_config.keys() else 'unknown'
    print(f'evaluating model from experiment with timestamp {timestamp}')

    DATADIR = experiment_config['data_directory']


    pred_stepsize = experiment_config['pred_stepsize']

    dataloader, val_dataloader, test_loader, *other = experiment_config['dataset'].get_data_loaders(experiment_config,
                                                                                                    return_other_loaders)

    model = experiment_config['model'](**experiment_config['model_params'],
                                       dynamic_channels=experiment_config['dynamic_channels'],
                                       im_dim=experiment_config['im_dim'],
                                       pred_stepsize=pred_stepsize)
    try:
        model.load_state_dict(torch.load(os.path.join('models', 'state_dicts', 'final/' +
                                                  experiment_config['experiment']['state_dict_fname'])))
    except FileNotFoundError:
        print('could not find model in final folder, falling back to root')
        model.load_state_dict(torch.load(os.path.join('models', 'state_dicts',
                                                      experiment_config['experiment']['state_dict_fname'])))

    if not return_test_loader:
        return model, pred_stepsize, dataloader, val_dataloader, *other

    return model, pred_stepsize, dataloader, val_dataloader, test_loader, *other  # other is the test loader with same init conds

@torch.no_grad()
def model_rollout(model: nn.Module, data_batch: torch.Tensor, pred_stepsize: int, rollout_length: int,
                  start_time:int = 0, use_posterior_sampling=False, verbose=False,
                  use_ground_truth_input=False, return_loglik=False, return_dists=False):


    random_x = data_batch[range(data_batch.size(0)), :, start_time]  # x at the specified start time

    trues = [random_x.cpu().numpy()]
    preds = [random_x.cpu().numpy()]
    preds_cont = [random_x.cpu().numpy()]
    dists = [torch.distributions.Normal(loc=random_x.cpu(), scale=1e-6)]
    kl = 0
    reconstr_loglik = 0
    miscellaneous = []

    start = time.time()
    for step in range(1, rollout_length + 1):
        target_time = start_time + pred_stepsize * step  # which x to get as a target x^{t+1}
        if np.any(target_time >= data_batch.shape[2]) and not use_posterior_sampling:
            random_y = None  # if the target time is too large and we are in sampling mode, we have no y to predict
            # is use_posterior_sampling is False, we also set random_y to None later
        else:
            random_y = data_batch[range(data_batch.size(0)), :, target_time]  # the x^{t+1} to predict
            trues.append(random_y.cpu().numpy())
        with torch.no_grad():
            # model.to('cpu')
            model.eval()
            random_x = random_x
            y = random_y if use_posterior_sampling else None  # only give x^{t+1} when we use posterior sampling
            try:
                pred_dist, kl_step, pred_sample, *miscellaneous = model(random_x, y, *miscellaneous)
            except Exception as e:
                print(repr(e), f'aborting further rollout at step {step}!')
                trues = trues[:-1]  # remove the last element in trues because we could not add a pred
                break  # stop the rollout here but still return what we got so far
            if return_loglik:
                assert random_y is not None, 'use posterior sampling should be set to true for loglik calculation!'
                kl += kl_step
                reconstr_loglik_step = pred_dist.log_prob(random_y) # note: not yet reduced over dimensions
                reconstr_loglik += reconstr_loglik_step

            preds.append(pred_sample.cpu().numpy())
            preds_cont.append(pred_dist.mean.cpu().numpy())  # mean prediction
            if return_dists:
                dists.append(pred_dist)

        random_x = torch.ones_like(pred_sample) * pred_sample  # quick hack to avoid modifying random_x in pred list inplace
        if use_ground_truth_input:
            random_x = random_y

    stop = time.time()
    if verbose:
        print(f'model rollout took {stop - start} seconds')

    if return_loglik:
        return trues, preds, preds_cont, reconstr_loglik, kl
    elif return_dists:
        return trues, dists
    else:
        return trues, preds, preds_cont


@torch.no_grad()
def log_likelihood_avg(model: nn.Module, dataloader, pred_stepsize: int, device='cpu', dynamic_channels=(1,2,3,4,5,6)):
    model.to(device)
    model.eval()
    summed_log_lik = 0
    count = 0

    for batch, data in enumerate(dataloader):
        data: torch.Tensor = data.to(device)  # shape(bs, num_nodes, time, num_feat)
        trues, preds_sample, _, reconstr_loglik, kl = model_rollout(model, data, pred_stepsize,
                                                                             rollout_length=(data.shape[2]-1)//pred_stepsize,
                                                                             use_posterior_sampling=True,
                                                                             use_ground_truth_input=True,
                                                                             return_loglik=True)

        if isinstance(model, Cellsort_Simulator):
            # we divide the kl by the number of cells in the decoder module of enc-prod-dec for the cellsort simulator,
            # in order to have the kl divergence magnitude invariant to the number of cells in the system.
            # but to get the true loglik, we need to re-multiply again by this number.
            # This is just a quick and dirty hack to get it right
            kl *= model.max_num_cells
            # also, internally, we have the dynamic channels at the number of types + 1 for the one-hot encoded cell in the graph encoding
            # but for the loglik, we only care about channel 0 which gives the cell id
            dynamic_channels = (0,)
        reconstr_loglik = index_channels_for_domain(reconstr_loglik,
                                                    dynamic_channels, domain_str=model.engine.domain_str)
        loglik = reconstr_loglik.sum() - kl
        count += data.shape[0]
        summed_log_lik += loglik

    return summed_log_lik / count

def plot_trajectories(list_of_lists_with_trajectories, save_path, plotting_stepsize, trajectory_kind, ylabel=None,
                      kind='cell'):

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if save_path[-1] != '/':
        save_path+= '/'

    with plt.rc_context(bundles.neurips2022()):

        ll = list_of_lists_with_trajectories
        for i, l in enumerate(ll):
            all_im_arrays = []
            for t, arr in enumerate(l[::plotting_stepsize]):
                if kind == 'cell':
                    im_arr = plot_cell_image(arr[:, 1:], no_title=True, show_plot=False)
                elif kind == 'n_body':
                    raise NotImplementedError('please use specialized n-body plotter instead (See sandbox_n_body.py')
                else:
                    raise NotImplementedError

                bar = np.ones_like(im_arr)[:, :3]  # bar of 3 pixels wide
                all_im_arrays.append(im_arr)
                all_im_arrays.append(bar)

            conc = np.concatenate(all_im_arrays, axis=1)
            plt.figure()
            if kind == 'cell':
                plt.imshow(conc)
            else:
                raise NotImplementedError()
            plt.gca().set_xticks([], [])
            plt.gca().set_yticks([], [])
            if ylabel is not None:
                plt.ylabel(ylabel)
            plt.tight_layout()
            plt.savefig(save_path + f'{trajectory_kind}_{i}.png', dpi=600)


def model_rollout_split_batch_on_crash(model: nn.Module, data_batch: torch.Tensor, pred_stepsize: int, rollout_length: int,
                  start_time:int = 0, use_posterior_sampling=False, verbose=False,
                  use_ground_truth_input=False, return_loglik=False, return_dists=False):

    # this is a method that comes in handy when testing stability of simulations where the model might produce NaNs
    # or throw errors during rollout. It recusrivelty splits the batch in half until it can complete the rollout
    # so that we can still exploit paralellism on the gpu as much as possible.
    # this is a bit of a hack, but it works well enough for now.
    assert not use_posterior_sampling and not use_ground_truth_input, 'Both should be False'
    _, preds_so_far, *rest = model_rollout(model, data_batch, pred_stepsize, rollout_length,
                                                                  start_time, use_posterior_sampling, verbose,
                                                                  use_ground_truth_input, return_loglik, return_dists)

    preds_so_far = np.array(preds_so_far)
    # continue from here onward
    start_from_here = preds_so_far[-1][:,:,None]
    start_from_here = torch.from_numpy(start_from_here).to(data_batch)
    rollout_remaining = rollout_length + 1 - preds_so_far.shape[0]
    if rollout_remaining > 0 and start_from_here.shape[0] > 1:
        if rollout_remaining == 505:
            print('debug')
        # we should try to splot the batch to simulate the remainder
        # this predicts the remainder, and merges it with what we already had along the time axis
        split_idx = start_from_here.shape[0] // 2

        # create two smaller batches
        start_from_here1 = start_from_here[:split_idx]
        start_from_here2 = start_from_here[split_idx:]

        preds_result1 = model_rollout_split_batch_on_crash(model, start_from_here1, pred_stepsize, rollout_remaining, 0,
                                                     use_posterior_sampling, verbose, use_ground_truth_input,
                                                     return_loglik, return_dists)
        preds_result2 = model_rollout_split_batch_on_crash(model, start_from_here2, pred_stepsize, rollout_remaining, 0,
                                                     use_posterior_sampling, verbose, use_ground_truth_input,
                                                     return_loglik, return_dists)
        # trues_result = np.concatenate([np.array(trues_result1), np.array(trues_result2)], axis=1)
        preds_result = np.concatenate([np.array(preds_result1), np.array(preds_result2)], axis=1)

        # trues_result = np.concatenate([trues_so_far, trues_result], axis=0)
        preds_result = np.concatenate([preds_so_far, preds_result[1:]], axis=0)

        return preds_result
    elif rollout_remaining > 0:
        # one sample left, so we cannot split further! simply append zeros if this still crashes
        # this returns the remainder as well as 1 preceding timestep we already had
        _, preds_result, *rest = model_rollout(model, start_from_here, pred_stepsize, rollout_remaining, 0,
                                                          use_posterior_sampling, verbose, use_ground_truth_input,
                                                          return_loglik, return_dists)

        preds_result = np.array(preds_result)
        preds_out = np.zeros(shape=(rollout_length+1, *preds_result.shape[1:]))
        preds_out[:preds_so_far.shape[0] - 1] += preds_so_far[:-1]  # only in case somehow retrying the model rollout above did work for some more steps... very edgy, probably some randomness in the pca decomp
        preds_out[preds_so_far.shape[0] - 1:preds_result.shape[0]] += preds_result
        assert preds_out.shape[0] == rollout_length + 1, 'this should match!'
        return preds_out
    else:
        # whole rollout is completed, so no need to split!
        # this returns the remainder that was to be predicted as well as 1 preceding timestep we already had
        assert preds_so_far.shape[0] == rollout_length + 1, 'this should match!'
        return preds_so_far



def _draw_borders_on_type_channel(arr):
    arr = np.asarray(arr)
    # upsample the array so we can draw the cell borders on the type channel in a bit higher resolution:
    arr = arr.repeat(2, axis=-2).repeat(2, axis=-1)
    borderx = np.diff(arr[..., 0,:,:], axis=-1, append=0) != 0
    bordery = np.diff(arr[..., 0, :, :], axis=-2, append=0) != 0
    arr[..., 1, :, :] += 1
    arr[..., 1, :, :][borderx] = 0
    arr[..., 1, :, :][bordery] = 0
    return arr
