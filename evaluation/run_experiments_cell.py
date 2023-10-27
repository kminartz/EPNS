import sys
sys.path.append(".")
sys.path.append("..")
import utils
import matplotlib.pyplot as plt
from experiment_utils import *
from utils import set_size_plots, put_legend_above_fig
import itertools
import torch
import torch.nn as nn
import numpy as np
import os
import time
import seaborn as sns
from tueplots import bundles, figsizes


# plotting sizes etc:
plot_bundle = bundles.neurips2023()
width = 397.48499  # width in points
# plot_bundle['axes.labelsize'] = 10
# plot_bundle['legend.fontsize'] = 10
plt.rcParams.update(plot_bundle)
figsize = figsizes.neurips2023(ncols=1)
# num_fig_columns = 2.5
# figsize['figure.figsize'] = set_size_plots(width, fraction=1/num_fig_columns, h_to_w_ratio=None)
num_fig_columns = 2
figsize['figure.figsize'] = set_size_plots(width, fraction=1/num_fig_columns, h_to_w_ratio=0.55)
# # figsize["figure.figsize"] = tuple(map(lambda x: x * 2/3, figsize["figure.figsize"]))  # two subplots per column
# figsize['figure.figsize'] = (figsize['figure.figsize'][0] / num_fig_columns, figsize['figure.figsize'][1] / num_fig_columns)

# figsize['figure.figsize'] = set_size_plots(width, fraction=1/num_fig_columns, h_to_w_ratio=None)
plt.rcParams.update(figsize)

# some more imports
import matplotlib.cm
from utils import create_circular_mask, load_config, make_gif
import argparse
from scipy.ndimage import label
from scipy.stats import kstest
from experiment_utils import _draw_borders_on_type_channel


# run this script to run experiments for the cell dynamics model
def run_experiments():
    # define the configs to load and the corresponding state dicts to load
    cfgs = ['cell_dynamics_EPNS', 'cell_dynamics_PNS']
    state_dicts = [None, None]  # None to use the state dict from the config file, string for any name
    # it will look for the state dicts in the models/state_dicts folder, with priority for those in the /final/ folder
    labels = ['EPNS', 'PNS']
    simulation_data = None
    # we could also load data from disk by doing something like this:
    # simulation_data = ['evaluation/samples_testset_cells_cell_dynamics_EPNS.pt.npy',
    # 'evaluation/samples_testset_cells_cell_dynamics_PNS.pt.npy']

    cfgs_loaded = []
    for i, c in enumerate(cfgs):
        cfgs_loaded.append(load_config(c, state_dicts[i]).config_dict)
    cfgs = cfgs_loaded

    # please check this function to comment out the experiments you don't want to run
    e1(cfgs,labels=labels , simulation_data=simulation_data)

@torch.no_grad()
def e1(cfgs, labels, simulation_data=None):

    if not os.path.exists('figures/cellsort'):
        os.makedirs('figures/cellsort')

    # uncomment the relevant methods below for the specific experiments
    print('running experiment 1 \n\n')
    e1_long_simulation_stability(cfgs, vol_lower_threshold=16, vol_upper_threshold=30, max_cells_unstable=12,
                                 labels=labels, simulation_data=simulation_data, run_from_config=False)
    for i, c in enumerate(cfgs):
        print(c)
        e1_single_loglik(c)
        # e1_single_plots(c, generate_gifs=True, num_reps=10, simulated_data=simulation_data[i] if simulation_data is not None else None)
        e1_agg_stats(c, num_samples=100, quantiles=(0.1,))
        # if simulation_data is None:
        #     simulated_data = None
        # else:
        #     simulated_data = simulation_data[i]
        # e1_agg_stats(c, num_samples=100, quantiles=(0.1,), simulated_data=simulated_data, label=labels[i])
        e1_check_equivariance(c, num_samples=1000)

    return

@torch.no_grad()
def e1_single_plots(cfg, generate_gifs=True, num_reps=10, save_individual_images=True, simulated_data=None):

    # make some visualizations of sample trajectories
    device = cfg['device'] if torch.cuda.is_available() else 'cpu'
    bs = cfg['batch_size']

    model, pred_stepsize, dataloader, _, test_dataloader, test_same_init_loader = load_model_and_get_dataloaders(cfg, True, ['test_same_init'])

    # random_data = next(iter(test_dataloader))
    # idxs = np.random.randint(0, 100, bs)
    idxs = [27, 32, 49, 25, 23, 63, 26, 13, 82, 67]
    print(f'data point idxs: {idxs}')
    all_random_data = []
    for idx in idxs:
        random_data = test_dataloader.dataset.__getitem__(idx).unsqueeze(0)
        all_random_data.append(random_data)
    random_data = torch.cat(all_random_data, dim=0)
    # random_data = test_same_init_loader.dataset.__getitem__(idx).unsqueeze(0)

    if simulated_data is None:
        # actually do the rollout as opposed to loading some data from disk
        model.to(device)

        num_reps = min(num_reps, random_data.shape[0])
        random_data = random_data[:num_reps]
        end_of_sim_time = random_data.size(2)

        rollout_length = (random_data.size(2) - 1) // pred_stepsize

        # start_time = np.random.randint(low=0, high=end_of_sim_time - pred_stepsize * rollout_length,
        #                                size=random_data.size(0))
        start_time = 0

        random_data = random_data.to(device)

        # initialize lazy layers by calling a fw pass:
        model(random_data[:1, :, 0].to(device), random_data[:1, :, 1].to(device))
        print(f'the model has {utils.count_parameters(model)} parameters.')

        # rollout_length = 40
        print(f'rollout length: {rollout_length}')
        # do the rollout with multi-step like steering:
        trues, preds, preds_cont = model_rollout(model, random_data, pred_stepsize, rollout_length,
                                                 start_time, use_posterior_sampling=True)

        trues = utils.from_time_batch_to_batch_time(trues)  #list with: (bs, t, 1, c, h, w)
        preds = utils.from_time_batch_to_batch_time(preds)
        path_prefix = 'reconstr'

    else:
        trues = random_data #(bs, c, time, h, w)
        preds = torch.from_numpy(np.load(simulated_data)) #(bs, c, time, h, w)
        # if len(preds.shape) > 4:
        #     preds = preds[1:2]  # pick first sample in batch
        # convert to the desired format for plotting
        trues = torch.movedim(trues, 1, 2).unsqueeze(2).numpy()
        preds = torch.movedim(preds, 1, 2).unsqueeze(2).numpy()
        path_prefix = 'ode2vae'  # change according to the model you used to load the data from disk

    all = np.concatenate([trues, preds], axis=-2)
    if save_individual_images:
        _save_individual_images(trues, preds, cfg, save_trues=True, save_every=5, path_prefix=path_prefix, idx=idxs)

    if simulated_data is None:
        # also generate samples without multi step-like steering -- these are actual model samples!
        num_rep = 3  # generate this amount of samples
        for j in range(num_rep):
            _, preds_sampled, _ = model_rollout(model, random_data, pred_stepsize, rollout_length,
                                                start_time, use_posterior_sampling=False)
            preds_sampled = utils.from_time_batch_to_batch_time(preds_sampled)
            if save_individual_images:
                _save_individual_images(trues, preds_sampled, cfg, save_trues=False, save_every=5, path_prefix=f'samples_{j}', idx=idxs)
            all = np.concatenate([all, preds_sampled], axis=-2)

        all = _draw_borders_on_type_channel(all)

        spath = f"figures/cellsort/trajectories/"
        if not os.path.exists(spath):
            os.makedirs(spath)
        plot_trajectories(all, os.path.join(spath, f"all_{cfg['experiment']['state_dict_fname'][:-3]}"),
                          plotting_stepsize=3,  #9
                          trajectory_kind='all',
                          ylabel=None)

        if generate_gifs:
            if not os.path.exists('figures/cellsort/gifs'):
                os.makedirs('figures/cellsort/gifs')
            for i in range(num_reps):

                trues_i = _draw_borders_on_type_channel(trues[i])
                preds_i = _draw_borders_on_type_channel(preds_sampled[i])

                trues_i = list(map(lambda x: x[:, 1:], trues_i))
                preds_i = list(map(lambda x: x[:, 1:], preds_i))
                # preds_cont_i = preds_cont[i]

                # plot_cell_image(random_x_i)
                # plot_cell_image(random_y_i, pred_cont_i)
                # plot_cell_image(true_i, pred_i, title_append=f'sample {i}, step {step}')
                time.sleep(0.1)
                make_gif(trues_i, preds_i, f"figures/cellsort/gifs/{cfg['experiment']['state_dict_fname'][:-3]}_{i}.gif",
                         # xlabels=quantiles_i
                         )


def e1_single_loglik(cfg):
    # calculate log likelihood of a single model, averaged over the number of samples it is calculated over (batch dim)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # device='cpu'
    model, pred_stepsize, dataloader, val_dataloader, test_dataloader = load_model_and_get_dataloaders(cfg, True)

    model.to(device)
    # model.beta = 1
    start = time.time()
    loglik = log_likelihood_avg(model, test_dataloader, pred_stepsize, device, dynamic_channels=cfg['dynamic_channels'])
    print(cfg, '\n\n -- loglik', loglik, f' -- took {round(time.time() - start, 2)} seconds')


def e1_agg_stats(cfg, num_samples=3, quantiles=(0.1,), simulated_data=None, label=None):

    # make some plots and calculate some statistics over relevant aggregate statistics, like the number of cell clusters
    # starting from the same initial condition ('test_same_init' loader)

    if label is None:
        label = 'model'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # device='cpu'
    model, pred_stepsize, dataloader, val_dataloader, test_dataloader, test_same_init_loader = load_model_and_get_dataloaders(
        cfg, True, ['test_same_init'])
    # data = test_dataloader.dataset.__getitem__(sim_idx).unsqueeze(0).repeat(num_samples, 1, 1, 1)
    data = torch.cat(
        [test_same_init_loader.dataset.__getitem__(i).unsqueeze(0) for i in range(len(test_same_init_loader.dataset))],
        dim=0)

    if simulated_data is None:  # do rollout as opposed to loading from disk

        data = torch.cat([data for _ in range((num_samples) // data.shape[0] + 1)], dim=0) if num_samples > data.shape[
            0] else data[:num_samples]
        rollout_length = (data.shape[2] - 1) // pred_stepsize

        model.to(device)
        data = data.to(device)
        bs = cfg['batch_size']

        # rollout_length = 50
        all_trues = []
        all_preds_samples = []
        start_rollout_time = time.time()
        for i in range(0, num_samples, bs):


            trues, preds, _ = model_rollout(model, data[i:min(i+bs, num_samples)], pred_stepsize,
                                            rollout_length, start_time=0, use_posterior_sampling=False,
                                            use_ground_truth_input=False)


            preds_samples = np.concatenate([i[:, :, None, ...] for i in preds], axis=2)  # (bs, c, time, h, w)
            trues = np.concatenate([i[:, :, None, ...] for i in trues], axis=2)  # (bs, c, time, h, w)
            all_trues.append(trues)
            all_preds_samples.append(preds_samples)

        print(
            f'rollout of length {rollout_length} took {float(np.round(time.time() - start_rollout_time, 2))} seconds for {int(data.shape[0])} samples')
        trues = torch.from_numpy(np.concatenate(all_trues, axis=0))  #(n_samples, c, time, h, w)
        preds_samples = torch.from_numpy(np.concatenate(all_preds_samples, axis=0))
        np.save(f"evaluation/samples_same_init_{cfg['experiment']['state_dict_fname'][:-3]}", preds_samples)
    else:
        # don't do the simulation but just load the simulated data from file
        trues = data
        sim_data = np.load(simulated_data)
        preds_samples = torch.from_numpy(sim_data)
        # some methods also reconstruct the x^0 and give it as their output, but since we assume the true x^0 is given,
        # we replace it by the ground-truth
        # preds_samples[:,:,0] = trues[:,:,0]


    ####################### calculate avg distance of cells traveled over time (compared to starting position)
    traveled_true = get_avg_cell_dist_traveled(trues)  #(samples, cells, time)
    traveled_samples = get_avg_cell_dist_traveled(preds_samples)
    # ignore nans for (bad) models where cells might disappear
    traveled_true_avg = np.nanmean(traveled_true, axis=1) #(samples, time), avg distance per cell at time point
    traveled_samples_avg = np.nanmean(traveled_samples, axis=1)

    r = range(traveled_true_avg.shape[-1])

    plt.figure()
    # plt.grid()
    plt.xlabel('Time step')
    plt.ylabel('Mean distance traveled per cell')
    pl = plt.plot(r, np.median(traveled_samples_avg, axis=0), '-', label=label, markevery=10)

    for q in quantiles:
        if q != 0.5:
            low = np.quantile(traveled_samples_avg, q=q, axis=0)
            high = np.quantile(traveled_samples_avg, q=1 - q, axis=0)
            plt.fill_between(r, low, high, color=pl[0].get_c(), alpha=0.5)
            # pl = plt.plot(r, , '--', color=pl[0].get_c(), label=f'model ({q}, {1-q}) quantile')
            # plt.plot(r, , '--', color=pl[0].get_c())

    pl = plt.plot(r, np.median(traveled_true_avg, axis=0), '-', label='ground truth', markevery=10)

    for q in quantiles:
        if q != 0.5:
            low = np.quantile(traveled_true_avg, q=q, axis=0)
            high = np.quantile(traveled_true_avg, q=1 - q, axis=0)
            plt.fill_between(r, low, high, color=pl[0].get_c(), alpha=0.5)
            # pl = plt.plot(r, np.quantile(furthest_true, q=q, axis=0), '--', color=pl[0].get_c(), label=f'data ({q}, {1-q}) quantile')
            # plt.plot(r, np.quantile(furthest_true, q=1-q, axis=0), '--', color=pl[0].get_c())

    # plt.title(f"model: {cfg['experiment']['state_dict_fname']}")
    plt.legend(loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.425),frameon=False)
    plt.savefig(f"figures/cellsort/mean_dist_traveled_{cfg['experiment']['state_dict_fname']}.pdf")
    plt.show()

    cell_speed_avg_true = np.mean((traveled_true_avg[:, -1] / traveled_true_avg.shape[1]))
    cell_speed_avg_model = np.mean((traveled_samples_avg[:, -1] / traveled_samples_avg.shape[1]))
    print('done')
    print('average cell speed: true and sample from model')
    print(cell_speed_avg_true)
    print(cell_speed_avg_model)
    #################################################

    # calculate num clusters of the same type
    clusters_true = get_num_clusters(trues)  #(samples, cells, time)
    clusters_samples = get_num_clusters(preds_samples)

    r = range(clusters_true.shape[-1])

    plt.figure()
    # plt.ylim(0, 30)
    # plt.grid()
    plt.xlabel('Time step')
    plt.ylabel('\# cell clusters\n')
    pl = plt.plot(r, np.median(clusters_samples, axis=0), '-', label=label, markevery=10)

    for q in quantiles:
        if q != 0.5:
            low = np.quantile(clusters_samples, q=q, axis=0)
            high = np.quantile(clusters_samples, q=1 - q, axis=0)
            plt.fill_between(r, low, high, color=pl[0].get_c(), alpha=0.5)
            # pl = plt.plot(r, , '--', color=pl[0].get_c(), label=f'model ({q}, {1-q}) quantile')
            # plt.plot(r, , '--', color=pl[0].get_c())

    pl = plt.plot(r, np.median(clusters_true, axis=0), '-', label='ground truth', markevery=10)

    for q in quantiles:
        if q != 0.5:
            low = np.quantile(clusters_true, q=q, axis=0)
            high = np.quantile(clusters_true, q=1 - q, axis=0)
            plt.fill_between(r, low, high, color=pl[0].get_c(), alpha=0.5)
            # pl = plt.plot(r, np.quantile(furthest_true, q=q, axis=0), '--', color=pl[0].get_c(), label=f'data ({q}, {1-q}) quantile')
            # plt.plot(r, np.quantile(furthest_true, q=1-q, axis=0), '--', color=pl[0].get_c())

    # plt.title(f"model: {cfg['experiment']['state_dict_fname']}")
    # locs, labels = plt.yticks()
    # for label_id, l in enumerate(labels):
    #     l_new = ' ' + l.get_text()
    #     labels[label_id] = l_new
    # plt.yticks(locs, labels)
    plt.legend(loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.425),frameon=False)
    plt.savefig(f"figures/cellsort/num_clusters_{cfg['experiment']['state_dict_fname']}.pdf")
    plt.show()
    print('done')

    for t in [15,30,45]:
        statistic, p = kstest(clusters_samples[:, t], clusters_true[:, t])
        print(f'\n\nKS test clusters at time {t}: stat - {statistic} p - {p}')


def get_avg_cell_dist_traveled(all_trajectories):
    # all trajectories: shape(samples, channels 0=id, 1=type, time, h, w)
    cell_id_onehot = utils.get_onehot_grid(all_trajectories[:, 0:1])  #samples, num_cells, time, h, w)
    h = cell_id_onehot.shape[-2]
    w = cell_id_onehot.shape[-1]
    h_range = torch.arange(0, h).view(1, -1, 1).to(cell_id_onehot)
    w_range = torch.arange(0, w).view(1, 1, -1).to(cell_id_onehot)

    coms_h = torch.sum(cell_id_onehot * h_range, dim=(-2, -1)) / torch.sum(cell_id_onehot, dim=(-2, -1))  # (samples, cells, time)
    coms_w = torch.sum(cell_id_onehot * w_range, dim=(-2, -1)) / torch.sum(cell_id_onehot, dim=(-2, -1))  # (samples, cells, time)
    coms = torch.cat([coms_h.unsqueeze(-1), coms_w.unsqueeze(-1)], dim=-1)  # (samples, cells, time, 2)
    dists = torch.cumsum(torch.sqrt(torch.sum((torch.diff(coms, dim=2))**2, dim=-1)), dim=-1)  #(samples, cells, time)
    return dists.numpy()

def get_num_clusters(all_trajectories):
    type_arr = all_trajectories[:, 1]  #(samples, time, h, w)
    num_clusters = []
    for s in range(type_arr.shape[0]):
        num_clusters_this_sample = []
        for t in range(type_arr.shape[1]):
            arr = type_arr[s,t]
            num_clusters_found = 0
            for type in np.unique(arr):
                if type == 0:  # medium
                    continue
                labeled_arr, num_clusters_found_this_type = label(arr == type)  # calculate num disconnected components
                num_clusters_found += num_clusters_found_this_type
            num_clusters_this_sample.append(num_clusters_found)
        num_clusters.append(num_clusters_this_sample)
    return np.array(num_clusters)

def calc_vol_for_all_cells(cell_id):
    # input shape(bs, 1, time, h, w)
    onehot = utils.get_onehot_grid(cell_id, None) # shape(bs, num_cells, time h, w)
    if isinstance(onehot, np.ndarray):
        vol_per_cell = onehot.sum(axis=(-1,-2))
    elif isinstance(onehot, torch.Tensor):
        vol_per_cell = onehot.sum(dim=(-1, -2))
    else:
        raise NotImplementedError('expected cell_id to be either a numpy array or a tensor')
    return vol_per_cell[:,1:]  #(bs, num_cells, time) -- ignore background cell

def rollout_and_plot_trajectory(cfg, save_path, start_time, plotting_stepsize, rollout_length=None, show_com=False):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model, pred_stepsize, cutoff_thresh, dataloader, _, test_dataloader = load_model_and_get_dataloaders(cfg, True)

    model.to(device)

    random_data = next(iter(test_dataloader))
    num_reps = min(10, random_data.shape[0])
    random_data = random_data[:num_reps]
    end_of_sim_time = random_data.size(2)

    if rollout_length is None:
        rollout_length = (random_data.size(2) - 1) // pred_stepsize
    print(f'rollout length: {rollout_length}')

    random_data = random_data.to(device)

    trues, preds, preds_cont = model_rollout(model, random_data, pred_stepsize,
                                             rollout_length, start_time, cutoff_thresh)



    trues = utils.from_time_batch_to_batch_time(trues)
    preds = utils.from_time_batch_to_batch_time(preds)
    # preds_cont = utils.from_time_batch_to_batch_time(preds_cont)

    plot_trajectories(trues, save_path, plotting_stepsize, 'true')
    plot_trajectories(preds, save_path, plotting_stepsize, 'sample')


@torch.no_grad()
def e1_long_simulation_stability(cfgs, vol_lower_threshold, vol_upper_threshold, max_cells_unstable=0, labels=None,
                                 simulation_data=None, run_from_config=True):

    # run_from_config: whether to actually run the model or just load data from disk
    plt.figure()
    data = None
    start_time = 'none'
    pred_stepsize = 'none'
    rollout_length = None
    model=None
    trues_batch = None
    i = 0
    for cfg_id, cfg in enumerate(cfgs):
        i = -1
        model, pred_stepsize, _, _, test_loader, = load_model_and_get_dataloaders(cfg, True,
                                                                                        return_other_loaders=[])
        print(f'succesfully loaded config {cfg_id}')
        device = cfg['device'] if torch.cuda.is_available() else 'cpu'
        data = torch.cat([d for d in test_loader], dim=0)
        if run_from_config:
            i = cfg_id
            # logistics:

            # data=data[:3]
            model.to(device)
            # data = data.to(device)
            start_time = 0
            rollout_length = (data.shape[2] - start_time - 1) // pred_stepsize
            # rollout_length = 10
            preds_batch_all = []
            trues_batch_all = []
            for b, batch in enumerate(test_loader):
                # if b > 1:
                #     break
                batch = batch.to(device)
                # rollout_length = 100
                trues_batch, preds_batch, _ = model_rollout(model, batch, pred_stepsize,
                                            rollout_length, start_time=start_time, use_posterior_sampling=False,
                                            use_ground_truth_input=False)
                preds_batch_all.append(preds_batch)
                trues_batch_all.append(trues_batch)

            # trues = np.swapaxes(np.array(trues), 1, 2)
            preds_batch = np.concatenate([np.concatenate([preds_batch_t[:,:,None] for preds_batch_t in preds_batch], axis=2) for preds_batch in preds_batch_all], axis=0)
            trues_batch = np.concatenate([np.concatenate([trues_batch_t[:,:,None] for trues_batch_t in trues_batch], axis=2) for trues_batch in trues_batch_all], axis=0)
            np.save(f"evaluation/samples_testset_cells_{cfg['experiment']['state_dict_fname'][:-3]}", preds_batch)
            # preds_samples = np.moveaxis(np.array(preds_batch), (0, 1, 2, 3), (2, 0, 1, 3))
            preds_samples = preds_batch[:, 0:1]
            vols = calc_vol_for_all_cells(preds_samples[:,0:1])  #out: bs, cells, time; in: bs, 1, time, h, w
            rollout_length = preds_samples.shape[2]
            print(f'found rollout length {rollout_length} for evaluation (model might have crashed during rollout if this is unexpectedly small)')

            unstable_per_cell = np.bitwise_or(vols < vol_lower_threshold, vols > vol_upper_threshold)  #(bs, cells, time)
            unstable = np.sum(unstable_per_cell, axis=1) > max_cells_unstable
            for t in range(1, rollout_length):
                unstable[:, t] = np.bitwise_or(unstable[:, t], unstable[:, t-1])
            frac_stable = 1 - unstable.mean(axis=0)  # (time,)
            if labels is None:
                label = 'model stability'
            else:
                label = labels[cfg_id]
            plt.plot(range(len(frac_stable)), frac_stable, label=label)

    # load from disk:
    if simulation_data is None:
        simulation_data = []
    linestyles = ['solid', 'dotted', 'dashed', 'dashdot']
    for j, sim_data_path in enumerate(simulation_data):  # skipped if simulation_data is None
        preds_batch = np.load(sim_data_path)
        # some baseline methods also reconstruct the x^0, which is in their simulation output. of course, since we
        # condition of x^0, we just have the true x^0 to start from:
        preds_batch[:,:,0] = data[:,:,0].cpu().numpy()  #bs, 1, time, h, w
        preds_samples = preds_batch[:, 0:1]
        vols = calc_vol_for_all_cells(preds_samples[:, 0:1])  # out: bs, cells, time; in: bs, 1, time, h, w
        rollout_length = preds_samples.shape[2]
        print(
            f'found rollout length {rollout_length} for evaluation (model might have crashed during rollout if this is unexpectedly small)')

        unstable_per_cell = np.bitwise_or(vols < vol_lower_threshold, vols > vol_upper_threshold)  # (bs, cells, time)
        unstable = np.sum(unstable_per_cell, axis=1) > max_cells_unstable
        for t in range(1, rollout_length):
            unstable[:, t] = np.bitwise_or(unstable[:, t], unstable[:, t - 1])
        frac_stable = 1 - unstable.mean(axis=0)  # (time,)
        if labels is None:
            label = 'model stability'
        else:
            label = labels[i + j + 1]

        np.save(f"evaluation/stab_{label}", frac_stable)
        plt.plot(range(len(frac_stable)), frac_stable, label=label, linestyle=linestyles[j])


    # plot
    plt.xlabel('Time step')
    plt.ylabel('Fraction of stable runs')
    plt.xlim(0, 60)
    plt.legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0, frameon=False)
    # plt.legend(frameon=False, loc='best')
    # put_legend_above_fig(frameon=False)
    plt.savefig(f"figures/cellsort/stability_all_cfgs_{labels}.pdf", bbox_inches='tight')
    print('saved stability plot!')
    # plt.show()


def e1_check_equivariance(cfg, num_samples):
    model, pred_stepsize, dataloader, val_dataloader, test_dataloader = load_model_and_get_dataloaders(cfg, True)

    # data = test_dataloader.dataset.__getitem__(sim_idx).unsqueeze(0).repeat(num_samples, 1, 1, 1)
    data = torch.cat([d for d in val_dataloader], dim=0)
    data = torch.cat([data for _ in range((num_samples) // data.shape[0] + 1)], dim=0) if num_samples > data.shape[0] else data[:num_samples]
    data = data.to('cuda')
    # data = data.to('cpu').double()  # note: we need high precision to prevent numerical errors accumulating during the rollout in equivariance checking
    data_permuted = torch.ones_like(data) * data
    bs = min(cfg['batch_size'], num_samples)

    # (n, c, h, w)
    # apply a random permutation to all datapoints:
    perm_idx = torch.Tensor(list(utils.get_permutation_idx(get_identity_permutation=False, num_elements=64))).long()
    perm_idx += 1  # keep background 0
    for j_temp, idx in enumerate(perm_idx):
        j = j_temp + 1 # keep background 0
        data_permuted[:, 0][data[:,0] == j] = idx


    start_time = 0
    rollout_length = (data.shape[2] - start_time - 1) // pred_stepsize
    # rollout_length = 25
    # model.to('cpu').double()
    model.to('cuda')

    all_trues = []
    all_preds_samples = []
    start_rollout_time = time.time()
    for i in range(0, num_samples, bs):
        trues, preds, _ = model_rollout(model, data[i:min(i + bs, num_samples)], pred_stepsize,
                                        rollout_length, start_time=0, use_posterior_sampling=False,
                                        use_ground_truth_input=False)

        preds_samples = np.concatenate([i[:, :, None, ...] for i in preds], axis=2)  # (bs, c, time, h, w)
        trues = np.concatenate([i[:, :, None, ...] for i in trues], axis=2)  # (bs, c, time, h, w)
        # all_trues.append(trues)
        all_preds_samples.append(preds_samples[:,:,-6:])  # only save last few timesteps

    print(
        f'rollout of length {rollout_length} took {float(np.round(time.time() - start_rollout_time, 2))} seconds for {int(data.shape[0])} samples')
    # trues = torch.from_numpy(np.concatenate(all_trues, axis=0))  # (n_samples, c, time, h, w)
    preds_samples_nonpermuted = torch.from_numpy(np.concatenate(all_preds_samples, axis=0))
    preds_samples_permuted_after = torch.ones_like(preds_samples_nonpermuted) * preds_samples_nonpermuted
    for j_temp, idx in enumerate(perm_idx):
        j = j_temp + 1 # keep background 0
        preds_samples_permuted_after[:, 0][preds_samples_nonpermuted[:,0] == j] = idx

    np.save(f"evaluation/samples_equi_test_after_{cfg['experiment']['state_dict_fname'][:-3]}", preds_samples_permuted_after.cpu().numpy())

    dist_cells_perm_after = get_avg_cell_dist_traveled(preds_samples_permuted_after)[..., -1]  #(samples, cells, time)  -> samples, cells

    all_trues = []
    all_preds_samples = []
    start_rollout_time = time.time()
    for i in range(0, num_samples, bs):
        trues, preds, _ = model_rollout(model, data_permuted[i:min(i + bs, num_samples)], pred_stepsize,
                                        rollout_length, start_time=0, use_posterior_sampling=False,
                                        use_ground_truth_input=False)

        preds_samples = np.concatenate([i[:, :, None, ...] for i in preds], axis=2)  # (bs, c, time, h, w)
        trues = np.concatenate([i[:, :, None, ...] for i in trues], axis=2)  # (bs, c, time, h, w)
        # all_trues.append(trues)
        all_preds_samples.append(preds_samples[:,:,-6:])

    print(
        f'rollout of length {rollout_length} took {float(np.round(time.time() - start_rollout_time, 2))} seconds for {int(data.shape[0])} samples')
    # trues = torch.from_numpy(np.concatenate(all_trues, axis=0))  # (n_samples, c, time, h, w)
    preds_samples = torch.from_numpy(np.concatenate(all_preds_samples, axis=0))
    preds_samples_permuted_before = preds_samples

    np.save(f"evaluation/samples_equi_test_before_{cfg['experiment']['state_dict_fname'][:-3]}",
            preds_samples_permuted_before.cpu().numpy())

    dist_cells_perm_before = get_avg_cell_dist_traveled(preds_samples_permuted_before)[...,-1]  #(samples, cells, time) -> samples, cells


    stat, p = kstest(dist_cells_perm_after[:, 42], dist_cells_perm_before[:, 42])
    print(f"model: {cfg['experiment']['state_dict_fname']}")
    print(f'distance traveled before / after permutation -- KS test stat: {stat} -- p-value: {p}\n -------------------')

    return




def _save_individual_images(trues, preds, cfg, save_trues=True, save_every=1, path_prefix='reconstr', idx=[0]):
    idx = list(idx)
    palette = np.array([[0, 0, 0],  # black
                        [0, 0, 60],  # darkish blue
                        [31, 119, 180],  # tab:blue
                        [44, 160, 44],  # tab:orange
                        [255, 215, 0],  # tab:green
                        [214, 39, 40]])  # tab:red
    for bidx, true_traj in enumerate(trues):
        pred_traj = preds[bidx]
        for t, true in enumerate(true_traj):
            if t % save_every != 0:
                continue
            pred = _draw_borders_on_type_channel(pred_traj[t])
            true = _draw_borders_on_type_channel(true)
            if save_trues:
                plt.figure(figsize=(5, 5), dpi=300)
                im_arr = utils.plot_cell_image(true[:, 1:], no_title=True, show_plot=False)
                rgb = palette[im_arr[..., 0].astype(int)]
                plt.imshow(rgb)
                spath = f"figures/cellsort/individual/{cfg['experiment']['state_dict_fname'][:-3]}/true_{idx}/"
                if not os.path.exists(spath):
                    os.makedirs(spath)
                plt.axis('off')
                plt.savefig(spath + f'b_{bidx}_t_{t}.png', dpi=300, bbox_inches='tight')
            plt.figure(figsize=(5, 5), dpi=300)
            im_arr = utils.plot_cell_image(pred[:, 1:], no_title=True, show_plot=False)
            rgb = palette[im_arr[..., 0].astype(int)]
            plt.imshow(rgb)
            spath = f"figures/cellsort/individual/{cfg['experiment']['state_dict_fname'][:-3]}/{path_prefix}_{idx}/"
            if not os.path.exists(spath):
                os.makedirs(spath)
            plt.axis('off')
            plt.savefig(spath + f'b_{bidx}_t_{t}.png', dpi=300, bbox_inches='tight')



if __name__ == '__main__':
    run_experiments()

