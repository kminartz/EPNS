import random

import matplotlib.pyplot as plt
import sys
sys.path.append(".")
sys.path.append("..")
import utils
from experiment_utils import *
from utils import set_size_plots
import itertools
import torch
import torch.nn as nn
import numpy as np
import os
import time
import seaborn as sns
from tueplots import bundles, figsizes
plot_bundle = bundles.neurips2023()
width = 397.48499  # width in points
# plot_bundle['axes.labelsize'] = 10
# plot_bundle['legend.fontsize'] = 10
plt.rcParams.update(plot_bundle)
figsize = figsizes.neurips2023(ncols=1)
num_fig_columns = 2.5
figsize['figure.figsize'] = set_size_plots(width, fraction=1/num_fig_columns, h_to_w_ratio=None)
# num_fig_columns = 2
# figsize['figure.figsize'] = set_size_plots(width, fraction=1/num_fig_columns, h_to_w_ratio=0.55)
# # figsize["figure.figsize"] = tuple(map(lambda x: x * 2/3, figsize["figure.figsize"]))  # two subplots per column
# figsize['figure.figsize'] = (figsize['figure.figsize'][0] / num_fig_columns, figsize['figure.figsize'][1] / num_fig_columns)


plt.rcParams.update(figsize)
import matplotlib.cm
from utils import create_circular_mask, load_config

from scipy.stats import kstest


def run_experiments():
    cfgs = ['n_body_dynamics_EPNS', 'n_body_dynamics_PNS', 'n_body_dynamics_PNS_MLP']
    state_dicts = [None, None, None]
    # state_dicts = [None, 'n_body_PNS_second_try.pt', 'crazy_MLP_model.pt']  # None for using the state dict from the config file, string for any name
    # it will look for the state dicts in the models/state_dicts folder, with priority for those in the /final/ folder
    labels = ['EPNS', 'PNS', 'PNS-MLP']


    cfgs_loaded = []
    for i, c in enumerate(cfgs):
        cfgs_loaded.append(load_config(c, state_dicts[i]).config_dict)
    cfgs = cfgs_loaded

    # please check this function to comment out the experiments you don't want to run:
    e1(cfgs, labels=labels)
    return

def e1(cfgs, labels=None, simulation_data=None):
    with torch.no_grad():
        print('running experiment 1 \n\n')
        for i, c in enumerate(cfgs):
            print('-' *20, '\n\n')
            print(c)
            # e1_timer(c, 'cpu', 1, 5)
            e1_single_loglik(c, use_05_std=False)
            e1_agg_stats(c, num_samples=100, quantiles=(0.1,), label=labels[i] if labels is not None else None)
            e1_check_equivariance(c, num_samples=1000)

        e1_long_simulation_stability(cfgs, delta_E_threshold=20, labels=labels, simulation_data=simulation_data)

    return

def e1_timer(cfg, device, batch_size, num_repetitions):
    model, pred_stepsize, dataloader, _, test_dataloader = load_model_and_get_dataloaders(cfg, True)

    model.to(device)

    random_data = next(iter(test_dataloader))
    num_reps = min(batch_size, random_data.shape[0])
    print(f'processing {batch_size} samples in a batch')

    random_data = random_data[:num_reps]

    rollout_length = (random_data.size(2)) // pred_stepsize

    start_time = 0

    random_data = random_data.to(device)

    print(f'rollout length: {rollout_length}')

    # initialize before starting time measurements -- makes sure we have memory allocated already!
    model_rollout(model, random_data, pred_stepsize, 2,
                  start_time)

    time.sleep(5)
    print('starting measurements')

    times = []
    total_start = time.time()
    for _ in range(num_repetitions):
        start = time.time()
        trues, preds, preds_cont = model_rollout(model, random_data, pred_stepsize, rollout_length,
                  start_time)
        times.append(time.time() - start)

    print(f'processing a batch of size {batch_size} on device {device} took {np.mean(times)} on avg'
          f' (std {np.std(times)}). Total exp time {time.time() - total_start} for {num_repetitions} repetitions')



def e1_single_loglik(cfg, use_05_std=False):
    # calculate loglik averaged over the batch
    # if use_05_std is set to true, the decoding distribution will have a fixed sigma that is not learned
    # such that the reconstruction loss is equivalent to MSE + a constant
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # device='cpu'
    model, pred_stepsize, dataloader, val_dataloader, test_dataloader = load_model_and_get_dataloaders(cfg, True)

    model.to(device)
    if use_05_std:
        model.use_05_std=True
    start = time.time()
    print(f'the model has {utils.count_parameters(model)} parameters!')
    loglik = log_likelihood_avg(model, test_dataloader, pred_stepsize, device, dynamic_channels=model.dynamic_channels)
    print(cfg, '\n\n -- loglik', loglik, f' -- took {round(time.time() - start, 2)} seconds')
    model.use_05_std=False


def e1_agg_stats(cfg, num_samples=1000, quantiles=(0.05, 0.5), simulated_data=None, label=None):

    # plot some aggregate properties and calculate some statistics -- phase space, distance from origin, energies

    if label is None:
        label = 'model'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # device='cpu'
    model, pred_stepsize, dataloader, val_dataloader, test_dataloader, test_same_init_loader = load_model_and_get_dataloaders(
        cfg,
        True,
        ('test_same_init', ))
    loader = test_same_init_loader
    # data = test_dataloader.dataset.__getitem__(sim_idx).unsqueeze(0).repeat(num_samples, 1, 1, 1)
    data = torch.cat([d.unsqueeze(0) for d in loader.dataset], dim=0)
    data = torch.cat([data for _ in range((num_samples) // data.shape[0] + 1)], dim=0) if num_samples > data.shape[
        0] else data[:num_samples]
    rollout_length = data.shape[2] // pred_stepsize
    if simulated_data is None:
        start_rollout_time = time.time()
        model.to(device)
        data = data.to(device)
        # rollout_length = 50
        trues, preds, _ = model_rollout(model, data, pred_stepsize,
                                                 rollout_length, start_time=0, use_posterior_sampling=False,
                                                 use_ground_truth_input=False)
        print(f'rollout of length {rollout_length} took {float(np.round(time.time()-start_rollout_time, 2))} seconds for {int(data.shape[0])} samples')

        preds_samples = np.concatenate([i[:, :, None, ...] for i in preds], axis=2)   # (bs, n_bodies, time, feat)
        trues = np.concatenate([i[:, :, None, ...] for i in trues], axis=2)  # (bs, n_bodies, time, feat)
    else:
        print('loading simulations from disk!')
        trues = data[:, :, ::pred_stepsize].numpy()
        preds_samples = np.load(simulated_data)
        if preds_samples.shape[-1] == 6:  #Nsde - did nto save mass, so put it back
            preds_samples = np.load(simulated_data)[:,:,:101]
            mass = torch.from_numpy(trues[..., 0:1, 0:1]).repeat(1, 1, preds_samples.shape[2], 1).numpy()
            preds_samples = np.concatenate([mass, preds_samples], axis=-1)

    ######################### phase space pdf per body ################################################
    # idx = 42
    linetypes = ['-', '--']
    for i in range(1, 7):  # x, y, z coordinate
        for b in range(1, min(preds_samples.shape[1], 5)):  # for each body
            plt.figure()
            for d, dataset in enumerate([trues, preds_samples]):
                data_label = ['ground truth pdf', 'model pdf'][d]
                ltyp = linetypes[d]
                coors_sampled = dataset[:, b, :, i].flatten()
                plt.hist(coors_sampled, density=True, label=data_label, histtype=u'step', linestyle=ltyp, bins=25)
            coor = ['x', 'y', 'z', 'v_x', 'v_y', 'v_z'][i-1]
            coor_latex = ['$x$', '$y$', '$z$', '$v_x$', '$v_y$', '$v_z$'][i-1]
            plt.title(f"{coor_latex}")
            # plt.legend(loc='best', frameon=False)
            plt.xlabel('Coordinate value')
            plt.ylabel('Density')
            # plt.yscale('log', nonposy='clip')
            p = f"figures/n_body/phase_space/{cfg['experiment']['state_dict_fname']}/"
            if not os.path.exists(p):
                os.makedirs(p)
            plt.savefig(p + f"body-{b}-{coor}.pdf")





    ############################## distance of body furthest from origin ###############################

    dist_to_center = np.sqrt(np.sum(preds_samples[..., 1:4]**2, axis=-1))  # (bs, n_bodies, time)
    furthest = np.max(dist_to_center, axis=1)  # (bs, time)

    furthest_median = np.median(furthest, axis=0)  # (time)
    r = range(len(furthest_median))

    plt.figure()
    # plt.grid()
    plt.xlabel('Time step')
    plt.ylabel('Distance of furthest body\n to origin')
    pl = plt.plot(r, furthest_median, '-', label=label, markevery=10)

    for q in quantiles:
        if q != 0.5:
            low = np.quantile(furthest, q=q, axis=0)
            high = np.quantile(furthest, q=1-q, axis=0)
            plt.fill_between(r, low, high, color=pl[0].get_c(), alpha=0.5)

        else:
            plt.plot(r, np.quantile(furthest, q=q, axis=0), '-.', color=pl[0].get_c(), label=label)


    dist_to_center_true = np.sqrt(np.sum(trues[..., 1:4] ** 2, axis=-1))  # (bs, n_bodies, time)
    furthest_true = np.max(dist_to_center_true, axis=1)

    furthest_true_median = np.median(furthest_true, axis=0)
    pl = plt.plot(r, furthest_true_median, '-', label='ground truth', markevery=10)

    for q in quantiles:
        if q != 0.5:
            low = np.quantile(furthest_true, q=q, axis=0)
            high = np.quantile(furthest_true, q=1-q, axis=0)
            plt.fill_between(r, low, high, color=pl[0].get_c(), alpha=0.5)
            # pl = plt.plot(r, np.quantile(furthest_true, q=q, axis=0), '--', color=pl[0].get_c(), label=f'data ({q}, {1-q}) quantile')
            # plt.plot(r, np.quantile(furthest_true, q=1-q, axis=0), '--', color=pl[0].get_c())
        else:
            plt.plot(r, np.quantile(furthest_true, q=q, axis=0), '-.',color=pl[0].get_c(), label='ground truth')

    # plt.title(f"model: {cfg['experiment']['state_dict_fname']}")
    plt.legend(loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.425),frameon=False)
    plt.savefig(f"figures/n_body/dist_furthest_body_{cfg['experiment']['state_dict_fname']}.pdf")
    plt.show()


    statistic, p = kstest(furthest[:, -1], furthest_true[:, -1])
    print(f'\n\nKS test distance at final time: stat - {statistic} p - {p}')

    ############################## energy levels ###############################

    plt.figure()
    KEs_model = np.zeros((data.shape[0], rollout_length))
    PEs_model = np.zeros((data.shape[0], rollout_length))
    KEs_true = np.zeros((data.shape[0], rollout_length))
    PEs_true = np.zeros((data.shape[0], rollout_length))


    for t in range(rollout_length):
        KE, PE = model.calc_energy(torch.Tensor(preds_samples[:,:, t, :]), idx_pos=(1,2,3), idx_vel=(4,5,6),
                                   return_split=True)
        KEs_model[:, t] = KE.cpu().numpy()
        PEs_model[:,t] = PE.cpu().numpy()

        KE_true, PE_true = model.calc_energy(torch.Tensor(trues[:,:,t,:]), idx_pos=(1,2,3), idx_vel=(4,5,6),
                                             return_split=True)
        KEs_true[:, t] = KE_true.cpu().numpy()
        PEs_true[:, t] = PE_true.cpu().numpy()

    r = list(range(rollout_length))
    energies_model = {'Kinetic energy': KEs_model, 'Potential energy': PEs_model, 'Total energy': KEs_model + PEs_model}
    energies_true = {'Kinetic energy': KEs_true, 'Potential energy': PEs_true, 'Total energy': KEs_true + PEs_true}
    for k in energies_model.keys():
        energy_model = energies_model[k]
        energy_true = energies_true[k]

        pl = plt.plot(r, np.median(energy_model, axis=0), '-', label=label, markevery=10)
        for q in quantiles:
            if q != 0.5:
                low = np.quantile(energy_model, q=q, axis=0)
                high = np.quantile(energy_model, q=1 - q, axis=0)
                plt.fill_between(r, low, high, color=pl[0].get_c(), alpha=0.5)
                # pl = plt.plot(r, np.quantile(energy_model, q=q, axis=0), '--', label=f'model ({q}, {1-q}) quantile', color=pl[0].get_c())
                # plt.plot(r, np.quantile(energy_model, q=1-q, axis=0), '--', color=pl[0].get_c())
            else:
                plt.plot(r, np.quantile(energy_model, q=q, axis=0), '-.',color=pl[0].get_c(), label=label)

        pl = plt.plot(r, np.median(energy_true, axis=0), '-', label='ground truth', markevery=10)
        for q in quantiles:
            if q != 0.5:
                low = np.quantile(energy_true, q=q, axis=0)
                high = np.quantile(energy_true, q=1 - q, axis=0)
                plt.fill_between(r, low, high, color=pl[0].get_c(), alpha=0.5)
                # pl = plt.plot(r, np.quantile(energy_true, q=q, axis=0), '--', label=f'data ({q}, {1-q}) quantile',color=pl[0].get_c())
                # plt.plot(r, np.quantile(energy_true, q=1-q, axis=0), '--', color=pl[0].get_c())
            else:
                plt.plot(r, np.quantile(energy_true, q=q, axis=0), '-.',color=pl[0].get_c(), label='ground truth')

        for t in [24, 49, 99]:
            statistic, p = kstest(energy_model[:, t], energy_true[:, t])
            print(f'\n\nKS test at time {t} -- {k}: stat - {statistic} p - {p}')



        # plt.title(f"model: {cfg['experiment']['state_dict_fname']}")
        plt.xlabel('Time step')
        plt.ylabel(k)
        # plt.tight_layout()
        # plt.grid()
        plt.legend(loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.425),frameon=False)
        plt.savefig(f"figures/n_body/{k}_{cfg['experiment']['state_dict_fname']}.pdf", )
        plt.show()

@torch.no_grad()
def e1_long_simulation_stability(cfgs, delta_E_threshold=20, labels=None, simulation_data=None):
    plt.figure()
    data = None
    start_time = 0
    pred_stepsize = 'none'
    rollout_length = None
    model=None
    i = 0
    linestyles = ['solid', 'dotted', 'dashed', 'dashdot']
    for cfg_id, cfg in enumerate(cfgs):
        i = cfg_id
        try:  # try loading a separate long simulation test file (not available for n=20 experiment)
            model, pred_stepsize, _, _, _, long_sim_loader = load_model_and_get_dataloaders(cfg, True,
                                                                                            return_other_loaders=('test_long_simulation', ))

            data = torch.cat([d for d in long_sim_loader], dim=0)
            rollout_length = (data.shape[2] - start_time - 1) // pred_stepsize
        except:  # n=20 experiment
            model, pred_stepsize, _, _, test_loader, same_init_loader = load_model_and_get_dataloaders(cfg, True,
                                                                                            return_other_loaders=('test_same_init', ))
            long_sim_loader = test_loader
            data = torch.cat([d for d in long_sim_loader], dim=0)
            # dirty hack for n=20: ground truth does not matter for this experiment as we only look at the model samples
            # data = torch.cat([data for _ in range(10)], dim=2)
            rollout_length = 1000
            # data = data[:,:,:10]


        # logistics:
        device = cfg['device'] if torch.cuda.is_available() else 'cpu'
        # data=data[:3]
        model.to(device)
        data = data.to(device)


        preds_batch = model_rollout_split_batch_on_crash(model, data, pred_stepsize,
                                    rollout_length, start_time=start_time, use_posterior_sampling=False,
                                    use_ground_truth_input=False)

        preds_samples = np.moveaxis(np.array(preds_batch), (0, 1, 2, 3), (2, 0, 1, 3))

        all_E = []
        rollout_length = min(rollout_length, preds_samples.shape[2])
        print(f'found rollout length {rollout_length} for evaluation (model might have crashed during rollout if this is unexpectedly small)')
        for t in range(rollout_length):
            KE, PE = model.calc_energy(torch.Tensor(preds_samples[:, :, t, :]), idx_pos=(1, 2, 3), idx_vel=(4, 5, 6),
                                       return_split=True)
            E_t = KE + PE  # (bs, )
            all_E.append(E_t[:, None].cpu().numpy())
        all_E = np.concatenate(all_E, axis=1)  # (bs, time)
        delta_E = np.diff(all_E, axis=1, prepend=all_E[:,0:1])
        unstable = delta_E > delta_E_threshold
        for t in range(1, rollout_length):
            unstable[:, t] = np.bitwise_or(unstable[:, t], unstable[:, t-1])
        frac_stable = 1 - unstable.mean(axis=0)  # (time,)
        if labels is None:
            label = 'model stability'
        else:
            label = labels[cfg_id]
        plt.plot(range(len(frac_stable)), frac_stable, label=label, linestyle=linestyles[i])

    # do not load from disk:
    if simulation_data is None:
        simulation_data = []

    # load some simulations from disk if there are some provided:
    for j, sim_data_path in enumerate(simulation_data):
        preds_samples = np.load(sim_data_path)

        if preds_samples.shape[-1] == 6:
            # in the case of nsde, re-append the constant mass data that we did not save to disk
            s = preds_samples.shape
            mass = data[...,0:1, 0:1].repeat(1,1, s[2], 1).cpu().numpy()
            preds_samples = np.concatenate([mass, preds_samples], axis=-1)

        all_E = []
        rollout_length = min(rollout_length, preds_samples.shape[2])
        print(
            f'found rollout length {rollout_length} for evaluation (model might have crashed during rollout if this is unexpectedly small)')
        for t in range(rollout_length):
            KE, PE = model.calc_energy(torch.Tensor(preds_samples[:, :, t, :]), idx_pos=(1, 2, 3), idx_vel=(4, 5, 6),
                                       return_split=True)
            E_t = KE + PE  # (bs, )
            all_E.append(E_t[:, None].cpu().numpy())
        all_E = np.concatenate(all_E, axis=1)  # (bs, time)
        delta_E = np.diff(all_E, axis=1, prepend=all_E[:, 0:1])
        unstable = delta_E > delta_E_threshold
        for t in range(1, rollout_length):
            unstable[:, t] = np.bitwise_or(unstable[:, t], unstable[:, t - 1])
        frac_stable = 1 - unstable.mean(axis=0)  # (time,)
        if labels is None:
            label = 'model stability'
        else:
            label = labels[j + i + 1]
        plt.plot(range(len(frac_stable)), frac_stable, label=label, linestyle=linestyles[j + i + 1])


    plt.legend(frameon=False, loc='best', ncol=2)
    plt.xlabel('Time step')
    plt.ylabel('Fraction of stable runs')
    # plt.ylim(0, 1)
    if not os.path.exists('figures/n_body/'):
        os.makedirs('figures/n_body/')
    plt.savefig(f"figures/n_body/stability_all_cfgs_nbody.pdf")
    # plt.show()

@torch.no_grad()
def e1_check_equivariance(cfg, num_samples):
    model, pred_stepsize, dataloader, val_dataloader, test_dataloader = load_model_and_get_dataloaders(cfg, True)
    # data = test_dataloader.dataset.__getitem__(sim_idx).unsqueeze(0).repeat(num_samples, 1, 1, 1)
    data = torch.cat([d for d in test_dataloader], dim=0)
    data = torch.cat([data for _ in range((num_samples) // data.shape[0] + 1)], dim=0) if num_samples > data.shape[0] else data[:num_samples]
    data = data.to('cuda')
    # data = data.to('cpu').double()  # note: we need high precision to prevent numerical errors accumulating during the rollout as much as possible
    data_rotated = torch.ones_like(data) * data
    # (n, nodes, t, [m, x, v])
    # apply a random rotation to all datapoints:
    matrices = utils.get_three_rotation_matrices(get_identity_rotation_matrices=False, rotate_only_2d=False)

    for mat in matrices:
        data_rotated[..., 1:4] = data_rotated[..., 1:4] @ mat.float().cuda()
        data_rotated[..., 4:] = data_rotated[..., 4:] @ mat.float().cuda()

    shuffle_idx = utils.get_permutation_idx(get_identity_permutation=True)
    data_rotated = data_rotated[:, shuffle_idx, ...]

    start_time = 0
    rollout_length = (data.shape[2] - start_time - 1) // pred_stepsize
    # rollout_length = 25
    model.to('cuda')
    # model.to('cpu').double()

    # torch.random.manual_seed(42)
    trues, preds, _ = model_rollout(model, data, pred_stepsize,
                                    rollout_length, start_time=start_time, use_posterior_sampling=False,
                                    use_ground_truth_input=False)
    # torch.random.manual_seed(42)
    _, preds_rotated, _ = model_rollout(model, data_rotated, pred_stepsize,
                                    rollout_length, start_time=start_time, use_posterior_sampling=False,
                                    use_ground_truth_input=False)

    preds_rotated_after_model = torch.Tensor(preds)
    for mat in matrices:
        preds_rotated_after_model[..., 1:4] = preds_rotated_after_model[..., 1:4] @ mat.float()
        preds_rotated_after_model[..., 4:] = preds_rotated_after_model[..., 4:] @ mat.float()

    preds_rotated_after_model = preds_rotated_after_model[:, :, shuffle_idx]  # permutation

    preds_rotated = torch.Tensor(preds_rotated)

    torch.random.seed()

    ks_tests_coors_p_values = []
    ks_tests_coors_stats = []
    for i in range(1, 4):
        # only consider coordinates at last timestep, not velocities (just for simplicity, in the end we care more about coors anyway for a prediction)
        stat, p = kstest(preds_rotated[-1, :, 1, i].flatten().numpy(), preds_rotated_after_model[-1, :, 1, i].flatten().numpy())
        ks_tests_coors_p_values.append(np.round(p, decimals=4))
        ks_tests_coors_stats.append(np.round(stat, decimals=4))
    print(f'p-values ks-test for each coordinate: {ks_tests_coors_p_values}')
    print(f'stats ks-test for each coordinate: {ks_tests_coors_stats}')
    return



if __name__ == '__main__':
    run_experiments()



