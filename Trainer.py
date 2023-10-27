import random
import os
import warnings

import torch
import torch.nn as nn
import time

import utils
from evaluation.experiment_utils import model_rollout
import numpy as np
import matplotlib.pyplot as plt
from utils import *
try:
    import wandb
    use_wandb = True
except:
    use_wandb = False
    print('wandb not found!')

class Trainer():

    def __init__(self, model: nn.Module, loss_func, opt, pred_steps, num_kl_annealing_cycles=1,
                 kl_increase_proportion_per_cycle=1, config=None, try_use_wand=True, beta=1,
                 clip_reconstr_loss_to=torch.inf):



        self.model = model
        self.loss_func = loss_func
        self.pred_stepsize = pred_steps
        self.opt = opt
        self.num_kl_annealing_cycles = num_kl_annealing_cycles
        self.kl_increase_proportion_per_cycle = kl_increase_proportion_per_cycle
        self.use_wandb = use_wandb if try_use_wand else False
        self.beta = beta
        self.reconstr_loss_clip = clip_reconstr_loss_to
        if self.use_wandb:
            try:
                wandb.init(project='EPNS', config=utils.make_dict_serializable(config))
                wandb.watch(self.model, log='all', log_freq=100)
            except Exception as e:
                warnings.warn('failed to connect to wandb somehow! Exception:')
                print(e)
                self.use_wandb = False



    def train(self, loader, val_loader, epochs, device, training_strategy='one-step',
              save_fname=None, start_from_epoch=0):
        if save_fname is None:
            save_fname = 'models/state_dicts/last_experiment.pt'
            print(f'saving weights to default path {save_fname}')
        train_losses = []
        train_acc = []
        val_losses = []
        val_acc = []
        val_rollout_losses = []
        val_rollout_acc = []
        print(f'using {training_strategy} training!')

        best_state_dict = None
        last_state_dict = None
        best_rollout_loss = torch.inf

        for ep in range(start_from_epoch, epochs):
            start_ep = time.time()
            beta = self._get_beta_kl_annealing_schedule(ep, epochs)

            current_weight_dict = self.model.state_dict()
            current_opt_dict = self.opt.state_dict()
            loss = None
            mean_correct = None
            mean_additional_loss = None

            # logic for catching errors due to crashes at an epoch, potentially due to bad minibatch - try to restart from last epoch
            max_num_crashes_per_epoch = 2
            for i in range(max_num_crashes_per_epoch + 1):
                try:
                    # try to do an epoch without crashing
                    loss, mean_correct, mean_additional_loss = self.train_one_ep(loader, device, training_strategy, ep, beta)
                    # if we succeeded, break the trial loop
                    break
                except Exception as e:
                    # if we failed, print the error
                    # if we didnt crash many times already, try again and pray for the best
                    print(
                        f"epoch {ep} training crashed for the {i + 1}'th time.")
                    print('Exception:')
                    print(e)
                    if i < max_num_crashes_per_epoch:
                        print("retrying from state dict before this epoch!")
                        self.model.load_state_dict(current_weight_dict)
                        self.opt.load_state_dict(current_opt_dict)
                    else:
                        print('canceling training due to too many crashes!')
                        raise e
            # statistics and validation:
            train_losses.append(loss), train_acc.append(mean_correct)
            val_loss = 0
            val_mean_correct = 0
            val_mean_additional_loss = 0
            if val_loader is not None:
                val_loss, val_mean_correct, val_mean_additional_loss = self.val(val_loader, device)
                val_losses.append(val_loss)
                val_acc.append(val_mean_correct)
            stop = time.time()
            print(
                f'epoch {ep} \t\t loss {loss:2f} \t\t KL {mean_additional_loss} \t\t mean correct {mean_correct:2f}'
                f'\t\t '
                f'val loss {val_loss} \t\t val KL {val_mean_additional_loss} \t\t val mean correct {val_mean_correct:2f}'
                f'\t\t took {stop - start_ep:2f} seconds'
            , flush=True)
            if self.use_wandb:
                # log to wandb if applicable
                wandb.log({'loss':loss, 'mean correct':mean_correct, 'KL': mean_additional_loss,
                           'reconstruction loss':loss - mean_additional_loss,
                           'val loss': val_loss, 'val mean correct': val_mean_correct, 'val KL': val_mean_additional_loss,
                           'val reconstruction loss': val_loss - val_mean_additional_loss,
                           'epoch': ep, 'beta': beta})
            # every 25 epochs, perform a rollout validation with multi-step training logic. save state dict if it is the best so far
            if ep % 25 == 0 and ep>0 and val_loader is not None:
                start_val_rollout = time.time()
                try:
                    val_rollout_loss, val_rollout_mean_correct = self.val_with_rollout(val_loader, device)
                    val_rollout_losses.append(val_rollout_loss)
                    val_rollout_acc.append(val_rollout_mean_correct)
                    print(
                        f'val rollout loss: {val_rollout_loss}, val rollout mean correct: {val_rollout_mean_correct}, took {time.time() - start_val_rollout:2f} seconds',
                        flush=True)
                    if val_rollout_loss <= best_rollout_loss:
                        print(f'found new best weights at epoch {ep}')
                        best_rollout_loss = val_rollout_loss
                        best_state_dict = self.model.state_dict()
                        torch.save(best_state_dict, os.path.join('models', 'state_dicts', save_fname))
                        if self.use_wandb:
                            art = wandb.Artifact(save_fname + 'best', type="model")
                            art.add_file(os.path.join('models', 'state_dicts', save_fname))
                            wandb.log_artifact(art)
                except utils.PCAException as e:
                    print(f'!!!! val rollout failed at epoch {ep} !!!!')
                    print('exception:', e)

            # save the last state dict every ten epochs and log to wandb if applicable
            if not os.path.exists(os.path.join('models', 'state_dicts', 'last')):
                os.makedirs(os.path.join('models', 'state_dicts', 'last'))
            last_state_dict = self.model.state_dict()
            torch.save(last_state_dict, os.path.join('models', 'state_dicts', 'last', save_fname))
            if self.use_wandb and (ep + 1) % 10 == 0:  # log the state dict every 10 epochs
                art = wandb.Artifact(save_fname + 'last', type="model")
                art.add_file(os.path.join('models', 'state_dicts', 'last', save_fname))
                wandb.log_artifact(art)
        return train_losses, train_acc, val_losses, val_acc, best_state_dict, last_state_dict

    def train_one_ep(self, loader, device, training_strategy, epoch, beta=1):

        self.model.to(device)
        self.model.train()

        mean_loss = 0
        mean_additional_loss = 0  # for now this is jsut KL loss
        count = 0
        mean_correct = 0
        max_rollout_len_set_flag = False
        max_rollout_len_multistep = 1

        for (batch, data) in enumerate(loader):
            if epoch > 50:
                # rollout length for multi-step training:
                max_rollout_len_multistep = min((epoch-40) // 10 + 1, (data.size(2) - 1) / 3 // self.pred_stepsize)
                max_rollout_len_multistep = int(max_rollout_len_multistep)
            if training_strategy == 'multi-step':
                rollout_length = max_rollout_len_multistep
            if not max_rollout_len_set_flag:
                max_rollout_len_set_flag = True
                print(f'training rollout len is {max_rollout_len_multistep}. Beta: {beta}')

            if training_strategy == 'one-step':
                loss, correct, additional_loss = self.train_step_onestep(data, device, beta)
            elif training_strategy == 'multi-step':
                loss, correct, additional_loss = self.train_step_multistep(data, device, rollout_length, beta)
            else:
                raise NotImplementedError()

            mean_loss += loss
            mean_additional_loss += additional_loss
            mean_correct += correct
            count += data.size(0)

        return mean_loss / count, mean_correct / count, mean_additional_loss / count

    @torch.no_grad()
    def val(self, val_loader, device):
        # validation loop with single-step training logic
        self.model.eval()
        mean_loss = 0
        mean_additional_loss = 0
        count = 0
        mean_correct = 0
        for data in val_loader:
            #  shape of data: (bs, channels, time, spatial_x, spatial_y)
            end_of_sim_time = data.size(2)
            start_time = np.random.randint(low=0, high=end_of_sim_time - self.pred_stepsize, size=data.size(0))
            target_time = start_time + self.pred_stepsize
            x = data[range(data.size(0)), :, start_time].to(device)
            y = data[range(data.size(0)), :, target_time].to(device)
            y_pred_dist, additional_loss, y_pred_disc, *miscellaneous = self.model(x, y)
            mean_additional_loss += additional_loss
            # print(y_pred - y.to(device))
            loss = -y_pred_dist.log_prob(y).sum() + additional_loss
            mean_loss += loss.item()
            count += data.size(0)
            mean_correct += self.model.get_additional_val_stats(y_pred_disc, y)
        return mean_loss / count, mean_correct / count, mean_additional_loss / count

    @torch.no_grad()
    def val_with_rollout(self, val_loader, device):
        # validation loop with multi-step training logic -- can be a bit expensive so dont call too often
        self.model.eval()
        mean_loss = 0
        count = 0
        mean_correct = 0
        rollout_length_max = 100
        for data in val_loader:
            #  shape of data: (bs, channels, time, spatial_x, spatial_y)
            data = data.to(device)
            start_time = 0
            rollout_length = (data.size(2) - 1) // self.pred_stepsize
            rollout_length = min(rollout_length_max, rollout_length)
            trues, pred, pred_cont = model_rollout(
                self.model, data, self.pred_stepsize, rollout_length, start_time, True)
            trues = torch.cat([torch.from_numpy(arr) for arr in trues[1:]], dim=0)
            pred = torch.cat([torch.from_numpy(arr) for arr in pred[1:]], dim=0)
            pred_cont = torch.cat([torch.from_numpy(arr) for arr in pred_cont[1:]], dim=0)
            loss = torch.sum(self.loss_func(pred_cont, trues))
            mean_loss += loss.item()
            count += data.size(0)
            mean_correct += self.model.get_additional_val_stats(pred, trues)
        return mean_loss / count / rollout_length, mean_correct / count / rollout_length

    def train_step_onestep(self, data, device, beta):
        # one-step training for one batch
        end_of_sim_time = data.size(2)
        start_time = np.random.randint(low=0, high=end_of_sim_time - self.pred_stepsize, size=data.size(0))
        target_time = start_time + self.pred_stepsize
        x = data[range(data.size(0)), :, start_time].to(device)
        y = data[range(data.size(0)), :, target_time].to(device)
        miscellaneous = []
        y_pred_dist, additional_loss, y_pred_disc, *miscellaneous = self.model(x, y, *miscellaneous)
        loss = - torch.clip(y_pred_dist.log_prob(y), -self.reconstr_loss_clip, self.reconstr_loss_clip).sum() + beta*additional_loss
        with torch.no_grad():
            # rounded_pred = torch.round(y_pred)
            add_stat = self.model.get_additional_val_stats(y_pred_disc, y)

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return loss.item(), add_stat, additional_loss.detach().item()

    def train_step_multistep(self, data, device, rollout_length, beta):
        # multi-step training with a single batch
        self.opt.zero_grad()
        # some trackers:
        loss_over_all_steps = 0
        add_loss_over_all_steps = 0
        add_stat_over_all_steps = 0

        # get initial conditions from random time
        end_of_sim_time = data.size(2)
        start_time = np.random.randint(low=0, high=end_of_sim_time - self.pred_stepsize * rollout_length, size=data.size(0))
        x = data[range(data.size(0)), :, start_time].to(device)

        # do multiple prediction steps with gradient tracking
        miscellaneous = []
        for step in range(1, rollout_length+1):
            target_time = start_time + self.pred_stepsize * step
            y = data[range(data.size(0)), :, target_time].to(device)

            try:
                y_pred_dist, additional_loss, y_pred_disc, *miscellaneous = self.model(x, y, *miscellaneous)
            except utils.PCAException as e:
                print(repr(e), f'aborting further rollout at step {step}!')
                print('min - mean - max of centered points for which the exception occurred:', e.pnts_centered.min(), e.pnts_centered.mean(), e.pnts_centered.max())
                rollout_length = step
                return loss_over_all_steps, add_stat_over_all_steps, add_loss_over_all_steps

            loss = - torch.clip(y_pred_dist.log_prob(y), -self.reconstr_loss_clip, self.reconstr_loss_clip).sum() + beta*additional_loss
            loss.backward()

            # the below two tensors are merely for tracking statistics
            loss_over_all_steps = loss_over_all_steps + loss.detach().item()
            add_loss_over_all_steps = add_loss_over_all_steps + additional_loss.detach().item()

            with torch.no_grad():
                add_stat = self.model.get_additional_val_stats(y_pred_disc, y)
                add_stat_over_all_steps += add_stat

            # now, the model input for the next steps becomes y_pred
            x = torch.ones_like(x) * y_pred_disc.detach()
            assert not x.requires_grad,\
                'we should not backpropagate through this for single-step training'

        loss_over_all_steps /= rollout_length
        add_loss_over_all_steps /= rollout_length
        add_stat_over_all_steps /= rollout_length

        # loss_over_all_steps.backward(), we already backpropagated all gradients in the loop above
        self.opt.step()

        return loss_over_all_steps, add_stat_over_all_steps, add_loss_over_all_steps


    def _get_beta_kl_annealing_schedule(self, current_ep, total_epochs):
        # calculate the beta for the annealing schedule

        cycle_length = total_epochs // self.num_kl_annealing_cycles
        ep_in_this_cycle = current_ep % cycle_length
        p_of_cycle = (ep_in_this_cycle + 1) / cycle_length

        if current_ep >= cycle_length * self.num_kl_annealing_cycles:
            beta = self.beta
        elif p_of_cycle > self.kl_increase_proportion_per_cycle:
            beta = self.beta
        else:
            beta = min(p_of_cycle / self.kl_increase_proportion_per_cycle * self.beta, self.beta)

        return beta

