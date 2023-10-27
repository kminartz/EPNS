
# !/usr/bin/env python3
#
# N-BODY DYNAMICS SIMULATOR
#
# SIMULATES THE DYNAMICS OF N MASSIVE BODIES UNDER THE GRAVITATIONAL FORCE
# MASSES ARE SIMULATED AS POINT MASSES AND PROPAGATED FORWARD IN TIME VIA
# KICK-DRIFT-KICK DISCRETE TIME PROPAGATION (SEE n_body.simulate() BELOW)
#
# AUTHOR: KYLE MORGENSTEIN (KYLEM@UTEXAS.EDU)
# DATE: 11/28/2020
#
# MIT License
#
# Copyright (c) 2020 KYLE MORGENSTEIN
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# imports
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import time
import datetime
import os
import torch
from evaluation.experiment_utils import load_model_and_get_dataloaders
# np.random.seed(42)
import evaluation.experiment_utils
from utils import load_config

class n_body_plotter:
    '''
    USAGE DOCUMENTATION

    =====================================================================

    #only necessary if the code is being run in a different file
    from n_body import *

    #otherwise copy and paste as follows:
    #(this code is supplied at the bottom of the file as well)

    #this example generates the movie file listed in the repository under "n_body_trials/*"
    #using:
    # G = 1
    # view_lim = 5
    # save = 100
    # run_time = 100

    #set initial conditions
    init_conds = np.array([
        [100,0,0,0,0,0,0],
        [1,0,-3,0,3,4,0],
        [1,0,3,0,-3,-4,0],
        [1,-3,0,0,0,-4,0],
        [1,3,0,0,0,4,0]
        ])

    #set experiment runtime
    run_time = 10 #[seconds]

    #initialize simulation class
    sim = n_body(init_conds,run_time)

    #run simulation
    sim.simulate()

    #plot the results
    save = 100 #sets framerate for saved simulation - set to 0 to playback only
    autoscroll = False #automatically adjusts view to keep all masses in view
    replay = False #better to just generate the video and watch it at full speed
    view_lim = 20 #scales view to [-view_lim, view_lim]

    sim.plot(save,autoscroll,replay,view_lim)

    #saved simulations take up ~2-20 MB depedning on run_time
    #generating the simulated video will take ~1-15 minutes
    #depending on the length of simulation and your hardware

    #WAIT UNTIL THE CODE FINISHES RUNNING COMPLETLY BEFORE TRYING TO OPEN THE VIDEO FILE

    =====================================================================

    OTHER HELPFUL USAGE TIPS

    #if you don'y care about selecting parameters,
    #the entire simulation can be run inline as
    n = 3 #number of masses to simulate
    sim = n_body(3).run()

    or

    #randomly generates 2-5 masses
    sim = n_body().run()

    The parameters you should focus on changing are:
    * initial_conds - be creative with your initial conditions!
    * self.G - strength of gravity
    * self.S - damping on collisions
    * run_time - simulation length
    * scale - scales maximum radius of uniform random distribution for random poisition generation
    * save - frames per second of simulation output video
    * view_lim - axes ranges for visualization

    DEV NOTES:
    * visualizations are mapped from X-Y-Z space to the X-Y plane for visualization
    * visualizations are shown in the Center-of-Mass (COM) frame
    * everything ~should~ work but I wrote this in three days so pls don't roast me on Twitter for my hacky code thx <3
    * lmk if something is broken though, thx!!!
    * have fun :)

    '''

    def __init__(self ,states, run_time=10,  dt=0.01, title=''):
        '''
        INITIALIZE N_BODY CLASS

        ARGS
        init_conds: initial data - may contain:
            int:
                0,1 => generates a random number of masses with randomized locations up to (max_pts-1)
                2+  => generates (init_conds) number of masses with randomized locations
            float:
                cast to int, see above
            ndarray:
                [n x 7] => uses as initial masses, positions, and velocties
                [n x 6] => uses as initial positions and velocties, with random mass vector (see mass_equal below)
                [n x 4] => uses as initial massses and positions, with random velocity vector (see init_vel below)
                [n x 3] => uses as initial poisitons, with randoml mass and velocity vector (see below)
                [n x 1] => uses as initial masses, with random position and velocity vector (see below)
                [empty] => generates a random number of masses with randomized locations up to (max_pts-1)
            list:
                cast to ndarray, see above
        mass_equal: bool
            1 => masses all contain equal mass fractions
            0 => masses contain random mass fractions
        init_vel: bool
            1 => masses have non-zero initial velocity
            0 => masses have zero initial velocity
        max_states: int
            maximum number of masses randomly generated

        STATE DESCRIPTION
        the entire state space is represented by an [n x 7] ndarray with each point containing
        [mass, x, y, z, vx, vy, vz]

        randomly generated initial states contain:
            x,y,z bounded within the unit sphere
            mass is represented as the normalized mass fraction
                all randomizedmasses sum to 1 and are constant in time

        TUNING PARAMETERS
        self.G: this is essentially the strength of gravity
            higher => raises the attraction between masses
            lower => lowers the attraction between masses
        self.S: this provides damping as masses approach each other
            higher => damps accelerations at close distances
            lower => allows acceleration to scale asymptotically
            0 => causes numerical errors, don't do this
        self.dt: time propagation step size
            higher => decreases expressiveness of model because the model is integrated over larger time steps
            lower => increases expressiveness of model at cost of computation time
            TBH I WOULDN"T TOUCH THIS IF I WERE YOU

        '''
        # CONSTANTS
        self.G = 1  # Gravitational constant, normalized
        self.S = 0.1  # softening
        self.t0 = 0  # start time [seconds]
        self.tf = run_time  # [seconds]
        self.dt = dt  # timestep size [seconds]
        self.T = int(np.ceil(self.tf /self.dt))  # number of total time steps
        self.save_every = max(1, round(0.01 / self.dt))

        # get state information
        self.states_ = states
        self.n = self.states_.shape[0]  # number of masses
        self.mass = self.states_[:, 0, 0].reshape((self.n,1))  # mass vector

        # get initial energy of the system
        self.KE, self.PE = self.calc_energy_for_all_states(self.states_)
        self.KE_ = self.KE
        self.PE_ = self.PE

        self.title=title

    def calc_energy_for_all_states(self, states):
        KE = np.zeros(states.shape[-1])
        PE = np.zeros(states.shape[-1])
        for i in range(states.shape[-1]):
            KE[i], PE[i] = self.get_energy(states[..., i])
        return [KE, PE]

    def get_energy(self, state):
        '''
        Gets the Kinetric and Potential energy of the system

        RETURNS
        [KE, PE]: list containing:
            KE: ndarray
            PE: ndarray

        adapted from code via Philip Mocz
        https://github.com/pmocz/nbody-python/blob/master/nbody.py
        '''
        # get Kinetic Energy .5*m*v**2
        KE = np.sum(self.mass * state[: ,4: ]**2) / 2.

        # get Potential Energy G*m1*m2/r**2
        x = state[: ,1].reshape((-1 ,1))
        y = state[: ,2].reshape((-1 ,1))
        z = state[: ,3].reshape((-1 ,1))

        dx ,dy ,dz = [x. T -x ,y. T -y ,z. T -z]

        r_inv = np.sqrt(dx**2 + dy**2 + dz**2)
        r_inv[r_inv >0] = 1/ r_inv[r_inv > 0]

        PE = self.G * np.sum(np.sum(np.triu(-(self.mass * self.mass.T) * r_inv, 1)))

        return [KE, PE]



    def plot(self, autoscroll=False, replay=False, view_lim=20, save_every_steps=None, save_screenshot_fname='test',
             plot_energies=True, start_time_label=0):
        save = 1 / self.dt
        if not os.path.exists('evaluation/n_body_gifs'):
            os.makedirs('evaluation/n_body_gifs')
        now = datetime.datetime.now()
        date_time = now.strftime("%m_%d_%Y")
        date_time_s = now.strftime("%H_%M_%S")
        sv = "evaluation/n_body_gifs/" + "simulation_" + str(date_time) + "__" + str(date_time_s) + ".mp4"
        if not os.path.exists('evaluation/n_body_gifs'):
            os.makedirs('evaluation/n_body_gifs')

        '''
        Simulation visualization method

        ARGS
        view_lim: int
            sets range [-view_lim, view_lim] for visualization
        save: int
            0 => do not save
                Warning: if save and replay are both False, the code won't do anything
            else => save file at [save] frames per second
                anything above 30 fps looks fine, 60-100 fps is ideal IMO
                this will largely determine how strong you set gravity
                the stronger you set gravity/the stronger the interactions -> the faster the masses will move -> the slower you'll want playback
        autoscroll: bool
            True => visualization automatically scrolls to keep all data in frame
                it looks pretty bad NGL
            False => centers plot on [-view_lim,view_lim] from the COM Reference Frame
        replay: bool
            True => replays simulation frame by frame
                good for testing initial confirgurations before setting up longer runs
            False => does not replay simulation
                Warning: if save and replay are both False, the code won't do anything
        '''
        # setup visualization
        plt.style.use('dark_background')
        self.fig = plt.figure(figsize=(10, 10), dpi=150)
        self.fig.suptitle(self.title, fontsize=26)
        if plot_energies:
            grid = plt.GridSpec(3, 1, wspace=0.0, hspace=0.3)
        else:
            grid = plt.GridSpec(2, 1, wspace=0.0, hspace=0.3)

        # create mass visualization axis
        self.ax1 = plt.subplot(grid[0:2, 0])
        self.ax1.set(xlim=(-view_lim, view_lim), ylim=(-view_lim, view_lim))
        if not plot_energies:
            self.ax1.set_axis_off()  # turn off axes to better see motion

        # automatically set axis bounds
        yl = self.get_bound()

        # create energy visualization axes
        if plot_energies:
            self.ax2 = plt.subplot(grid[2, 0])
            xmax = int(np.floor(self.T / self.save_every))
            self.ax2.set(xlim=(0, xmax), ylim=(-yl, yl))
            self.ax2.set_xticks([0, xmax / 4, xmax / 2, 3 * xmax / 4, xmax])
            self.ax2.set_yticks([-yl, -yl / 2, 0, yl / 2, yl])
            # self.ax2.set_xlabel(f"Time [{self.save_every * self.dt} s]")
            self.ax2.set_xlabel(f"Timestep")
            self.ax2.set_ylabel("Energy")

        ti = int(1 / self.dt)  # tail length

        # define color scheme for masses and tails
        cs = ['tab:blue', 'tab:red', 'tab:green', 'tab:purple', 'tab:orange', 'tab:pink', 'tab:cyan']
        cs_pts = [cs[j % len(cs)] for j in range(self.n)]
        cs_tail = [cs_pts[j % len(cs_pts)] for j in range(self.n) for k in range(ti)]

        # define marker sizes for masses and tails
        s = np.linspace(8, 16, ti).reshape((ti,)) * 5
        s_tail = np.vstack(tuple([s for p in range(self.n)]))
        s_pts = self.normalize_mass() * 5

        # create collections objects for animation
        trails = self.ax1.scatter([np.random.randn(*s_tail.shape)], [np.random.randn(*s_tail.shape)], c=cs_tail,
                                  s=s_tail, zorder=1)
        # trails = self.ax1.scatter([],[],c=cs_tail,s=8,zorder=1)
        pts = self.ax1.scatter(np.random.randn(*s_pts.shape), np.random.randn(*s_pts.shape), c=cs_pts, s=s_pts,
                               zorder=2)

        if plot_energies:
            KE_line, = self.ax2.plot([], [], 'b', lw=2)
            PE_line, = self.ax2.plot([], [], 'darkorange', lw=2)
            E_line, = self.ax2.plot([], [], 'lawngreen', lw=2)
        else:
            KE_line = None
            PE_line = None
            E_line = None

        sim_length = len(self.PE_)

        def init():
            '''
            Initialize animation
            '''
            if plot_energies:
                KE_line.set_data([], [])
                PE_line.set_data([], [])
                E_line.set_data([], [])

            self.KE_step = []
            self.PE_step = []
            self.E_step = []

            if plot_energies:
                return pts, trails, KE_line, PE_line, E_line,
            return pts, trails

        def animate(i):
            '''
            Animation loop

            NOTE: no actual computation is occuring during animation
            all values are precomputed during the n_body.simulate() routine

            '''
            # draw tails
            if i - ti < 0:
                trail_i = np.vstack(tuple([np.vstack(
                    (self.states_[j, 1:3, 0] * np.ones((ti - i, 2)), self.states_[j, 1:3, max(i - ti, 0):i].T)) for j in
                                           range(self.n)]))
            else:
                trail_i = np.vstack(tuple([self.states_[j, 1:3, max(i - ti, 0):i].T for j in range(self.n)]))
            trails.set_offsets(trail_i)

            # draw masses
            xy = [self.states_[j, 1:3, i].T for j in range(self.n)]
            pts_i = np.vstack(tuple(xy))
            pts.set_offsets(pts_i)

            # get energy
            KE = self.KE_[i]
            PE = self.PE_[i]
            E = KE + PE

            # save energy states
            self.KE_step.append(KE)
            self.PE_step.append(PE)
            self.E_step.append(E)
            t_steps = np.linspace(0, i, len(self.E_step))

            # draw energy curves
            if plot_energies:
                KE_line.set_data(t_steps, self.KE_step)
                PE_line.set_data(t_steps, self.PE_step)
                E_line.set_data(t_steps, self.E_step)

            # sets autoscroll
            if autoscroll:
                # cast xy to an array and take its transpose
                xyT = np.array(xy).T  # new dims [2 x n]

                # set the maximum scrolling behavior
                xymax = max(np.amax(xyT[1]), np.amax(xyT[0]))
                xymax_lim = max(view_lim, max(1.1 * xymax,
                                              xymax + 5))  # applies additive scrolling near the origin and mutiplicative scrolling farther away

                # set the minimum scrolling behavior
                xymin = min(np.amin(xyT[1]), np.amin(xyT[0]))
                xymin_lim = min(-view_lim, min(-1.1 * xymin, xymin - 5))  # see above

                # apply autoscroll
                self.ax1.set(xlim=(xymin_lim, xymax_lim), ylim=(xymin_lim, xymax_lim))

            if plot_energies:
                return pts, trails, KE_line, PE_line, E_line,
            return pts, trails


        if save_every_steps is not None:
            spath = f'figures/n_body/individual/{save_screenshot_fname}/'
            if not os.path.exists(spath):
                os.makedirs(spath)
            # we want to save some screenshots
            if plot_energies:
                pts, trails, KE_line, PE_line, E_line = init()
            else:
                pts, trails, = init()
            # plt.savefig(f'screenshots/{save_screenshot_fname}_t{0}.png')
            sim_len = len(self.KE_)
            for t in range(sim_len):
                if plot_energies:
                    pts, trails, KE_line, PE_line, E_line, = animate(t)
                else:
                    pts, trails, = animate(t)
                if t % save_every_steps == 0:
                    plt.savefig(spath + f't{t + start_time_label}.png', dpi=150, bbox_inches='tight')



        # animate plots
        ani = animation.FuncAnimation(self.fig, animate, frames=int(np.floor(self.T / self.save_every)), interval=1,
                                      blit=True, init_func=init, repeat=False)

        # apply legends
        # self.ax1.legend(self.make_mass_legend()) #not currently working
        if plot_energies:
            self.ax2.legend(("KE", "PE", "E"))

        # saves simulation run
        # set up formatting
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=save, metadata=dict(artist='Anonymous artist'))



        print("SAVING SIMULATION AS....      ", "./" + sv)
        ani.save(sv, writer=writer)
        print("done.")
    print("\n")
    print("animation COMPLETE")
    print("\n")

    def make_plot(self, **kwargs):

        self.plot(**kwargs)

    # =====================================================================
    # HELPER FUNCTIONS
    # =====================================================================

    def make_mass_legend(self):
        '''
        Method for generating legend markers for each mass

        Not currently used
        Functionality will be provided in a future push

        '''
        mass_string = []
        mass_str = "Mass: "
        for m in range(self.n):
            mass_string.append(mass_str + str(self.mass[m][0])[:5])
        return mass_string

    def get_bound(self):
        '''
        Method to automatically scale energy plot bounds
        '''
        # get the largest energy reading from the simulated run and add 1
        b = int(max(np.amax(np.abs(self.KE_)), np.amax(np.abs(self.PE_)))) + 1

        # if less than 10, bound to the max energy + 1
        if b < 10:
            return b
        # if less than 100, bound to the nearest 20
        elif b < 100:
            return (b + 19) // 20 * 20
        # if less than 1000 bound to the nearest 50
        elif b < 1000:
            return (b + 49) // 50 * 50
        # else bound to the nearest 100
        else:
            return (b + 99) // 100 * 100

    def normalize_mass(self):
        '''
        Method to scale visulaization marker areas to mass

        '''
        # get mass vector
        m = self.mass

        # get the average mass
        m_avg = np.ones((m.shape)) * np.sum((m ** 2) ** .5) / m.shape[0]

        # get the standard deviation
        m_std = (np.std(m) ** 2) ** .5

        # account for std ~= 0 (happens when all masses are equal)
        if m_std < 0.0001:
            return 225  # marker size is determined by area i.e. 225 => size 15

        # get z-score for each mass
        m_score = (m - m_avg) / m_std

        # bound minimum size marker
        m_score[np.sign(m_score) * (10 * m_score) ** 2 < -144] = -144  # 225-144 = 81 => size 9 minimumm

        # z score scaled to enforce distribution of marker sizes
        return np.minimum(225 + np.sign(m_score) * (10 * m_score) ** 2, 625)



if __name__ == '__main__':


    cfg = load_config('n_body_dynamics_EPNS').config_dict

    model, pred_stepsize, dataloader, \
    val_loader, test_loader,\
    test_same_init_loader = evaluation.experiment_utils.load_model_and_get_dataloaders(cfg, True,
                                                                                       return_other_loaders=('test_same_init', ))
    model_name = cfg['experiment']['state_dict_fname'][:-3]
    dt = 1. / 100 * pred_stepsize

    idx = 27
    example_trajectory = test_loader.dataset.__getitem__(idx).unsqueeze(0)
    example_trajectory_rescaled = example_trajectory

    start_time = 100
    rollout_length = (example_trajectory.shape[2] - start_time) // pred_stepsize
    # rollout_length = 30

    true_test = torch.movedim(example_trajectory_rescaled[0], (1,2), (2,1))[..., :rollout_length*pred_stepsize:pred_stepsize].cpu().numpy()

    trues, preds_samples, preds_dist = evaluation.experiment_utils.model_rollout(model, example_trajectory, pred_stepsize,
                                                     rollout_length=rollout_length, start_time=start_time,
                                                     use_posterior_sampling=False)
    preds_samples = np.concatenate([i[0][:, None, ...] for i in preds_samples], axis=1)
    preds_samples_rescaled = np.ones_like(preds_samples) * preds_samples
    preds_samples_rescaled[..., 1:] = preds_samples_rescaled[..., 1:]
    sim = np.moveaxis(preds_samples_rescaled, (1,2), (2,1))

    sim_true = np.moveaxis(example_trajectory_rescaled[0,:, ::pred_stepsize].numpy(), (1,2), (2,1))

    plotter_true = n_body_plotter(sim_true, run_time=(example_trajectory.shape[2]) // pred_stepsize*dt, dt=dt,
                             # title=f'Simulation idx {idx}, ground truth'
                             )


    sim = np.concatenate([sim_true[...,:start_time//pred_stepsize], sim], axis=2)

    plotter_sample = n_body_plotter(sim, run_time=rollout_length*dt, dt=dt,
                             #title=f"Simulation idx {idx}, model {cfg['experiment']['state_dict_fname']}"
                             )
    view_lim = max(1.1 * np.max(np.abs(sim[:, 1:3])), 1.1 * np.max(np.abs(sim_true[:, 1:3])))

    plotter_true.plot(save_every_steps=5, plot_energies=False, view_lim=view_lim, save_screenshot_fname=f'{model_name}/true/{idx}/',
                      start_time_label=-start_time//pred_stepsize)
    plotter_sample.plot(save_every_steps=5, plot_energies=False, view_lim=view_lim, save_screenshot_fname=f'{model_name}/sample/{idx}/',
                        start_time_label=-start_time//pred_stepsize)
    print('all done')




