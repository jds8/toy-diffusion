#!/usr/bin/env python3

import warnings

import re
from typing import List, Callable
from collections import namedtuple
import numpy as np
import torch
import torch.distributions as dist
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def suppresswarning():
    warnings.warn("user", UserWarning)


SDE = namedtuple('SDE', 'drift diffusion')
Trajectories = namedtuple('Trajectories', 'W llk is_rare dt')

def get_end_time(trajs: Trajectories):
    return trajs.dt * (trajs.W.shape[1] - 1)

def integrate(sde: SDE, timesteps: torch.Tensor, end_time: torch.Tensor, n_samples=torch.tensor([100], device=device)):
    """
    Integrates the SDE to time "end_time" independently "n_samples" times over "timesteps" steps
    """
    dt = end_time/timesteps
    W = torch.zeros(n_samples, timesteps+1, device=n_samples.device)
    llk = torch.zeros(n_samples, timesteps+1, device=n_samples.device)
    is_rare = torch.zeros(n_samples, timesteps+1, device=n_samples.device)
    for timestep in range(1, timesteps+1):
        dW = dist.Normal(0, torch.sqrt(dt)).sample(n_samples)
        mu = W[:, timestep-1] + sde.drift * dt
        W[:, timestep] = mu + sde.diffusion * dW
        llk[:, timestep] = llk[:, timestep-1] + dist.Normal(mu, sde.diffusion ** 2 * dt).log_prob(W[:, timestep])
        is_rare[:, timestep] = torch.abs(W[:, timestep] - sde.drift * dt * timestep) > 3 * torch.sqrt(dt * timestep)
    return Trajectories(W, llk, is_rare, dt)

def index_trajs(trajs: Trajectories, idx: List[int]):
    W = trajs.W[idx].reshape(len(idx), -1)
    llk = trajs.llk[idx].reshape(len(idx), -1)
    is_rare = trajs.is_rare[idx].reshape(len(idx), -1)
    return Trajectories(W, llk, is_rare, trajs.dt)

def plot_motions(trajs, savefig=False):
    """
    Plots all trajectories and colors trajectories within 3 standard deviations of the mean trajectory at all times as green
    Plots trajectories that have one state outside of 3 standard deviations of the mean as orange
    Plots trajectories that whose end state is outside of 3 standard deviations of the mean as red
    States lying outside of 3 standard deviations are blue
    """
    end_time = get_end_time(trajs)
    greens_llk = []
    oranges_llk = []
    reds_llk = []
    for idx, motion in enumerate(trajs.W):
        if trajs.is_rare[idx].any():
            if trajs.is_rare[idx, -1]:
                color = 'red'
                reds_llk.append(trajs.llk[idx, -1])
            else:
                color = 'orange'
                oranges_llk.append(trajs.llk[idx, -1])
            alpha = 1.
            for jdx, is_rare in reversed(list(enumerate(trajs.is_rare[idx]))):
                if is_rare:
                    plt.plot(jdx * trajs.dt, trajs.W[idx, jdx], marker='o', markeredgecolor='blue', markerfacecolor='blue')
        else:
            color = 'green'
            alpha = 0.05
            greens_llk.append(trajs.llk[idx, -1])
        plt.plot(torch.linspace(0, end_time, len(motion)), motion, color=color, alpha=alpha)

    if greens_llk:
        print('({}) green llk: {}'.format(len(greens_llk), torch.stack(greens_llk).logsumexp(dim=0) - torch.tensor(len(greens_llk)).log()))
    if oranges_llk:
        print('({}) orange llk: {}'.format(len(oranges_llk), torch.stack(oranges_llk).logsumexp(dim=0) - torch.tensor(len(oranges_llk)).log()))
    if reds_llk:
        print('({}) red llk: {}'.format(len(reds_llk), torch.stack(reds_llk).logsumexp(dim=0) - torch.tensor(len(reds_llk)).log()))

    if savefig:
        plt.savefig('figs/trajectories.pdf')
        plt.clf()

def plot_rarities(trajs, savefig=False):
    """
    Plots the proportion of trajectories which are outside 3 standard deviations of end_time at every step, dt
    """
    end_time = get_end_time(trajs)
    plt.plot(torch.linspace(0, end_time, trajs.is_rare.shape[1]-1), trajs.is_rare[:, 1:].mean(dim=0))

    if savefig:
        plt.savefig('figs/rarity.pdf')
        plt.clf()

def analytical_log_likelihood(x: torch.Tensor, sde: SDE, dt: torch.Tensor):
    '''
    Computes log p(x_2,x_3,\dots, x_n) using consecutive conditionals.
    Note that this code makes no assumption about the distribution of x_1.
    It only assumes that the process is Markovian with $x_t | x_{t-1} \sim Normal(x_{t-1}, dt.sqrt())$
    transitions
    '''
    llk = torch.zeros((x.shape[0],) + x.shape[2:], device=device)
    x_prev = x[:, 0]
    for xn in x.split(dim=1, split_size=1)[1:]:
        x_next = xn[:, 0]
        llk_prev = dist.Normal(x_prev + sde.drift * dt, sde.diffusion * dt.sqrt()).log_prob(x_next)
        llk += llk_prev
        x_prev = x_next
    return llk

def continuous_failure_prob(a: float):
    '''
    According to https://math.stackexchange.com/questions/2336266/exit-probability-on-a-brownian-motion-from-an-interval
    the following is the probability that brownian motion ending at time t=1
    exits [-a, a] at any time from t=0 to t=1
    '''
    return 2 * np.sqrt(2)/(a * np.sqrt(np.pi)) * np.exp(-a**2/2)

def score_function_heat_map(
        score_function: Callable,
        version: int,
        mu: float = 0.,
        sigma: float = 1.
):
    lwr = -5 * sigma + mu
    upr = 5 * sigma + mu
    xs = torch.linspace(lwr, upr, 100, device=device)
    ts = torch.linspace(0, 1, 100, device=device)
    grid_t, grid_x = torch.meshgrid(ts, xs, indexing='ij')
    input_x = grid_x.reshape(-1, 1, 1)
    input_t = grid_t.reshape(-1)
    scores = score_function(x=input_x, time=input_t)

    fig, ax = plt.subplots()
    mesh_x = xs.cpu().numpy()
    mesh_t = ts.cpu().numpy()
    mesh_scores = scores.reshape(100, 100).detach().cpu().numpy()
    score_min = mesh_scores.min()
    score_max = mesh_scores.max()

    ax.pcolormesh(mesh_x, mesh_t, mesh_scores, cmap='RdBu', vmin=score_min, vmax=score_max)
    fig.savefig('figs/heat_maps/nn_score_v{}.jpg'.format(version))

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

def create_gif(file_dir, gif_name=None):
    import imageio
    import os
    filenames = os.listdir(file_dir)
    filenames.sort(key=natural_keys)
    gif_name = gif_name if gif_name is not None else filenames[0]
    images = []
    for filename in filenames:
        jpg = os.path.join(file_dir, filename)
        if jpg.endswith('.jpg'):
            images.append(imageio.imread(jpg))
    imageio.mimsave('{}/{}.gif'.format(file_dir, gif_name), images)


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        suppresswarning()

    sde = SDE(0., 1.)
    timesteps = torch.tensor(100, device=device)
    n_samples = torch.tensor([1000], device=device)
    end_time = torch.tensor(1., device=device)
    trajs = integrate(sde=sde, timesteps=timesteps, end_time=end_time, n_samples=n_samples)
    plot_motions(trajs)
    plt.show()
