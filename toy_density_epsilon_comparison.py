#!/usr/bin/env python3

import warnings
import os
import logging
from typing import Callable, List, Tuple
import time
import re
from collections import namedtuple
import einops
import math
from copy import deepcopy

import hydra
from hydra.core.config_store import ConfigStore
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
import torch
import numpy as np
import scipy
import matplotlib.pyplot as plt

from toy_configs import register_configs
from toy_sample import ContinuousEvaluator, compute_transformed_ode, compute_perimeter, \
    line_circle_intersection, vertical_line_circle_intersection, plot_boundary
from toy_train_config import SampleConfig, get_run_type, MultivariateGaussianExampleConfig, \
    BrownianMotionDiffExampleConfig, EpsilonComparisonConfig
from models.toy_diffusion_models_config import ContinuousSamplerConfig
from compute_quadratures import get_2d_pdf

from importance_sampling import BrownianMotionDiffTarget

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

ErrorData = namedtuple('ErrorData', 'bins samples median error_bars label color')
HistOutput = namedtuple('HistOutput', 'hist bins')

#########################
#########################
def suppresswarning():
    warnings.warn("user", UserWarning)

def save_icov_samples(
        abscissa: torch.Tensor,
        transformed_ode_lk: torch.Tensor,
        errors: torch.Tensor,
        equiv_saps: torch.Tensor,
        model_name: str
):
    icov_dir = 'icov_likelihoods'
    abs_dir = f'{HydraConfig.get().run.dir}/{icov_dir}'
    os.makedirs(abs_dir, exist_ok=True)
    abs_filename = f'{abs_dir}/{model_name}.pt'
    torch.save({
        'Abscissa': abscissa,
        'TransformedOdeLLK': transformed_ode_lk,
        'Errors': errors,
        'EquivSaps': equiv_saps,
        'ModelName': model_name
    }, abs_filename)

def compute_fake_bm_arcs(
        r,
        alpha,
        dt,
        num_angles,
):
    dt_sqrt = dt.sqrt()
    output_diagonal = line_circle_intersection(r, alpha, dt_sqrt)
    output_right = vertical_line_circle_intersection(r, alpha, dt_sqrt)
    output = torch.concat([output_diagonal, output_right])
    angle_bounds = torch.atan2(output[:, 1], output[:, 0]).sort().values
    angle_min, angle_max = angle_bounds[0], angle_bounds[-1]
    angles = torch.linspace(angle_min, angle_max, num_angles)
    values = r * (angles * 1j).exp()
    return torch.stack([values.real, values.imag]).T.unsqueeze(-1)

def compute_fake_gaussian_arcs(
        r,
        num_angles,
):
    angles = torch.linspace(0, 2*torch.pi, num_angles)
    values = r * (angles * 1j).exp()
    return torch.stack([values.real, values.imag]).T.unsqueeze(-1)

def compute_icov_error_vs_bins(
        stds: List[ContinuousEvaluator],
        cfg: SampleConfig,
) -> ErrorData:
    if type(stds[0].example) == MultivariateGaussianExampleConfig:
        dim = cfg.example.d
        dd = scipy.stats.chi(dim)
        analytical_tail = 1. #1 - dd.cdf(alpha)
    elif type(stds[0].example) == BrownianMotionDiffExampleConfig:
        dim = cfg.example.sde_steps-1
        dd = None
        dt = torch.tensor(1 / dim)
    else:
        raise NotImplementedError
    max_sample = 3.
    ode_lks = []
    errors = []
    quantiles_list = []

    fig, ax = plt.subplots()
    plot_boundary(stds[0], cfg, ax)

    alpha = stds[0].likelihood.alpha
    num_bins = 5
    r = alpha + cfg.r_delta
    if type(stds[0].example) == BrownianMotionDiffExampleConfig:
        pdf_array = get_2d_pdf(stds[0].example.sde_steps, torch.tensor([r]), alpha.item())
        pdf = pdf_array.item()
        upper_fake_traj_ND1 = compute_fake_bm_arcs(
            r,
            alpha,
            dt,
            num_angles=cfg.num_icov_samples//2,
        )
        lower_fake_traj_ND1 = compute_fake_bm_arcs(
            r,
            -alpha,
            dt,
            num_angles=cfg.num_icov_samples//2,
        )
        fake_traj_ND1 = torch.cat([upper_fake_traj_ND1, lower_fake_traj_ND1])
    else:
        pdf = dd.pdf(r)/(1-dd.cdf(alpha)).item()
        fake_traj_ND1 = compute_fake_gaussian_arcs(
            r,
            num_angles=cfg.num_icov_samples,
        )
    smallest_t_eps = 1.
    for std in stds:
        if std.sampler.t_eps < smallest_t_eps:
            smallest_t_eps = std.sampler.t_eps
            smallest_std = std
    ode_llk = smallest_std.ode_log_likelihood(
        fake_traj_ND1.to(device),
        cond=torch.tensor([1.]),
        alpha=torch.tensor([alpha]),
        exact=cfg.compute_exact_trace,
    )
    start_time = smallest_std.sampler.t_eps
    integrator_dt = (1 - start_time) / smallest_std.sampler.diffusion_timesteps
    epsilon_abscissas = []
    num_steps = ode_llk[0].shape[0]
    curr_num_to_subtract = 1
    abscissa = fake_traj_ND1.norm(dim=[1]).cpu().squeeze()
    first_transformed_ode_lk_N = None
    while curr_num_to_subtract < smallest_std.sampler.diffusion_timesteps:
        ode_llk_Nb = ode_llk[0][-curr_num_to_subtract]
        if type(stds[0].example) == BrownianMotionDiffExampleConfig:
            transformed_ode_lk_N = compute_transformed_ode(
                abscissa,
                ode_llk_Nb,
                alpha=alpha,
                dt=dt
            )
        else:
            transformed_ode_llk_Nb = ode_llk_Nb.cpu() + (dim / 2) * torch.tensor(2 * torch.pi).log() + \
                (dim - 1) * abscissa.log() - (dim / 2 - 1) * \
                torch.tensor(2.).log() - scipy.special.loggamma(dim / 2)
            transformed_ode_lk_N = transformed_ode_llk_Nb.exp()
        if first_transformed_ode_lk_N is None:
            first_transformed_ode_lk_N = transformed_ode_lk_N
        ode_lks.append(transformed_ode_lk_N)
        error_N = (transformed_ode_lk_N - torch.tensor(pdf)) / torch.tensor(pdf)
        errors.append(error_N)
        epsilon_abscissas.append(1 - integrator_dt * (ode_llk[0].shape[0] - curr_num_to_subtract))
        curr_num_to_subtract *= 10

    # cmap = plt.cm.viridis   # or magma / plasma for heavy-tailed χ
    cmap = plt.get_cmap("hsv")
    colors = cmap(np.linspace(0, 1., cfg.num_icov_samples))
    for point, clr in zip(fake_traj_ND1, colors):
        angle = torch.atan2(point[1], point[0]).item()
        plt.scatter(point[0], point[1], color=clr, label='{:.2f}'.format(angle), alpha=0.3)
    if isinstance(stds[0].example, BrownianMotionDiffExampleConfig):
        lim = torch.sqrt(torch.tensor(5.)) * alpha / dt.sqrt()
    else:
        lim = alpha * 2
    plt.ylim((-lim, lim))
    plt.xlim((-lim, lim))
    plt.gca().set_aspect('equal')
    plt.title(r'$\theta$ Values Along Level Curve')
    plt.legend()
    if type(stds[0].example) == BrownianMotionDiffExampleConfig:
        plt.xlabel(r'$\Delta X1/\sqrt{\Delta t}$')
        plt.ylabel(r'$\Delta X2/\sqrt{\Delta t}$')
    else:
        plt.xlabel(r'$X_1$')
        plt.ylabel(r'$X_2$')
    plt.savefig(f'{HydraConfig.get().run.dir}/angles.pdf')

    plt.clf()
    plt.axhline(y=pdf, color='r', linestyle='-', label='True PDF')
    plt.ylim((0., pdf+0.05))
    angles = []
    for lk, line in zip(first_transformed_ode_lk_N, fake_traj_ND1):
        angle = torch.atan2(line[1], line[0]).item()
        angles.append(angle)
        plt.axhline(y=lk, label='{:.2f}'.format(angle))
    plt.xlabel('Radius')
    plt.ylabel('Density')
    plt.title('Densities')
    plt.savefig(f'{HydraConfig.get().run.dir}/densities.pdf')

    epsilons = torch.tensor(epsilon_abscissas)
    all_epsilons = einops.repeat(epsilons, 'b -> (n b)', n=error_N.shape[0])
    all_errors = torch.cat(errors)
    # colors1=['red'] * len(epsilons)
    # colors2=['orange'] *  len(epsilons)
    # colors3=['yellow'] *  len(epsilons)
    # colors4=['green'] *  len(epsilons)
    # colors5=['blue'] *  len(epsilons)
    # colors6=['indigo'] *  len(epsilons)
    # colors7=['violet'] *  len(epsilons)
    # colors8=['black'] *  len(epsilons)
    # # colors = colors1 + colors2 + colors3 + colors4 + colors5 + colors6 + colors7
    # color_list = [colors1, colors2, colors3, colors4, colors5, colors6, colors7, colors8]

    angles_tensor = torch.tensor(angles)
    plt.clf()
    ax = plt.gca()

    error_data = ErrorData(
        all_epsilons,
        all_epsilons,
        all_errors,
        torch.zeros(len(errors)),
        'ICOV',
        colors,
    )
    title = r"Signed Relative Error of Density vs. $\epsilon$" + "\n(r={0:.2f})".format(r)
    angles = torch.atan2(fake_traj_ND1[:, 1], fake_traj_ND1[:, 0])
    errors_NE = torch.stack(errors).T  # N is number of radii; E is number of epsilons
    for i in range(errors_NE.shape[0]):
        plt.scatter(
            epsilons,
            errors_NE[i],
            label='{:.2f}'.format(angles[i].item()),
            color=colors[i],
            alpha=0.3
        )
    ax.set_xlabel(r'$\epsilon$')
    ax.set_ylabel('Signed Relative Error')
    ax.set_title(title)
    ax.set_xscale("log")
    ax.set_ylim((-1, 1))
    plt.grid(which='both', axis='y')

    _, run_type = get_run_type(cfg)
    run_type = run_type.replace(' ', '_')
    plt.savefig('{}/{}_tail_integral_error_vs_training.pdf'.format(
        HydraConfig.get().run.dir,
        run_type,
    ))

@hydra.main(version_base=None, config_path="conf", config_name="continuous_is_config")
def sample(cfg):
    logger = logging.getLogger("main")
    logger.info('run type: density_epsilon_comparison')
    cfg_str = OmegaConf.to_yaml(cfg)
    logger.info(f"CONFIG\n{cfg_str}")
    logger.info(f'OUTPUT\n{HydraConfig.get().run.dir}\n')

    os.system('echo git commit: $(git rev-parse HEAD)')

    torch.manual_seed(cfg.random_seed)

    omega_sampler = OmegaConf.to_object(cfg.sampler)
    if isinstance(omega_sampler, ContinuousSamplerConfig):
        stds = []
        for epsilon in cfg.t_epses:
            new_cfg = deepcopy(cfg)
            new_cfg.sampler.t_eps = epsilon
            std = ContinuousEvaluator(new_cfg)
            stds.append(std)
    else:
        raise NotImplementedError

    cfg_obj = OmegaConf.to_object(cfg)
    with torch.no_grad():
        compute_icov_error_vs_bins(
            stds,
            cfg_obj,
        )


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        suppresswarning()

    cs = ConfigStore.instance()
    cs.store(name="vpsde_sample_config", node=EpsilonComparisonConfig)
    register_configs()

    with torch.no_grad():
        sample()
