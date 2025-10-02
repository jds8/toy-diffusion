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
from toy_sample import ContinuousEvaluator, compute_transformed_ode
from toy_train_config import SampleConfig, get_run_type, MultivariateGaussianExampleConfig, \
    BrownianMotionDiffExampleConfig, AlphaComparisonConfig
from models.toy_diffusion_models_config import ContinuousSamplerConfig
from compute_quadratures import pdf_2d_quadrature_bm


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

ErrorData = namedtuple('ErrorData', 'bins samples median error_bars label color')
HistOutput = namedtuple('HistOutput', 'hist bins')

#########################
#########################
def suppresswarning():
    warnings.warn("user", UserWarning)

def compute_tail_estimate(
        subsap: torch.Tensor,
        alpha: float,
        dd
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    # Construct a histogram from subsap data
    # and compute an estimate of the tail integral
    # of being greater than alpha from the histogram
    # Freedman-Diaconis
    bin_width = 2. * scipy.stats.iqr(subsap) * subsap.shape[0] ** (-1/3)
    num_bins = int((subsap.max()) / bin_width)

    # Create the histogram
    hist, bins = torch.histogram(
        subsap,
        bins=num_bins,
        density=True,
    )
    smallest_idx = max((bins < alpha).sum() - 1, 0)
    # empirical_bin_width = bins[1] - bins[0]
    # tail_estimate = hist[smallest_idx:].sum() * empirical_bin_width
    lwr_bins = bins[:-1]
    upr_bins = bins[1:]
    med_bins = (lwr_bins + upr_bins) / 2
    # scipy.integrate.trapezoid(hist[smallest_idx:], med_bins[smallest_idx:])
    error = scipy.integrate.simpson(
        np.abs(hist[smallest_idx:] - dd.pdf(med_bins[smallest_idx:])/(1-dd.cdf(alpha))),
        x=med_bins[smallest_idx:]
    )
    return hist, bins, error

def compute_sample_error_vs_samples(
        rearranged_trajs_list: List[torch.Tensor],
        alphas: torch.Tensor,
        std: ContinuousEvaluator,
        cfg: SampleConfig,
) -> Tuple[ErrorData, List[List[HistOutput]]]:
    if type(std.example) == MultivariateGaussianExampleConfig:
        dim = cfg.example.d
        dd = scipy.stats.chi(dim)
        analytical_tail = 1.#1 - dd.cdf(alpha)
    elif type(std.example) == BrownianMotionDiffExampleConfig:
        dim = cfg.example.sde_steps
        # cfg_obj = OmegaConf.to_object(cfg)
        # target = get_target(cfg_obj)
        # analytical_tail = target.analytical_prob(alpha)
        analytical_tail = 1.
    else:
        raise NotImplementedError
    quantiles = []
    all_bins = []
    for rearranged_traj, alpha in zip(rearranged_trajs_list, alphas):
        sample_levels = rearranged_traj.norm(dim=[2, 3])
        subsample_bins = []
        errors = []
        for subsap_idx, subsap in enumerate(sample_levels):
            hist, bins, error = compute_tail_estimate(
                subsap.cpu(),
                alpha,
                dd
            )
            subsample_bins.append(HistOutput(hist, bins))
            errors.append(torch.tensor(error))
        all_bins.append(subsample_bins)
        errors_tensor = torch.stack(errors)
        quantile = errors_tensor.quantile(
            torch.tensor([0.05, 0.5, 0.95], dtype=errors_tensor.dtype)
        )
        quantiles.append(quantile)
    quantiles_tensor = torch.stack(quantiles)
    error_data = ErrorData(
        alphas,
        alphas,
        quantiles_tensor[:, 1],
        quantiles_tensor[:, [0, 2]].movedim(0, 1),
        'Histogram Approximation',
        'blue'
    )
    return error_data, all_bins

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

def compute_fake_gaussian_trajs(
        abscissa: torch.Tensor,
        num_sample_batches: int,
        dim: int
):
    vecs = torch.randn(1, num_sample_batches, dim, 1)
    normed_vecs_1BD1 = vecs / vecs.norm(dim=2, keepdim=True)
    abscissa_repeat_NBD1 = einops.repeat(
        abscissa,
        'n 1 -> n b d 1',
        b=num_sample_batches,
        d=dim
    )
    fake_trajs_NBD1 = abscissa_repeat_NBD1.cpu() * normed_vecs_1BD1
    flattened_fake_trajs_NbD1 = einops.rearrange(fake_trajs_NBD1, 'n b d 1 -> (n b) d 1')
    return flattened_fake_trajs_NbD1

def compute_fake_bm_trajs(
    abscissa_tensor: torch.Tensor,
    dim: int,
    alpha: float,
    dt: torch.Tensor,
    num_trajs: int
):
    points = []
    angle_points_list = [compute_perimeter(r, alpha, dt.sqrt())[1:] for r in abscissa_tensor]
    for (angles, angle_points), r in zip(angle_points_list, abscissa_tensor.cpu()):
        top_points = get_points_along_angle(
            angles, angle_points, 0, r, num_trajs, dim, alpha
        )
        bottom_points = get_points_along_angle(
            angles, angle_points, 1, r, num_trajs, dim, alpha
        )
        points.append(torch.cat([top_points, bottom_points]))
    all_points = torch.cat(points).unsqueeze(-1)
    return all_points

def compute_icov_error_vs_bins(
        sample_trajs: torch.Tensor,
        stds: List[ContinuousEvaluator],
        cfg: SampleConfig,
        alphas: torch.Tensor,
) -> ErrorData:
    if type(stds[0].example) == MultivariateGaussianExampleConfig:
        dim = cfg.example.d
        dd = scipy.stats.chi(dim)
        analytical_tail = 1. #1 - dd.cdf(alpha)
    elif type(stds[0].example) == BrownianMotionDiffExampleConfig:
        dim = cfg.example.sde_steps
        # cfg_obj = OmegaConf.to_object(cfg)
        # target = get_target(cfg_obj)
        # analytical_tail = target.analytical_prob(alpha)
        analytical_tail = 1.
        dt = torch.tensor(1 / (dim-1))
    else:
        raise NotImplementedError
    sample_trajs = einops.rearrange(
        sample_trajs,
        'b c h w -> (b c) h w',
        b=cfg.num_sample_batches
    ).norm(dim=[1, 2])
    IQR = scipy.stats.iqr(sample_trajs.cpu())
    bin_width = (2 * IQR) * stds[0].cfg.num_samples ** (-1/3)
    max_sample = sample_trajs.max()
    ode_lks = []
    errors = []
    quantiles_list = []
    for std, alpha in zip(stds, alphas):
        num_bins = ((max_sample - alpha) / bin_width).int()
        abscissa_N1 = torch.linspace(alpha, max_sample, num_bins+1).reshape(-1, 1).to(device)
        if type(stds[0].example) == MultivariateGaussianExampleConfig:
            fake_traj_NbD1 = compute_fake_gaussian_trajs(
                abscissa_N1,
                cfg.num_sample_batches,
                dim
            )
        elif type(stds[0].example) == BrownianMotionDiffExampleConfig:
            fake_traj_NbD1 = compute_fake_bm_trajs(
                abscissa_N1,
                dim,
                alpha,
                dt,
                num_trajs=cfg.num_sample_batches//2
            )
        else:
            raise NotImplementedError
        ode_llk = std.ode_log_likelihood(
            fake_traj_NbD1.to(device),
            cond=torch.tensor([1.]),
            alpha=torch.tensor([alpha]),
            exact=cfg.compute_exact_trace,
        )
        ode_llk_Nb = ode_llk[0][-1]
        if type(stds[0].example) == MultivariateGaussianExampleConfig:
            ode_llk_NB = einops.rearrange(ode_llk_Nb.cpu(), '(n b) -> n b', n=abscissa_N1.shape[0])
            transformed_ode_llk_NB = ode_llk_NB.cpu() + (dim / 2) * torch.tensor(2 * torch.pi).log() + \
                (dim - 1) * abscissa_N1.cpu().log() - (dim / 2 - 1) * \
                torch.tensor(2.).log() - scipy.special.loggamma(dim / 2)
            transformed_ode_lk_NB = transformed_ode_llk_NB.exp()
        elif type(stds[0].example) == BrownianMotionDiffExampleConfig:
            transformed_ode_lk_NB = compute_transformed_ode(
                abscissa_N1.cpu().squeeze(),
                ode_llk[0][-1],
                alpha=alpha,
                dt=dt
            )
        else:
            raise NotImplementedError
        ode_lks.append(transformed_ode_lk_NB)
        errors_B_list = []
        for b in range(transformed_ode_lk_NB.shape[1]):
            error_N = scipy.integrate.simpson(
                np.abs(transformed_ode_lk_NB[:, b].cpu().numpy() - dd.pdf(abscissa_N1.squeeze().cpu())/(1-dd.cdf(alpha))),
                x=abscissa_N1.squeeze().cpu()
            )
            errors_B_list.append(torch.tensor(error_N))
        errors_B = torch.stack(errors_B_list)
        model_quantiles = torch.quantile(errors_B,
                                        torch.tensor([0.05, 0.5, 0.95],
                                                    device=errors_B.device,
                                                    dtype=errors_B.dtype))
        quantiles_list.append(model_quantiles)
        save_icov_samples(
            abscissa_N1.cpu(),
            transformed_ode_lk_NB,
            errors_B,
            std.cfg.num_samples,
            std.cfg.model_name
        )
    quantiles = torch.stack(quantiles_list)

    error_data = ErrorData(
        alphas,
        alphas,
        quantiles[:, 1],
        quantiles[:, [0, 2]].movedim(0, 1),
        'ICOV',
        'orange'
    )
    return error_data

def save_error_data(error_data: ErrorData, title: str):
    rel_filename = f'{title}_{error_data.label}'.replace(' ', '_').replace('\n', '_')
    abs_filename = f'{HydraConfig.get().run.dir}/{rel_filename}.pt'
    torch.save({
        'Abscissa_bins': error_data.bins,
        'Abscissa_samples': error_data.samples,
        'Median': error_data.median,
        '5%': error_data.error_bars[0],
        '95%': error_data.error_bars[1]
    }, abs_filename)

def plot_errors(error_data: ErrorData, title: str):
    plt.scatter(
        error_data.samples,
        error_data.median,
        label=error_data.label,
        color=error_data.color
    )
    plt.fill_between(
        error_data.samples,
        error_data.error_bars[0],
        error_data.error_bars[1],
        color=error_data.color,
        alpha=0.2
    )
    save_error_data(error_data, title)

def make_error_vs_samples(
        sample_error_data: ErrorData,
        icov_error_data: ErrorData,
        cfg: SampleConfig
):
    title = f'Absolute Error of Tail Integral vs. Alpha\n(eta={cfg.eta}, N={cfg.num_samples})'
    plot_errors(sample_error_data, title)
    plot_errors(icov_error_data, title)
    plt.xlabel('Alpha')
    plt.ylabel('Absolute Error')
    plt.legend()
    plt.title(title)

def make_plots(
        rearranged_trajs_list: List[torch.Tensor],
        cfg: SampleConfig,
        stds: List[ContinuousEvaluator],
        alphas: torch.Tensor
):
    plt.clf()

    all_bins = make_error_vs_samples_plot(
        rearranged_trajs_list,
        cfg,
        stds,
        alphas
    )

    plt.yscale("log")
    plt.grid(which='both', axis='y')

    _, run_type = get_run_type(cfg)
    run_type = run_type.replace(' ', '_')
    plt.savefig('{}/{}_tail_integral_error_vs_training.pdf'.format(
        HydraConfig.get().run.dir,
        run_type,
    ))

def make_error_vs_samples_plot(
        rearranged_trajs_list: List[torch.Tensor],
        cfg: SampleConfig,
        stds: List[ContinuousEvaluator],
        alphas: torch.Tensor
):
    hist_error_vs_samples, all_bins = compute_sample_error_vs_samples(
        rearranged_trajs_list,
        alphas,
        stds[0],
        cfg,
    )
    dim = rearranged_trajs_list[0].shape[2]
    icov_error_vs_samples = compute_icov_error_vs_bins(
        rearranged_trajs_list[0],
        stds,
        cfg,
        alphas
    )
    make_error_vs_samples(
        hist_error_vs_samples,
        icov_error_vs_samples,
        cfg
    )
    return all_bins

def get_num_samples(model_name: str) -> int:
    result = re.search(r'_v([0-9]+)', model_name)
    return int(result[1])

@hydra.main(version_base=None, config_path="conf", config_name="continuous_is_config")
def sample(cfg):
    logger = logging.getLogger("main")
    logger.info('run type: train_comparison')
    cfg_str = OmegaConf.to_yaml(cfg)
    logger.info(f"CONFIG\n{cfg_str}")
    logger.info(f'OUTPUT\n{HydraConfig.get().run.dir}\n')

    os.system('echo git commit: $(git rev-parse HEAD)')

    torch.manual_seed(cfg.random_seed)

    omega_sampler = OmegaConf.to_object(cfg.sampler)
    if isinstance(omega_sampler, ContinuousSamplerConfig):
        stds = []
        for alpha in cfg.alphas:
            new_cfg = deepcopy(cfg)
            new_cfg.likelihood.alpha = alpha
            std = ContinuousEvaluator(new_cfg)
            stds.append(std)
    else:
        raise NotImplementedError

    cfg_obj = OmegaConf.to_object(cfg)
    if type(std.example) == MultivariateGaussianExampleConfig:
        dim = cfg.example.d
    elif type(std.example) == BrownianMotionDiffExampleConfig:
        dim = cfg.example.sde_steps
    else:
        raise NotImplementedError
    with torch.no_grad():
        rearranged_trajs_list = []
        for std in stds:
            alpha = std.likelihood.alpha.reshape(-1, 1)
            sample_traj_out = std.sample_trajectories(
                cond=torch.tensor([1.]),
                alpha=alpha
            )
            sample_trajs = sample_traj_out.samples
            trajs = sample_trajs[-1]
            rearranged_trajs = einops.rearrange(
                trajs,
                '(b c) h w -> b c h w',
                b=cfg.num_sample_batches
            )
            rearranged_trajs_list.append(rearranged_trajs)
        make_plots(
            rearranged_trajs_list,
            cfg_obj,
            stds,
            torch.tensor(cfg.alphas)
        )


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        suppresswarning()

    cs = ConfigStore.instance()
    cs.store(name="vpsde_sample_config", node=AlphaComparisonConfig)
    register_configs()

    with torch.no_grad():
        sample()
