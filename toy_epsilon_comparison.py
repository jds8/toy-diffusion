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
    get_points_along_angle, compute_fake_gaussian_trajs, compute_fake_bm_trajs
from toy_train_config import SampleConfig, get_run_type, MultivariateGaussianExampleConfig, \
    BrownianMotionDiffExampleConfig, EpsilonComparisonConfig
from models.toy_diffusion_models_config import ContinuousSamplerConfig
from compute_quadratures import get_2d_pdf


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

ErrorData = namedtuple('ErrorData', 'bins samples median error_bars label color')
HistOutput = namedtuple('HistOutput', 'hist bins')

#########################
#########################
def suppresswarning():
    warnings.warn("user", UserWarning)

def compute_tail_error(
        std: ContinuousEvaluator,
        subsap: torch.Tensor,
        alpha: torch.Tensor,
        dd
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    # Construct a histogram from subsap data
    # and compute an estimate of the tail integral
    # of being greater than alpha from the histogram
    # Freedman-Diaconis
    bin_width = 2. * scipy.stats.iqr(subsap) * subsap.shape[0] ** std.cfg.histogram_bin_factor
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
    if dd is None:
        # Brownian motion case
        pdf = get_2d_pdf(std.example.sde_steps, med_bins[smallest_idx:], alpha.item())
        pdf = np.concat([np.zeros(smallest_idx), pdf])
        if len(pdf) == 0:
            error = np.nan
        else:
            # error = scipy.integrate.simpson(
            #     np.abs(hist[smallest_idx:] - torch.tensor(pdf)),
            #     x=med_bins[smallest_idx:]
            # )
            error = scipy.integrate.simpson(
                np.abs(hist.numpy() - pdf),
                x=med_bins
            )
    else:
        # scipy.integrate.trapezoid(hist[smallest_idx:], med_bins[smallest_idx:])
        # error = scipy.integrate.simpson(
        #     np.abs(hist[smallest_idx:] - dd.pdf(med_bins[smallest_idx:])/(1-dd.cdf(alpha))),
        #     x=med_bins[smallest_idx:]
        # )
        pdf = dd.pdf(med_bins)/(1-dd.cdf(alpha)) * (med_bins > alpha).numpy()
        error = scipy.integrate.simpson(
            np.abs(hist.numpy() - pdf),
            x=med_bins
        )
    return hist, bins, error

def compute_sample_error_vs_samples(
        rearranged_trajs_list: List[torch.Tensor],
        epsilons: torch.Tensor,
        std: ContinuousEvaluator,
        cfg: SampleConfig,
) -> Tuple[ErrorData, List[List[HistOutput]]]:
    if type(std.example) == MultivariateGaussianExampleConfig:
        dim = cfg.example.d
        dd = scipy.stats.chi(dim)
        analytical_tail = 1.#1 - dd.cdf(alpha)
    elif type(std.example) == BrownianMotionDiffExampleConfig:
        dim = cfg.example.sde_steps
        dd = None
        # cfg_obj = OmegaConf.to_object(cfg)
        # target = get_target(cfg_obj)
        # analytical_tail = target.analytical_prob(alpha)
        analytical_tail = 1.
    else:
        raise NotImplementedError
    quantiles = []
    all_bins = []
    alpha = std.likelihood.alpha
    for rearranged_traj, epsilon in zip(rearranged_trajs_list, epsilons):
        sample_levels = rearranged_traj.norm(dim=[2, 3])
        subsample_bins = []
        errors = []
        for subsap_idx, subsap in enumerate(sample_levels):
            hist, bins, error = compute_tail_error(
                std,
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
        epsilons,
        epsilons,
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

def compute_icov_error_vs_bins(
        sample_trajs: torch.Tensor,
        stds: List[ContinuousEvaluator],
        cfg: SampleConfig,
        epsilons: torch.Tensor,
        all_bins: List[List[HistOutput]]
) -> ErrorData:
    if type(stds[0].example) == MultivariateGaussianExampleConfig:
        dim = cfg.example.d
        dd = scipy.stats.chi(dim)
        analytical_tail = 1. #1 - dd.cdf(alpha)
    elif type(stds[0].example) == BrownianMotionDiffExampleConfig:
        dim = cfg.example.sde_steps
        dd = None
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
    bin_width = (2 * IQR) * stds[0].cfg.num_samples ** stds[0].cfg.histogram_bin_factor
    max_sample = sample_trajs.max()
    ode_lks = []
    errors = []
    quantiles_list = []
    for std in stds:
        alpha = std.likelihood.alpha
        num_bins = ((max_sample - alpha) / bin_width).int()
        # abscissa_N1 = torch.linspace(alpha, max_sample, num_bins+1).reshape(-1, 1).to(device)
        abscissa_N1 = all_bins[idx][0].bins.unsqueeze(-1)
        if type(stds[0].example) == MultivariateGaussianExampleConfig:
            fake_traj_NbD1, _ = compute_fake_gaussian_trajs(
                abscissa_N1,
                cfg.num_sample_batches,
                dim
            )
        elif type(stds[0].example) == BrownianMotionDiffExampleConfig:
            fake_traj_NbD1, _ = compute_fake_bm_trajs(
                abscissa_N1,
                dim,
                alpha,
                dt,
                num_trajs=cfg.num_sample_batches//2,
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
            transformed_ode_llk_Nb = ode_llk_Nb.cpu() + (dim / 2) * torch.tensor(2 * torch.pi).log() + \
                (dim - 1) * fake_traj_NbD1.norm(dim=[1]).cpu().log() - (dim / 2 - 1) * \
                torch.tensor(2.).log() - scipy.special.loggamma(dim / 2)
            transformed_ode_lk_Nb = transformed_ode_llk_Nb.exp()
            transformed_ode_lk_NB = einops.rearrange(transformed_ode_lk_Nb, '(n b) -> n b', n=abscissa_N1.shape[0])
        elif type(stds[0].example) == BrownianMotionDiffExampleConfig:
            transformed_ode_lk_N = compute_transformed_ode(
                fake_traj_NbD1.norm(dim=[1]).cpu().squeeze(),
                ode_llk[0][-1],
                alpha=alpha,
                dt=dt
            )
            transformed_ode_lk_NB = einops.rearrange(
                transformed_ode_lk_N,
                '(n b) -> n b',
                b=cfg.num_sample_batches
            )
        else:
            raise NotImplementedError
        ode_lks.append(transformed_ode_lk_NB)
        errors_B_list = []
        xs = abscissa_N1.squeeze().cpu()
        if dd is None:
            pdf = get_2d_pdf(std.example.sde_steps, xs, alpha.item())
        for b in range(transformed_ode_lk_NB.shape[1]):
            if dd is None:
                # Brownian motion case
                error_N = scipy.integrate.simpson(
                    np.abs(transformed_ode_lk_NB[:, b] - torch.tensor(pdf)),
                    x=xs
                )
            else:
                error_N = scipy.integrate.simpson(
                    np.abs(
                        transformed_ode_lk_NB[:, b].cpu().numpy() -
                        dd.pdf(xs)/(1-dd.cdf(alpha))
                    ),
                    x=xs
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
        epsilons,
        epsilons,
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
    title = f'Absolute Error of Tail Integral vs. Epsilon\n(eta={cfg.eta}, N={cfg.num_samples})'
    plot_errors(sample_error_data, title)
    plot_errors(icov_error_data, title)
    plt.xlabel('Epsilon')
    plt.ylabel('Absolute Error')
    plt.legend()
    plt.title(title)

def make_plots(
        rearranged_trajs_list: List[torch.Tensor],
        cfg: SampleConfig,
        stds: List[ContinuousEvaluator],
        epsilons: torch.Tensor
):
    plt.clf()

    all_bins = make_error_vs_samples_plot(
        rearranged_trajs_list,
        cfg,
        stds,
        epsilons
    )

    plt.xscale("log")
    # plt.yscale("log")
    plt.ylim((0, 1))
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
        epsilons: torch.Tensor
):
    hist_error_vs_samples, all_bins = compute_sample_error_vs_samples(
        rearranged_trajs_list,
        epsilons,
        stds[0],
        cfg,
    )
    dim = rearranged_trajs_list[0].shape[2]
    icov_error_vs_samples = compute_icov_error_vs_bins(
        rearranged_trajs_list[0],
        stds,
        cfg,
        epsilons,
        all_bins
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
    logger.info('run type: epsilon_comparison')
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
            torch.tensor(cfg.t_epses)
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
