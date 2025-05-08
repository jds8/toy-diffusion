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

import hydra
from hydra.core.config_store import ConfigStore
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
import torch
import scipy
import matplotlib.pyplot as plt

from toy_configs import register_configs
from toy_sample import ContinuousEvaluator
from toy_train_config import SampleConfig, get_run_type, MultivariateGaussianExampleConfig, \
    BrownianMotionDiffExampleConfig
from models.toy_diffusion_models_config import ContinuousSamplerConfig
from compute_quadratures import pdf_2d_quadrature_bm


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

ErrorData = namedtuple('ErrorData', 'x median error_bars label color')
HistOutput = namedtuple('HistOutput', 'hist bins')

#########################
#########################
def suppresswarning():
    warnings.warn("user", UserWarning)

def compute_tail_estimate(
        subsap: torch.Tensor,
        alpha: float
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
    tail_estimate = scipy.integrate.simpson(
        hist[smallest_idx:],
        x=med_bins[smallest_idx:]
    )
    return hist, bins, torch.tensor(tail_estimate)

def compute_pfode_tail_estimate(
        subsap: torch.Tensor,
        ode_llk: torch.Tensor,
        bins: torch.Tensor,
        alpha: float
) -> torch.Tensor:
    ordinates = []
    abscissas = []
    bin_idx = (bins < alpha).sum()
    num_out = 0
    while bin_idx < len(bins)-1:
        max_idx = subsap <= bins[bin_idx+1]
        min_idx = bins[bin_idx] <= subsap
        and_idx = (max_idx & min_idx)
        if and_idx.any():
            subsap_idx = and_idx.nonzero()[0].item()
            abscissa = subsap[subsap_idx]
            abscissas.append(abscissa)
            ordinate = ode_llk[subsap_idx].exp()
            ordinates.append(ordinate)
        else:
            num_out += 1
        bin_idx += 1
    abscissas_sorted, sorted_idx = torch.stack(abscissas).sort()
    ordinates_sorted = torch.stack(ordinates)[sorted_idx]
    # tail_estimate = scipy.integrate.trapezoid(ordinates_sorted, abscissas_sorted)
    # tail_estimate = (ordinates_sorted[:-1] * abscissas_sorted.diff()).sum()
    tail_estimate = scipy.integrate.simpson(ordinates_sorted, x=abscissas_sorted)
    ys = [pdf_2d_quadrature_bm(a, alpha) for a in abscissas_sorted]
    plt.clf()
    plt.plot(abscissas_sorted, ys)
    plt.scatter(abscissas_sorted, ordinates_sorted)
    t = time.time()
    plt.savefig('{}/pfode_tail_estimate_plot_{}'.format(
        HydraConfig.get().run.dir,
        int(t)
    ))
    plt.clf()
    return torch.tensor(tail_estimate)

def compute_pfode_tail_estimate_from_bins(
        subsap: torch.Tensor,
        ode_llk: torch.Tensor,
        num_bins: int,
        alpha: float
) -> torch.Tensor:
    max_value = subsap.max()
    bins = torch.linspace(alpha, max_value, num_bins+1)

    bin_indices = torch.bucketize(subsap, bins, right=False) - 1  # shift so bins[i] <= x < bins[i+1]

    # Initialize output
    selected_samples_list = []
    ordinates_list = []

    # For each bin, find one sample
    for bin_idx in range(len(bins) - 1):
        in_bin = (bin_indices == bin_idx).nonzero(as_tuple=True)[0]
        if len(in_bin) > 0:
            selected_samples_list.append(subsap[in_bin[0]])
            ordinates_list.append(ode_llk[in_bin[0]].exp())

    selected_samples = torch.tensor(selected_samples_list)
    ordinates = torch.tensor(ordinates_list)

    # tail_estimate = scipy.integrate.trapezoid(ordinates, selected_samples)
    # tail_estimate = ordinates_sorted[:-1] * selected_samples_sorted.diff()
    selected_samples_sorted, sorted_idx = selected_samples.sort()
    ordinates_sorted = ordinates[sorted_idx]
    tail_estimate = scipy.integrate.simpson(ordinates_sorted, x=selected_samples_sorted)
    tail_estimate_tensor = torch.tensor(tail_estimate)
    return tail_estimate_tensor

def sample(std):
    sample_traj_out = std.sample_trajectories(
        cond=-1.,
        alpha=std.likelihood.alpha.reshape(-1, 1),
    )

    sample_trajs = sample_traj_out.samples
    trajs = sample_trajs[-1]
    out_trajs = trajs
    return out_trajs

def compute_sample_error_vs_samples(
        trajs: torch.Tensor,
        alpha: float,
        subsample_sizes: torch.Tensor,
) -> Tuple[ErrorData, List[List[HistOutput]]]:
    dim = trajs.shape[2]
    dd = scipy.stats.chi(dim)
    analytical_tail = 1 - dd.cdf(alpha)
    sample_levels = trajs.norm(dim=[2, 3])
    quantiles = torch.zeros(len(subsample_sizes), 3)
    all_bins = []
    for size_idx, subsample_size in enumerate(subsample_sizes):
        all_subsaps = sample_levels[:, :subsample_size]
        subsample_bins = []
        rel_errors = []
        for subsap_idx, subsap in enumerate(all_subsaps):
            hist, bins, tail_estimate = compute_tail_estimate(
                subsap.cpu(),
                alpha
            )
            subsample_bins.append(HistOutput(hist, bins))
            rel_error = (tail_estimate - analytical_tail).abs() / analytical_tail
            rel_errors.append(rel_error)
        all_bins.append(subsample_bins)
        rel_errors_tensor = torch.stack(rel_errors)
        quantile = rel_errors_tensor.quantile(
            torch.tensor([0.05, 0.5, 0.95], dtype=rel_errors_tensor.dtype)
        )
        quantiles[size_idx] = quantile
    error_data = ErrorData(
        subsample_sizes,
        quantiles[:, 1],
        quantiles[:, [0, 2]].movedim(0, 1),
        'Histogram Approximation',
        'blue'
    )
    return error_data, all_bins

def compute_pfode_error_vs_samples(
        trajs: torch.Tensor,
        all_bins: List,
        alpha: float,
        ode_llk: torch.Tensor,
        subsample_sizes: torch.Tensor,
) -> ErrorData:
    dim = trajs.shape[2]
    dd = scipy.stats.chi(dim)
    analytical_tail = 1 - dd.cdf(alpha)
    sample_levels = trajs.norm(dim=[2, 3])
    quantiles = torch.zeros(len(subsample_sizes), 3, device='cpu')
    for size_idx, _ in enumerate(subsample_sizes):
        rel_errors = []
        for subsap_idx, subsap in enumerate(sample_levels):
            hist_output = all_bins[size_idx][subsap_idx]
            tail_estimate = compute_pfode_tail_estimate(
                subsap.cpu(),
                ode_llk[subsap_idx].cpu(),
                hist_output.bins,
                alpha
            )
            rel_error = (tail_estimate - analytical_tail).abs() / analytical_tail
            rel_errors.append(rel_error)
        rel_errors_tensor = torch.stack(rel_errors)
        quantile = rel_errors_tensor.quantile(
            torch.tensor([0.05, 0.5, 0.95], dtype=rel_errors_tensor.dtype)
        )
        quantiles[size_idx] = quantile.cpu()
    error_data = ErrorData(
        subsample_sizes.cpu(),
        quantiles[:, 1],
        quantiles[:, [0, 2]].movedim(0, 1),
        'PFODE Approximation',
        'orange'
    )
    return error_data

def compute_sample_error_vs_bins(
        trajs: torch.Tensor,
        all_bins: List,
        alpha: float,
) -> ErrorData:
    dim = trajs.shape[2]
    dd = scipy.stats.chi(dim)
    analytical_tail = 1 - dd.cdf(alpha)
    sample_levels = trajs.norm(dim=[2, 3])
    quantiles = torch.zeros(len(all_bins), 3, device='cpu')
    bin_sizes = []
    for bin_idx, bins in enumerate(all_bins):
        hist_output = all_bins[bin_idx][0]
        num_bins = len(hist_output.bins)
        bin_sizes.append(num_bins)
        all_subsaps = sample_levels[:, :num_bins]
        rel_errors = []
        for subsap_idx, subsap in enumerate(all_subsaps):
            hist, bins, tail_estimate = compute_tail_estimate(
                subsap.cpu(),
                alpha
            )
            rel_error = (tail_estimate - analytical_tail).abs() / analytical_tail
            rel_errors.append(rel_error)
        rel_errors_tensor = torch.stack(rel_errors)
        quantile = rel_errors_tensor.quantile(
            torch.tensor([0.05, 0.5, 0.95], dtype=rel_errors_tensor.dtype)
        )
        quantiles[bin_idx] = quantile.cpu()
    error_data = ErrorData(
        bin_sizes,
        quantiles[:, 1],
        quantiles[:, [0, 2]].movedim(0, 1),
        'Histogram Approximation',
        'blue'
    )
    return error_data

def compute_pfode_error_vs_bins(
        trajs: torch.Tensor,
        ode_llk: torch.Tensor,
        all_bins: List,
        alpha: float,
) -> ErrorData:
    dim = trajs.shape[2]
    dd = scipy.stats.chi(dim)
    analytical_tail = 1 - dd.cdf(alpha)
    sample_levels = trajs.norm(dim=[2, 3])
    quantiles = torch.zeros(len(all_bins), 3, device='cpu')
    bin_sizes = []
    for bin_idx, bins in enumerate(all_bins):
        hist_output = all_bins[bin_idx][0]
        num_bins = len(hist_output.bins)
        bin_sizes.append(num_bins)
        rel_errors = []
        for subsap_idx, subsap in enumerate(sample_levels):
            tail_estimate = compute_pfode_tail_estimate_from_bins(
                subsap.cpu(),
                ode_llk[subsap_idx].cpu(),
                num_bins,
                alpha
            )
            rel_error = (tail_estimate - analytical_tail).abs() / analytical_tail
            rel_errors.append(rel_error)
        rel_errors_tensor = torch.stack(rel_errors)
        quantile = rel_errors_tensor.quantile(
            torch.tensor([0.05, 0.5, 0.95], dtype=rel_errors_tensor.dtype)
        )
        quantiles[bin_idx] = quantile.cpu()
    error_data = ErrorData(
        bin_sizes,
        quantiles[:, 1],
        quantiles[:, [0, 2]].movedim(0, 1),
        'PFODE Approximation',
        'orange'
    )
    return error_data

def save_error_data(error_data: ErrorData, title: str):
    rel_filename = f'{title}_{error_data.label}'.replace(' ', '_')
    abs_filename = f'{HydraConfig.get().run.dir}/{rel_filename}.pt'
    torch.save({
        'Abscissa': error_data.x,
        'Median': error_data.median,
        '5%': error_data.error_bars[0],
        '95%': error_data.error_bars[1]
    }, abs_filename)

def plot_errors(ax, error_data: ErrorData, title: str):
    ax.scatter(
        error_data.x,
        error_data.median,
        label=error_data.label,
        color=error_data.color
    )
    ax.fill_between(
        error_data.x,
        error_data.error_bars[0],
        error_data.error_bars[1],
        color=error_data.color,
        alpha=0.2
    )

    save_error_data(error_data, title)

def make_error_vs_samples(
        ax,
        sample_error_data: ErrorData,
        pfode_error_data: ErrorData,
        alpha: float
):
    title = f'Relative Error of Tail Integral (alpha={alpha}) vs. Sample Size'
    plot_errors(ax, sample_error_data, title)
    plot_errors(ax, pfode_error_data, title)
    ax.set_xlabel('Sample Size')
    ax.set_ylabel('Relative Error')
    ax.set_title(title)

def make_error_vs_bins(
        ax,
        sample_error_data: ErrorData,
        pfode_error_data: ErrorData,
        alpha: float
):
    print(f'x axis {sample_error_data.x}')
    title = f'Relative Error of Tail Integral (alpha={alpha}) vs. Number of Bins'
    plot_errors(ax, sample_error_data, title)
    plot_errors(ax, pfode_error_data, title)
    ax.set_xlabel('Number of Bins')
    ax.set_ylabel('Relative Error')
    ax.set_title(title)

def make_plots(
        trajs: torch.Tensor,
        ode_llk: torch.Tensor,
        alpha: float,
        cfg: SampleConfig
):
    plt.clf()
    fig, (ax1, ax2) = plt.subplots(2, 1, sharey=True)

    subsample_sizes = torch.logspace(
        math.log10(500),
        math.log10(cfg.num_samples),
        10,
        dtype=int
    )

    all_bins = make_error_vs_samples_plot(
        ax1,
        trajs,
        ode_llk,
        alpha,
        cfg,
        subsample_sizes
    )
    # ax1.set_xscale("log")
    # ax3 = ax1.twiny()
    # ax3.set_xlim(ax1.get_xlim())
    # ax3.set_xlabel('Num Bins')

    make_error_vs_bins_plot(
        ax2,
        trajs,
        ode_llk,
        alpha,
        cfg,
        all_bins
    )

    plt.legend()
    fig.tight_layout()

    _, run_type = get_run_type(cfg)
    run_type = run_type.replace(' ', '_')
    plt.savefig('{}/{}_{}_tail_integral_error.pdf'.format(
        HydraConfig.get().run.dir,
        run_type,
        alpha
    ))

def make_error_vs_samples_plot(
        ax,
        trajs: torch.Tensor,
        ode_llk: torch.Tensor,
        alpha: float,
        cfg: SampleConfig,
        subsample_sizes: torch.Tensor
):
    hist_error_vs_samples, all_bins = compute_sample_error_vs_samples(
        trajs,
        alpha,
        subsample_sizes
    )
    pfode_error_vs_samples = compute_pfode_error_vs_samples(
        trajs,
        all_bins,
        alpha,
        ode_llk,
        subsample_sizes,
    )
    make_error_vs_samples(
        ax,
        hist_error_vs_samples,
        pfode_error_vs_samples,
        alpha
    )
    return all_bins

def make_error_vs_bins_plot(
        ax,
        trajs: torch.Tensor,
        ode_llk: torch.Tensor,
        alpha: float,
        cfg: SampleConfig,
        all_bins: torch.Tensor
):
    hist_error_vs_bins = compute_sample_error_vs_bins(
        trajs,
        all_bins,
        alpha,
    )
    pfode_error_vs_bins = compute_pfode_error_vs_bins(
        trajs,
        ode_llk,
        all_bins,
        alpha,
    )
    make_error_vs_bins(
        ax,
        hist_error_vs_bins,
        pfode_error_vs_bins,
        alpha
    )

@hydra.main(version_base=None, config_path="conf", config_name="continuous_is_config")
def sample(cfg):
    logger = logging.getLogger("main")
    logger.info('run type: importance sampling')
    cfg_str = OmegaConf.to_yaml(cfg)
    logger.info(f"CONFIG\n{cfg_str}")

    os.system('echo git commit: $(git rev-parse HEAD)')

    omega_sampler = OmegaConf.to_object(cfg.sampler)
    if isinstance(omega_sampler, ContinuousSamplerConfig):
        std = ContinuousEvaluator(cfg=cfg)
    else:
        raise NotImplementedError

    with torch.no_grad():
        alpha = std.likelihood.alpha.reshape(-1, 1)
        sample_traj_out = std.sample_trajectories(
            cond=torch.tensor([-1.]),
            alpha=alpha
        )
        sample_trajs = sample_traj_out.samples
        trajs = sample_trajs[-1]
        rearranged_trajs = einops.rearrange(
            trajs,
            '(b c) h w -> b c h w',
            b=cfg.num_sample_batches
        )
        ode_llk = std.ode_log_likelihood(
            trajs,
            cond=torch.tensor([-1.]),
            alpha=alpha,
            exact=cfg.compute_exact_trace,
        )
        cfg_obj = OmegaConf.to_object(cfg)
        alpha_float = alpha.cpu().item()
        sample_levels = trajs.norm(dim=[1, 2])

        if type(std.example) == MultivariateGaussianExampleConfig:
            dim = cfg.example.d
        elif type(std.example) == BrownianMotionDiffExampleConfig:
            dim = cfg.example.sde_steps
        else:
            raise NotImplementedError

        chi_ode_llk = ode_llk[0][-1] + (dim / 2) * torch.tensor(2 * torch.pi).log() + \
                (dim - 1) * sample_levels.log() - (dim / 2 - 1) * \
                torch.tensor(2.).log() - scipy.special.loggamma(dim / 2)
        rearranged_odes = einops.rearrange(
            chi_ode_llk,
            '(b c) -> b c',
            b=cfg.num_sample_batches
        )
        make_plots(rearranged_trajs, rearranged_odes, alpha_float, cfg_obj)


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        suppresswarning()

    cs = ConfigStore.instance()
    cs.store(name="vpsde_sample_config", node=SampleConfig)
    register_configs()

    with torch.no_grad():
        sample()
