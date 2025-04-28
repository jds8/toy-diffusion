#!/usr/bin/env python3
import warnings
import os
import logging
from typing import Callable, List
import time
import re
from collections import namedtuple
import einops

import hydra
from hydra.core.config_store import ConfigStore
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
import torch
import scipy
import matplotlib.pyplot as plt

from toy_configs import register_configs
from toy_sample import ContinuousEvaluator
from toy_train_config import SampleConfig, get_run_type
from models.toy_diffusion_models_config import ContinuousSamplerConfig


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

ErrorData = namedtuple('ErrorData', 'x median error_bars label')
HistOutput = namedtuple('HistOutput', 'hist bins')

#########################
#########################
def suppresswarning():
    warnings.warn("user", UserWarning)

def compute_tail_estimate(subsap: torch.Tensor, alpha: float):
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
    smallest_idx = (bins < alpha).sum()
    empirical_bin_width = bins[1] - bins[0]
    tail_estimate = hist[smallest_idx:].sum() * empirical_bin_width
    return hist, bins, tail_estimate

def compute_pfode_tail_estimate(
        subsap: torch.Tensor,
        ode_llk: torch.Tensor,
        bins: torch.Tensor,
        alpha: float
):
    ordinates = []
    abscissas = []
    idx = (bins < alpha).sum()
    while idx < len(bins)-1:
        max_idx = subsap < bins[idx+1]
        min_idx = bins[idx] < subsap
        idx = int((max_idx + min_idx)/2)
        abscissa = subsap[idx]
        abscissas.append(abscissa)
        ordinate = ode_llk[idx].exp()
        ordinates.append(ordinate)
        idx += 1
        tail_estimate = scipy.trapezoid(ordinates, abscissas)
    return tail_estimate

def compute_pfode_tail_estimate_from_bins(
        subsap: torch.Tensor,
        num_bins: int,
        alpha: float
):
    max_value = subsap.max()
    bins = torch.linspace(alpha, max_value, num_bins+1)
    # lower_bins = einops.repeat(bins[:-1], 'a -> a b', b=subsap.shape[0])
    # upper_bins = einops.repeat(bins[1:], 'a -> a b', b=subsap.shape[0])
    # subsap_repeat = einops.repeat(subsap, 'b -> a b', a=bins.shape[0]-1)
    # between = lower_bins < subsap_repeat < upper_bins

    bin_indices = torch.bucketize(subsap, bins, right=False) - 1  # shift so bins[i] <= x < bins[i+1]

    # Initialize output
    selected_samples = torch.full(
        (len(bins) - 1,),
        float('nan'),
        dtype=subsap.dtype,
        device=subsap.device
    )

    # For each bin, find one sample
    for bin_idx in range(len(bins) - 1):
        in_bin = (bin_indices == bin_idx).nonzero(as_tuple=True)[0]
        if len(in_bin) > 0:
            selected_samples[bin_idx] = subsap[in_bin[0]]
    return selected_samples

def sample(std):
    sample_traj_out = std.sample_trajectories(
        cond=-1.,
        alpha=std.likelihood.alpha.reshape(-1, 1),
    )

    sample_trajs = sample_traj_out.samples
    trajs = sample_trajs[-1]
    out_trajs = trajs
    return out_trajs

def compute_sample_error_vs_bins(
        trajs: torch.Tensor,
        alpha: float,
        cfg: SampleConfig,
        subsample_sizes: torch.Tensor,
):
    dim = trajs.shape[2]
    dd = scipy.stats.chi(dim)
    analytical_tail = 1 - dd.cdf(alpha)
    sample_levels = trajs.norm(dim=[2, 3])
    quantiles = torch.zeros(len(subsample_sizes), 3)
    all_bins = []
    for size_idx, subsample_size in enumerate(subsample_sizes):
        all_subsaps = sample_levels[:, :subsample_size]
        subsample_bins = []
        rel_errors = torch.zeros(trajs.shape[:2])
        for subsap_idx, subsap in enumerate(all_subsaps):
            hist, bins, tail_estimate = compute_tail_estimate(subsap, alpha)
            subsample_bins.append(HistOutput(hist, bins))
            rel_error = (tail_estimate - analytical_tail).abs() / analytical_tail
            rel_errors[subsap_idx, size_idx] = rel_error
        all_bins.append(subsample_bins)
        quantile = rel_errors.quantile(
            torch.tensor([0.05, 0.5, 0.95], dtype=rel_errors.dtype)
        )
        quantiles[size_idx] = quantile
    error_data = ErrorData(
        subsample_sizes,
        quantiles[:, 1],
        quantiles[:, [0, 2]].movedim(0, 1),
        'Histogram Approximation'
    )
    return error_data, all_bins

def compute_pfode_error_vs_bins(
        trajs: torch.Tensor,
        all_bins: List,
        alpha: float,
        cfg: SampleConfig,
        ode_llk: torch.Tensor,
        subsample_sizes: torch.Tensor,
):
    dim = trajs.shape[2]
    dd = scipy.stats.chi(dim)
    analytical_tail = 1 - dd.cdf(alpha)
    sample_levels = trajs.norm(dim=[2, 3])
    quantiles = torch.zeros(len(subsample_sizes), 3)
    for size_idx, subsample_size in enumerate(subsample_sizes):
        all_subsaps = sample_levels[:, :subsample_size]
        rel_errors = torch.zeros(trajs.shape[:2])
        for subsap_idx, subsap in enumerate(all_subsaps):
            hist_output = all_bins[size_idx][subsap_idx]
            tail_estimate = compute_pfode_tail_estimate(
                subsap,
                ode_llk[subsap_idx, :subsample_size],
                hist_output.bins,
                alpha
            )
            rel_error = (tail_estimate - analytical_tail).abs() / analytical_tail
            rel_errors[subsap_idx, size_idx] = rel_error
        quantile = rel_errors.quantile(
            torch.tensor([0.05, 0.5, 0.95], dtype=rel_errors.dtype)
        )
        quantiles[size_idx] = quantile
    error_data = ErrorData(
        subsample_sizes,
        quantiles[:, 1],
        quantiles[:, [0, 2]].movedim(0, 1),
        'PFODE Approximation'
    )
    return error_data

def compute_sample_error_vs_samples(
        trajs: torch.Tensor,
        all_bins: List,
        alpha: float,
        cfg: SampleConfig,
        subsample_sizes: torch.Tensor
):
    dim = trajs.shape[2]
    dd = scipy.stats.chi(dim)
    analytical_tail = 1 - dd.cdf(alpha)
    sample_levels = trajs.norm(dim=[2, 3])
    quantiles = torch.zeros(len(all_bins), 3)
    for bin_idx, bins in enumerate(all_bins):
        hist_output = all_bins[bin_idx][0]
        num_bins = len(hist_output.bins)
        all_subsaps = sample_levels[:, :num_bins]
        rel_errors = []
        for subsap_idx, subsap in enumerate(all_subsaps):
            hist, bins, tail_estimate = compute_tail_estimate(subsap, alpha)
            rel_error = (tail_estimate - analytical_tail).abs() / analytical_tail
            rel_errors.append(rel_error)
        rel_errors_tensor = torch.stack(rel_errors)
        quantile = rel_errors_tensor.quantile(
            torch.tensor([0.05, 0.5, 0.95], dtype=rel_errors.dtype)
        )
        quantiles[bin_idx] = quantile
    error_data = ErrorData(
        subsample_sizes,
        quantiles[:, 1],
        quantiles[:, [0, 2]].movedim(0, 1),
        'Histogram Approximation'
    )
    return error_data

def compute_pfode_error_vs_samples(
        trajs: torch.Tensor,
        ode_llk: torch.Tensor,
        all_bins: List,
        alpha: float,
        cfg: SampleConfig,
        subsample_sizes: torch.Tensor
):
    dim = trajs.shape[2]
    dd = scipy.stats.chi(dim)
    analytical_tail = 1 - dd.cdf(alpha)
    sample_levels = trajs.norm(dim=[2, 3])
    quantiles = torch.zeros(len(all_bins), 3)
    for bin_idx, bins in enumerate(all_bins):
        hist_output = all_bins[bin_idx][0]
        num_bins = len(hist_output.bins)
        rel_errors = []
        for subsap_idx, subsap in enumerate(sample_levels):
            hist, bins, tail_estimate = compute_pfode_tail_estimate_from_bins(
                subsap,
                num_bins,
                alpha
            )
            rel_error = (tail_estimate - analytical_tail).abs() / analytical_tail
            rel_errors.append(rel_error)
        rel_errors_tensor = torch.stack(rel_errors)
        quantile = rel_errors_tensor.quantile(
            torch.tensor([0.05, 0.5, 0.95], dtype=rel_errors.dtype)
        )
        quantiles[bin_idx] = quantile
    error_data = ErrorData(
        subsample_sizes,
        quantiles[:, 1],
        quantiles[:, [0, 2]].movedim(0, 1),
        'Histogram Approximation'
    )
    return error_data, all_bins

def plot_errors(ax, error_data: ErrorData):
    ax.errorbar(
        error_data.x,
        error_data.median,
        yerr=error_data.error_bars,
        label=error_data.label,
        color='black',
    )

def make_error_vs_bins(
        ax,
        sample_error_data: ErrorData,
        pfode_error_data: ErrorData,
        alpha: float
):
    plot_errors(ax, sample_error_data)
    plot_errors(ax, pfode_error_data)
    ax.set_xlabel('Number of Bins')
    ax.set_ylabel('Relative Error')
    ax.set_title(f'Relative Error of Tail Integral (alpha={alpha}) vs. Number of Bins')

def make_error_vs_samples(
        ax,
        sample_error_data: ErrorData,
        pfode_error_data: ErrorData,
        alpha: float
):
    plot_errors(ax, sample_error_data)
    plot_errors(ax, pfode_error_data)
    ax.set_xlabel('Sample Size')
    ax.set_ylabel('Relative Error')
    ax.set_title(f'Relative Error of Tail Integral (alpha={alpha}) vs. Sample Size')

def make_plots(
        trajs: torch.Tensor,
        ode_llk: torch.Tensor,
        alpha: float,
        cfg: SampleConfig
):
    plt.clf()
    fig, (ax1, ax2) = plt.subplots(2, 1)

    subsample_sizes = torch.linspace(5, cfg.num_samples, 10, dtype=int)

    all_bins = make_error_vs_bins_plot(
        ax1,
        trajs,
        ode_llk,
        alpha,
        cfg,
        subsample_sizes
    )
    make_error_vs_samples_plot(
        ax2,
        trajs,
        ode_llk,
        alpha,
        cfg,
        subsample_sizes,
        all_bins
    )

    fig.tight_layout()

    _, run_type = get_run_type(cfg)
    run_type = run_type.replace(' ', '_')
    plt.savefig('{}/{}_{}_tail_integral_error.pdf'.format(
        HydraConfig.get().run.dir,
        run_type,
        alpha
    ))

def make_error_vs_bins_plot(
        ax,
        trajs: torch.Tensor,
        ode_llk: torch.Tensor,
        alpha: float,
        cfg: SampleConfig,
        subsample_sizes: torch.Tensor
):
    hist_error_vs_bins, all_bins = compute_sample_error_vs_bins(
        trajs,
        alpha,
        cfg,
        subsample_sizes
    )
    pfode_error_vs_bins = compute_pfode_error_vs_bins(
        trajs,
        all_bins,
        alpha,
        cfg,
        ode_llk,
        subsample_sizes,
    )
    make_error_vs_bins(
        ax,
        hist_error_vs_bins,
        pfode_error_vs_bins,
        alpha
    )
    return all_bins

def make_error_vs_samples_plot(
        ax,
        trajs: torch.Tensor,
        ode_llk: torch.Tensor,
        alpha: float,
        cfg: SampleConfig,
        subsample_sizes: torch.Tensor,
        all_bins
):
    hist_error_vs_samples = compute_sample_error_vs_samples(
        trajs,
        all_bins,
        alpha,
        cfg,
        subsample_sizes
    )
    pfode_error_vs_samples = compute_pfode_error_vs_samples(
        trajs,
        ode_llk,
        all_bins,
        alpha,
        cfg,
        subsample_sizes
    )
    make_error_vs_samples(
        ax,
        hist_error_vs_samples,
        pfode_error_vs_samples,
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
            cond=std.cond,
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
            cond=std.cond,
            alpha=alpha,
            exact=cfg.compute_exact_trace,
        )
        cfg_obj = OmegaConf.to_object(cfg)
        alpha_float = alpha.cpu().item()
        rearranged_odes = einops.rearrange(
            ode_llk[0][-1],
            '(b c) -> b c',
            b=cfg.num_sample_batches
        )
        make_plots(rearranged_trajs, ode_llk[0][-1], alpha_float, cfg_obj)


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        suppresswarning()

    cs = ConfigStore.instance()
    cs.store(name="vpsde_sample_config", node=SampleConfig)
    register_configs()

    with torch.no_grad():
        sample()
