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

ErrorData = namedtuple('ErrorData', 'bins samples median error_bars label color')
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
    # ys = [pdf_2d_quadrature_bm(a.cpu().numpy(), alpha) for a in selected_samples_sorted]
    # plt.clf()
    # plt.plot(selected_samples_sorted, ys)
    # plt.scatter(selected_samples_sorted, ordinates_sorted)
    # t = time.time()
    # plt.savefig('{}/pfode_tail_estimate_from_bins_plot_{}'.format(
    #     HydraConfig.get().run.dir,
    #     int(t)
    # ))
    # plt.clf()
    return tail_estimate_tensor

def sample(std: ContinuousEvaluator):
    sample_traj_out = std.sample_trajectories(
        cond=-1.,
        alpha=std.likelihood.alpha.reshape(-1, 1),
    )

    sample_trajs = sample_traj_out.samples
    trajs = sample_trajs[-1]
    out_trajs = trajs
    return out_trajs

def compute_sample_error_vs_samples(
        rearranged_trajs_list: List[torch.Tensor],
        alpha: float,
        subsample_sizes: torch.Tensor,
) -> Tuple[ErrorData, List[List[HistOutput]]]:
    dim = rearranged_trajs_list[0].shape[2]
    dd = scipy.stats.chi(dim)
    analytical_tail = 1 - dd.cdf(alpha)
    quantiles = torch.zeros(len(subsample_sizes), 3)
    all_bins = []
    for rearranged_traj in rearranged_trajs_list:
        sample_levels = rearranged_traj.norm(dim=[2, 3])
        subsample_bins = []
        rel_errors = []
        for subsap_idx, subsap in enumerate(sample_levels):
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
        bin_sizes,
        quantiles[:, 1],
        quantiles[:, [0, 2]].movedim(0, 1),
        'Histogram Approximation',
        'blue'
    )
    return error_data

def compute_pfode_error_vs_bins(
        dim: float,
        all_bins: List,
        alpha: float,
        stds: List[ContinuousEvaluator],
        cfg: SampleConfig,
        IQR: float,
) -> ErrorData:
    dd = scipy.stats.chi(dim)
    max_sample = dd.ppf(0.99999)
    analytical_tail = 1 - dd.cdf(alpha)
    bin_sizes = torch.tensor([len(bins[0].bins) for bins in all_bins])
    bin_sizes = torch.cat([bin_sizes, torch.logspace(
        math.log10(len(all_bins[-1][0].bins)),
        math.log10(1000),
        5,
        dtype=int
    )[1:]]).unique()
    abscissas = []
    for num_bins in bin_sizes:
        abscissa = torch.linspace(alpha, max_sample, num_bins+1)
        abscissas.append(abscissa)
    abscissa_tensor = torch.cat(abscissas).reshape(-1, 1).to(device)
    fake_traj = torch.cat([
        torch.zeros(abscissa_tensor.shape[0], dim-1, device=device),
        abscissa_tensor
    ], 1).unsqueeze(-1)
    ode_llks = []
    rel_errors = []
    for std in stds:
        ode_llk = std.ode_log_likelihood(
            fake_traj,
            cond=torch.tensor([-1.]),
            alpha=torch.tensor([alpha]),
            exact=cfg.compute_exact_trace,
        )
        chi_ode_llk = ode_llk[0][-1] + (dim / 2) * torch.tensor(2 * torch.pi).log() + \
            (dim - 1) * abscissa_tensor.squeeze().log() - (dim / 2 - 1) * \
            torch.tensor(2.).log() - scipy.special.loggamma(dim / 2)
        ode_llks.append(chi_ode_llk)
        augmented_all_num_bins = torch.cat([torch.tensor([0]), bin_sizes+1])
        augmented_cumsum = augmented_all_num_bins.cumsum(dim=0)
        x = abscissas[-1]
        pdf = dd.pdf(x)
        equivalents = []
        bin_width = int((max_sample - alpha) / num_bins)
        equiv_saps = (bin_width / (2 * IQR)) ** -3
        equivalents.append(equiv_saps)

        abscissa = abscissas[i]
        abscissa_count = len(abscissa)
        idx = augmented_cumsum[i]
        tail_estimate = scipy.integrate.simpson(
            ode_llk_subsample.cpu().exp(),
            x=abscissa
        )
        rel_error = torch.tensor(tail_estimate - analytical_tail).abs() / analytical_tail
        rel_errors.append(rel_error)
        # save_pfode_samples(abscissa, ode_llk_subsample)
        # plt.plot(x, pdf, color='blue')
        # plt.scatter(abscissa, ode_llk_subsample.cpu().exp(), color='red')
        # plt.savefig('{}/bin_comparison_density_estimates_{}'.format(
        #     HydraConfig.get().run.dir,
        #     i
        # ))
        # plt.clf()
    median_tensor = torch.stack(rel_errors)
    zeros_tensor = torch.zeros_like(median_tensor)
    conf_int_tensor = torch.stack([zeros_tensor, zeros_tensor])

    error_data = ErrorData(
        bin_sizes,
        equivalents,
        median_tensor,
        conf_int_tensor,
        'PFODE Approximation',
        'orange'
    )
    return error_data

def old_compute_pfode_error_vs_bins(
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
        'Abscissa_bins': error_data.bins,
        'Abscissa_samples': error_data.samples,
        'Median': error_data.median,
        '5%': error_data.error_bars[0],
        '95%': error_data.error_bars[1]
    }, abs_filename)

def plot_errors(ax, error_data: ErrorData, title: str):
    ax.scatter(
        error_data.samples,
        error_data.median,
        label=error_data.label,
        color=error_data.color
    )
    ax.fill_between(
        error_data.samples,
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
    title = f'Relative Error of Tail Integral (alpha={alpha}) vs. Number of Bins'
    plot_errors(ax, sample_error_data, title)
    plot_errors(ax, pfode_error_data, title)
    ax.set_xlabel('Number of Bins')
    ax.set_ylabel('Relative Error')
    ax.set_title(title)

def make_plots(
        rearranged_trajs_list: List[torch.Tensor],
        rearranged_odes_list: List[torch.Tensor],
        alpha: float,
        cfg: SampleConfig,
        stds: List[ContinuousEvaluator],
):
    plt.clf()

    subsample_sizes = torch.logspace(
        math.log10(500),
        math.log10(cfg.num_samples),
        10,
        dtype=int
    )

    all_bins = make_error_vs_samples_plot(
        rearranged_trajs_list,
        rearranged_odes_list,
        alpha,
        cfg,
        subsample_sizes,
        stds,
    )
    plt.xscale("log")
    # ax3 = ax1.twiny()
    # ax3.set_xlim(ax1.get_xlim())
    # ax3.set_xlabel('Num Bins')

    _, run_type = get_run_type(cfg)
    run_type = run_type.replace(' ', '_')
    plt.savefig('{}/{}_{}_tail_integral_error.pdf'.format(
        HydraConfig.get().run.dir,
        run_type,
        alpha
    ))

def make_error_vs_samples_plot(
        rearranged_trajs_list: List[torch.Tensor],
        alpha: float,
        cfg: SampleConfig,
        subsample_sizes: torch.Tensor,
        stds: List[ContinuousEvaluator]
):
    hist_error_vs_samples, all_bins = compute_sample_error_vs_samples(
        rearranged_trajs_list,
        alpha,
        subsample_sizes
    )
    dim = rearranged_trajs_list[0].shape[2]
    pfode_error_vs_samples = compute_pfode_error_vs_bins(
        dim,
        all_bins,
        alpha,
        stds,
        cfg,
    )
    make_error_vs_samples(
        hist_error_vs_samples,
        pfode_error_vs_samples,
        alpha
    )
    return all_bins

@hydra.main(version_base=None, config_path="conf", config_name="continuous_is_config")
def sample(cfg):
    logger = logging.getLogger("main")
    logger.info('run type: importance sampling')
    cfg_str = OmegaConf.to_yaml(cfg)
    logger.info(f"CONFIG\n{cfg_str}")

    os.system('echo git commit: $(git rev-parse HEAD)')

    omega_sampler = OmegaConf.to_object(cfg.sampler)
    if isinstance(omega_sampler, ContinuousSamplerConfig):
        stds = []  
        for trained_model in cfg.trained_models:
            cfg.model_name = trained_model
            std = ContinuousEvaluator(cfg)
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
            rearranged_trajs_list.append(rearranged_trajs)
            ode_llk = std.ode_log_likelihood(
                trajs,
                cond=torch.tensor([-1.]),
                alpha=alpha,
                exact=cfg.compute_exact_trace,
            )
            sample_levels = trajs.norm(dim=[1, 2])
        alpha_float = alpha.cpu().item()
        make_plots(
            rearranged_trajs_list,
            alpha_float,
            cfg_obj,
            stds
        )


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        suppresswarning()

    cs = ConfigStore.instance()
    cs.store(name="vpsde_sample_config", node=SampleConfig)
    register_configs()

    with torch.no_grad():
        sample()
