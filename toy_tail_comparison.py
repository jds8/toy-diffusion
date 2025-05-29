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
from toy_sample import ContinuousEvaluator, compute_transformed_ode
from toy_train_config import SampleConfig, get_run_type, MultivariateGaussianExampleConfig, \
    BrownianMotionDiffExampleConfig, get_target
from models.toy_diffusion_models_config import ContinuousSamplerConfig
from compute_quadratures import pdf_2d_quadrature_bm, pdf_3d_quadrature_bm


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

ErrorData = namedtuple('ErrorData', 'bins samples median error_bars label color')
HistOutput = namedtuple('HistOutput', 'hist bins')

#########################
#########################
def suppresswarning():
    warnings.warn("user", UserWarning)

def compute_tail_estimate(
        std: ContinuousEvaluator,
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
    plt.clf()
    hist, bins, _ = plt.hist(
        subsap,
        bins=num_bins,
        density=True,
    )
    x = torch.linspace(alpha, 4.0, 100)
    if type(std.example) == MultivariateGaussianExampleConfig:
        dim = std.example.d
        dd = scipy.stats.chi(dim)
        pdf = [dd.pdf(a) for a in x]
    elif type(std.example) == BrownianMotionDiffExampleConfig:
        pdf = [pdf_2d_quadrature_bm(a.cpu().item(), alpha) for a in x]
    else:
        raise NotImplementedError
    plt.plot(x, pdf, color='blue', label='analytical')
    plt.savefig('{}/histogram_plot_{}'.format(
        HydraConfig.get().run.dir,
        subsap.nelement()
    ))
    plt.clf()
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

def compute_sample_error_vs_samples(
        trajs: torch.Tensor,
        alpha: float,
        subsample_sizes: torch.Tensor,
        std: ContinuousEvaluator,
        cfg: SampleConfig,
) -> Tuple[ErrorData, List[List[HistOutput]]]:
    if type(std.example) == MultivariateGaussianExampleConfig:
        dim = cfg.example.d
        dd = scipy.stats.chi(dim)
        analytical_tail = 1#1 - dd.cdf(alpha)
    elif type(std.example) == BrownianMotionDiffExampleConfig:
        dim = cfg.example.sde_steps
        target = get_target(cfg)
        analytical_tail = 1# target.analytical_prob(alpha)
    else:
        raise NotImplementedError
    sample_levels = trajs.norm(dim=[2, 3])
    quantiles = torch.zeros(len(subsample_sizes), 3)
    all_bins = []
    for size_idx, subsample_size in enumerate(subsample_sizes):
        all_subsaps = sample_levels[:, :subsample_size]
        subsample_bins = []
        rel_errors = []
        for subsap_idx, subsap in enumerate(all_subsaps):
            hist, bins, tail_estimate = compute_tail_estimate(
                std,
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

def compute_sample_error_vs_bins(
        trajs: torch.Tensor,
        all_bins: List,
        alpha: float,
        std: ContinuousEvaluator,
        cfg: SampleConfig,
) -> ErrorData:
    dim = trajs.shape[2]
    dd = scipy.stats.chi(dim)
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
                std,
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
        trajs: torch.Tensor,
        all_bins: List,
        alpha: float,
        std: ContinuousEvaluator,
        cfg: SampleConfig,
) -> ErrorData:
    norm_trajs = trajs.norm(dim=[2, 3]).cpu()
    IQR = scipy.stats.iqr(norm_trajs)
    max_sample = norm_trajs.max()

    if type(std.example) == MultivariateGaussianExampleConfig:
        dim = cfg.example.d
        dd = scipy.stats.chi(dim)
        analytical_tail = 1
    elif type(std.example) == BrownianMotionDiffExampleConfig:
        dim = cfg.example.sde_steps
        dt = torch.tensor(1/(dim-1))
        target = get_target(cfg)
        analytical_tail = 1
    else:
        raise NotImplementedError

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
    if type(std.example) == MultivariateGaussianExampleConfig:
        intermediate_traj_elements = (abscissa_tensor**2/dim).sqrt()
        fake_traj = intermediate_traj_elements.repeat(1, dim).unsqueeze(-1)
    elif type(std.example) == MultivariateGaussianExampleConfig:
        intermediate_traj_elements = (abscissa_tensor**2/(dim-1)).sqrt()
        fake_traj = intermediate_traj_elements.repeat(1, dim-1).unsqueeze(-1)
    else:
        raise NotImplementedError
    ode_llk = std.ode_log_likelihood(
        fake_traj,
        cond=torch.tensor([1.]),
        alpha=torch.tensor([alpha]),
        exact=cfg.compute_exact_trace,
    )
    if type(std.example) == MultivariateGaussianExampleConfig:
        transformed_ode_llk = ode_llk[0][-1] + (dim / 2) * torch.tensor(2 * torch.pi).log() + \
            (dim - 1) * abscissa_tensor.squeeze().log() - (dim / 2 - 1) * \
            torch.tensor(2.).log() - scipy.special.loggamma(dim / 2)
        transformed_ode = transformed_ode_llk.exp()
    elif type(std.example) == BrownianMotionDiffExampleConfig:
        transformed_ode = compute_transformed_ode(
            abscissa_tensor.squeeze(),
            ode_llk[0][-1],
            alpha=alpha,
            dt=dt
        )
    rel_errors = []
    augmented_all_num_bins = torch.cat([torch.tensor([0]), bin_sizes+1])
    augmented_cumsum = augmented_all_num_bins.cumsum(dim=0)
    x = abscissas[-1]
    if type(std.example) == MultivariateGaussianExampleConfig:
        pdf = dd.pdf(x)
    elif type(std.example) == BrownianMotionDiffExampleConfig:
        if cfg.example.sde_steps == 3:
            pdf = [pdf_2d_quadrature_bm(a.cpu().item(), alpha) for a in x]
        elif cfg.example.sde_steps == 4:
            pdf = [pdf_3d_quadrature_bm(a.cpu().item(), alpha) for a in x]
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError
    equivalents = []
    for i, num_bins in enumerate(bin_sizes):
        bin_width = (max_sample - alpha) / num_bins
        equiv_saps = (bin_width / (2 * IQR)) ** -3
        equivalents.append(equiv_saps)

        abscissa = abscissas[i]
        abscissa_count = len(abscissa)
        idx = augmented_cumsum[i]
        ode_lk_subsample = transformed_ode[idx:idx+abscissa_count]
        tail_estimate = scipy.integrate.simpson(
            ode_lk_subsample.cpu(),
            x=abscissa
        )
        rel_error = torch.tensor(tail_estimate - analytical_tail).abs() / analytical_tail
        rel_errors.append(rel_error)
        plt.plot(x, pdf, color='blue', label='analytical')
        plt.scatter(abscissa, ode_lk_subsample.cpu(), color='red', label='pfode')
        plt.xlabel('Radius')
        plt.ylabel('Density')
        plt.title(f'PFODE abscissa with estimate: '
            f'{round(tail_estimate.item(), 2)} and error: '
            f'{round(rel_error.item(), 2)}\n'
            f'ESS: {equiv_saps}')
        plt.legend()
        plt.savefig('{}/bin_comparison_density_estimates_{}'.format(
            HydraConfig.get().run.dir,
            i
        ))
        plt.clf()
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

def plot_errors(error_data: ErrorData, title: str):
    plt.scatter(
        error_data.samples,
        error_data.median,
        label=error_data.label,
        color=error_data.color
    )
    if error_data.error_bars[0].sum() + error_data.error_bars[1].sum():
        plt.fill_between(
            error_data.samples,
            error_data.error_bars[0],
            error_data.error_bars[1],
            color=error_data.color,
            alpha=0.2
        )

    save_error_data(error_data, title)

def find_order_of_magnitude_subtensor(x: torch.Tensor) -> torch.Tensor:
    vals = []
    idxs = []
    prev_val = 1.
    for i, val in enumerate(x):
        if val / prev_val >= 10:
            vals.append(val.item())
            prev_val = val
            idxs.append(i)

    return vals, idxs

def make_error_vs_samples(
        sample_error_data: ErrorData,
        pfode_error_data: ErrorData,
        alpha: float
):
    title = f'Relative Error of Tail Integral (alpha={alpha}) vs. Sample Size'
    plot_errors(sample_error_data, title)
    plot_errors(pfode_error_data, title)
    plt.xlabel('Sample Size')
    plt.ylabel('Relative Error')
    plt.title(title)
    plt.xscale('log')

    # add PFODE bin axis
    ax = plt.gca()
    ax_top = ax.secondary_xaxis('top')
    ticklabels, mask = find_order_of_magnitude_subtensor(pfode_error_data.bins)
    ax_top.set_xticks(torch.stack(pfode_error_data.samples)[mask])
    ax_top.set_xticklabels(ticklabels)
    ax_top.set_xlabel('Number of Bins')

def make_plots(
        trajs: torch.Tensor,
        alpha: float,
        cfg: SampleConfig,
        std: ContinuousEvaluator,
):
    plt.clf()

    subsample_sizes = torch.logspace(
        math.log10(500),
        math.log10(cfg.num_samples),
        10,
        dtype=int
    )

    all_bins, hist_error_vs_samples, pfode_error_vs_samples = make_error_vs_samples_plot(
        trajs,
        alpha,
        cfg,
        subsample_sizes,
        std,
    )

    plt.legend()
    plt.tight_layout()

    _, run_type = get_run_type(cfg)
    run_type = run_type.replace(' ', '_')
    plt.savefig('{}/{}_{}_tail_integral_error_vs_sample_size.pdf'.format(
        HydraConfig.get().run.dir,
        run_type,
        alpha
    ))

def make_error_vs_samples_plot(
        trajs: torch.Tensor,
        alpha: float,
        cfg: SampleConfig,
        subsample_sizes: torch.Tensor,
        std: ContinuousEvaluator
):
    hist_error_vs_samples, all_bins = compute_sample_error_vs_samples(
        trajs,
        alpha,
        subsample_sizes,
        std,
        cfg,
    )
    pfode_error_vs_samples = compute_pfode_error_vs_bins(
        trajs,
        all_bins,
        alpha,
        std,
        cfg,
    )
    make_error_vs_samples(
        hist_error_vs_samples,
        pfode_error_vs_samples,
        alpha
    )
    return all_bins, hist_error_vs_samples, pfode_error_vs_samples

@hydra.main(version_base=None, config_path="conf", config_name="continuous_is_config")
def sample(cfg):
    logger = logging.getLogger("main")
    logger.info('run type: tail_comparison')
    cfg_str = OmegaConf.to_yaml(cfg)
    logger.info(f"CONFIG\n{cfg_str}")
    logger.info(f'OUTPUT\n{HydraConfig.get().run.dir}\n')

    os.system('echo git commit: $(git rev-parse HEAD)')

    omega_sampler = OmegaConf.to_object(cfg.sampler)
    if isinstance(omega_sampler, ContinuousSamplerConfig):
        std = ContinuousEvaluator(cfg=cfg)
    else:
        raise NotImplementedError

    with torch.no_grad():
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
        cfg_obj = OmegaConf.to_object(cfg)
        alpha_float = alpha.cpu().item()
        make_plots(
            rearranged_trajs,
            alpha_float,
            cfg_obj,
            std
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
