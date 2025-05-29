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
    try:
        tail_estimate = scipy.integrate.simpson(
            hist[smallest_idx:],
            x=med_bins[smallest_idx:]
        )
    except:
        tail_estimate = 0.
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

def save_pfode_samples(
        abscissa: torch.Tensor,
        transformed_ode_lk: torch.Tensor,
        equiv_saps: torch.Tensor,
        model_name: str
):
    pfode_dir = 'pfode_likelihoods'
    abs_dir = f'{HydraConfig.get().run.dir}/{pfode_dir}'
    os.makedirs(abs_dir, exist_ok=True)
    abs_filename = f'{abs_dir}/{model_name}.pt'
    torch.save({
        'Abscissa': abscissa,
        'TransformedOdeLLK': transformed_ode_lk,
        'EquivSaps': equiv_saps,
        'ModelName': model_name
    }, abs_filename)

def compute_pfode_error_vs_bins(
        sample_trajs: torch.Tensor,
        stds: List[ContinuousEvaluator],
        cfg: SampleConfig,
        alphas: torch.Tensor,
) -> ErrorData:
    if type(stds[0].example) == MultivariateGaussianExampleConfig:
        dim = cfg.example.d
        dd = scipy.stats.chi(dim)
        analytical_tail = 1.#1 - dd.cdf(alpha)
    elif type(stds[0].example) == BrownianMotionDiffExampleConfig:
        dim = cfg.example.sde_steps
        # cfg_obj = OmegaConf.to_object(cfg)
        # target = get_target(cfg_obj)
        # analytical_tail = target.analytical_prob(alpha)
        analytical_tail = 1.
        dt = torch.tensor(1 / (dim-1))
    else:
        raise NotImplementedError
    ode_lks = []
    rel_errors = []
    for std in stds:
        dd = scipy.stats.chi(dim)
        max_sample = dd.ppf(0.99999)
        num_bins = 1000
        bin_width = (max_sample - std.likelihood.alpha) / num_bins
        sample_trajs = einops.rearrange(
            sample_trajs,
            'b c h w -> (b c) h w',
            b=cfg.num_sample_batches
        ).norm(dim=[1, 2])
        IQR = scipy.stats.iqr(sample_trajs.cpu())
        equiv_saps = (bin_width / (2 * IQR)) ** -3
        abscissa = torch.linspace(std.likelihood.alpha, max_sample, num_bins+1).to(device)
        intermediate_traj_elements = (abscissa**2/(dim-1)).sqrt()
        fake_traj = intermediate_traj_elements.repeat(1, dim-1).unsqueeze(-1)
        x = abscissa
        # pdf = dd.pdf(x)
        ode_llk = std.ode_log_likelihood(
            fake_traj,
            cond=torch.tensor([1.]),
            alpha=torch.tensor([std.likelihood.alpha]),
            exact=cfg.compute_exact_trace,
        )
        if type(stds[0].example) == MultivariateGaussianExampleConfig:
            transformed_ode_llk = ode_llk[0][-1].cpu() + (dim / 2) * torch.tensor(2 * torch.pi).log() + \
                (dim - 1) * abscissa.cpu().squeeze().log() - (dim / 2 - 1) * \
                torch.tensor(2.).log() - scipy.special.loggamma(dim / 2)
            transformed_ode_lk = transformed_ode_llk.exp()
        elif type(stds[0].example) == BrownianMotionDiffExampleConfig:
            transformed_ode_lk = compute_transformed_ode(
                abscissa.cpu().squeeze(),
                ode_llk[0][-1],
                alpha=std.likelihood.alpha,
                dt=dt
            )
        else:
            raise NotImplementedError
        ode_lks.append(transformed_ode_lk)
        tail_estimate = scipy.integrate.simpson(
            transformed_ode_lk.cpu(),
            x=abscissa.cpu()
        )
        rel_error = torch.tensor(tail_estimate - analytical_tail).abs() / analytical_tail
        rel_errors.append(rel_error)
        save_pfode_samples(abscissa.cpu(), transformed_ode_lk, equiv_saps, std.cfg.model_name)
    median_tensor = torch.stack(rel_errors)
    zeros_tensor = torch.zeros_like(median_tensor)
    conf_int_tensor = torch.stack([zeros_tensor, zeros_tensor])

    error_data = ErrorData(
        alphas,
        alphas,
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
        pfode_error_data: ErrorData,
        alpha: float
):
    title = f'Relative Error of Tail Integral (alpha={alpha}) vs. Training Samples'
    plot_errors(sample_error_data, title)
    plot_errors(pfode_error_data, title)
    plt.xlabel('Training Samples')
    plt.ylabel('Relative Error')
    plt.title(title)

def make_plots(
        rearranged_trajs_list: List[torch.Tensor],
        cfg: SampleConfig,
        stds: List[ContinuousEvaluator],
        alphas: torch.Tensor
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
        cfg,
        stds,
        alphas
    )
    plt.xscale("log")

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
    pfode_error_vs_samples = compute_pfode_error_vs_bins(
        rearranged_trajs_list[0],
        stds,
        cfg,
        alphas
    )
    make_error_vs_samples(
        hist_error_vs_samples,
        pfode_error_vs_samples,
        alphas
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
