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
from toy_sample import ContinuousEvaluator, compute_transformed_ode, compute_derivatives, plot_pfode, \
    compute_fake_bm_trajs_random, compute_fake_gaussian_trajs
from toy_train_config import SampleConfig, get_run_type, MultivariateGaussianExampleConfig, \
    BrownianMotionDiffExampleConfig, IntegratorComparisonConfig, Integrator, \
    get_reduction_op
from models.toy_diffusion_models_config import ContinuousSamplerConfig
from compute_quadratures import get_2d_pdf, pdf_2d_quadrature_bm


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
        alpha: float,
        dd,
        dim: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
        sde_steps = dim+1
        # pdf = get_2d_pdf(sde_steps, med_bins[smallest_idx:], alpha)
        # tail_error = scipy.integrate.simpson(
        #     np.abs(hist[smallest_idx:] - pdf),
        #     x=med_bins[smallest_idx:]
        # )
        pdf = get_2d_pdf(sde_steps, med_bins[smallest_idx:], alpha)
        pdf = np.concat([np.zeros(smallest_idx), pdf])
        tail_error = scipy.integrate.simpson(
            np.abs(hist.numpy() - pdf),
            x=med_bins,
        )
    else:
        # scipy.integrate.trapezoid(hist[smallest_idx:], med_bins[smallest_idx:])
        # tail_error = scipy.integrate.simpson(
        #     np.abs(hist[smallest_idx:].numpy() - dd.pdf(med_bins[smallest_idx:])/(1-dd.cdf(alpha))),
        #     x=med_bins[smallest_idx:]
        # )
        pdf = dd.pdf(med_bins)/(1-dd.cdf(alpha)) * (med_bins > alpha).numpy()
        tail_error = scipy.integrate.simpson(
            np.abs(hist.numpy() - pdf),
            x=med_bins
        )
    return hist, bins, torch.tensor(tail_error)

def compute_sample_error_vs_samples(
        rearranged_trajs_list: List[torch.Tensor],
        alpha: float,
        stds: List[ContinuousEvaluator],
        cfg: SampleConfig,
) -> Tuple[ErrorData, List[List[HistOutput]]]:
    if type(stds[0].example) == MultivariateGaussianExampleConfig:
        dim = cfg.example.d
        dd = scipy.stats.chi(dim)
    else:
        dim = stds[0].example.sde_steps-1
        dd = None
    quantiles = []
    all_bins = []
    for rearranged_traj in rearranged_trajs_list:
        sample_levels = rearranged_traj.norm(dim=[2, 3])
        subsample_bins = []
        errors = []
        for subsap_idx, subsap in enumerate(sample_levels):
            hist, bins, error = compute_tail_estimate(
                stds[0],
                subsap.cpu(),
                alpha,
                dd,
                dim,
            )
            subsample_bins.append(HistOutput(hist, bins))
            errors.append(error)
        all_bins.append(subsample_bins)
        errors_tensor = torch.stack(errors)
        quantile = errors_tensor.quantile(
            torch.tensor([0.05, 0.5, 0.95], dtype=errors_tensor.dtype)
        )
        quantiles.append(quantile)
    quantiles_tensor = torch.stack(quantiles)
    error_data = ErrorData(
        stds[0].cfg.timesteps,
        stds[0].cfg.timesteps,
        quantiles_tensor[:, 1],
        quantiles_tensor[:, [0, 2]].movedim(0, 1),
        f'Histogram (N={cfg.num_samples})',
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
        alpha: float,
        stds: List[ContinuousEvaluator],
        cfg: SampleConfig,
        all_bins: List[List[torch.Tensor]]
) -> ErrorData:
    if type(stds[0].example) == MultivariateGaussianExampleConfig:
        dim = cfg.example.d
        dd = scipy.stats.chi(dim)
        leftover = (1 - dd.cdf(sample_trajs.max().cpu())) / (1 - dd.cdf(alpha+cfg.eta))
        analytical_tail = 1. - leftover #1 - dd.cdf(alpha)
    elif type(stds[0].example) == BrownianMotionDiffExampleConfig:
        dim = cfg.example.sde_steps-1
        dd = None
        dt = torch.tensor(1 / dim)
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
    num_bins = ((max_sample - (alpha+cfg.eta)) / bin_width).int()
    # abscissa_N1 = torch.linspace(alpha+cfg.eta, max_sample, num_bins+1).reshape(-1, 1).to(device)
    ode_lks = []
    errors = []
    quantiles_list = []
    reduction_op = get_reduction_op(cfg)
    num_bins = []
    for idx, std in enumerate(stds):
        abscissa_N1 = all_bins[idx][0].bins.unsqueeze(-1)
        if type(stds[0].example) == MultivariateGaussianExampleConfig:
            fake_traj_NbD1, _ = compute_fake_gaussian_trajs(
                abscissa_N1,
                cfg.num_sample_batches,
                dim
            )
            sample_levels = fake_traj_NbD1.norm(dim=1).squeeze()
        elif type(stds[0].example) == BrownianMotionDiffExampleConfig:
            fake_traj_NbD1, _ = compute_fake_bm_trajs_random(
                abscissa_N1,
                dim,
                alpha,
                dt,
                num_trajs=cfg.num_sample_batches//2,
                num_samples=cfg.num_icov_samples,
            )
            sample_levels = einops.repeat(abscissa_N1, 'n 1 -> (n b)', b=cfg.num_sample_batches)
        else:
            raise NotImplementedError
        ode_llk = std.ode_log_likelihood(
            fake_traj_NbD1.to(device),
            cond=torch.tensor([1.]),
            alpha=torch.tensor([alpha]),
            exact=cfg.compute_exact_trace,
        )
        if type(stds[0].example) == BrownianMotionDiffExampleConfig:
            reduction_op = get_reduction_op(cfg)
            new_llk = einops.reduce(ode_llk[0], 't (b i) -> t b', reduction_op, i=cfg.num_icov_samples)
            ode_llk = (new_llk, *ode_llk[1:])
        ode_llk_Nb = ode_llk[0][-1]
        if type(stds[0].example) == MultivariateGaussianExampleConfig:
            transformed_ode_llk_Nb = ode_llk_Nb.cpu() + (dim / 2) * torch.tensor(2 * torch.pi).log() + \
                (dim - 1) * sample_levels.cpu().log() - (dim / 2 - 1) * \
                torch.tensor(2.).log() - scipy.special.loggamma(dim / 2)
            transformed_ode_lk_Nb = transformed_ode_llk_Nb.exp()
            transformed_ode_lk_NB = einops.rearrange(transformed_ode_lk_Nb, '(n b) -> n b', n=abscissa_N1.shape[0])
        elif type(stds[0].example) == BrownianMotionDiffExampleConfig:
            transformed_ode_lk_Nb = compute_transformed_ode(
                sample_levels.cpu().squeeze(),
                ode_llk_Nb,
                alpha=alpha,
                dt=dt
            )
            transformed_ode_lk_NB = einops.reduce(
                transformed_ode_lk_Nb,
                '(n b s) -> n b',
                reduction_op,
                n=abscissa_N1.shape[0],
                b=cfg.num_sample_batches,
            )
        else:
            raise NotImplementedError
        ode_lks.append(transformed_ode_lk_NB)
        errors_B_list = []
        xs = abscissa_N1.squeeze(dim=1).cpu()
        if dd is None:
            sde_steps = dim + 1
            pdf = get_2d_pdf(sde_steps=sde_steps, abscissa=xs, alpha=alpha)
        else:
            pdf = dd.pdf(xs)/(1-dd.cdf(alpha))
        for b in range(transformed_ode_lk_NB.shape[1]):
            estimate = transformed_ode_lk_NB[:, b].cpu().numpy()
            error_N = scipy.integrate.simpson(
                np.abs(estimate - pdf),
                x=xs
            )
            errors_B_list.append(torch.tensor(error_N))

        plt.scatter(xs, transformed_ode_lk_NB[:, b].cpu().numpy(), label='Estimate')
        plt.plot(xs, pdf, label='Analytical')
        plt.legend()
        plt.savefig('{}/estimates_{}_{}.pdf'.format(
            HydraConfig.get().run.dir,
            std.sampler.diffusion_timesteps,
            std.cfg.model_name
        ))
        plt.clf()

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
        num_bins.append(xs.nelement())
        # plt.plot(abscissa_N1.cpu(), pdf, color='blue')
        # plt.scatter(abscissa_N1, ode_llk_subsample.cpu().exp(), color='red')
        # plt.savefig('{}/bin_comparison_density_estimates_{}'.format(
        #     HydraConfig.get().run.dir,
        #     i
        # ))
        # plt.clf()
    quantiles = torch.stack(quantiles_list)

    min_bins = min(num_bins)
    max_bins = max(num_bins)
    if min_bins < max_bins:
        label = f'bins=[{min_bins}, {max_bins}]'
    else:
        label = f'bins={min_bins}'
    error_data = ErrorData(
        stds[0].cfg.timesteps,
        stds[0].cfg.timesteps,
        quantiles[:, 1],
        quantiles[:, [0, 2]].movedim(0, 1),
        f'ICOV ({label})',
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

def plot_errors(error_data: ErrorData):
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

def make_error_vs_samples(
        sample_error_data: ErrorData,
        icov_error_data: ErrorData,
        alpha: float,
        cfg: SampleConfig,
):
    title = f'Integrated ({cfg.density_integrator}) Absolute Error vs. Num. Diffusion Steps\n(alpha={alpha})'
    plot_errors(sample_error_data)
    plot_errors(icov_error_data)
    plt.xlabel('Diffusion Steps')
    plt.ylabel('Integrated Absolute Error')
    plt.legend()
    plt.title(title)

def make_plots(
        rearranged_trajs_list: List[torch.Tensor],
        alpha: float,
        cfg: SampleConfig,
        stds: List[ContinuousEvaluator],
):
    plt.clf()

    all_bins = make_error_vs_samples_plot(
        rearranged_trajs_list,
        alpha,
        cfg,
        stds,
    )
    plt.xscale("log")
    plt.yscale("log")
    # bottom, top = plt.ylim()
    # new_top = min(top, 10**4)
    lwr, upr = plt.get_ylim()
    lwr = min(lwr, 1e-2)
    plt.ylim((lwr, 1e4))
    # plt.grid(which='both', axis='y')
    # ax3 = ax1.twiny()
    # ax3.set_xlim(ax1.get_xlim())
    # ax3.set_xlabel('Num Bins')

    _, run_type = get_run_type(cfg)
    run_type = run_type.replace(' ', '_')
    plt.savefig('{}/{}_{}_tail_integral_error_vs_integrator.pdf'.format(
        HydraConfig.get().run.dir,
        run_type,
        alpha
    ))

def make_error_vs_samples_plot(
        rearranged_trajs_list: List[torch.Tensor],
        alpha: float,
        cfg: SampleConfig,
        stds: List[ContinuousEvaluator],
):
    hist_error_vs_samples, all_bins = compute_sample_error_vs_samples(
        rearranged_trajs_list,
        alpha,
        stds,
        cfg,
    )
    dim = rearranged_trajs_list[0].shape[2]
    icov_error_vs_samples = compute_icov_error_vs_bins(
        rearranged_trajs_list[0],
        alpha,
        stds,
        cfg,
        all_bins
    )
    make_error_vs_samples(
        hist_error_vs_samples,
        icov_error_vs_samples,
        alpha,
        cfg
    )
    return all_bins

def get_num_samples(model_name: str) -> int:
    result = re.search(r'_v([0-9]+)', model_name)
    return int(result[1])

@hydra.main(version_base=None, config_path="conf", config_name="continuous_is_config")
def sample(cfg):
    logger = logging.getLogger("main")
    logger.info('run type: integrator_comparison')
    cfg_str = OmegaConf.to_yaml(cfg)
    logger.info(f"CONFIG\n{cfg_str}")
    logger.info(f'OUTPUT\n{HydraConfig.get().run.dir}\n')

    os.system('echo git commit: $(git rev-parse HEAD)')

    # Here I set the random seed so that, for the same sample size,
    # the same histogram is generated.
    # Also note that the histogram is generated before the ICOV,
    # so any random in constructing the ICOV, namely due to the
    # aggregation (e.g. mean, median) happens after the histogram
    # has already been constructed.
    torch.manual_seed(cfg.random_seed)

    omega_sampler = OmegaConf.to_object(cfg.sampler)
    if isinstance(omega_sampler, ContinuousSamplerConfig):
        stds = []  
        for timestep in cfg.timesteps:
            new_cfg = deepcopy(cfg)
            new_cfg.sampler.diffusion_timesteps = timestep
            std = ContinuousEvaluator(new_cfg)
            stds.append(std)
    else:
        raise NotImplementedError

    cfg_obj = OmegaConf.to_object(cfg)
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

            small_idx = torch.topk(trajs.norm(dim=-2).squeeze(), k=7, largest=False).indices
            traj_subset = sample_trajs[:, small_idx, :, 0].to('cpu')
            derivatives, times = compute_derivatives(std, traj_subset)
            diffusion_steps = std.cfg.sampler.diffusion_timesteps
            plot_title = '{} Time Steps'.format(diffusion_steps)
            plot_pfode(
                traj_subset,
                derivatives,
                times,
                'pfode_{}_{}'.format(
                    diffusion_steps,
                    std.cfg.model_name
                ),
                plot_title,
            )

            rearranged_trajs = einops.rearrange(
                trajs,
                '(b c) h w -> b c h w',
                b=cfg.num_sample_batches
            )
            rearranged_trajs_list.append(rearranged_trajs)
        alpha_float = alpha.cpu().item()
        make_plots(
            rearranged_trajs_list,
            alpha_float,
            cfg_obj,
            stds,
        )


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        suppresswarning()

    cs = ConfigStore.instance()
    cs.store(name="vpsde_sample_config", node=IntegratorComparisonConfig)
    register_configs()

    with torch.no_grad():
        sample()
