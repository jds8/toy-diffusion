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
    compute_fake_bm_trajs_random, compute_fake_gaussian_trajs, \
    line_circle_intersection, vertical_line_circle_intersection, plot_boundary
from toy_train_config import SampleConfig, get_run_type, MultivariateGaussianExampleConfig, \
    BrownianMotionDiffExampleConfig, TrainComparisonConfig, Integrator
from models.toy_diffusion_models_config import ContinuousSamplerConfig
from compute_quadratures import pdf_2d_quadrature_bm, get_2d_pdf


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

ErrorData = namedtuple('ErrorData', 'bins samples median error_bars label color')
HistOutput = namedtuple('HistOutput', 'hist bins')

TITLE = 'Integrated Absolute Error of Tail Integral vs. Training Samples\n(alpha={})'

#########################
#########################
def suppresswarning():
    warnings.warn("user", UserWarning)

def compute_tail_error(
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
        # tail_estimate = scipy.integrate.simpson(
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
        training_samples: torch.Tensor,
        std: ContinuousEvaluator,
        cfg: SampleConfig,
) -> Tuple[ErrorData, List[List[HistOutput]]]:
    if type(std.example) == MultivariateGaussianExampleConfig:
        dim = cfg.example.d
        dd = scipy.stats.chi(dim)
    else:
        dim = std.example.sde_steps-1
        dd = None
    quantiles = []
    all_bins = []
    for rearranged_traj in rearranged_trajs_list:
        sample_levels = rearranged_traj.norm(dim=[2, 3])
        subsample_bins = []
        errors = []
        for subsap_idx, subsap in enumerate(sample_levels):
            hist, bins, error = compute_tail_error(
                std,
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

        save_histogram_samples(
            errors,
            std.cfg.model_name,
            subsample_bins
        )
    quantiles_tensor = torch.stack(quantiles)
    error_data = ErrorData(
        training_samples,
        training_samples,
        quantiles_tensor[:, 1],
        quantiles_tensor[:, [0, 2]].movedim(0, 1),
        'Histogram',
        'blue'
    )
    title = TITLE.format(alpha)
    save_error_data(error_data, title)
    return error_data, all_bins

def save_histogram_samples(
        errors: torch.Tensor,
        model_name: str,
        subsample_bins: List[HistOutput],
):
    hist_dir = 'histogram'
    abs_dir = f'{HydraConfig.get().run.dir}/{hist_dir}'
    os.makedirs(abs_dir, exist_ok=True)
    abs_filename = f'{abs_dir}/{model_name}.pt'
    torch.save({
        'Errors': errors,
        'ModelName': model_name,
        'Hist': torch.stack([hist.hist for hist in subsample_bins]),
        'Bins': torch.stack([hist.bins for hist in subsample_bins]),
    }, abs_filename)

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

def compute_icov_error_vs_bins(
        sample_trajs: torch.Tensor,
        alpha: float,
        stds: List[ContinuousEvaluator],
        cfg: SampleConfig,
        training_samples: torch.Tensor,
        all_bins: List[List[torch.Tensor]]
) -> ErrorData:
    if type(stds[0].example) == MultivariateGaussianExampleConfig:
        dim = cfg.example.d
        dd = scipy.stats.chi(dim)
        leftover = (1 - dd.cdf(sample_trajs.max().cpu())) / (1 - dd.cdf(alpha))
        analytical_tail = 1. - leftover #1 - dd.cdf(alpha)
    elif type(stds[0].example) == BrownianMotionDiffExampleConfig:
        dim = cfg.example.sde_steps-1
        # cfg_obj = OmegaConf.to_object(cfg)
        # target = get_target(cfg_obj)
        # analytical_tail = target.analytical_prob(alpha)
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
    num_bins = ((max_sample - (alpha)) / bin_width).int()
    abscissa_N1 = torch.linspace(alpha, max_sample, num_bins+1).reshape(-1, 1).to(device)
    ode_lks = []
    errors = []
    quantiles_list = []
    for idx, std in enumerate(stds):
        abscissa_N1 = all_bins[idx][0].bins.unsqueeze(-1)
        if type(stds[0].example) == MultivariateGaussianExampleConfig:
            fake_traj_NbD1, _ = compute_fake_gaussian_trajs(
                abscissa_N1,
                cfg.num_sample_batches,
                dim
            )
        elif type(stds[0].example) == BrownianMotionDiffExampleConfig:
            fake_traj_NbD1, _ = compute_fake_bm_trajs_random(
                abscissa_N1,
                dim,
                alpha,
                dt,
                num_trajs=cfg.num_sample_batches//2,
                num_samples=cfg.num_icov_samples,
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
        xs = abscissa_N1.squeeze().cpu()
        if dd is None:
            sde_steps = dim + 1
            pdf = get_2d_pdf(sde_steps=sde_steps, abscissa=xs, alpha=alpha)
        else:
            pdf = dd.pdf(xs)/(1-dd.cdf(alpha))
        for b in range(transformed_ode_lk_NB.shape[1]):
            error_N = scipy.integrate.simpson(
                np.abs(transformed_ode_lk_NB[:, b].cpu().numpy() - pdf),
                x=xs,
            )
            errors_B_list.append(torch.tensor(error_N))
        plt.plot(abscissa_N1.squeeze().cpu(), dd.pdf(abscissa_N1.squeeze().cpu())/(1-dd.cdf(alpha)))
        plt.scatter(abscissa_N1.squeeze().cpu(), transformed_ode_lk_NB[:, b].cpu().numpy())
        plt.savefig('{}/{}_estimates.pdf'.format(
            HydraConfig.get().run.dir,
            std.cfg.model_name
        ))
        plt.clf()

        if cfg.density_integrator == Integrator.EULER:
            small_idx = torch.topk(fake_traj_NbD1.norm(dim=-2).squeeze(), k=7, largest=False).indices
            sol = ode_llk[2]
            p = sol[1]
            start_time = std.sampler.t_eps if std.cfg.test == TestType.Test else 0.
            times = torch.linspace(
                start_time,
                1.,
                std.sampler.diffusion_timesteps,
            )
            dp_dt = sol[1].diff(dim=0) / times.diff()[0]

            plt.clf()
            fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
            for i in small_idx:
                ax1.plot(times, p[:, i].to('cpu'))
                ax2.plot(times[:-1], dp_dt[:, i].to('cpu'))
            ax1.set_ylabel(f"log p")
            ax2.set_ylabel(f"(log p)'")
            ax2.set_xlabel(f'Times')
            plt.savefig('{}/icov_plot.pdf'.format(HydraConfig.get().run.dir))
            plt.close()

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
        # plt.plot(abscissa_N1.cpu(), pdf, color='blue')
        # plt.scatter(abscissa_N1, ode_llk_subsample.cpu().exp(), color='red')
        # plt.savefig('{}/bin_comparison_density_estimates_{}'.format(
        #     HydraConfig.get().run.dir,
        #     i
        # ))
        # plt.clf()
    quantiles = torch.stack(quantiles_list)
    # import pdb; pdb.set_trace()
    all_bins_flattened = torch.cat(
        [bins.bins for all_bins_lst in all_bins for bins in all_bins_lst],
        dim=0
    )
    bins_quantiles = torch.quantile(all_bins_flattened,
                                    torch.tensor([0.0, 0.5, 1.0],
                                                 device=all_bins_flattened.device,
                                                 dtype=all_bins_flattened.dtype))

    error_data = ErrorData(
        bins_quantiles,
        training_samples,
        quantiles[:, 1],
        quantiles[:, [0, 2]].movedim(0, 1),
        'ICOV',
        'orange'
    )
    title = TITLE.format(alpha)
    save_error_data(error_data, title)
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

def plot_errors(error_data: ErrorData, label: str):
    plt.scatter(
        error_data.samples,
        error_data.median,
        label=error_data.label + f'({label})',
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
    plot_errors(sample_error_data, f"N={cfg.num_samples}")
    max_bin_diff = max(icov_error_data.bins.diff())
    plot_errors(icov_error_data, f"bins={icov_error_data.bins[1]} +/- {max_bin_diff}")
    plt.xlabel('Training Samples')
    plt.ylabel('Absolute Error')
    plt.legend()
    plt.title(title)

def make_plots(
        rearranged_trajs_list: List[torch.Tensor],
        alpha: float,
        cfg: SampleConfig,
        stds: List[ContinuousEvaluator],
        training_samples: torch.Tensor
):
    plt.clf()

    all_bins = make_error_vs_samples_plot(
        rearranged_trajs_list,
        alpha,
        cfg,
        stds,
        training_samples
    )
    plt.xscale("log")
    plt.yscale("log")
    plt.grid(which='both', axis='y')
    # ax3 = ax1.twiny()
    # ax3.set_xlim(ax1.get_xlim())
    # ax3.set_xlabel('Num Bins')

    _, run_type = get_run_type(cfg)
    run_type = run_type.replace(' ', '_')
    plt.savefig('{}/{}_{}_tail_integral_error_vs_training.pdf'.format(
        HydraConfig.get().run.dir,
        run_type,
        alpha
    ))

def make_error_vs_samples_plot(
        rearranged_trajs_list: List[torch.Tensor],
        alpha: float,
        cfg: SampleConfig,
        stds: List[ContinuousEvaluator],
        training_samples: torch.Tensor
):
    hist_error_vs_samples, all_bins = compute_sample_error_vs_samples(
        rearranged_trajs_list,
        alpha,
        training_samples,
        stds[0],
        cfg,
    )
    dim = rearranged_trajs_list[0].shape[2]
    icov_error_vs_samples = compute_icov_error_vs_bins(
        rearranged_trajs_list[-1],
        alpha,
        stds,
        cfg,
        training_samples,
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
    logger.info('run type: train_comparison')
    cfg_str = OmegaConf.to_yaml(cfg)
    logger.info(f"CONFIG\n{cfg_str}")
    logger.info(f'OUTPUT\n{HydraConfig.get().run.dir}\n')

    os.system('echo git commit: $(git rev-parse HEAD)')

    torch.manual_seed(cfg.random_seed)

    omega_sampler = OmegaConf.to_object(cfg.sampler)
    if isinstance(omega_sampler, ContinuousSamplerConfig):
        stds = []
        for trained_model in cfg.trained_models:
            new_cfg = deepcopy(cfg)
            new_cfg.model_name = trained_model
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
            plot_pfode(traj_subset, derivatives, times, 'subset_{}'.format(std.cfg.model_name))

            rearranged_trajs = einops.rearrange(
                trajs,
                '(b c) h w -> b c h w',
                b=cfg.num_sample_batches
            )
            rearranged_trajs_list.append(rearranged_trajs)
        alpha_float = alpha.cpu().item()
        training_samples = torch.tensor([get_num_samples(s.cfg.model_name) for s in stds])
        make_plots(
            rearranged_trajs_list,
            alpha_float,
            cfg_obj,
            stds,
            training_samples
        )


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        suppresswarning()

    cs = ConfigStore.instance()
    cs.store(name="vpsde_sample_config", node=TrainComparisonConfig)
    register_configs()

    with torch.no_grad():
        sample()
