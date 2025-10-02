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
import matplotlib.animation as animation
import matplotlib.colors as colors
import numpy as np
from scipy.interpolate import griddata

from toy_configs import register_configs
from toy_sample import ContinuousEvaluator, compute_transformed_ode, compute_perimeter, get_raw, \
    compute_derivatives, plot_pfode, get_points_along_angle, plot_boundary
from toy_train_config import SampleConfig, get_run_type, MultivariateGaussianExampleConfig, \
    BrownianMotionDiffExampleConfig, get_target, get_error_metric, ErrorMetric, \
    TestType, Integrator
from models.toy_diffusion_models_config import ContinuousSamplerConfig
from compute_quadratures import get_2d_pdf, pdf_2d_quadrature_bm


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

ErrorData = namedtuple('ErrorData', 'bins samples median error_bars label color')
HistOutput = namedtuple('HistOutput', 'hist bins')

#########################
#########################
def suppresswarning():
    warnings.warn("user", UserWarning)

def interp(tensor: torch.Tensor, num_points: int=20):
    start = tensor[:-1]
    end = tensor[1:]
    
    # Linearly interpolate between each pair
    # Shape will be (len-1, num_points)
    interp_output = torch.stack([
        torch.linspace(s.item(), e.item(), num_points)
        for s, e in zip(start, end)
    ])
    return interp_output

def plot_ode_error(ode_lk_subsample, abscissa, abscissa_interp, alpha, num_bins):
    flattened_abscissa = abscissa_interp.reshape(-1)
    abscissa_pdf = [pdf_2d_quadrature_bm(a.cpu().item(), alpha) for a in flattened_abscissa]
    abscissa_pdf = torch.tensor(abscissa_pdf).reshape(abscissa_interp.shape)
    analytical_integrals = [scipy.integrate.simpson(
        abscissa_pdf_values,
        x=abscissa_interp_values
    ) for abscissa_interp_values, abscissa_pdf_values in zip(abscissa_interp, abscissa_pdf)]
    ode_integrals = [scipy.integrate.trapezoid(ode_lk_subsample[i:i+2], abscissa[i:i+2]) for i in range(len(abscissa)-1)]
    errors = [ode_value - analytical_value for ode_value, analytical_value in zip(ode_integrals, analytical_integrals)]
    plt.scatter(abscissa[:-1], errors, color='red', label='ICOV (Trapezoid) - Analytical')
    plt.xlabel('Radius')
    plt.ylabel('Error')
    plt.title(f'Error between ICOV and Analytical at each Radius')
    plt.legend()
    plt.savefig('{}/pfode_error_{}'.format(
        HydraConfig.get().run.dir,
        num_bins
    ))
    plt.clf()

def compute_tail_estimate(
        std: ContinuousEvaluator,
        subsap: torch.Tensor,
        alpha: float,
        x: torch.Tensor,
        pdf: torch.Tensor,
        dd,
) -> Tuple[torch.Tensor, torch.Tensor, float]:
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
    plt.plot(x, pdf, color='blue', label='analytical')
    smallest_idx = max((bins < alpha).sum() - 1, 0)
    # empirical_bin_width = bins[1] - bins[0]
    # tail_estimate = hist[smallest_idx:].sum() * empirical_bin_width
    lwr_bins = bins[:-1]
    upr_bins = bins[1:]
    med_bins = torch.tensor((lwr_bins + upr_bins) / 2)
    if dd is None:
        # Brownian motion case
        pdf = get_2d_pdf(std.example.sde_steps, med_bins[smallest_idx:], alpha)
        tail_error = scipy.integrate.simpson(
            np.abs(hist[smallest_idx:] - pdf),
            x=med_bins[smallest_idx:]
        )
    else:
        # scipy.integrate.trapezoid(hist[smallest_idx:], med_bins[smallest_idx:])
        tail_error = scipy.integrate.simpson(
            np.abs(hist[smallest_idx:] - dd.pdf(med_bins[smallest_idx:])/(1-dd.cdf(alpha))),
            x=med_bins[smallest_idx:]
        )
    plt.scatter(med_bins[smallest_idx:], hist[smallest_idx:], color='red', label='approximation')
    plt.savefig('{}/histogram_plot_{}'.format(
        HydraConfig.get().run.dir,
        subsap.nelement()
    ))
    plt.clf()
    return hist, bins, tail_error

def compute_sample_error_vs_samples(
        trajs: torch.Tensor,
        alpha: float,
        subsample_sizes: torch.Tensor,
        std: ContinuousEvaluator,
        cfg: SampleConfig,
        error_metric: ErrorMetric,
) -> Tuple[ErrorData, List[List[HistOutput]]]:
    if type(std.example) == MultivariateGaussianExampleConfig:
        dim = cfg.example.d
        dd = scipy.stats.chi(dim)
        leftover = (1 - dd.cdf(trajs.max().cpu())) / (1 - dd.cdf(alpha))
        analytical_tail = 1 - leftover #1 - dd.cdf(alpha)
    elif type(std.example) == BrownianMotionDiffExampleConfig:
        dim = cfg.example.sde_steps
        target = get_target(cfg)
        analytical_tail = 1# target.analytical_prob(alpha)
    else:
        raise NotImplementedError
    sample_levels = trajs.norm(dim=[2, 3])

    quantiles = torch.zeros(len(subsample_sizes), 3)
    all_bins = []
    max_value = sample_levels.max()
    x = torch.linspace(alpha, max_value, 100)
    if type(std.example) == MultivariateGaussianExampleConfig:
        dim = std.example.d
        dd = scipy.stats.chi(dim)
        pdf = [dd.pdf(a) / (1 - dd.cdf(alpha)) for a in x]
    elif type(std.example) == BrownianMotionDiffExampleConfig:
        dd = None
        pdf = get_2d_pdf(std.example.sde_steps, x, alpha)
    else:
        raise NotImplementedError
    for size_idx, subsample_size in enumerate(subsample_sizes):
        all_subsaps = sample_levels[:, :subsample_size]
        subsample_bins = []
        errors = []
        for subsap_idx, subsap in enumerate(all_subsaps):
            hist, bins, tail_error = compute_tail_estimate(
                std,
                subsap.cpu(),
                alpha,
                x,
                pdf,
                dd
            )
            subsample_bins.append(HistOutput(hist, bins))
            # error = error_metric(tail_estimate, analytical_tail)
            error = torch.tensor(tail_error)
            errors.append(error)
        all_bins.append(subsample_bins)
        errors_tensor = torch.stack(errors)
        quantile = errors_tensor.quantile(
            torch.tensor([0.05, 0.5, 0.95], dtype=errors_tensor.dtype)
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

def plot_bin_comparison_estimates(
    x: torch.Tensor,
    pdf: torch.Tensor,
    abscissa: torch.Tensor,
    ode_lk_subsample: torch.Tensor,
    tail_estimate: float,
    error: torch.Tensor,
    equiv_saps: int,
    num_bins: int,
):
    plt.plot(x, pdf, color='blue', label='analytical')
    plt.scatter(abscissa, ode_lk_subsample.cpu(), color='red', label='pfode')
    plt.xlabel('Radius')
    plt.ylabel('Density')
    plt.title(f'ICOV abscissa with estimate: '
        f'{round(tail_estimate, 2)} and error: '
        f'{round(error.item(), 2)}\n'
        f'ESS: {equiv_saps}')
    plt.legend()
    plt.savefig('{}/bin_comparison_density_estimates_{}'.format(
        HydraConfig.get().run.dir,
        num_bins
    ))
    plt.clf()

def plot_line(ax, r: float, alpha: float, dt: torch.Tensor):
    x_vals = torch.linspace(-r*1.2, r*1.2, 200)
    C = alpha / dt
    y_vals = C - x_vals
    ax.plot(x_vals, y_vals, color='red')

# Vertical Line x = alpha/dt
def plot_vertical_line(ax, r: float, alpha: float, dt: torch.Tensor):
    x_val = (alpha / dt).item()
    y_vals = torch.linspace(-r*1.2, r*1.2, 200)
    ax.plot([x_val]*len(y_vals), y_vals, color='red')
    
def generate_color_gradient(N, cmap_name='viridis'):
    cmap = plt.get_cmap(cmap_name)  # e.g., 'viridis', 'plasma', 'inferno', etc.
    colors = [cmap(i / (N - 1)) for i in range(N)]
    return colors

def generate_circle(r):
    thetas = torch.linspace(0, 2*torch.pi, 200)
    x = r * thetas.cos()
    y = r * thetas.sin()
    return x, y

def plot_circle(r, color):
    x, y = generate_circle(r)
    plt.plot(x, y, c=color)

def plot_circle_from_prob(r, prob_min, prob_denom, bins, colors, normalizing_factor):
    x, y = generate_circle(r)
    fake_data = torch.stack([x[0], y[0]])
    dist = torch.distributions.MultivariateNormal(torch.zeros(2), torch.eye(2))
    uncond_density = dist.log_prob(fake_data).exp()
    cond_density = uncond_density / normalizing_factor
    normalized_prob = (cond_density - prob_min) / prob_denom
    color_idx = (normalized_prob >= bins).sum() - 1
    color_idx = min(color_idx, len(colors)-1)
    color = colors[color_idx]
    plt.plot(x, y, c=color)

def plot_fake_trajs(fake_trajs, num_trajs, alpha, prob, subtitle, normalizing_factor):
    # Convert to NumPy for matplotlib/scipy
    x_np = fake_trajs[:, 0].numpy()
    y_np = fake_trajs[:, 1].numpy()
    z_np = prob.cpu().numpy()
    
    # 1. Create a regular grid to interpolate onto
    xi = np.linspace(x_np.min(), x_np.max(), 100)
    yi = np.linspace(y_np.min(), y_np.max(), 100)
    print('making meshgrid')
    X, Y = np.meshgrid(xi, yi)
    
    # 2. Interpolate the scattered data onto the grid
    # Z = griddata((x_np, y_np), z_np, (X, Y), method='cubic')
    
    # 3. Plot
    plt.figure(figsize=(6, 5))
    # cp = plt.contourf(X, Y, Z, levels=50, cmap='viridis')
    # plt.colorbar(cp)
    num_colors = fake_trajs.shape[0] // num_trajs
    print('generating color gradient')
    colors = generate_color_gradient(num_colors, cmap_name='plasma')
    prob_denom = prob.max() - prob.min()
    normalized_prob = (prob - prob.min()) / prob_denom
    bins = torch.linspace(0., 1., num_colors+1)
    bins[-1] += 1e-1
    print('einops')
    normalized_prob_repeat = einops.repeat(normalized_prob, 'n -> n d', d=bins.shape[0])
    bins_repeat = einops.repeat(bins, 'd -> n d', n=normalized_prob.shape[0])
    bin_idx = (normalized_prob_repeat >= bins_repeat).sum(dim=1) - 1
    print('scattering')
    plt.scatter(fake_trajs[:, 0], fake_trajs[:, 1], c=torch.tensor(colors)[bin_idx])
    print('plotting circles')
    r = alpha
    plot_circle_from_prob(r, prob.min(), prob_denom, bins, colors, normalizing_factor)
    r = alpha / torch.tensor(1/2).sqrt()
    plot_circle_from_prob(r, prob.min(), prob_denom, bins, colors, normalizing_factor)
    r = torch.tensor(5.).sqrt() * alpha / torch.tensor(1/2).sqrt()
    plot_circle_from_prob(r, prob.min(), prob_denom, bins, colors, normalizing_factor)
    ax = plt.gca()
    rs = fake_trajs.norm(dim=1)
    r = rs.max()
    dt = torch.tensor(1 / 2).sqrt()
    print('plotting lines')
    plot_line(ax, r, alpha, dt)
    plot_line(ax, r, -alpha, dt)
    plot_vertical_line(ax, r, alpha, dt)
    plot_vertical_line(ax, r, -alpha, dt)
    plt.ylim((-4, 4))
    plt.xlim((-4, 4))
    plt.title("Contour Plot")
    plt.xlabel("x")
    plt.ylabel("y")
    ax.set_aspect('equal')
    plt.savefig('{}/fake_trajs_with_{}.pdf'.format(
        HydraConfig.get().run.dir,
        subtitle
    ))

def plot_fake_trajs_with_analytical(fake_trajs, num_trajs, alpha, normalizing_factor):
    dim = fake_trajs.shape[1]
    prob = torch.distributions.MultivariateNormal(
        torch.zeros(dim),
        torch.eye(dim)
    ).log_prob(fake_trajs).exp()
    plot_fake_trajs(fake_trajs, num_trajs, alpha, prob, 'analytical', normalizing_factor)

def plot_fake_trajs_with_pfode(fake_trajs, num_trajs, alpha, pfode, normalizing_factor):
    plot_fake_trajs(fake_trajs, num_trajs, alpha, pfode.cpu(), 'pfode', normalizing_factor)

def plot_histogram_pdf_approximation(
    subsaps: torch.Tensor,
    num_sample_batches: int,
    num_bins: int,
    abscissa: torch.Tensor,
    analytical_pdf: torch.Tensor,
):
    plt.clf()
    hists = []
    bins = torch.linspace(abscissa[0], subsaps.max(), num_bins+1)
    total_subsaps = len(subsaps)
    for i, subsap in enumerate(subsaps):
        print('generating histogram {i} of {total_subsaps}')
        hist, _ = torch.histogram(
            subsap,
            bins=bins,
            density=True
        )
        hists.append(hist)
    hists_tensor = torch.stack(hists)
    quantiles = torch.quantile(
        hists_tensor, 
        torch.tensor([0.05, 0.5, 0.95], 
                     device=hists_tensor.device, 
                     dtype=hists_tensor.dtype),
        dim=0,
    )
    midpoints = (bins[:-1] + bins[1:]) / 2
    plt.scatter(midpoints, quantiles[1], color='green', label='Median')
    plt.scatter(midpoints, hists_tensor.mean(dim=0), color='yellow', label='Mean')
    plt.fill_between(
        midpoints,
        quantiles[0],
        quantiles[2],
        color='blue',
        alpha=0.3,
        label='5%-95% Quantile'
    )
    plt.plot(abscissa, analytical_pdf, color='red', label='Analytical')
    plt.xlabel('Radius')
    plt.ylabel('Density (Histogram)')
    plt.legend()
    plt.savefig('{}/histogram_pdf_approximation.pdf'.format(
        HydraConfig.get().run.dir,
    ))
    plt.clf()
    return plt.gca().get_ylim()

def plot_pfode_pdf_approximation(
    abscissa_repeat: torch.Tensor,
    transformed_ode: torch.Tensor,
    num_sample_batches: int,
    analytical_pdf: List[torch.Tensor],
    ylim: Tuple=None,
    xlim: Tuple=None,
):
    plt.clf()
    flattened_abscissa = abscissa_repeat.flatten()
    for i in range(flattened_abscissa.shape[0] // num_sample_batches):
        idx = int(i * num_sample_batches)
        radius = flattened_abscissa[idx]
        pdfs = transformed_ode[idx:idx + num_sample_batches]
        quantiles = torch.quantile(
            pdfs, 
            torch.tensor([0.05, 0.5, 0.95], 
                         device=pdfs.device, 
                         dtype=pdfs.dtype)
        )
        mean = pdfs.mean()
        plt.scatter(radius, quantiles[1], color='green', label='Median')
        plt.scatter(radius, mean, color='yellow', label='Mean')
        plt.fill_between(
            radius,
            quantiles[:1],
            quantiles[2:],
            color='blue',
            alpha=0.3,
            label='5%-95% Quantile'
        )
    plt.plot(abscissa_repeat[:, 0], analytical_pdf, color='red', label='Analytical')
    plt.xlabel('Radius')
    plt.ylabel('Density (PFODE)')
    ax = plt.gca()
    handles, labels = ax.get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    ax.legend(*zip(*unique))
    plt.ylim(ylim)
    plt.xlim(xlim)
    plt.grid(axis='y')
    plt.savefig('{}/pfode_pdf_approximation.pdf'.format(
        HydraConfig.get().run.dir,
    ))
    plt.clf()

def compute_fake_gaussian_trajs(
    abscissa: torch.Tensor,
    num_sample_batches: int,
    dim: int
):
    vecs = torch.randn(1, num_sample_batches, dim, 1)
    normed_vecs = vecs / vecs.norm(dim=2, keepdim=True)
    abscissa_repeat = einops.repeat(
        abscissa, 
        'n 1 -> n b d 1',
        b=num_sample_batches,
        d=dim
    )
    fake_trajs = abscissa_repeat.cpu() * normed_vecs
    flattened_fake_trajs = einops.rearrange(fake_trajs, 'a n d 1 -> (a n) d 1')
    return flattened_fake_trajs
    
def compute_pfode_error_vs_bins(
        trajs: torch.Tensor,
        all_bins: List,
        alpha: float,
        std: ContinuousEvaluator,
        cfg: SampleConfig,
        error_metric: ErrorMetric,
) -> ErrorData:
    norm_trajs = trajs.norm(dim=[2, 3]).cpu()
    IQR = scipy.stats.iqr(norm_trajs)
    max_sample = norm_trajs.max()
    xlim = (alpha, math.ceil(max_sample))

    normalizing_factor = get_target(std).analytical_prob(torch.tensor(alpha))
    if type(std.example) == MultivariateGaussianExampleConfig:
        dim = cfg.example.d
        dd = scipy.stats.chi(dim)
        leftover = (1-dd.cdf(max_sample)) / normalizing_factor
        analytical_tail = float((1-dd.cdf(alpha+cfg.eta)) / normalizing_factor - leftover)
    elif type(std.example) == BrownianMotionDiffExampleConfig:
        dim = cfg.example.sde_steps
        dt = torch.tensor(1/(dim-1))
        target = get_target(cfg)
        bm_cdf = target.bm_cdf(dim)
        bm_cdf_keys = list(bm_cdf.keys())
        idx = int((torch.tensor(bm_cdf_keys) < max_sample).sum() - 1)
        alpha_key = bm_cdf_keys[idx]
        approx_leftover = bm_cdf[alpha_key]
        analytical_tail = float(1 - approx_leftover)
        dd = None
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
        abscissa = torch.linspace(alpha+cfg.eta, max_sample, num_bins+1)
        abscissas.append(abscissa)
    abscissa_tensor = torch.cat(abscissas).reshape(-1, 1).to(device)
    abscissa_repeat = einops.repeat(abscissa_tensor, 'd 1 -> d n', n=cfg.num_sample_batches)
    if type(std.example) == MultivariateGaussianExampleConfig:
        fake_trajs = compute_fake_gaussian_trajs(
            abscissa_tensor,
            cfg.num_sample_batches,
            dim
        )
    elif type(std.example) == BrownianMotionDiffExampleConfig:
        fake_trajs = compute_fake_bm_trajs(
            abscissa_tensor, 
            dim,
            alpha,
            dt,
            num_trajs=cfg.num_sample_batches//2
        )
    else:
        raise NotImplementedError
    flattened_trajs = fake_trajs.squeeze()
    torch.save(flattened_trajs.cpu(), f'{HydraConfig.get().run.dir}/flattened_trajs.pt')
    # plot_fake_trajs_with_analytical(
    #     flattened_trajs,
    #     cfg.num_sample_batches,
    #     alpha,
    #     normalizing_factor
    # )
    x = abscissas[-1]
    if type(std.example) == MultivariateGaussianExampleConfig:
        pdf = dd.pdf(x) / normalizing_factor
    elif type(std.example) == BrownianMotionDiffExampleConfig:
        pdf = get_2d_pdf(cfg.example.sde_steps, x, alpha)
    else:
        raise NotImplementedError
    bin_width = 2. * IQR * norm_trajs.shape[1] ** (-1/3)  # Freedman-Diaconis
    num_bins = int((norm_trajs.max() - alpha) / bin_width)
    print('plotting histogram pdf')
    ylim = (0, max(pdf)+0.1)
    # ylim = plot_histogram_pdf_approximation(
    #     norm_trajs,
    #     cfg.num_sample_batches,
    #     num_bins,
    #     abscissa,
    #     pdf,
    # )
    ode_llk = std.ode_log_likelihood(
        fake_trajs.to(device),
        cond=torch.tensor([1.]),
        alpha=torch.tensor([alpha]),
        exact=cfg.compute_exact_trace,
    )
    torch.save(ode_llk[0][-1].cpu(), f'{HydraConfig.get().run.dir}/ode_llk.pt')
    plot_fake_trajs_with_pfode(
        flattened_trajs,
        cfg.num_sample_batches,
        alpha,
        ode_llk[0][-1].exp(),
        normalizing_factor
    )

    if cfg.density_integrator == Integrator.EULER:
        small_idx = torch.topk(fake_trajs.norm(dim=-2).squeeze(), k=7, largest=False).indices
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

    if type(std.example) == MultivariateGaussianExampleConfig:
        transformed_ode_llk = ode_llk[0][-1] + (dim / 2) * torch.tensor(2 * torch.pi).log() + \
            (dim - 1) * abscissa_repeat.flatten().squeeze().log() - (dim / 2 - 1) * \
            torch.tensor(2.).log() - scipy.special.loggamma(dim / 2)
        transformed_ode = transformed_ode_llk.exp().cpu()
        if cfg.test == TestType.MultivariateGaussian:
            transformed_ode /= normalizing_factor
    elif type(std.example) == BrownianMotionDiffExampleConfig:
        transformed_ode = compute_transformed_ode(
            abscissa_repeat.flatten(),
            ode_llk[0][-1],
            alpha=alpha,
            dt=dt
        ).cpu()
        if cfg.test == TestType.BrownianMotionDiff:
            bm_cdf = target.bm_cdf(dim)
            cdf = bm_cdf[alpha]
            normalizing_constant = torch.tensor(
                1-cdf,
                device=transformed_ode.device
            )
            transformed_ode /= normalizing_constant
    errors = []
    augmented_all_num_bins = torch.cat([torch.tensor([0]), bin_sizes+1])
    augmented_cumsum = augmented_all_num_bins.cumsum(dim=0)
    plot_pfode_pdf_approximation(
        abscissa_repeat[-len(x):].cpu(),
        transformed_ode[-len(x)*cfg.num_sample_batches:],
        cfg.num_sample_batches,
        pdf,
        ylim,
        xlim,
    )
    # import pdb; pdb.set_trace()
    torch.save(abscissa_repeat.flatten(), f'{HydraConfig.get().run.dir}/abscissa_repeat.pt')
    torch.save(transformed_ode, f'{HydraConfig.get().run.dir}/transformed_ode.pt')
    equivalents = []
    conf_int_list = []
    nsb = cfg.num_sample_batches
    for i, num_bins in enumerate(bin_sizes):
        bin_width = (max_sample - alpha) / num_bins
        equiv_saps = (bin_width / (2 * IQR)) ** -3
        equivalents.append(equiv_saps)

        abscissa = abscissas[i]
        abscissa_count = len(abscissa)
        idx = augmented_cumsum[i] * nsb
        ode_lk_subsample_list = []
        for k in range(nsb):
            ode_lk_subsample = transformed_ode[idx+k:idx+k+nsb*abscissa_count:nsb]
            ode_lk_subsample_list.append(ode_lk_subsample)
        abscissa_interp = interp(abscissa)
        # if type(std.example) == BrownianMotionDiffExampleConfig:
        #     plot_ode_error(
        #         ode_lk_subsample, 
        #         abscissa,
        #         abscissa_interp,
        #         alpha,
        #         num_bins
        #     )
        local_errors = []
        worst_error = 0
        worst_error_idx = 0
        worst_tail_estimate = 0
        for j in range(nsb):
            approx = ode_lk_subsample_list[j].cpu().numpy()
            # mask = abscissa < (alpha + 0.)
            # approx[mask] = dd.pdf(abscissa[mask])/(1-dd.cdf(alpha))
            if dd is None:
                # Brownian motion case
                pdf = get_2d_pdf(std.example.sde_steps, abscissa, alpha)
                tail_error = scipy.integrate.simpson(
                    np.abs(approx - pdf),
                    x=abscissa
                )
            else:
                # scipy.integrate.trapezoid(hist[smallest_idx:], med_bins[smallest_idx:])
                tail_error = scipy.integrate.simpson(
                    np.abs(approx - dd.pdf(abscissa)/(1-dd.cdf(alpha))),
                    x=abscissa
                )
            # error = error_metric(tail_estimate, analytical_tail)
            error = torch.tensor(tail_error)
            if error > worst_error:
                worst_error = error
                worst_error_idx = j
                worst_tail_estimate = tail_error
            local_errors.append(error)
        local_errors_tensor = torch.stack(local_errors)
        quantiles = torch.quantile(local_errors_tensor, torch.tensor([0.05, 0.5, 0.95], 
                                      device=local_errors_tensor.device, 
                                      dtype=local_errors_tensor.dtype))
        errors.append(quantiles[1])
        conf_int_list.append(quantiles[[0, 2]])
        try:
            plot_bin_comparison_estimates(
                x,
                pdf,
                abscissa,
                ode_lk_subsample_list[worst_error_idx],
                worst_tail_estimate,
                worst_error,
                equiv_saps,
                num_bins
            )
        except Exception as e:
            print(f'failed bin comparison: {e}')
            pass
    median_tensor = torch.stack(errors)
    conf_int_tensor = torch.stack(conf_int_list)

    error_data = ErrorData(
        bin_sizes,
        equivalents,
        median_tensor,
        conf_int_tensor.T,
        'Density Approximation',
        'orange'
    )
    return error_data, ylim, xlim

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
        alpha: float,
        cfg: SampleConfig,
):
    title = f'Absolute Error of Tail Integral vs. Sample Size\n(alpha={alpha}, eta={cfg.eta})'
    plot_errors(sample_error_data, title)
    plot_errors(pfode_error_data, title)
    plt.xlabel('Sample Size')
    plt.ylabel('Absolute Error')
    plt.title(title)
    plt.xscale('log')

    # add ICOV bin axis
    ax = plt.gca()
    ax_top = ax.secondary_xaxis('top')
    ticklabels, mask = find_order_of_magnitude_subtensor(pfode_error_data.bins)
    ax_top.set_xticks(torch.stack(pfode_error_data.samples)[mask])
    ax_top.set_xticklabels(ticklabels)
    ax_top.set_xlabel('Number of Bins')

    plt.legend()

    ymax = min(sample_error_data.error_bars[1].max(), 0.2)
    # ymax = 0.2
    plt.ylim((0., ymax))
    # plt.yscale('log')

    _, run_type = get_run_type(cfg)
    run_type = run_type.replace(' ', '_')
    plt.savefig('{}/{}_{}_tail_integral_error_vs_sample_size.pdf'.format(
        HydraConfig.get().run.dir,
        run_type,
        alpha
    ))

def make_plots(
        trajs: torch.Tensor,
        alpha: float,
        cfg: SampleConfig,
        std: ContinuousEvaluator,
        error_metric: ErrorMetric,
):
    plt.clf()

    subsample_sizes = torch.logspace(
        math.log10(500),
        math.log10(trajs.shape[1]),
        10,
        dtype=int
    )

    all_bins, hist_error_vs_samples, pfode_error_vs_samples = make_error_vs_samples_plot(
        trajs,
        alpha,
        cfg,
        subsample_sizes,
        std,
        error_metric,
    )

def make_error_vs_samples_plot(
        trajs: torch.Tensor,
        alpha: float,
        cfg: SampleConfig,
        subsample_sizes: torch.Tensor,
        std: ContinuousEvaluator,
        error_metric: ErrorMetric
):
    hist_error_vs_samples, all_bins = compute_sample_error_vs_samples(
        trajs,
        alpha,
        subsample_sizes,
        std,
        cfg,
        error_metric,
    )
    pfode_error_vs_samples, ylim, xlim = compute_pfode_error_vs_bins(
        trajs,
        all_bins,
        alpha,
        std,
        cfg,
        error_metric,
    )
    make_error_vs_samples(
        hist_error_vs_samples,
        pfode_error_vs_samples,
        alpha,
        cfg,
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

    torch.manual_seed(cfg.random_seed)

    omega_sampler = OmegaConf.to_object(cfg.sampler)
    if isinstance(omega_sampler, ContinuousSamplerConfig):
        std = ContinuousEvaluator(cfg=cfg)
    else:
        raise NotImplementedError

    with torch.no_grad():
        alpha = std.likelihood.alpha.reshape(-1, 1)
        sample_traj_out = std.sample_trajectories(
            cond=torch.tensor([cfg.cond]),
            alpha=alpha
        )
        sample_trajs = sample_traj_out.samples
        trajs = sample_trajs[-1]

        # plot pfode trajectories for 7 trajectories closest to the boundary
        small_idx = torch.topk(trajs.norm(dim=-2).squeeze(), k=7, largest=False).indices
        traj_subset = sample_trajs[:, small_idx, :, 0].to('cpu')
        derivatives, times = compute_derivatives(std, traj_subset)
        plot_pfode(traj_subset, derivatives, times, 'subset')



        samples = sample_trajs.to('cpu')[:, small_idx.cpu()].squeeze()

        fig, ax = plt.subplots()

        # Set up plot limits based on data range
        ymin, ymax = samples[..., 1].min(), samples[..., 1].max()
        xmin, xmax = samples[..., 0].min(), samples[..., 0].max()
        margin = 0.1 * max(xmax - xmin, ymax - ymin)
        ax.set_xlim(xmin - margin, xmax + margin)
        ax.set_ylim(ymin - margin, ymax + margin)
        cfg_obj = OmegaConf.to_object(cfg)
        plot_boundary(std, cfg_obj, ax)

        # Create color normalization based on likelihood values
        norm = colors.Normalize(vmin=0, vmax=1)

        # Initialize scatter plot
        cmap = plt.get_cmap('hsv')
        scat = ax.scatter([], [], c=[], cmap=cmap, norm=norm, s=1)
        clr = torch.zeros_like(small_idx.cpu())
        fig.colorbar(scat, label='Meaningless')

        def update(frame):
            # Update positions and colors for current frame
            scat.set_offsets(samples[frame, :, :2])
            scat.set_array(clr)
            return scat,

        # Create animation
        frames = len(samples)
        anim = animation.FuncAnimation(
            fig,
            update,
            frames=frames,
            interval=100,  # 100ms between frames
            blit=True
        )

        # Save animation
        cond = 'conditional' if std.cond else 'unconditional'
        anim.save(f'{HydraConfig.get().run.dir}/{cond}_large_derivative.gif', writer='pillow')
        plt.savefig(f'{HydraConfig.get().run.dir}/{cond}_large_derivative_final_frame.pdf'.format(
            HydraConfig.get().run.dir,
        ))
        plt.close()



        cfg_obj = OmegaConf.to_object(cfg)
        rearranged_trajs = einops.rearrange(
            trajs,
            '(b c) h w -> b c h w',
            b=cfg.num_sample_batches
        )
        alpha_float = alpha.cpu().item()
        error_metric = get_error_metric(std.cfg.error_metric)
        make_plots(
            rearranged_trajs,
            alpha_float,
            cfg_obj,
            std,
            error_metric,
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
