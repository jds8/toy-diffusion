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
import numpy as np
from scipy.interpolate import griddata

from toy_configs import register_configs
from toy_sample import ContinuousEvaluator, compute_transformed_ode, compute_perimeter
from toy_train_config import SampleConfig, get_run_type, MultivariateGaussianExampleConfig, \
    BrownianMotionDiffExampleConfig, get_target, get_error_metric, ErrorMetric
from models.toy_diffusion_models_config import ContinuousSamplerConfig
from compute_quadratures import pdf_2d_quadrature_bm, pdf_3d_quadrature_bm


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
    plt.scatter(abscissa[:-1], errors, color='red', label='PFODE (Trapezoid) - Analytical')
    plt.xlabel('Radius')
    plt.ylabel('Error')
    plt.title(f'Error between PFODE and Analytical at each Radius')
    plt.legend()
    plt.savefig('{}/pfode_error_{}'.format(
        HydraConfig.get().run.dir,
        num_bins
    ))
    plt.clf()

def compute_tail_estimate(
        std: ContinuousEvaluator,
        subsap: torch.Tensor,
        alpha: float
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
    # tail_estimate = scipy.integrate.trapezoid(hist[smallest_idx:], med_bins[smallest_idx:])
    tail_estimate = scipy.integrate.simpson(
        hist[smallest_idx:],
        x=med_bins[smallest_idx:]
    )
    print(f'histogram estimate: {tail_estimate}')
    return hist, bins, tail_estimate

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
        errors = []
        for subsap_idx, subsap in enumerate(all_subsaps):
            hist, bins, tail_estimate = compute_tail_estimate(
                std,
                subsap.cpu(),
                alpha
            )
            subsample_bins.append(HistOutput(hist, bins))
            error = error_metric(tail_estimate, analytical_tail)
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

def get_points_along_angle(
    angles: torch.Tensor,
    angle_points: torch.Tensor,
    idx: int,
    r: torch.Tensor,
    num_trajs: int,
    dim: int,
    alpha: float,
):
    angle = angles[idx]
    if angle > 0:
        angle_points = angle_points[idx]
        dangles = torch.linspace(0, angle, num_trajs)
        x,y = angle_points[:, 0], angle_points[:, 1]
        theta = torch.atan2(y, x)
        complex_points = r * torch.exp(torch.complex(
            torch.tensor(0.), 
            theta[~idx] + dangles
            ))
        points = torch.stack([torch.tensor([point.real, point.imag]) for point in complex_points])
        dt = torch.tensor(1 / 2).sqrt()
        if (points.cumsum(dim=1).abs() < alpha/dt-1e-3).all(dim=1).any():
            plt.scatter(points[:, 0], points[:, 1])
            r1 = alpha
            plot_circle(r1, 'r')
            ax = plt.gca()
            plot_line(ax, alpha, alpha, dt)
            plot_line(ax, alpha, -alpha, dt)
            plot_vertical_line(ax, alpha, alpha, dt)
            plot_vertical_line(ax, alpha, -alpha, dt)
            ax.set_xlim((-3, 3))
            ax.set_ylim((-3, 3))
            plt.savefig('{}/trial.pdf'.format(
                HydraConfig.get().run.dir
            ))
            import pdb; pdb.set_trace()
        return points
    return (torch.ones(num_trajs, 2) * r**2 / (dim-1)).sqrt()
    
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
    plt.title(f'PFODE abscissa with estimate: '
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
    X, Y = np.meshgrid(xi, yi)
    
    # 2. Interpolate the scattered data onto the grid
    Z = griddata((x_np, y_np), z_np, (X, Y), method='cubic')
    
    # 3. Plot
    plt.figure(figsize=(6, 5))
    # cp = plt.contourf(X, Y, Z, levels=50, cmap='viridis')
    # plt.colorbar(cp)
    num_colors = fake_trajs.shape[0] // num_trajs
    colors = generate_color_gradient(num_colors, cmap_name='plasma')
    prob_denom = prob.max() - prob.min()
    normalized_prob = (prob - prob.min()) / prob_denom
    bins = torch.linspace(0., 1., num_colors+1)
    bins[-1] += 1e-1
    normalized_prob_repeat = einops.repeat(normalized_prob, 'n -> n d', d=bins.shape[0])
    bins_repeat = einops.repeat(bins, 'd -> n d', n=normalized_prob.shape[0])
    bin_idx = (normalized_prob_repeat >= bins_repeat).sum(dim=1) - 1
    for color_idx, traj in zip(bin_idx, fake_trajs):
        plt.scatter(traj[0], traj[1], color=colors[color_idx])
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
    prob = torch.distributions.MultivariateNormal(torch.zeros(2), torch.eye(2)).log_prob(fake_trajs).exp()
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
    for subsap in subsaps:
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
    ylim,
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
    plt.savefig('{}/pfode_pdf_approximation.pdf'.format(
        HydraConfig.get().run.dir,
    ))
    plt.clf()

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
    abscissa_repeat = einops.repeat(abscissa_tensor, 'd 1 -> d n', n=cfg.num_sample_batches)
    if type(std.example) == MultivariateGaussianExampleConfig:
        dangles = torch.linspace(0, 2*torch.pi, cfg.num_sample_batches)
        dangles_repeat = einops.repeat(dangles, 'd -> n d', n=abscissa_tensor.shape[0])
        complex_fake_trajs = abscissa_repeat * torch.exp(torch.complex(torch.tensor(0.), dangles_repeat))
        fake_trajs = torch.stack([torch.tensor([x.real, x.imag]) for x in complex_fake_trajs.flatten()])
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
    normalizing_factor = get_target(std).analytical_prob(alpha)
    plot_fake_trajs_with_analytical(
        flattened_trajs,
        cfg.num_sample_batches,
        alpha,
        normalizing_factor
    )
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
    bin_width = 2. * IQR * norm_trajs.shape[1] ** (-1/3)  # Freedman-Diaconis
    num_bins = int((norm_trajs.max() - alpha) / bin_width)
    ylim = plot_histogram_pdf_approximation(
        norm_trajs,
        cfg.num_sample_batches,
        num_bins,
        abscissa,
        pdf,
    )
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
    if type(std.example) == MultivariateGaussianExampleConfig:
        transformed_ode_llk = ode_llk[0][-1] + (dim / 2) * torch.tensor(2 * torch.pi).log() + \
            (dim - 1) * abscissa_tensor.squeeze().log() - (dim / 2 - 1) * \
            torch.tensor(2.).log() - scipy.special.loggamma(dim / 2)
        transformed_ode = transformed_ode_llk.exp()
    elif type(std.example) == BrownianMotionDiffExampleConfig:
        transformed_ode = compute_transformed_ode(
            abscissa_repeat.flatten(),
            ode_llk[0][-1],
            alpha=alpha,
            dt=dt
        )
    errors = []
    augmented_all_num_bins = torch.cat([torch.tensor([0]), bin_sizes+1])
    augmented_cumsum = augmented_all_num_bins.cumsum(dim=0)
    plot_pfode_pdf_approximation(
        abscissa_repeat[-len(x):].cpu(),
        transformed_ode[-len(x)*cfg.num_sample_batches:].cpu(),
        cfg.num_sample_batches,
        pdf,
        ylim,
    )
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
            abscissa_repeat.flatten()[idx+k:idx+k+nsb*abscissa_count:nsb]
            ode_lk_subsample_list.append(ode_lk_subsample)
        abscissa_interp = interp(abscissa)
        plot_ode_error(
            ode_lk_subsample, 
            abscissa,
            abscissa_interp,
            alpha,
            num_bins
        )
        # tail_estimate = scipy.integrate.trapezoid(ode_lk_subsample.cpu(), abscissa)
        local_errors = []
        for j in range(nsb):
            tail_estimate = scipy.integrate.simpson(
                ode_lk_subsample_list[j].cpu(),
                x=abscissa
            )
            error = error_metric(tail_estimate, analytical_tail)
            local_errors.append(error)
        print(f'pfode estimate: {tail_estimate}')
        print(f'error: {error}')
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
                ode_lk_subsample_list[0],
                tail_estimate,
                error,
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
    try:
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
    except:
        import pdb; pdb.set_trace()

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
        error_metric: ErrorMetric,
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
        error_metric,
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
    pfode_error_vs_samples = compute_pfode_error_vs_bins(
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
