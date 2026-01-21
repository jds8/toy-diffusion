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
from omegaconf import OmegaConf, dictconfig
import torch
import numpy as np
import scipy
import matplotlib.pyplot as plt

from toy_configs import register_configs
from toy_sample import ContinuousEvaluator, compute_transformed_ode, compute_derivatives, plot_pfode, \
    compute_fake_bm_trajs, compute_fake_gaussian_trajs, line_circle_intersection, vertical_line_circle_intersection
from toy_train_config import SampleConfig, get_run_type, MultivariateGaussianExampleConfig, get_reduction_op, \
    BrownianMotionDiffExampleConfig, IntegratorComparisonConfig, Integrator
from models.toy_diffusion_models_config import ContinuousSamplerConfig
from compute_quadratures import get_2d_pdf, pdf_2d_quadrature_bm


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#########################
#########################
def suppresswarning():
    warnings.warn("user", UserWarning)

def compute_fake_bm_arcs(
        r,
        alpha,
        dt,
        num_angles=8,
):
    dt_sqrt = dt.sqrt()
    output_diagonal = line_circle_intersection(r, alpha, dt_sqrt)
    output_right = vertical_line_circle_intersection(r, alpha, dt_sqrt)
    output = torch.concat([output_diagonal, output_right])
    angle_bounds = torch.atan2(output[:, 1], output[:, 0]).sort().values
    angle_min, angle_max = angle_bounds[0], angle_bounds[-1]
    angles = torch.linspace(angle_min, angle_max, num_angles)
    values = r * (angles * 1j).exp()
    fake_trajsBnD1 = torch.stack([values.real, values.imag], dim=-1).unsqueeze(-1)
    return fake_trajsBnD1


def compute_fake_gaussian_arcs(
        r,
        num_angles=8,
):
    """ Output has shape r.shape[0] * num_angles x dim x 1 """
    # r has shape nx1
    angles = torch.linspace(0, 2*torch.pi, num_angles)
    values = r * (angles * 1j).exp()
    # values has shape n x num_angles x d x 1
    fake_trajsBND1 = torch.stack([values.real, values.imag], dim=-1).unsqueeze(-1)
    fake_trajsBnD1 = einops.rearrange(fake_trajsBND1, 'n num_angles dim 1 -> (n num_angles) dim 1')
    return fake_trajsBnD1


@hydra.main(version_base=None, config_path="conf", config_name="continuous_is_config")
def sample(cfg):
    logger = logging.getLogger("main")
    logger.info('run type: plot_samples')
    cfg_str = OmegaConf.to_yaml(cfg)
    logger.info(f"CONFIG\n{cfg_str}")
    logger.info(f'OUTPUT\n{HydraConfig.get().run.dir}\n')

    os.system('echo git commit: $(git rev-parse HEAD)')

    torch.manual_seed(cfg.random_seed)

    omega_sampler = OmegaConf.to_object(cfg.sampler)

    cfg_obj = OmegaConf.to_object(cfg)
    if isinstance(omega_sampler, ContinuousSamplerConfig):
        std = ContinuousEvaluator(cfg=cfg)
        alpha = std.likelihood.alpha.reshape(-1, 1)
        minimum_radius = cfg.minimum_radius if cfg.minimum_radius >= 0 else alpha
        maximum_radius = cfg.maximum_radius if cfg.maximum_radius >= 0 else alpha+0.2
        r = torch.arange(minimum_radius.item(), maximum_radius.item(), cfg.increment).unsqueeze(-1)
        analytic_radii = torch.arange(r[0].item(), r[-1].item(), cfg.increment/cfg.density_factor)
        if isinstance(cfg_obj.example, MultivariateGaussianExampleConfig):
            dim = cfg.example.d
            dd = scipy.stats.chi(dim)
            fake_trajs = compute_fake_gaussian_arcs(
                r,
                num_angles=cfg.num_icov_samples,
            )
            pdf = dd.pdf(analytic_radii) / (1-dd.cdf(alpha)) * (analytic_radii > alpha).numpy()
            label = 'Analytical'
        elif isinstance(cfg_obj.example, BrownianMotionDiffExampleConfig):
            dim = cfg.example.sde_steps-1
            dt = 1 / torch.tensor(dim)
            fake_trajs = compute_fake_bm_arcs(
                r,
                alpha,
                dt,
                num_angles=cfg.num_icov_samples,
            )
            pdf = get_2d_pdf(cfg.example.sde_steps, analyic_radii.squeeze(), alpha.item()) * (analytic_radii > alpha).numpy()
            label = 'Quadrature'
        else:
            raise NotImplementedError
        old_ode_llk = std.ode_log_likelihood(
            fake_trajs.to(device),
            cond=torch.tensor([1.]),
            alpha=torch.tensor([alpha]),
            exact=cfg.compute_exact_trace,
        )
        reduction_op = get_reduction_op(cfg)
        new_llk = einops.reduce(old_ode_llk[0], 'c (b n) -> c b', reduction_op, n=cfg.num_icov_samples)
        ode_llk = (new_llk, *old_ode_llk[1:])
        if isinstance(cfg_obj.example, MultivariateGaussianExampleConfig):
            transformed_ode_llk = ode_llk[0][-1].to('cpu') + (dim / 2) * torch.tensor(2 * torch.pi).log() + \
                (dim - 1) * r.flatten().squeeze().log() - (dim / 2 - 1) * \
                torch.tensor(2.).log() - scipy.special.loggamma(dim / 2)
            transformed_ode = transformed_ode_llk.exp()
        elif isinstance(cfg_obj.example, BrownianMotionDiffExampleConfig):
            transformed_ode = compute_transformed_ode(
                r.flatten(),
                ode_llk[0][-1],
                alpha=alpha,
                dt=dt
            ).cpu()
        else:
            raise NotImplementedError
        dense_r = analytic_radii.squeeze()
        pdf = pdf.squeeze()
        plt.plot(dense_r, pdf, label=label, color='orange')
        plt.scatter(dense_r[::cfg.density_factor], pdf[::cfg.density_factor], marker='x', color='orange')
        plt.scatter(r, transformed_ode, label='Estimate')
        plt.xlabel('Radius')
        plt.ylabel(f'Density')
        plt.title('Density vs. Radius')
        plt.legend()
        plt.savefig('{}/density.pdf'.format(
            HydraConfig.get().run.dir,
        ))
    else:
        raise NotImplementedError


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        suppresswarning()

    cs = ConfigStore.instance()
    cs.store(name="vpsde_sample_config", node=PlotSamplesConfig)
    register_configs()

    with torch.no_grad():
        sample()
