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

import pandas as pd
import hydra
from hydra.core.config_store import ConfigStore
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
import torch
import scipy
import matplotlib.pyplot as plt

from toy_configs import register_configs
from toy_sample import ContinuousEvaluator, compute_perimeter
from toy_train_config import BinComparisonConfig, get_run_type, MultivariateGaussianExampleConfig, \
    BrownianMotionDiffExampleConfig, get_target
from models.toy_diffusion_models_config import ContinuousSamplerConfig
from compute_quadratures import pdf_2d_quadrature_bm, pdf_3d_quadrature_bm


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

ErrorData = namedtuple('ErrorData', 'x median error_bars label color')
HistOutput = namedtuple('HistOutput', 'hist bins')

#########################
#########################
def suppresswarning():
    warnings.warn("user", UserWarning)

def get_condition_idx(xx, yy, alpha, std):
    if type(std.example) == MultivariateGaussianExampleConfig:
        condition_idx = xx**2 + yy**2 < alpha**2
    elif type(std.example) == BrownianMotionDiffExampleConfig:
        dt = torch.tensor(1/2)
        x1 = xx * torch.sqrt(dt)
        x2 = (xx + yy) * torch.sqrt(dt)
        condition_idx = (torch.abs(x1) < alpha) & (torch.abs(x2) < alpha)
    return condition_idx

@hydra.main(version_base=None, config_path="conf", config_name="continuous_is_config")
def sample(cfg):
    logger = logging.getLogger("main")
    logger.info('run type: bin_comparison')
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
        x_steps = 50
        y_steps = 50
        max_val = torch.tensor(5.) / torch.tensor(1/2) + 0.5
        x = torch.linspace(-max_val, max_val, steps=x_steps)
        y = torch.linspace(-max_val, max_val, steps=y_steps)
        xx, yy = torch.meshgrid(x, y, indexing='xy')
        fake_traj = torch.stack([xx.reshape(-1), yy.reshape(-1)]).T.unsqueeze(-1).to(device)

        # compute analytical likelihood
        normal = torch.distributions.Normal(0., 1.)
        analytical = (normal.log_prob(xx) + normal.log_prob(yy)).exp()

        # compute approximate likelihood
        alpha = std.likelihood.alpha.reshape(-1, 1)
        ode_llk = std.ode_log_likelihood(
            fake_traj,
            cond=torch.tensor([cfg.cond]),
            alpha=alpha,
            exact=cfg.compute_exact_trace,
        )
        approx = einops.rearrange(ode_llk[0][-1].exp().cpu(), '(i j) -> i j', i=x_steps)

        # compute error
        # rel_error = torch.abs(approx - analytical)
        rel_error = approx
        condition_idx = get_condition_idx(xx, yy, alpha, std)
        rel_error[condition_idx] = torch.nan

        # plot error
        # plt.figure(figsize=(6, 5))
        plt.pcolormesh(xx, yy, rel_error, shading='auto', cmap='viridis')
        plt.colorbar(label='Error')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Heatmap')
        plt.tight_layout()
        plt.savefig('{}/{}_alpha={}_error_heatmap.pdf'.format(
            HydraConfig.get().run.dir,
            cfg.model_name,
            alpha.item(),
        ))


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        suppresswarning()

    cs = ConfigStore.instance()
    cs.store(name="vpsde_sample_config", node=BinComparisonConfig)
    register_configs()

    with torch.no_grad():
        sample()
