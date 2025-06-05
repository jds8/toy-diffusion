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
import numpy as np

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

def get_condition_idx(samples, alpha, std):
    if type(std.example) == MultivariateGaussianExampleConfig:
        condition_idx = samples.norm(dim=1) < alpha**2
    elif type(std.example) == BrownianMotionDiffExampleConfig:
        dt = torch.tensor(1/2)
        x = samples * torch.sqrt(dt)
        x_cumsum = x.cumsum(dim=1)
        condition_idx = x_cumsum < alpha
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
        alpha = std.likelihood.alpha.reshape(-1, 1)
        max_val = torch.tensor(5.).sqrt() * alpha.squeeze() / torch.tensor(1/2).sqrt() + .1
        sample_traj_out = std.sample_trajectories(
            cond=std.cond,
            alpha=std.likelihood.alpha.reshape(-1, 1),
        )
        samples = sample_traj_out.samples[-1].cpu().squeeze()

        # compute analytical likelihood
        bin_width = 2. * scipy.stats.iqr(samples) * samples.shape[0] ** (-1/3)
        num_bins = int((samples.max()) / bin_width)
        hist, x_edges, y_edges = np.histogram2d(samples[:, 0], samples[:, 1], bins=num_bins, density=True)
        mid_x = x_edges[:-1] + x_edges[1:]
        mid_y = y_edges[:-1] + y_edges[1:]
        xx, yy = torch.meshgrid(torch.tensor(mid_x), torch.tensor(mid_y), indexing='xy')

        # plot error
        # plt.figure(figsize=(6, 5))
        # plt.pcolormesh(xx, yy, hist, shading='auto', cmap='viridis')
        plt.contourf(xx, yy, hist, cmap='viridis')
        plt.colorbar(label='Density')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Density')
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
