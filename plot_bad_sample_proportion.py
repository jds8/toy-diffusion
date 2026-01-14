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
    compute_fake_bm_trajs, compute_fake_gaussian_trajs
from toy_train_config import SampleConfig, get_run_type, MultivariateGaussianExampleConfig, \
    BrownianMotionDiffExampleConfig, IntegratorComparisonConfig, Integrator
from models.toy_diffusion_models_config import ContinuousSamplerConfig
from compute_quadratures import get_2d_pdf, pdf_2d_quadrature_bm


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#########################
#########################
def suppresswarning():
    warnings.warn("user", UserWarning)

def get_example(
        cfg_obj: SampleConfig,
        model_name: str,
        model_dim_str: str
) -> dictconfig.DictConfig:
    if isinstance(cfg_obj.example, MultivariateGaussianExampleConfig):
        config = f'conf/continuous_multivariate_{model_dim_str}_sample_config.yaml'
        example = OmegaConf.load(config).example
    elif isinstance(cfg_obj.example, BrownianMotionDiffExampleConfig):
        example = cfg_obj.example
    else:
        raise NotImplementedError
    return example

def get_dim_mults(model_dim: int) -> List[int]:
    result = []
    p = 1
    while p <= model_dim:
        result.append(p)
        p *= 2
    return result

def get_diffusion_dim(model: str) -> int:
    m = re.search(r'dim_(\d+)_', model)
    if m:
        return int(m.group(1))
    return None

def get_model_dim(model: str) -> str:
    m = re.search(r'(\d+)Example', model)
    if m:
        return m.group(1)
    return None


@hydra.main(version_base=None, config_path="conf", config_name="continuous_is_config")
def sample(cfg):
    logger = logging.getLogger("main")
    logger.info('run type: plot_bad_sample_proportion')
    cfg_str = OmegaConf.to_yaml(cfg)
    logger.info(f"CONFIG\n{cfg_str}")
    logger.info(f'OUTPUT\n{HydraConfig.get().run.dir}\n')

    os.system('echo git commit: $(git rev-parse HEAD)')

    torch.manual_seed(cfg.random_seed)

    omega_sampler = OmegaConf.to_object(cfg.sampler)

    cfg_obj = OmegaConf.to_object(cfg)
    if isinstance(omega_sampler, ContinuousSamplerConfig):
        if isinstance(cfg_obj.example, MultivariateGaussianExampleConfig):
            models = [
                'VPSDEVelocitySampler_TemporalUnetAlpha_dim_120_MultivariateGaussian2ExampleConfig_v10891111424',
                'VPSDEVelocitySampler_TemporalUnetAlpha_dim_80_MultivariateGaussian4ExampleConfig_v10240002048',
                'VPSDEVelocitySampler_TemporalUnetAlpha_dim_32_MultivariateGaussian8ExampleConfig_v10240001048',
            ]
        elif isinstance(cfg_obj.example, BrownianMotionDiffExampleConfig):
            models = [
                'VPSDEVelocitySampler_TemporalUnetAlpha_dim_120_BrownianMotionDiff3ExampleConfigNoTernary_v4096001792'
            ]
        else:
            raise NotImplementedError
        alphas = torch.arange(0., 4.5, 0.5)
        for model in reversed(models):
            model_dim = get_model_dim(model)
            model_dim_label = model_dim + 'D'
            diffusion_dim = get_diffusion_dim(model)
            dim_mults = get_dim_mults(int(model_dim))
            example = get_example(cfg_obj, model, model_dim)
            model_prop_bad_samples = []
            for next_alpha in alphas:
                new_cfg = deepcopy(cfg)
                new_cfg.likelihood.alpha = next_alpha.item()
                new_cfg.model_name = model
                new_cfg.diffusion.dim = diffusion_dim
                new_cfg.diffusion.dim_mults = dim_mults
                example_target = OmegaConf.structured(type(cfg_obj.example)())
                example_cfg = OmegaConf.merge(example_target, example)
                new_cfg.example = example_cfg
                std = ContinuousEvaluator(new_cfg)
                with torch.no_grad():
                    alpha = std.likelihood.alpha.reshape(-1, 1)
                    sample_traj_out = std.sample_trajectories(
                        cond=torch.tensor([1.]),
                        alpha=alpha
                    )
                    sample_trajs = sample_traj_out.samples
                    trajs = sample_trajs[-1]

                    prop_bad_samples = (trajs.norm(dim=-2).cpu() < alpha).to(float).mean()
                    model_prop_bad_samples.append(prop_bad_samples)
            plt.scatter(alphas, model_prop_bad_samples, label=model_dim_label)
            plt.xlabel('Alpha')
            plt.ylabel(f'Proportion (of {std.cfg.num_samples})')
            plt.title('Proportion of Samples Outside alpha')
            plt.legend()
        plt.savefig('{}/bad_sample_proportions.pdf'.format(
            HydraConfig.get().run.dir,
        ))
    else:
        raise NotImplementedError


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        suppresswarning()

    cs = ConfigStore.instance()
    cs.store(name="vpsde_sample_config", node=SampleConfig)
    register_configs()

    with torch.no_grad():
        sample()
