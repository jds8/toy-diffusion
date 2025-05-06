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
from toy_sample import ContinuousEvaluator
from toy_train_config import SampleConfig, get_run_type, MultivariateGaussianExampleConfig, \
    BrownianMotionDiffExampleConfig
from models.toy_diffusion_models_config import ContinuousSamplerConfig


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

ErrorData = namedtuple('ErrorData', 'x median error_bars label color')
HistOutput = namedtuple('HistOutput', 'hist bins')

#########################
#########################
def suppresswarning():
    warnings.warn("user", UserWarning)

def save_pfode_errors(
        all_num_bins: torch.Tensor,
        rel_errors_tensor: torch.Tensor
):
    filename = f'{HydraConfig.get().run.dir}/pfode_bin_comparison_errors.csv'
    torch.save({
        'NumBins': all_num_bins,
        'Errors': rel_errors_tensor
    }, filename)

def save_pfode_samples(
        abscissa: torch.Tensor,
        chi_ode_llk: torch.Tensor
):
    num_bins = abscissa.shape[0]
    rel_directory = 'pfode_bin_comparison_data'
    abs_directory = f'{HydraConfig.get().run.dir}/{rel_directory}'
    os.makedirs(abs_directory, exist_ok=True)
    filename = f'{abs_directory}/pfode_bin_comparison_data_{num_bins}_bins.csv'
    torch.save({
        'Abscissa': abscissa,
        'ChiOdeLlk': chi_ode_llk
    }, filename)

@hydra.main(version_base=None, config_path="conf", config_name="continuous_is_config")
def sample(cfg):
    logger = logging.getLogger("main")
    logger.info('run type: importance sampling')
    cfg_str = OmegaConf.to_yaml(cfg)
    logger.info(f"CONFIG\n{cfg_str}")

    os.system('echo git commit: $(git rev-parse HEAD)')

    omega_sampler = OmegaConf.to_object(cfg.sampler)
    if isinstance(omega_sampler, ContinuousSamplerConfig):
        std = ContinuousEvaluator(cfg=cfg)
    else:
        raise NotImplementedError

    with torch.no_grad():
        alpha = std.likelihood.alpha.reshape(-1, 1)

        if type(std.example) == MultivariateGaussianExampleConfig:
            dim = cfg.example.d
        elif type(std.example) == BrownianMotionDiffExampleConfig:
            dim = cfg.example.sde_steps
        else:
            raise NotImplementedError

        dd = scipy.stats.chi(dim)
        analytical_tail = 1 - dd.cdf(alpha.item())
        max_sample = dd.ppf(0.99999)
        all_num_bins = torch.logspace(
            math.log10(100),
            math.log10(1000),
            5,
            dtype=int
        )
        abscissas = []
        for num_bins in all_num_bins:
            abscissa = torch.linspace(alpha.item(), max_sample, num_bins+1)
            abscissas.append(abscissa)
        abscissa_tensor = torch.cat(abscissas).reshape(-1, 1)
        fake_traj = torch.cat([
            torch.zeros(abscissa_tensor.shape[0], dim-1),
            abscissa_tensor
        ], 1).unsqueeze(-1)
        ode_llk = std.ode_log_likelihood(
            fake_traj,
            cond=torch.tensor([-1.]),
            alpha=alpha,
            exact=cfg.compute_exact_trace,
        )
        chi_ode_llk = ode_llk[0][-1] + (dim / 2) * torch.tensor(2 * torch.pi).log() + \
            (dim - 1) * abscissa_tensor.squeeze().log() - (dim / 2 - 1) * \
            torch.tensor(2.).log() - scipy.special.loggamma(dim / 2)
        rel_errors = []
        augmented_all_num_bins = torch.cat([torch.tensor([0]), all_num_bins+1])
        for i, abscissa_count in enumerate(augmented_all_num_bins[1:]):
            abscissa = abscissas[i]
            idx = augmented_all_num_bins[i]
            ode_llk_subsample = chi_ode_llk[idx:idx+abscissa_count]
            tail_estimate = scipy.integrate.simpson(ode_llk_subsample.exp(), x=abscissa)
            rel_error = torch.tensor(tail_estimate - analytical_tail).abs() / analytical_tail
            rel_errors.append(rel_error)
            save_pfode_samples(abscissa, ode_llk_subsample)
        rel_errors_tensor = torch.stack(rel_errors)
        save_pfode_errors(all_num_bins, rel_errors_tensor)
        plt.plot(all_num_bins, rel_errors_tensor)
        plt.xlabel('Number of Bins')
        plt.ylabel('Relative Error')
        plt.title(f'Relative Error of Tail Integral (alpha={alpha.item()}) vs. Sample Size')
        _, run_type = get_run_type(cfg)
        run_type = run_type.replace(' ', '_')
        plt.savefig('{}/{}_{}_tail_integral_bin_comparison.pdf'.format(
            HydraConfig.get().run.dir,
            run_type,
            alpha
        ))


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        suppresswarning()

    cs = ConfigStore.instance()
    cs.store(name="vpsde_sample_config", node=SampleConfig)
    register_configs()

    with torch.no_grad():
        sample()
