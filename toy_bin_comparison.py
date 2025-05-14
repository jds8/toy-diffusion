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
from toy_train_config import BinComparisonConfig, get_run_type, MultivariateGaussianExampleConfig, \
    BrownianMotionDiffExampleConfig
from models.toy_diffusion_models_config import ContinuousSamplerConfig
from compute_quadratures import pdf_2d_quadrature_bm


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
    filename = f'{HydraConfig.get().run.dir}/pfode_bin_comparison_errors.pt'
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
    filename = f'{abs_directory}/pfode_bin_comparison_data_{num_bins}_bins.pt'
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
            math.log10(10),
            math.log10(1000),
            9,
            dtype=int
        )
        abscissas = []
        for num_bins in all_num_bins:
            abscissa = torch.linspace(alpha.item(), max_sample, num_bins+1)
            abscissas.append(abscissa)
        abscissa_tensor = torch.cat(abscissas).reshape(-1, 1).to(device)
        fake_traj = torch.cat([
            torch.zeros(abscissa_tensor.shape[0], dim-1, device=device),
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
        augmented_cumsum = augmented_all_num_bins.cumsum(dim=0)
        x = abscissas[-1]
        if type(std.example) == MultivariateGaussianExampleConfig:
            pdf = dd.pdf(x)
        elif type(std.example) == BrownianMotionDiffExampleConfig:
            pdf = [pdf_2d_quadrature_bm(a.cpu().item(), alpha.item()) for a in x]
        else:
            raise NotImplementedError
        for i in range(len(all_num_bins)):
            abscissa = abscissas[i]
            abscissa_count = len(abscissa)
            idx = augmented_cumsum[i]
            ode_llk_subsample = chi_ode_llk[idx:idx+abscissa_count]
            tail_estimate = scipy.integrate.simpson(
                ode_llk_subsample.cpu().exp(),
                x=abscissa
            )
            rel_error = torch.tensor(tail_estimate - analytical_tail).abs() / analytical_tail
            rel_errors.append(rel_error)
            save_pfode_samples(abscissa, ode_llk_subsample)
            plt.plot(x, pdf, color='blue')
            plt.scatter(abscissa.cpu(), ode_llk_subsample.exp().cpu(), color='red')
            plt.savefig('{}/bin_comparison_density_estimates_{}'.format(
                HydraConfig.get().run.dir,
                i
            ))
            plt.clf()
        import pdb; pdb.set_trace()
        rel_errors_tensor = torch.stack(rel_errors)
        save_pfode_errors(all_num_bins, rel_errors_tensor)
        plt.scatter(all_num_bins, rel_errors_tensor, label='PFODE')
        try:
            histogram_data = torch.load(cfg.histogram_bins_filename)
            histogram_bins = histogram_data['Abscissa_bins']
            histogram_error_medians = histogram_data['Median']
            histogram_5 = histogram_data['5%']
            histogram_95 = histogram_data['95%']
            plt.scatter(
                histogram_bins,
                histogram_error_medians,
                label='Histogram',
                color='blue',
            )
            plt.fill_between(
                histogram_bins,
                histogram_5,
                histogram_95,
                color='blue',
                alpha=0.2
            )
        except:
            pass
        plt.xlabel('Number of Bins')
        plt.ylabel('Relative Error')
        plt.title(f'Relative Error of Tail Integral (alpha={alpha.item()}) vs. Sample Size')
        plt.xscale("log")
        plt.legend()
        cfg_obj = OmegaConf.to_object(cfg)
        _, run_type = get_run_type(cfg_obj)
        run_type = run_type.replace(' ', '_')
        plt.savefig('{}/{}_{}_tail_integral_bin_comparison.pdf'.format(
            HydraConfig.get().run.dir,
            run_type,
            alpha.item()
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
