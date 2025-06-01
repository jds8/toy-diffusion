#!/usr/bin/env python3
import warnings
import os
import logging
from typing import Callable
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
import matplotlib.pyplot as plt
import scipy

from toy_configs import register_configs
from toy_sample import ContinuousEvaluator
from models.toy_diffusion_models_config import ContinuousSamplerConfig
from toy_train_config import ISConfig, get_target, get_proposal, \
    SampleConfig, ErrorMetric, get_error_metric
from importance_sampling import Proposal, Target


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class RoundData:
    def __init__(
            self,
            curr_sap=0,
            target_estimate=0,
            diffusion_estimate=0,
            target_N=0,
            total_num_saps_not_in_region=0
    ):
        self.curr_sap = curr_sap
        self.target_estimate = torch.tensor(
            [target_estimate],
            dtype=torch.float32,
            device=device
        )
        self.diffusion_estimate = torch.tensor(
            [diffusion_estimate],
            dtype=torch.float32,
            device=device
        )
        self.target_N = target_N
        self.total_num_saps_not_in_region = total_num_saps_not_in_region

    @staticmethod
    def get_save_file(save_dir: str, alpha: str, i: int) -> str:
        return f'{save_dir}/alpha={alpha}_round={i}_data'

    def save(self, save_dir: str, alpha: str, i: int):
        data = {
            'curr_sap': self.curr_sap,
            'target_estimate': self.target_estimate,
            'diffusion_estimate': self.diffusion_estimate,
            'target_N': self.target_N,
            'total_num_saps_not_in_region': self.total_num_saps_not_in_region,
        }
        torch.save(data, RoundData.get_save_file(save_dir, alpha, i))

    @staticmethod
    def load(save_dir: str, alpha: str, i: int):
        try:
            data = torch.load(RoundData.get_save_file(save_dir, alpha, i))
            return RoundData(
                curr_sap=data['curr_sap'],
                target_estimate=data['target_estimate'],
                diffusion_estimate=data['diffusion_estimate'],
                target_N=data['target_N'],
                total_num_saps_not_in_region=data['total_num_saps_not_in_region'],
            )
        except Exception:
            return RoundData()

def suppresswarning():
    warnings.warn("user", UserWarning)

def importance_estimate(
        test_fn: Callable,
        saps_raw: torch.Tensor,
        saps: torch.Tensor,
        log_probs: torch.Tensor,
        log_qrobs: torch.Tensor
):
    log_ws = log_probs - log_qrobs
    max_log_w = log_ws.max(dim=-1).values
    w_bars = (log_ws - max_log_w).exp()
    phis = test_fn(saps_raw, saps).squeeze()
    expectation = ((phis * w_bars).mean(dim=-1).log() + max_log_w).exp()
    logN = torch.tensor(log_ws.shape[0]).log()
    log_std = (torch.logsumexp((phis * w_bars - expectation) ** 2, dim=0) - logN)/2
    std = log_std.exp()
    return expectation, std, phis

def iterative_importance_estimate(
        test_fn: Callable,
        saps_raw: torch.Tensor,
        saps: torch.Tensor,
        log_probs: torch.Tensor,
        log_qrobs: torch.Tensor,
        cur_expectation: torch.Tensor,
        cur_N: torch.Tensor,
):
    log_ws = log_probs - log_qrobs
    max_log_w = log_ws.max()
    w_bars = (log_ws - max_log_w).exp()
    phis = test_fn(saps_raw, saps).squeeze()
    new_expectation = ((phis * w_bars).mean().log() + max_log_w).exp()
    total_sum = cur_expectation * cur_N + new_expectation * log_probs.nelement()
    total_N = cur_N + log_probs.nelement()
    total_expectation = total_sum / total_N
    return total_expectation, total_N, phis

def compute_is_diagnostics(
    log_probs: torch.Tensor,
    log_qrobs: torch.Tensor,
    phis: torch.Tensor = None
):
    """
    Compute diagnostics for importance sampling
    Args:
        log_probs: log probabilities from target distribution
        log_qrobs: log probabilities from proposal distribution
        phis: indicator function values (optional)
    Returns:
        dict containing diagnostic metrics
    """
    # Compute importance weights
    log_ws = log_probs - log_qrobs
    max_log_w = log_ws.max()
    w_bars = (log_ws - max_log_w).exp()

    diagnostics = {}

    # 1. Effective Sample Size (ESS)
    ess = w_bars.sum() ** 2 / (w_bars ** 2).sum()
    diagnostics['ess'] = ess
    diagnostics['ess_ratio'] = ess / len(w_bars)

    # 2. Weight statistics
    diagnostics['max_weight'] = w_bars.max()
    diagnostics['min_weight'] = w_bars.min()
    diagnostics['weight_range'] = diagnostics['max_weight'] / diagnostics['min_weight']
    diagnostics['max_weight_over_sum'] = diagnostics['max_weight'] / w_bars.sum()

    # 3. Coefficient of Variation (CV) of weights
    weight_mean = w_bars.mean()
    weight_std = w_bars.std()
    diagnostics['weight_cv'] = weight_std / weight_mean

    # 4. Perplexity (alternative measure of sample efficiency)
    perplexity = torch.exp(-torch.sum(w_bars * torch.log(w_bars + 1e-10)))
    diagnostics['perplexity'] = perplexity

    # 5. KL divergence estimate between q and p
    kl_div = (w_bars * log_ws).sum()
    diagnostics['kl_divergence'] = kl_div

    if phis is not None:
        # 6. Region statistics
        diagnostics['prop_samples_in_region'] = phis.mean()
        diagnostics['effective_samples_in_region'] = (phis * w_bars).sum()

    return diagnostics

def compute_is(
        proposal: Proposal,
        diffusion_target: Proposal,
        target: Target,
        alpha: torch.Tensor,
        cfg_obj: SampleConfig,
        std: ContinuousEvaluator,
        error_metric: ErrorMetric,
):
    saps_raw, saps = proposal.sample()
    log_qrobs = proposal.log_prob(saps)
    test_fn = std.likelihood.get_condition
    tail_prob = target.analytical_prob(alpha)
    true_log_probs = target.log_prob(saps_raw).squeeze()
    rearranged_saps_raw = einops.rearrange(
        saps_raw,
        '(b c) d h -> b c d h',
        b=cfg_obj.num_sample_batches
    )
    rearranged_saps = einops.rearrange(
        saps,
        '(b c) d h -> b c d h',
        b=cfg_obj.num_sample_batches
    )
    rearranged_log_qrobs = einops.rearrange(
        log_qrobs,
        '(b c) -> b c',
        b=cfg_obj.num_sample_batches
    )
    rearranged_true_probs = einops.rearrange(
        true_log_probs,
        '(b c) -> b c',
        b=cfg_obj.num_sample_batches
    )
    log_drobs = diffusion_target.log_prob(saps)
    rearranged_log_drobs = einops.rearrange(
        log_drobs,
        '(b c) -> b c',
        b=cfg_obj.num_sample_batches
    )
    subsample_sizes = torch.logspace(
        math.log10(500),
        math.log10(cfg_obj.num_samples // cfg_obj.num_sample_batches),
        10,
        dtype=int
    )
    is_errors = []
    quadrature_errors = []
    for subsample_size in subsample_sizes:
        expectation, std, phis = importance_estimate(
            test_fn,
            rearranged_saps_raw[:, :subsample_size],
            rearranged_saps[:, :subsample_size],
            rearranged_true_probs[:, :subsample_size],
            rearranged_log_qrobs[:, :subsample_size],
        )
        is_errors.append(error_metric(expectation, tail_prob))
    plt.plot(subsample_sizes, is_errors, 'Importance Sampling Error')
    plt.scatter(subsample_sizes, is_errors)
    # plt.plot(subsample_sizes, quadrature_errors, 'Quadrature Error')
    # plt.scatter(subsample_sizes, quadrature_errors)
    plt.xlabel('Sample Size')
    plt.ylabel(error_metric.name())
    plt.legend()
    plt.savefig('{}/mc_integral_plot{}'.format(
        HydraConfig.get().run.dir,
    ))


@hydra.main(version_base=None, config_path="conf", config_name="continuous_is_config")
def importance_sample(cfg):
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
        cfg_obj = OmegaConf.to_object(cfg)
        proposal = get_proposal(cfg_obj.example, std)
        diffusion_target = get_proposal(cfg_obj.example, std)
        target = get_target(cfg_obj)
        std.cfg.num_samples *= std.cfg.num_sample_batches
        error_metric = get_error_metric(std.cfg.error_metric)
        compute_is(
            proposal,
            diffusion_target,
            target,
            alpha,
            cfg_obj,
            std,
            error_metric
        )


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        suppresswarning()

    cs = ConfigStore.instance()
    cs.store(name="vpsde_is_config", node=ISConfig)
    register_configs()

    with torch.no_grad():
        importance_sample()
