#!/usr/bin/env python3
import warnings
import logging
from typing import Callable
import time

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
import torch

from toy_configs import register_configs
from toy_sample import ContinuousEvaluator
from toy_train_config import ISConfig, get_target, get_proposal


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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
    max_log_w = log_ws.max()
    w_bars = (log_ws - max_log_w).exp()
    phis = test_fn(saps_raw, saps).squeeze()
    return ((phis * w_bars).mean().log() + max_log_w).exp()


@hydra.main(version_base=None, config_path="conf", config_name="continuous_is_config")
def importance_sample(cfg):
    logger = logging.getLogger("main")
    logger.info(f"CONFIG\n{OmegaConf.to_yaml(cfg)}")

    start = time.time()
    std = ContinuousEvaluator(cfg=cfg)
    cfg_obj = OmegaConf.to_object(cfg)

    ##################################################
    # true tail probability under target
    ##################################################
    target = get_target(cfg_obj)
    tail_prob = target.analytical_prob(torch.tensor(cfg_obj.likelihood.alpha))
    logger.info(f'true tail probability: {tail_prob}')
    ##################################################

    finish = time.time()
    logger.info(f'total time: {finish-start}')

    ##################################################
    # IS estimate using target
    ##################################################
    proposal = get_proposal(cfg_obj.example, std)
    saps_raw, saps = proposal.sample()
    log_proposal = proposal.log_prob(saps).squeeze()
    log_qrobs, log_drobs = log_proposal[:cfg.num_samples], log_proposal[cfg.num_samples:]
    log_probs = target.log_prob(saps_raw).squeeze()

    # z_score = lambda x: (x - cfg.example.mu) / cfg.example.sigma
    # test_fn = lambda x: (z_score(x).abs() < std.cond).to(torch.float)
    test_fn = std.likelihood.get_condition

    target_estimate = importance_estimate(
        test_fn=test_fn,
        saps_raw=saps_raw,
        saps=saps,
        log_probs=log_probs,
        log_qrobs=log_qrobs
    )
    logger.info(f'IS estimate with target: {target_estimate}')
    ##################################################

    finish = time.time()
    logger.info(f'total time: {finish-start}')

    ##################################################
    # IS estimate using unconditional diffusion model
    ##################################################
    std.set_no_guidance()
    diffusion_target = get_proposal(cfg_obj.example, std)
    log_drobs = diffusion_target.log_prob(saps).squeeze()
    diffusion_estimate = importance_estimate(
        test_fn=test_fn,
        saps_raw=saps_raw,
        saps=saps,
        log_probs=log_drobs,
        log_qrobs=log_qrobs
    )
    logger.info(f'IS estimate with diffusion: {diffusion_estimate}')
    ##################################################

    finish = time.time()
    logger.info(f'total time: {finish-start}')
    import pdb; pdb.set_trace()


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        suppresswarning()

    cs = ConfigStore.instance()
    cs.store(name="vpsde_is_config", node=ISConfig)
    register_configs()

    with torch.no_grad():
        importance_sample()
