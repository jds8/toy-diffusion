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
    expectation = ((phis * w_bars).mean().log() + max_log_w).exp()
    logN = torch.tensor(log_ws.shape[0]).log()
    log_std = (torch.logsumexp((phis * w_bars - expectation) ** 2, dim=0) - logN)/2
    std = log_std.exp()
    return expectation, std


@hydra.main(version_base=None, config_path="conf", config_name="continuous_is_config")
def importance_sample(cfg):
    logger = logging.getLogger("main")
    logger.info(f"CONFIG\n{OmegaConf.to_yaml(cfg)}")

    start = time.time()
    std = ContinuousEvaluator(cfg=cfg)
    cfg_obj = OmegaConf.to_object(cfg)

    save_dir = 'figs/{}'.format(cfg.model_name)

    ##################################################
    # true tail probability under target
    ##################################################
    target = get_target(cfg_obj)
    tail_prob = target.analytical_prob(torch.tensor(cfg_obj.likelihood.alpha))
    alpha = '%.1f' % cfg_obj.likelihood.alpha
    torch.save(tail_prob, '{}/alpha={}_tail_prob.pt'.format(
        save_dir,
        alpha,
    ))
    logger.info(f'true tail probability: {tail_prob}')
    ##################################################

    finish = time.time()
    logger.info(f'total time: {finish-start}')

    ##################################################
    # IS estimate using target
    ##################################################
    proposal = get_proposal(cfg_obj.example, std)
    start_sample = time.time()
    saps_raw, saps = proposal.sample()
    finish_sample = time.time()
    log_proposal = proposal.log_prob(saps).squeeze()
    finish_evaluate = time.time()
    logger.info(f'total sample time: {finish_sample-start_sample}')
    logger.info(f'total time: {finish_evaluate-finish_sample}')
    log_qrobs, log_drobs = log_proposal[:cfg.num_samples], log_proposal[cfg.num_samples:]
    log_probs = target.log_prob(saps_raw).squeeze()

    test_fn = std.likelihood.get_condition

    target_estimate, target_std = importance_estimate(
        test_fn=test_fn,
        saps_raw=saps_raw,
        saps=saps,
        log_probs=log_probs,
        log_qrobs=log_qrobs
    )
    target_is_stats = torch.stack([target_estimate, target_std])
    torch.save(target_is_stats, '{}/alpha={}_target_is_stats.pt'.format(
        save_dir,
        alpha,
    ))
    logger.info('IS estimate with target: {} and std. dev.: {}'.format(
        target_estimate,
        target_std,
    ))
    ##################################################

    finish = time.time()
    logger.info(f'total time: {finish-start}')

    ##################################################
    # IS estimate using unconditional diffusion model
    ##################################################
    std.set_no_guidance()
    diffusion_target = get_proposal(cfg_obj.example, std)
    log_drobs = diffusion_target.log_prob(saps).squeeze()
    diffusion_estimate, diffusion_std = importance_estimate(
        test_fn=test_fn,
        saps_raw=saps_raw,
        saps=saps,
        log_probs=log_drobs,
        log_qrobs=log_qrobs
    )
    diffusion_is_stats = torch.stack([diffusion_estimate, diffusion_std])
    torch.save(diffusion_is_stats, '{}/alpha={}_diffusion_is_stats.pt'.format(
        save_dir,
        alpha,
    ))
    logger.info('IS estimate with diffusion: {} and std. dev.: {}'.format(
        diffusion_estimate,
        diffusion_std,
    ))
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
