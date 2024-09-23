#!/usr/bin/env python3
import warnings
import os
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
    import pdb; pdb.set_trace()
    return total_expectation, total_N

@hydra.main(version_base=None, config_path="conf", config_name="continuous_is_config")
def importance_sample(cfg):
    logger = logging.getLogger("main")
    logger.info(f"CONFIG\n{OmegaConf.to_yaml(cfg)}")

    with torch.no_grad():
        start = time.time()
        std = ContinuousEvaluator(cfg=cfg)
        cfg_obj = OmegaConf.to_object(cfg)

        save_dir = 'figs/{}'.format(cfg.model_name)
        os.makedirs(save_dir, exist_ok=True)

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

        guidance = std.cfg.guidance
        for i in range(cfg.num_rounds):
            print('round {}'.format(i))
            ##################################################
            # IS estimate using target
            ##################################################
            std.cfg.guidance = guidance
            proposal = get_proposal(cfg_obj.example, std)
            start_sample = time.time()
            num_full_splits, num_leftover = cfg_obj.num_splits()
            std.cfg.num_samples = std.cfg.split_size
            num_samples_list = []
            saps_raw_list = []
            saps_list = []
            for j in range(num_full_splits):
                saps_raw, saps = proposal.sample()
                saps_raw_list.append(saps_raw)
                saps_list.append(saps)
                num_samples_list.append(std.cfg.split_size)
            if num_leftover:
                std.cfg.num_samples = num_leftover
                saps_raw, saps = proposal.sample()
                saps_raw_list.append(saps_raw)
                saps_list.append(saps)
                num_samples_list.append(num_leftover)
            finish_sample = time.time()
            log_proposal_list = []
            true_log_probs_list = []
            for j in range(num_full_splits + int(num_leftover > 0)):
                log_proposal = proposal.log_prob(saps_list[j]).squeeze()
                log_proposal_list.append(log_proposal)
                true_log_probs = target.log_prob(saps_raw_list[j]).squeeze()
                true_log_probs_list.append(true_log_probs)
            finish_evaluate = time.time()
            logger.info(f'total sample time: {finish_sample-start_sample}')
            logger.info(f'total eval time: {finish_evaluate-finish_sample}')

            test_fn = std.likelihood.get_condition
            target_estimate = torch.tensor([0.], device=device)
            target_N = 0
            for j in range(num_full_splits + int(num_leftover > 0)):
                num_samples = num_samples_list[j]
                log_qrobs = log_proposal_list[j][:num_samples]
                log_drobs = log_proposal_list[j][num_samples:]
                true_log_probs = true_log_probs_list[j]
                saps = saps_list[j]
                saps_raw = saps_raw_list[j]
                target_estimate, target_N = iterative_importance_estimate(
                    test_fn=test_fn,
                    saps_raw=saps_raw,
                    saps=saps,
                    log_probs=true_log_probs,
                    log_qrobs=log_qrobs,
                    cur_expectation=target_estimate,
                    cur_N=target_N,
                )

            target_std = torch.tensor([0.], device=device)
            target_is_stats = torch.stack([target_estimate, target_std])
            torch.save(target_is_stats, '{}/alpha={}_target_is_stats_round_{}.pt'.format(
                save_dir,
                alpha,
                i
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
            num_full_splits, num_leftover = cfg_obj.num_splits()
            std.cfg.num_samples = std.cfg.split_size
            log_droposal_list = []
            for j in range(num_full_splits + int(num_leftover > 0)):
                log_droposal = diffusion_target.log_prob(saps_list[j]).squeeze()
                log_droposal_list.append(log_droposal)

            test_fn = std.likelihood.get_condition
            diffusion_estimate = torch.tensor([0.], device=device)
            target_N = 0
            for j in range(num_full_splits + int(num_leftover > 0)):
                num_samples = num_samples_list[j]
                log_qrobs = log_proposal_list[j][:num_samples]
                log_drobs = log_droposal_list[j]
                saps = saps_list[j]
                saps_raw = saps_raw_list[j]
                diffusion_estimate, target_N = iterative_importance_estimate(
                    test_fn=test_fn,
                    saps_raw=saps_raw,
                    saps=saps,
                    log_probs=log_drobs,
                    log_qrobs=log_qrobs,
                    cur_expectation=diffusion_estimate,
                    cur_N=target_N,
                )
            diffusion_std = torch.tensor([0.], device=device)
            diffusion_is_stats = torch.stack([diffusion_estimate, diffusion_std])
            torch.save(diffusion_is_stats, '{}/alpha={}_diffusion_is_stats_round_{}.pt'.format(
                save_dir,
                alpha,
                i
            ))
            logger.info('IS estimate with diffusion: {} and std. dev.: {}'.format(
                diffusion_estimate,
                diffusion_std,
            ))
            ##################################################

        finish = time.time()
        logger.info(f'total time: {finish-start}')


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        suppresswarning()

    cs = ConfigStore.instance()
    cs.store(name="vpsde_is_config", node=ISConfig)
    register_configs()

    with torch.no_grad():
        importance_sample()
