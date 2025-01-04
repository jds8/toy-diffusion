#!/usr/bin/env python3
import warnings
import os
import logging
from typing import Callable
import time
import re
from collections import namedtuple

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
import torch

from toy_configs import register_configs
from toy_sample import ContinuousEvaluator
from toy_train_config import ISConfig, get_target, get_proposal


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
    max_log_w = log_ws.max()
    w_bars = (log_ws - max_log_w).exp()
    phis = test_fn(saps_raw, saps).squeeze()
    num_saps_not_in_region = (1-phis).sum()
    expectation = ((phis * w_bars).mean().log() + max_log_w).exp()
    logN = torch.tensor(log_ws.shape[0]).log()
    log_std = (torch.logsumexp((phis * w_bars - expectation) ** 2, dim=0) - logN)/2
    std = log_std.exp()
    return expectation, std, num_saps_not_in_region

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
    num_saps_not_in_region = (1-phis).sum()
    new_expectation = ((phis * w_bars).mean().log() + max_log_w).exp()
    total_sum = cur_expectation * cur_N + new_expectation * log_probs.nelement()
    total_N = cur_N + log_probs.nelement()
    total_expectation = total_sum / total_N
    return total_expectation, total_N, num_saps_not_in_region


@hydra.main(version_base=None, config_path="conf", config_name="continuous_is_config")
def importance_sample(cfg):
    logger = logging.getLogger("main")
    logger.info('run type: importance sampling')
    cfg_str = OmegaConf.to_yaml(cfg)
    logger.info(f"CONFIG\n{cfg_str}")

    os.system('echo git commit: $(git rev-parse HEAD)')

    with torch.no_grad():
        start = time.time()
        std = ContinuousEvaluator(cfg=cfg)
        cfg_obj = OmegaConf.to_object(cfg)

        save_dir = '{}/{}'.format(cfg.figs_dir, cfg.model_name)
        os.makedirs(save_dir, exist_ok=True)

        # get alpha level
        alpha = '%.1f' % cfg_obj.likelihood.alpha

        # save config
        torch.save(cfg_str, f'{save_dir}/alpha={alpha}_config.txt')

        ##################################################
        # true tail probability under target
        ##################################################
        target = get_target(cfg_obj)
        tail_prob = target.analytical_prob(torch.tensor(cfg_obj.likelihood.alpha))
        torch.save(tail_prob, '{}/alpha={}_tail_prob.pt'.format(
            save_dir,
            alpha,
        ))
        logger.info(f'true tail probability: {tail_prob}')
        ##################################################

        finish = time.time()
        logger.info(f'total time: {finish-start}')

        old_guidance = std.cfg.guidance
        for i in range(cfg.num_rounds):
            print('round {}'.format(i))
            ##################################################
            # IS estimate using target
            ##################################################
            start_sample = time.time()
            num_full_splits, num_leftover = cfg_obj.num_splits()
            num_samples_list = [std.cfg.split_size] * num_full_splits + [num_leftover]
            test_fn = std.likelihood.get_condition
            round_data = RoundData.load(save_dir, alpha, i)
            saps_idx = round_data.curr_sap
            target_estimate = round_data.target_estimate
            diffusion_estimate = round_data.diffusion_estimate
            target_N = round_data.target_N
            total_num_saps_not_in_region = round_data.total_num_saps_not_in_region
            for j in range(num_full_splits + int(num_leftover > 0)):
                proposal = get_proposal(cfg_obj.example, std)
                std.cfg.num_samples = num_samples_list[j]
                sample_start = time.time()
                saps_raw, saps = proposal.sample()
                sample_end = time.time()
                sample_time = sample_end - sample_start
                saps_new_idx = saps_idx + len(saps)
                saps_filename = '{}/alpha={}_saps_{}_{}_round_{}'.format(
                    save_dir,
                    alpha,
                    saps_idx,
                    saps_new_idx,
                    cfg.start_round+i
                )
                torch.save(
                    saps,
                    saps_filename
                )
                saps_time_filename = '{}/alpha={}_time_for_saps_{}_{}_round_{}'.format(
                    save_dir,
                    alpha,
                    saps_idx,
                    saps_new_idx,
                    cfg.start_round+i
                )
                torch.save(
                    sample_time,
                    saps_time_filename
                )
                log_qrobs_start = time.time()
                log_qrobs = proposal.log_prob(saps)
                log_qrobs_end = time.time()
                log_qrobs_time = log_qrobs_end - log_qrobs_start
                log_qrobs = log_qrobs.squeeze()
                log_qrobs_filename = '{}/alpha={}_log_qrobs_{}_{}_round_{}'.format(
                    save_dir,
                    alpha,
                    saps_idx,
                    saps_new_idx,
                    cfg.start_round+i
                )
                torch.save(
                    log_qrobs,
                    log_qrobs_filename
                )
                log_qrobs_time_filename = '{}/alpha={}_time_for_log_qrobs_{}_{}_round_{}'.format(
                    save_dir,
                    alpha,
                    saps_idx,
                    saps_new_idx,
                    cfg.start_round+i
                )
                torch.save(
                    log_qrobs_time,
                    log_qrobs_time_filename
                )
                old_guidance = std.set_no_guidance()
                diffusion_target = get_proposal(cfg_obj.example, std)
                log_drobs_start = time.time()
                log_drobs = diffusion_target.log_prob(saps)
                log_drobs_end = time.time()
                log_drobs_time = log_drobs_end - log_drobs_start
                log_drobs = log_drobs.squeeze()
                log_drobs_filename = '{}/alpha={}_log_drobs_{}_{}_round_{}'.format(
                    save_dir,
                    alpha,
                    saps_idx,
                    saps_new_idx,
                    cfg.start_round+i
                )
                torch.save(
                    log_drobs,
                    log_drobs_filename
                )
                log_drobs_time_filename = '{}/alpha={}_time_for_log_drobs_{}_{}_round_{}'.format(
                    save_dir,
                    alpha,
                    saps_idx,
                    saps_new_idx,
                    cfg.start_round+i
                )
                torch.save(
                    log_drobs_time,
                    log_drobs_time_filename
                )
                saps_idx = saps_new_idx
                true_log_probs = target.log_prob(saps_raw).squeeze()
                target_estimate, _, num_saps_not_in_region = iterative_importance_estimate(
                    test_fn=test_fn,
                    saps_raw=saps_raw,
                    saps=saps,
                    log_probs=true_log_probs,
                    log_qrobs=log_qrobs,
                    cur_expectation=target_estimate,
                    cur_N=target_N,
                )
                ##################################################
                # IS estimate using unconditional diffusion model
                ##################################################
                diffusion_estimate, target_N, _ = iterative_importance_estimate(
                    test_fn=test_fn,
                    saps_raw=saps_raw,
                    saps=saps,
                    log_probs=log_drobs,
                    log_qrobs=log_qrobs,
                    cur_expectation=diffusion_estimate,
                    cur_N=target_N,
                )
                std.cfg.guidance = old_guidance
                total_num_saps_not_in_region += num_saps_not_in_region

            round_data = RoundData(
                curr_sap=saps_idx,
                target_estimate=target_estimate,
                diffusion_estimate=diffusion_estimate,
                target_N=target_N,
                total_num_saps_not_in_region=total_num_saps_not_in_region
            )
            round_data.save(save_dir, alpha, i)
            pct_saps_not_in_region = torch.tensor([100 * total_num_saps_not_in_region / target_N])
            logger.info(f'pct saps not in region: {pct_saps_not_in_region}')
            torch.save(pct_saps_not_in_region, \
            '{}/alpha={}_pct_saps_not_in_region_round_{}.pt'.format(
                save_dir,
                alpha,
                cfg.start_round+i
            ))

            finish_sample = time.time()
            logger.info(f'total sample+eval time: {finish_sample-start_sample}')
            zero_std = torch.tensor([0.], device=device)
            target_is_stats = torch.stack([target_estimate, zero_std])
            torch.save(target_is_stats, '{}/alpha={}_target_is_stats_round_{}.pt'.format(
                save_dir,
                alpha,
                cfg.start_round+i
            ))
            logger.info('IS estimate with target: {} and std. dev.: {}'.format(
                target_estimate,
                zero_std,
            ))

            finish = time.time()
            logger.info(f'total time: {finish-start}')

            diffusion_is_stats = torch.stack([diffusion_estimate, zero_std])
            torch.save(diffusion_is_stats, '{}/alpha={}_diffusion_is_stats_round_{}.pt'.format(
                save_dir,
                alpha,
                cfg.start_round+i
            ))
            logger.info('IS estimate with diffusion: {} and std. dev.: {}'.format(
                diffusion_estimate,
                zero_std,
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
