#!/usr/bin/env python3
import warnings
import os
import logging
import matplotlib.pyplot as plt
import re

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
import torch
import torch.distributions as dist

from toy_configs import register_configs
from toy_sample import ContinuousEvaluator
from toy_train_config import MSEPlotConfig, BrownianMotionDiffExampleConfig, GaussianExampleConfig
from models.toy_diffusion_models_config import GuidanceType


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def suppresswarning():
    warnings.warn("user", UserWarning)

def sample_models(cfg):
    all_mses = []
    for model in cfg.models:
        cfg.model_name = model
        std = ContinuousEvaluator(cfg=cfg)
        sample_traj_out = std.sample_trajectories(
            cond=std.cond,
            alpha=std.likelihood.alpha.reshape(-1, 1),
        )
        sample_trajs = sample_traj_out.samples[-1]
        mse = compute_mse(cfg, sample_trajs, std)
        print(f'model_{model}_mse_{mse}')
        all_mses.append(mse)
    return all_mses

def compute_mse(cfg, sample_trajs, std):
    if isinstance(std.example, BrownianMotionDiffExampleConfig):
        return compute_bm_mse(std, sample_trajs, std)
    elif isinstance(std.example, GaussianExampleConfig):
        return compute_gaussian_mse(std, sample_trajs, std)
    else:
        raise NotImplementedError

def compute_bm_mse(cfg, sample_trajs, std):
    alpha = torch.tensor([std.likelihood.alpha])
    ode_llk = std.ode_log_likelihood(
        sample_trajs,
        cond=0,
        alpha=alpha
    )
    end_time = torch.tensor(1., device=device)
    dt = end_time / (cfg.example.sde_steps-1)
    scaled_ode_llk = ode_llk[0] - dt.sqrt().log() * (cfg.example.sde_steps-1)
    analytical_llk = (dist.Normal(0, 1).log_prob(sample_trajs) - dt.sqrt().log()).sum(1).squeeze()
    mse_llk = torch.nn.MSELoss()(analytical_llk, scaled_ode_llk)
    return mse_llk

def compute_gaussian_mse(cfg, sample_trajs, std):
    alpha = torch.tensor([0.])
    ode_llk = std.ode_log_likelihood(
        sample_trajs,
        cond=0,
        alpha=alpha
    )[0] - torch.tensor(cfg.example.sigma).log()

    datapoint_dist = torch.distributions.Normal(
        cfg.example.mu, cfg.example.sigma
    )
    tail = 2 * datapoint_dist.cdf(torch.tensor(cfg.example.mu))
    traj = sample_trajs * cfg.example.sigma + cfg.example.mu
    analytical_llk = datapoint_dist.log_prob(traj) - tail.log()
    a_llk = analytical_llk.exp().squeeze()

    mse_llk = torch.nn.MSELoss()(a_llk, ode_llk)
    return mse_llk

@hydra.main(version_base=None, config_path="conf", config_name="continuous_mse_plot_config")
def make_mse_plot(cfg):
    assert cfg.guidance == GuidanceType.NoGuidance and cfg.cond == 0.

    logger = logging.getLogger("main")
    logger.info('run type: mse plot')
    logger.info(f"CONFIG\n{OmegaConf.to_yaml(cfg)}")

    os.system('echo git commit: $(git rev-parse HEAD)')

    pattern = re.compile(r'.*v(\d+)')
    training_sample_list = [int(pattern.search(model).group(1)) for model in cfg.models]
    with torch.no_grad():
        all_pcts = sample_models(cfg)
        plt.plot(training_sample_list, torch.stack(all_pcts))
        plt.title('Mean Squared Error of Log-Likelihood vs. Computational Effort')
        plt.xlabel('Num. Training Samples')
        plt.ylabel('MSE with ground truth density')

        save_dir = 'figs/{}'.format(cfg.model_name)
        os.makedirs(save_dir, exist_ok=True)
        alpha = torch.tensor([cfg.likelihood.alpha])
        plt.savefig('{}/alpha={}_mse_vs_effort.pdf'.format(
            save_dir,
            alpha.item(),
        ))


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        suppresswarning()

    cs = ConfigStore.instance()
    cs.store(name="vpsde_mse_plot_config", node=MSEPlotConfig)
    register_configs()

    with torch.no_grad():
        make_mse_plot()
