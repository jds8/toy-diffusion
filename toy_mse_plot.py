#!/usr/bin/env python3
import warnings
import os
import logging
import matplotlib.pyplot as plt
import re
from datetime import datetime

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
    all_prop_exiteds = []
    for model in cfg.models:
        cfg.model_name = model
        std = ContinuousEvaluator(cfg=cfg)
        sample_traj_out = std.sample_trajectories(
            cond=std.cond,
            alpha=std.likelihood.alpha.reshape(-1, 1),
        )
        sample_trajs = sample_traj_out.samples[-1]
        mse, prop_exited = compute_mse(cfg, sample_trajs, std)
        print(f'\nmodel: {model}\nmse: {mse}\nprop_exited: {prop_exited}\n')
        all_mses.append(mse)
        all_prop_exiteds.append(prop_exited)
    return all_mses, all_prop_exiteds

def compute_mse(cfg, sample_trajs, std):
    if isinstance(std.example, BrownianMotionDiffExampleConfig):
        return compute_bm_mse(std, sample_trajs, std)
    elif isinstance(std.example, GaussianExampleConfig):
        return compute_gaussian_mse(std, sample_trajs, std)
    else:
        raise NotImplementedError

def compute_bm_mse(cfg, sample_trajs, std):
    end_time = torch.tensor(1., device=device)
    dt = end_time / (cfg.example.sde_steps-1)
    # de-standardize data
    trajs = sample_trajs * dt.sqrt()
    bm_trajs = torch.cat([
        torch.zeros(trajs.shape[0], 1, 1, device=trajs.device),
        trajs.cumsum(dim=-2)
    ], dim=1)
    exited = (bm_trajs.abs() > std.likelihood.alpha).any(dim=1).to(float)
    prop_exited = exited.mean()

    alpha = torch.tensor([std.likelihood.alpha])
    ode_llk = std.ode_log_likelihood(
        sample_trajs,
        cond=0,
        alpha=alpha
    )
    scaled_ode_llk = (ode_llk[0] - dt.sqrt().log() * (cfg.example.sde_steps-1)).squeeze()
    analytical_llk = (dist.Normal(0, 1).log_prob(sample_trajs) - dt.sqrt().log()).sum(1).squeeze()
    mse_llk = torch.nn.MSELoss()(analytical_llk, scaled_ode_llk)
    return mse_llk, prop_exited

def compute_gaussian_mse(cfg, sample_trajs, std):
    exited = (sample_trajs.abs() > std.likelihood.alpha).any(dim=1).to(float)
    prop_exited = exited.mean()
    alpha = torch.tensor([std.likelihood.alpha])
    pre_ode_llk = std.ode_log_likelihood(
        sample_trajs,
        cond=std.cond,
        alpha=alpha
    )[0] - torch.tensor(cfg.example.sigma).log()

    datapoint_dist = torch.distributions.Normal(
        cfg.example.mu, cfg.example.sigma
    )
    tail = 2 * datapoint_dist.cdf(cfg.example.mu-alpha*cfg.example.sigma)
    traj = sample_trajs * cfg.example.sigma + cfg.example.mu
    analytical_llk_w_nan = torch.where(
        torch.abs(traj - cfg.example.mu) > alpha * cfg.example.sigma,
        datapoint_dist.log_prob(traj) - tail.log(),
        torch.nan
    )
    non_nan_idx = ~torch.any(analytical_llk_w_nan.isnan(), dim=1)
    a_llk = analytical_llk_w_nan[non_nan_idx].squeeze()
    ode_llk = pre_ode_llk[non_nan_idx.squeeze()].squeeze()

    mse_llk = torch.nn.MSELoss()(a_llk, ode_llk)
    return mse_llk, prop_exited

@hydra.main(version_base=None, config_path="conf", config_name="continuous_mse_plot_config")
def make_mse_plot(cfg):
    # assert cfg.guidance == GuidanceType.NoGuidance and cfg.cond == 0.

    logger = logging.getLogger("main")
    logger.info('run type: mse plot')
    logger.info(f"CONFIG\n{OmegaConf.to_yaml(cfg)}")

    os.system('echo git commit: $(git rev-parse HEAD)')

    pattern = re.compile(r'.*v(\d+)')
    training_sample_list = [int(pattern.search(model).group(1)) for model in cfg.models]
    with torch.no_grad():
        all_pcts, all_prop_exiteds = sample_models(cfg)
        plt.plot(training_sample_list, torch.stack(all_pcts), marker='*')
        plt.title('Mean Squared Error of Log-Density vs. Computational Effort')
        plt.xlabel('Model Trained on X Num. Training Samples')
        plt.ylabel('MSE with ground truth density')

        save_dir = 'figs/mse_plots/{}'.format(datetime.now().isoformat())
        os.makedirs(save_dir, exist_ok=True)
        alpha = torch.tensor([cfg.likelihood.alpha])
        plt.savefig('{}/alpha={}_num_saps_{}_mse_vs_effort.pdf'.format(
            save_dir,
            alpha.item(),
            cfg.num_samples,
        ))

        plt.clf()

        plt.plot(training_sample_list, torch.stack(all_prop_exiteds), marker='*')
        plt.title('Prop. Samples in Tail vs. Computational Effort')
        plt.xlabel('Num. Training Samples')
        plt.ylabel('Proportion')
        plt.ylim((0., 1.))

        os.makedirs(save_dir, exist_ok=True)
        alpha = torch.tensor([cfg.likelihood.alpha])
        plt.savefig('{}/alpha={}_num_saps_{}_pct_in_tail_vs_effort.pdf'.format(
            save_dir,
            alpha.item(),
            cfg.num_samples,
        ))

        plt.clf()

        model_name_csv = ','.join(cfg.models)
        torch.save(model_name_csv, f'{save_dir}/models.csv')

    import pdb; pdb.set_trace()

if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        suppresswarning()

    cs = ConfigStore.instance()
    cs.store(name="vpsde_mse_plot_config", node=MSEPlotConfig)
    register_configs()

    with torch.no_grad():
        make_mse_plot()
