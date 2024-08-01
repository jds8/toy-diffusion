#!/usr/bin/env python3
import warnings
import logging

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
import torch

from toy_configs import register_configs
from toy_sample import ContinuousEvaluator, GaussianProposal
from importance_sampling import ImportanceSampler
from toy_train_config import SampleConfig


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def suppresswarning():
    warnings.warn("user", UserWarning)


@hydra.main(version_base=None, config_path="conf", config_name="continuous_sample_config")
def importance_sample(cfg):
    logger = logging.getLogger("main")
    logger.info(f"CONFIG\n{OmegaConf.to_yaml(cfg)}")

    std = ContinuousEvaluator(cfg=cfg)

    target = torch.distributions.Normal(
        cfg.example.mu,
        cfg.example.sigma
    )
    proposal = GaussianProposal(std)
    importance_sampler = ImportanceSampler(
        proposal=proposal,
        target=target
    )
    z_score = lambda x: (x - cfg.example.mu) / cfg.example.sigma
    test_fn = lambda x: (z_score(x).abs() < std.cond).to(torch.float)
    estimate = importance_sampler.estimate(test_fn)
    logger.info(f'IS estimate: {estimate}')
    import pdb; pdb.set_trace()


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        suppresswarning()

    cs = ConfigStore.instance()
    cs.store(name="vpsde_sample_config", node=SampleConfig)
    register_configs()

    with torch.no_grad():
        importance_sample()
