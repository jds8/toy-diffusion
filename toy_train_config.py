#!/usr/bin/env python3

from typing import Optional
from dataclasses import dataclass
from models.toy_diffusion_models_config import BaseSamplerConfig, ModelConfig, GuidanceType
from toy_likelihood_configs import LikelihoodConfig

from omegaconf import OmegaConf


@dataclass
class BaseConfig:
    sampler: BaseSamplerConfig = BaseSamplerConfig()
    diffusion: ModelConfig = ModelConfig()
    likelihood: LikelihoodConfig = LikelihoodConfig()
    sde_drift: float = 0.
    sde_diffusion: float = 1.
    sde_steps: int = 1000
    model_dir: str = 'diffusion_models/'
    model_name: str = ''


@dataclass
class TrainConfig(BaseConfig):
    batch_size: int = 1024
    lr: float = 0.0001
    no_wandb: bool = 1
    delete_local_model: bool = False
    max_gradient: float = 1.
    loss_fn: str = 'l2'
    p_uncond: float = 1.


def get_model_path(cfg: TrainConfig):
    if cfg.model_name:
        model_name = cfg.model_name
    else:
        sampler_name = OmegaConf.to_object(cfg.sampler).name()
        diffusion_name = OmegaConf.to_object(cfg.diffusion).name()
        model_name = "{}_{}".format(sampler_name, diffusion_name)
    return "{}/{}".format(
        cfg.model_dir,
        model_name,
    )


@dataclass
class SampleConfig(BaseConfig):
    num_samples: int = 10
    cond: Optional[float] = None
    guidance: GuidanceType = GuidanceType.Classifier
