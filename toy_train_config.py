#!/usr/bin/env python3

from enum import Enum
from typing import Optional
from dataclasses import dataclass
from models.toy_diffusion_models_config import BaseSamplerConfig, ModelConfig, GuidanceType
from toy_likelihood_configs import LikelihoodConfig

from omegaconf import OmegaConf


@dataclass
class ExampleConfig:
    def name(self):
        return 'ExampleConfig'


@dataclass
class GaussianExampleConfig(ExampleConfig):
    mu: float = 1.
    sigma: float = 2.

    def name(self):
        return 'GaussianExampleConfig'


@dataclass
class BrownianMotionExampleConfig(ExampleConfig):
    sde_drift: float = 0.
    sde_diffusion: float = 1.
    sde_steps: int = 1000

    def name(self):
        return 'BrownianMotionExampleConfig'


@dataclass
class BrownianMotionDiffExampleConfig(BrownianMotionExampleConfig):
    def name(self):
        return 'BrownianMotionDiffExampleConfig'


@dataclass
class BaseConfig:
    sampler: BaseSamplerConfig = BaseSamplerConfig()
    diffusion: ModelConfig = ModelConfig()
    likelihood: LikelihoodConfig = LikelihoodConfig()
    model_dir: str = 'diffusion_models/'
    model_name: str = ''
    example: ExampleConfig = GaussianExampleConfig()


@dataclass
class TrainConfig(BaseConfig):
    batch_size: int = 1024
    lr: float = 0.0001
    classifier_lr: float = 0.0001
    no_wandb: bool = 1
    delete_local_model: bool = False
    max_gradient: float = 1.
    loss_fn: str = 'l2'
    p_uncond: float = 1.


def get_path(cfg: TrainConfig, model_name):
    return "{}/{}".format(
        cfg.model_dir,
        model_name,
    )

def get_model_path(cfg: TrainConfig):
    if cfg.model_name:
        model_name = cfg.model_name
    else:
        sampler_name = OmegaConf.to_object(cfg.sampler).name()
        diffusion_name = OmegaConf.to_object(cfg.diffusion).name()
        model_name = "{}_{}".format(sampler_name, diffusion_name)
    return get_path(cfg, model_name)

def get_classifier_path(cfg: TrainConfig):
    if cfg.likelihood.classifier_name:
        model_name = cfg.likelihood.classifier_name
    else:
        sampler_name = OmegaConf.to_object(cfg.sampler).name()
        diffusion_name = OmegaConf.to_object(cfg.diffusion).name()
        model_name = "{}_{}_classifier".format(sampler_name, diffusion_name)
    return get_path(cfg, model_name)


class TestType(Enum):
    Gaussian = 'gaussian'
    BrownianMotion = 'brownian_motion'
    BrownianMotionDiff = 'brownian_motion_diff'
    Uniform = 'uniform'
    Test = 'test'


class IntegratorType(Enum):
    ProbabilityFlow = 'probability_flow'
    EulerMaruyama = 'euler_maruyama'


@dataclass
class SampleConfig(BaseConfig):
    num_samples: int = 10
    cond: Optional[float] = None
    guidance: GuidanceType = GuidanceType.Classifier
    test: TestType = TestType.Test
    integrator_type: IntegratorType = IntegratorType.ProbabilityFlow
