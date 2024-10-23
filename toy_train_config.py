#!/usr/bin/env python3

from enum import Enum
from typing import Optional
from dataclasses import dataclass, field
from models.toy_diffusion_models_config import BaseSamplerConfig, ModelConfig, GuidanceType
from toy_likelihood_configs import LikelihoodConfig
from importance_sampling import \
    GaussianTarget, GaussianProposal, \
    BrownianMotionDiffTarget, BrownianMotionDiffProposal, \
    StudentTTarget, StudentTProposal \

import torch
from omegaconf import OmegaConf


@dataclass
class ExampleConfig:
    def name(self):
        return 'ExampleConfig'


@dataclass
class GaussianExampleConfig(ExampleConfig):
    # mu: list = field(default_factory=lambda: [0., 0.])
    # sigma: list = field(default_factory=lambda: [[1., 0.], [0., 1.]])
    mu: float = 1.
    sigma: float = 2.

    def name(self):
        return 'GaussianExampleConfig'


@dataclass
class BrownianMotionExampleConfig(ExampleConfig):
    sde_drift: float = 0.
    sde_diffusion: float = 1.
    sde_steps: int = 104

    def name(self):
        return 'BrownianMotionExampleConfig'


@dataclass
class BrownianMotionDiffExampleConfig(BrownianMotionExampleConfig):
    def name(self):
        return 'BrownianMotionDiffExampleConfig'


@dataclass
class UniformExampleConfig(ExampleConfig):
    lower: float = -1.
    upper: float = 1.

    def name(self):
        return 'UniformExampleConfig'


@dataclass
class StudentTExampleConfig(ExampleConfig):
    nu: float = 3.

    def name(self):
        return 'StudentTExampleConfig'


@dataclass
class StudentTDiffExampleConfig(ExampleConfig):
    nu: float = 2.1
    sde_steps: int = 104

    def name(self):
        return 'StudentTDiffExampleConfig'


@dataclass
class BaseConfig:
    sampler: BaseSamplerConfig = field(default_factory=BaseSamplerConfig)
    diffusion: ModelConfig = field(default_factory=ModelConfig)
    likelihood: LikelihoodConfig = field(default_factory=LikelihoodConfig)
    model_dir: str = 'diffusion_models/'
    model_name: str = ''
    example: ExampleConfig = field(default_factory=ExampleConfig)


class SaveParadigm(Enum):
    Iterations = 'iterations'
    Epochs = 'epochs'
    TrainingSamples = 'training_samples'


@dataclass
class TrainConfig(BaseConfig):
    batch_size: int = 1024
    lr: float = 0.001
    classifier_lr: float = 0.0001
    no_wandb: bool = 1
    delete_local_model: bool = False
    max_gradient: float = 1.
    loss_fn: str = 'l2'
    p_uncond: float = 1.
    iterations_before_save: int = 1000
    upsample: bool = False
    max_alpha: float = 5.
    use_fixed_dataset: bool = False
    epochs_before_save: int = 1
    save_paradigm: SaveParadigm = SaveParadigm.TrainingSamples
    training_samples_before_save: int = 100000
    last_training_sample: int = -1


def get_path(cfg: TrainConfig, model_name):
    return "{}/{}".format(
        cfg.model_dir,
        model_name,
    )

def get_model_path(cfg: TrainConfig, dim: int):
    if cfg.model_name:
        model_name = cfg.model_name
    else:
        cfg_obj = OmegaConf.to_object(cfg)
        sampler_name = cfg_obj.sampler.name()
        diffusion_name = cfg_obj.diffusion.name()
        example_name = cfg_obj.example.name()
        model_name = "{}_{}_dim_{}_{}".format(
            sampler_name,
            diffusion_name,
            dim,
            example_name,
        )
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
    num_samples: int = 100
    cond: Optional[float] = None
    guidance: GuidanceType = GuidanceType.NoGuidance
    test: TestType = TestType.Test
    integrator_type: IntegratorType = IntegratorType.ProbabilityFlow


@dataclass
class SMCSampleConfig(SampleConfig):
    time_steps: int = 5


@dataclass
class ISConfig(SampleConfig):
    likelihood: LikelihoodConfig = field(default_factory=LikelihoodConfig)
    example: ExampleConfig = field(default_factory=ExampleConfig)
    num_rounds: int = 1
    start_round: int = 0
    split_size: int = 10000

    def num_splits(self):
        num_full_splits = self.num_samples // self.split_size
        num_leftover = self.num_samples % self.split_size
        return num_full_splits, num_leftover


def get_target(cfg):
    if isinstance(cfg.example, GaussianExampleConfig):
        return GaussianTarget(cfg)
    elif isinstance(cfg.example, BrownianMotionDiffExampleConfig):
        return BrownianMotionDiffTarget(cfg)
    elif isinstance(cfg.example, StudentTExampleConfig):
        return StudentTTarget(cfg)
    else:
        raise NotImplementedError

def get_proposal(example, std):
    if isinstance(example, GaussianExampleConfig):
        return GaussianProposal(std)
    elif isinstance(example, BrownianMotionDiffExampleConfig):
        return BrownianMotionDiffProposal(std)
    elif isinstance(example, StudentTExampleConfig):
        return StudentTProposal(std)
    else:
        raise NotImplementedError
