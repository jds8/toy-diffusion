#!/usr/bin/env python3

from enum import Enum

from dataclasses import dataclass


class BetaSchedule(Enum):
    LinearSchedule = 'linear'
    CosineSchedule = 'cosine'
    QuadraticSchedule = 'quadratic'
    SigmoidSchedule = 'sigmoid'


class SamplerType(Enum):
    EpsilonSamplerType = 'epsilon'
    MuSamplerType = 'mu'
    XstartSamplerType = 'xstart'
    ScoreFunctionSamplerType = 'score_function'
    VelocitySamplerType = 'velocity'


@dataclass
class BaseSamplerConfig:

    def name(self):
        return 'BaseSampler'


@dataclass
class SamplerConfig(BaseSamplerConfig):
    beta_schedule: BetaSchedule = BetaSchedule.CosineSchedule
    diffusion_timesteps: int = 1000
    guidance_coef: float = 1.

    def name(self):
        return 'Sampler'


@dataclass
class EpsilonSamplerConfig(SamplerConfig):
    _target_: str = 'models.sampler.EpsilonSampler'

    def name(self):
        return 'EpsilonSampler'


@dataclass
class MuSamplerConfig(SamplerConfig):
    _target_: str = 'models.sampler.MuSampler'

    def name(self):
        return 'MuSampler'


@dataclass
class XstartSamplerConfig(SamplerConfig):
    _target_: str = 'models.toy_sampler.XstartSampler'

    def name(self):
        return 'XstartSampler'


@dataclass
class ScoreFunctionSamplerConfig(SamplerConfig):
    _target_: str = 'models.toy_sampler.ScoreFunctionSampler'

    def name(self):
        return 'ScoreFunctionSampler'


@dataclass
class VelocitySamplerConfig(SamplerConfig):
    _target_: str = 'models.toy_sampler.VelocitySampler'

    def name(self):
        return 'VelocitySampler'


class GuidanceType(Enum):
    Classifier = 'classifier'
    ClassifierFree = 'classifier_free'


@dataclass
class ModelConfig:
    traj_length: int = 40


@dataclass
class TemporalUnetConfig(ModelConfig):
    cond_dim: int = 1
    _target_: str = 'models.toy_temporal.TemporalUnet'

    def name(self):
        return 'TemporalUnet'


@dataclass
class TemporalTransformerUnetConfig(ModelConfig):
    cond_dim: int = 1
    _target_: str = 'models.toy_temporal.TemporalTransformerUnet'

    def name(self):
        return 'TemporalTransformerUnet'
