#!/usr/bin/env python3

from enum import Enum

from dataclasses import dataclass


class BetaSchedule(Enum):
    LinearSchedule = 'linear'
    CosineSchedule = 'cosine'
    QuadraticSchedule = 'quadratic'
    SigmoidSchedule = 'sigmoid'


class SamplerType(Enum):
    VPSDEEpsilonSamplerType = 'vpsde_epsilon'
    VPSDEVelocitySamplerType = 'vpsde_velocity'
    EpsilonSamplerType = 'epsilon'
    MuSamplerType = 'mu'
    XstartSamplerType = 'xstart'
    ScoreFunctionSamplerType = 'score_function'
    VelocitySamplerType = 'velocity'


@dataclass
class BaseSamplerConfig:
    diffusion_timesteps: int = 1000
    guidance_coef: float = 1.

    def name(self):
        return 'BaseSampler'


@dataclass
class DiscreteSamplerConfig(BaseSamplerConfig):
    beta_schedule: BetaSchedule = BetaSchedule.CosineSchedule

    def name(self):
        return 'DiscreteSampler'


@dataclass
class VPSDESamplerConfig(BaseSamplerConfig):
    beta0: float = 0.1
    beta1: float = 20.
    t_eps: float = 1e-5

    def name(self):
        return 'VPSDESampler'


@dataclass
class VPSDEEpsilonSamplerConfig(VPSDESamplerConfig):
    _target_: str = 'models.toy_sampler.VPSDEEpsilonSampler'

    def name(self):
        return 'VPSDEEpsilonSampler'


@dataclass
class VPSDEVelocitySamplerConfig(VPSDESamplerConfig):
    _target_: str = 'models.toy_sampler.VPSDEVelocitySampler'

    def name(self):
        return 'VPSDEVelocitySampler'


@dataclass
class VPSDEScoreFunctionSamplerConfig(VPSDESamplerConfig):
    _target_: str = 'models.toy_sampler.VPSDEScoreFunctionSampler'

    def name(self):
        return 'VPSDEScoreFunctionSampler'


@dataclass
class VPSDEGaussianScoreFunctionSamplerConfig(VPSDESamplerConfig):
    _target_: str = 'models.toy_sampler.VPSDEGaussianScoreFunctionSampler'

    def name(self):
        return 'VPSDEGaussianScoreFunctionSampler'


@dataclass
class EpsilonSamplerConfig(DiscreteSamplerConfig):
    _target_: str = 'models.toy_sampler.EpsilonSampler'

    def name(self):
        return 'EpsilonSampler'


@dataclass
class MuSamplerConfig(DiscreteSamplerConfig):
    _target_: str = 'models.toy_sampler.MuSampler'

    def name(self):
        return 'MuSampler'


@dataclass
class XstartSamplerConfig(DiscreteSamplerConfig):
    _target_: str = 'models.toy_sampler.XstartSampler'

    def name(self):
        return 'XstartSampler'


@dataclass
class ScoreFunctionSamplerConfig(DiscreteSamplerConfig):
    _target_: str = 'models.toy_sampler.ScoreFunctionSampler'

    def name(self):
        return 'ScoreFunctionSampler'


@dataclass
class VelocitySamplerConfig(DiscreteSamplerConfig):
    _target_: str = 'models.toy_sampler.VelocitySampler'

    def name(self):
        return 'VelocitySampler'


class GuidanceType(Enum):
    Classifier = 'classifier'
    ClassifierFree = 'classifier_free'


@dataclass
class ModelConfig:

    def name(self):
        return 'ModelConfig'


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
