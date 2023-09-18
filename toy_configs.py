#!/usr/bin/env python3

from hydra.core.config_store import ConfigStore
from models.toy_diffusion_models_config import EpsilonSamplerConfig, MuSamplerConfig, \
    XstartSamplerConfig, ScoreFunctionSamplerConfig, VelocitySamplerConfig, TemporalUnetConfig, \
    TemporalTransformerUnetConfig
from toy_likelihood_configs import DistLikelihoodConfig


def register_configs() -> None:
    cs = ConfigStore.instance()
    cs.store(
        group='sampler',
        name='base_epsilon_sampler',
        node=EpsilonSamplerConfig,
    )
    cs.store(
        group='sampler',
        name='base_mu_sampler',
        node=MuSamplerConfig,
    )
    cs.store(
        group='sampler',
        name='base_xstart_sampler',
        node=XstartSamplerConfig,
    )
    cs.store(
        group='sampler',
        name='base_score_function_sampler',
        node=ScoreFunctionSamplerConfig,
    )
    cs.store(
        group='sampler',
        name='base_velocity_sampler',
        node=VelocitySamplerConfig,
    )
    cs.store(
        group='diffusion',
        name='base_temporal_unet',
        node=TemporalUnetConfig,
    )
    cs.store(
        group='diffusion',
        name='temporal_transformer_unet',
        node=TemporalTransformerUnetConfig,
    )
    cs.store(
        group="likelihood",
        name="base_likelihood",
        node=DistLikelihoodConfig,
    )
