#!/usr/bin/env python3

from hydra.core.config_store import ConfigStore
from models.toy_diffusion_models_config import \
    VESDEEpsilonSamplerConfig, \
    VESDEVelocitySamplerConfig, \
    VESDEScoreFunctionSamplerConfig, \
    VPSDEEpsilonSamplerConfig, \
    VPSDEVelocitySamplerConfig, \
    VPSDEScoreFunctionSamplerConfig, \
    VPSDEGaussianScoreFunctionSamplerConfig, \
    EpsilonSamplerConfig, MuSamplerConfig, \
    XstartSamplerConfig, ScoreFunctionSamplerConfig, VelocitySamplerConfig, \
    TemporalUnetConfig, TemporalNnetConfig, TemporalTransformerUnetConfig, TemporalIDKConfig
from toy_likelihood_configs import DistLikelihoodConfig, GeneralDistLikelihoodConfig, \
    RLAILikelihoodConfig, ClassifierLikelihoodConfig, GaussianTailsLikelihoodConfig, \
    BrownianMotionDiffTailsLikelihoodConfig
from toy_train_config import GaussianExampleConfig, \
    BrownianMotionDiffExampleConfig, BrownianMotionExampleConfig, \
    UniformExampleConfig


def register_configs() -> None:
    cs = ConfigStore.instance()
    cs.store(
        group='sampler',
        name='vesde_epsilon_sampler',
        node=VESDEEpsilonSamplerConfig,
    )
    cs.store(
        group='sampler',
        name='vesde_velocity_sampler',
        node=VESDEVelocitySamplerConfig,
    )
    cs.store(
        group='sampler',
        name='vesde_score_function_sampler',
        node=VESDEScoreFunctionSamplerConfig,
    )
    cs.store(
        group='sampler',
        name='vpsde_epsilon_sampler',
        node=VPSDEEpsilonSamplerConfig,
    )
    cs.store(
        group='sampler',
        name='vpsde_velocity_sampler',
        node=VPSDEVelocitySamplerConfig,
    )
    cs.store(
        group='sampler',
        name='vpsde_score_function_sampler',
        node=VPSDEScoreFunctionSamplerConfig,
    )
    cs.store(
        group='sampler',
        name='vpsde_gaussian_score_function_sampler',
        node=VPSDEGaussianScoreFunctionSamplerConfig,
    )
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
        name='temporal_n_net',
        node=TemporalNnetConfig,
    )
    cs.store(
        group='diffusion',
        name='temporal_transformer_unet',
        node=TemporalTransformerUnetConfig,
    )
    cs.store(
        group='diffusion',
        name='temporal_idk',
        node=TemporalIDKConfig,
    )
    cs.store(
        group="likelihood",
        name="base_likelihood",
        node=DistLikelihoodConfig,
    )
    cs.store(
        group="likelihood",
        name="gaussian_tails_likelihood",
        node=GaussianTailsLikelihoodConfig,
    )
    cs.store(
        group="likelihood",
        name="brownian_motion_diff_tails_likelihood",
        node=BrownianMotionDiffTailsLikelihoodConfig,
    )
    cs.store(
        group='likelihood',
        name='general_dist_likelihood',
        node=GeneralDistLikelihoodConfig,
    )
    cs.store(
        group="likelihood",
        name="rlai_likelihood",
        node=RLAILikelihoodConfig,
    )
    cs.store(
        group='likelihood',
        name='classifier_likelihood',
        node=ClassifierLikelihoodConfig,
    )
    cs.store(
        group='example',
        name='gaussian_example',
        node=GaussianExampleConfig,
    )
    cs.store(
        group='example',
        name='brownian_motion_diff_example',
        node=BrownianMotionDiffExampleConfig,
    )
    cs.store(
        group='example',
        name='brownian_motion_example',
        node=BrownianMotionExampleConfig,
    )
    cs.store(
        group='example',
        name='uniform_example',
        node=UniformExampleConfig,
    )
