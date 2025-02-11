#!/usr/bin/env python3

import hydra

from dataclasses import dataclass
from enum import Enum
from models.toy_diffusion_models_config import BetaSchedule

class DistanceFunction(Enum):
    Curve = 'curve'
    Linear = 'linear'
    Traj = 'traj'
    FinalState = 'final_state'


@dataclass
class LikelihoodConfig:
    _target_: str = ''

    def name(self):
        return 'Lik'


@dataclass
class GaussianTailsLikelihoodConfig(LikelihoodConfig):
    alpha: float = 3.
    _target_: str = 'toy_likelihoods.GaussianTailsLikelihood'

    def name(self):
        return 'GTailsLik'


class LikelihoodCondition(Enum):
    Hard = 'hard'
    Soft = 'soft'


@dataclass
class MultivariateGaussianTailsLikelihoodConfig(LikelihoodConfig):
    alpha: float = 3.
    which_condition: LikelihoodCondition = LikelihoodCondition.Hard
    _target_: str = 'toy_likelihoods.MultivariateGaussianTailsLikelihood'

    def name(self):
        return 'MVGTailsLik{}'.format(
            self.which_condition.value.capitalize()
        )


@dataclass
class BrownianMotionDiffTailsLikelihoodConfig(LikelihoodConfig):
    alpha: float = 3.
    _target_: str = 'toy_likelihoods.BrownianMotionDiffTailsLikelihood'

    def name(self):
        return 'BMTailsLik'


@dataclass
class DistLikelihoodConfig(LikelihoodConfig):
    dist_fun_type: DistanceFunction = DistanceFunction.FinalState
    sigma: float = 0.3
    symmetric_llk_condition: bool = True
    _target_: str = 'toy_likelihoods.DistLikelihood'


@dataclass
class GeneralDistLikelihoodConfig(LikelihoodConfig):
    beta_schedule: BetaSchedule = '${sampler.beta_schedule}'
    timesteps: int = '${sampler.diffusion_timesteps}'
    dist_fun_type: DistanceFunction = DistanceFunction.Curve
    _target_: str = 'toy_likelihoods.GeneralDistLikelihood'


@dataclass
class RLAILikelihoodConfig(LikelihoodConfig):
    dist_fun_type: DistanceFunction = DistanceFunction.FinalState
    _target_: str = 'toy_likelihoods.RLAILikelihood'


@dataclass
class ClassifierLikelihoodConfig(LikelihoodConfig):
    classifier_name: str = ''
    cond_dim: int = 1
    num_classes: int = 2
    _target_: str = 'toy_likelihoods.ClassifierLikelihood'
