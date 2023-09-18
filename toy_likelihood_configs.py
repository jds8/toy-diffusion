#!/usr/bin/env python3

from dataclasses import dataclass
from enum import Enum


class DistanceFunction(Enum):
    Curve = 'curve'
    Linear = 'linear'
    Traj = 'traj'
    FinalState = 'final_state'


@dataclass
class LikelihoodConfig:
    _target_: str = ''


@dataclass
class DistLikelihoodConfig(LikelihoodConfig):
    dist_fun_type: DistanceFunction = DistanceFunction.FinalState
    sigma: float = 0.3
    symmetric_llk_condition: bool = True
    _target_: str = 'toy_likelihoods.DistLikelihood'
