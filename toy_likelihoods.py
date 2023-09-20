#!/usr/bin/env python3

import torch
import numpy as np
from toy_likelihood_configs import DistanceFunction

def final_state_dist(traj: torch.Tensor) -> torch.Tensor:
    return traj[..., -1, :].sum(-1)

def final_state_dist_from_cond(traj: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
    return ((traj[..., -1, :] - cond) ** 2).sqrt()

# def linear_dist(traj: torch.Tensor) -> torch.Tensor:
#     return ( (traj[..., 0, :2] - traj[..., -1, :2]) ** 2 ).sum(dim=-1).sqrt()

# def traj_dist(traj: torch.Tensor, cond_traj: torch.Tensor):
#     return ( (traj[..., :2] - cond_traj[..., :2]) ** 2 ).sum(dim=(-1, -2)).sqrt()

# def curve_dist(traj: torch.Tensor, traj_mean: torch.Tensor, traj_std: torch.Tensor) -> torch.Tensor:
#     curve_traj_file = 'speed_mega_trajectory_cluster_16.csv'
#     curve_traj = np.loadtxt(curve_traj_file)
#     curve_tensor = torch.tensor(curve_traj, dtype=traj.dtype, device=traj.device)[:, :2].diff(dim=-2)
#     curve_tensor = (curve_tensor - traj_mean[..., :2]) / traj_std[..., :2]
#     return traj_dist(traj, curve_tensor)


class Likelihood:
    def grad_log_lik(self, y, x, wrt):
        raise NotImplementedError


class DistLikelihood(Likelihood):
    def __init__(self, dist_fun_type, sigma, symmetric_llk_condition):
        self.dist_fun_type = dist_fun_type
        self.dist_fun = self.get_dist_fun_from_type(dist_fun_type)
        self.sigma = sigma
        self.symmetric_llk_condition = symmetric_llk_condition

    def get_dist_fun_from_type(self, dist_fun_type):
        if dist_fun_type == DistanceFunction.FinalState:
            return final_state_dist
        else:
            raise NotImplementedError

    def condition(self, *x):
        mu = self.dist_fun(*x)
        normals = torch.distributions.Normal(mu, self.sigma)
        return normals

    def get_condition(self, *x):
        return self.condition(*x).mean

    def grad_log_lik(self, y, wrt, *x):
        # computes gradient of log p(y|x) with respect to wrt
        if y is None:
            return torch.tensor(0.)
        normals = self.condition(*x)
        if self.symmetric_llk_condition:
            y = normals.loc.sign() * torch.abs(y)
        log_probs = normals.log_prob(y)
        sum_grad = torch.autograd.grad(log_probs.sum(), wrt, retain_graph=True)[0]
        return sum_grad


class RLAILikelihood(Likelihood):
    def __init__(self, dist_fun_type):
        self.dist_fun_type = dist_fun_type
        self.dist_fun = self.get_dist_fun_from_type(dist_fun_type)

    def get_dist_fun_from_type(self, dist_fun_type):
        if dist_fun_type == DistanceFunction.FinalState:
            return final_state_dist
        else:
            raise NotImplementedError

    def get_condition(self, *x):
        return self.dist_fun(*x)

    def grad_log_lik(self, y, wrt, *x):
        # computes gradient of log p(y|x) with respect to wrt
        if y is None:
            return torch.tensor(0.)

        r = final_state_dist_from_cond(*x, y)
        bernoullis = torch.distributions.Bernoulli(torch.exp(-r))
        log_probs = bernoullis.log_prob(torch.tensor(1., device=y.device))
        sum_grad = torch.autograd.grad(log_probs.sum(), wrt, retain_graph=True)[0]
        return sum_grad
