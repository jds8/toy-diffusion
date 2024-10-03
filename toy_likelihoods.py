#!/usr/bin/env python3

import torch
import numpy as np
from toy_likelihood_configs import DistanceFunction

from models.toy_temporal import NewTemporalClassifier
from models.toy_sampler import get_beta_schedule

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def final_state_dist(traj: torch.Tensor) -> torch.Tensor:
    return traj[..., -1, :].sum(-1)

def final_state_dist_from_cond(traj: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
    return ((traj[..., -1, :] - cond) ** 2).sqrt()

# def linear_dist(traj: torch.Tensor) -> torch.Tensor:
#     return ( (traj[..., 0, :2] - traj[..., -1, :2]) ** 2 ).sum(dim=-1).sqrt()

def traj_dist(traj: torch.Tensor, cond_traj: torch.Tensor):
    return ( (traj[..., :2] - cond_traj[..., :2]) ** 2 ).sum(dim=(-1, -2)).sqrt()

def curve_dist(traj: torch.Tensor) -> torch.Tensor:
    curve_traj_file = 'condition_traj.csv'
    curve_traj = np.loadtxt(curve_traj_file)
    curve_tensor = torch.tensor(curve_traj, dtype=traj.dtype, device=traj.device)[:, :2].diff(dim=-2)
    return traj_dist(traj, curve_tensor)


class Likelihood:
    def __init__(self, alpha: float):
        self.alpha = torch.tensor(alpha)

    def set_alpha(self, alpha: torch.Tensor):
        self.alpha = alpha

    def grad_log_lik(self, y, x, wrt):
        raise NotImplementedError

    def get_rarity(self, _, x0):
        x0 = x0 if x0 is not None else torch.zeros_like(x0)
        return x0.abs().max(dim=1).values


class GaussianTailsLikelihood(Likelihood):
    def get_condition(self, _, x0):
        # x0 has mean 0 std 1
        return (x0.abs() > self.alpha).to(torch.float)


class BrownianMotionDiffTailsLikelihood(Likelihood):
    def get_condition(self, x0_raw, _):
        # x0_raw is the brownian motion trajectory
        return (x0_raw.abs() > self.alpha).any(dim=1, keepdim=True).float()

    def get_rarity(self, x0_raw, _):
        x0_raw = x0_raw if x0_raw is not None else torch.zeros_like(x0_raw)
        return x0_raw.abs().max(dim=1).values


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


class GeneralDistLikelihood(Likelihood):
    def __init__(self, beta_schedule, timesteps, dist_fun_type):
        self.beta_schedule = beta_schedule
        beta_schedule_fn = get_beta_schedule(beta_schedule)
        self.timesteps = timesteps
        self.betas = beta_schedule_fn(timesteps)
        self.betas = torch.ones_like(self.betas) * 0.2
        self.cond = self.get_rare_traj()

    def condition(self, *x):
        mu, t = x
        if mu.isnan().any():
            import pdb; pdb.set_trace()
        normals = torch.distributions.Normal(mu.squeeze(-1), self.betas[t])
        return normals

    def get_rare_traj(self):
        rare_traj_file = 'rare_traj.pt'
        rare_traj = torch.load(rare_traj_file).to(device)
        return rare_traj.diff(dim=-1)

    def grad_log_lik(self, _, wrt, *x):
        # computes gradient of log p(y|x) with respect to wrt
        normals = self.condition(*x)
        log_probs = normals.log_prob(self.cond)
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


class ClassifierLikelihood(Likelihood):
    def __init__(self, classifier_name, cond_dim, num_classes):
        self.classifier = NewTemporalClassifier(
            traj_length=1000,
            d_model=torch.tensor(1),
            cond_dim=cond_dim,
            num_classes=num_classes,
        ).to(device)
        self.classifier.load_state_dict(torch.load('{}'.format(classifier_name)))

    def get_condition(self, *x):
        return torch.nn.Softmax(dim=-1)(self.classifier(*x)).argmax(dim=-1)

    def grad_log_lik(self, y, wrt, *x):
        predicted_unnormalized_logits = self.classifier(*x)
        probs = torch.nn.Softmax(dim=-1)(predicted_unnormalized_logits)
        cats = torch.distributions.Categorical(probs)
        log_probs = cats.log_prob(y)
        sum_grad = torch.autograd.grad(log_probs.sum(), wrt, retain_graph=True)[0]
        return sum_grad
