#!/usr/bin/env python3

import torch
import numpy as np
from toy_likelihood_configs import DistanceFunction, LikelihoodCondition

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


class MultivariateGaussianTailsLikelihood(Likelihood):
    def __init__(
            self,
            alpha: float,
            which_condition: LikelihoodCondition,
            gamma: float,
            sampler=None,
    ):
        super().__init__(alpha)
        self.sigmas = torch.linspace(1., 0.001, 1000)
        self.dist = torch.distributions.Normal(alpha, self.sigmas)
        self.gamma = gamma
        self.which_condition = which_condition
        self.sampler = sampler

    def set_alpha(self, alpha: torch.Tensor):
        self.alpha = alpha[:, 0]

    def get_hard_condition(self, _, x0):
        return (x0.norm(dim=1) > self.alpha).to(torch.float)

    def get_soft_condition(self, _, x0):
        x0_norm = x0.norm(dim=1)
        soft_cond = torch.sigmoid(self.gamma * (x0_norm - self.alpha))
        return soft_cond

    def get_condition(self, x0_raw, x0):
        if self.which_condition == LikelihoodCondition.Hard:
            return self.get_hard_condition(x0_raw, x0)
        elif self.which_condition == LikelihoodCondition.Soft:
            return self.get_soft_condition(x0_raw, x0)
        else:
            raise NotImplementedError

    # TODO: Remove
    def gaussian_approx_grad_log_lik(self, xt, t):
        # approximates the score function of the indicator,
        # which is a Dirac delta, as a sequence of Gaussians
        idx = (t <= self.sigmas).nonzero().max()
        xt_norm = xt.norm(dim=[1, 2])
        approx = self.dist.log_prob(xt_norm).exp()[idx]
        return approx

    def sigmoid_approx_grad_log_lik(self, xt, t):
        # approximates the indicator function as a sigmoid,
        # so the score function approximation is the gradient
        # of a LogSigmoid
        xt_norm = xt.norm(dim=[1, 2])
        # gammas = torch.linspace(1., 10., 1000)
        # idx = (t <= self.sigmas).nonzero().max()
        # gamma = gammas[idx]
        gamma = 0.8
        llk = torch.nn.LogSigmoid()(gamma * (xt_norm - self.alpha)).sum()
        approx = torch.autograd.grad(llk, xt)[0]
        assert approx.shape == xt.shape
        return approx

    # TODO: Remove
    def gaussian_grad_log_lik(self, xt: torch.Tensor, t: float):
        """
        score function of likelihood
        p(y_1, y_2|x_1 + x_2) = N(y_1, y_2; mu=x_1 + x_2, Sigma=2*torch.eye(2))
        and we observe that y_1 = 3, y_2 = 3
        Y = X_0 + 2W, so
        E[Y] = E[X_0]+ 2E[W] = 0 + 0 = 0
        Var[Y] = Var[X_0] + Var[2W] = I + 4I since X_0 and W are iid N(0, I)
        let alpha = log_mean_coeff.exp(), beta = std with alpha ** 2 + beta ** 2 = 1
        X_t = alpha * X_0 + beta * V, so
        E[X_t] = alpha * E[X_0] + beta * E[V] = 0 + 0 = 0
        Var[X_t] = alpha ** 2 * I + beta ** 2 * I since X_0 and V are iid N(0, I)
                 = I since alpha ** 2 + beta ** 2 = 1
        Cov[Y, X_t] = Cov[X_0 + 2W, alpha X_0 + beta V] = alpha Var[X_0] since X_0, W, V are iid
                    = alpha * I
        """
        y_obs = 3.
        y_var = 2.
        with torch.no_grad():
            _, log_mean_coeff, std = self.sampler.marginal_prob(
                xt,
                torch.tensor(t)
            )
            std_dev = std[0]
            alpha = log_mean_coeff.exp()[0]
            mu_x0 = torch.zeros(2, 1)
            sigma_x0 = torch.eye(2)
            # this means the observation y = 3 and
            # marginal mean of xt
            mu_xt = alpha * mu_x0
            # marginal covariance of xt
            # sigma_xt = (alpha ** 2 * sigma_x0 + std_dev ** 2) * torch.eye(2)
            sigma_xt = torch.eye(2)
            # marginal mean of y
            mu_y = torch.tensor([[0], [0]])
            sigma_y = sigma_x0 + y_var ** 2 * torch.eye(2)
            sigma_y_xt = alpha * sigma_x0
            sigma_xt_inv = sigma_xt.pinverse()
            sigma_y_given_xt = sigma_y - torch.matmul(
                sigma_y_xt,
                torch.matmul(
                    sigma_xt_inv,
                    sigma_y_xt.T
                )
            )
            mu_y_given_xt = mu_y + torch.matmul(
                torch.matmul(
                    sigma_y_xt,
                    sigma_xt_inv,
                ),
                xt - mu_xt
            )
            score = torch.matmul(sigma_y_given_xt.pinverse(), mu_y_given_xt - y_obs)
            return score

    def grad_log_lik(self, xt, t):
        return self.sigmoid_approx_grad_log_lik(xt, t)
        # return self.gaussian_grad_log_lik(xt, t)


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
