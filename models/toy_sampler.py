#!/usr/bin/env python3

from collections import namedtuple

import torch
import torch.distributions as dist
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
from models.toy_diffusion_models_config import BetaSchedule

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.9):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return torch.tensor(betas, device=device)

# def cosine_beta_schedule(timesteps, s=0.008):
#     """
#     cosine schedule
#     as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
#     """
#     steps = timesteps + 1
#     x = np.linspace(0, steps, steps)
#     alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
#     alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
#     betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
#     betas_clipped = np.clip(betas, a_min=0, a_max=0.999)
#     return torch.tensor(betas_clipped, dtype=torch.float32, device=device)

def cosine_beta_schedule(timesteps):
    """
    Get a pre-defined beta schedule for the given name.
    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.
    """
    return betas_for_alpha_bar(
        timesteps,
        lambda t: torch.cos(torch.tensor(t + 0.008) / 1.008 * torch.pi / 2) ** 2,
    )

def linear_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)

def quadratic_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start**0.5, beta_end**0.5, timesteps) ** 2

def sigmoid_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    betas = torch.linspace(-6, 6, timesteps)
    return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start

def get_beta_schedule(beta_schedule: BetaSchedule):
    if beta_schedule == BetaSchedule.LinearSchedule:
        return linear_beta_schedule
    elif beta_schedule == BetaSchedule.CosineSchedule:
        return cosine_beta_schedule
    elif beta_schedule == BetaSchedule.QuadraticSchedule:
        return quadratic_beta_schedule
    elif beta_schedule == BetaSchedule.SigmoidSchedule:
        return sigmoid_beta_schedule
    else:
        raise NotImplementedError

def interpolate_schedule(diffusion_time, schedule, decreasing=False):
    floor = torch.floor(diffusion_time)
    ceil = torch.ceil(diffusion_time)
    ceil_idx = torch.where(
        ceil < len(schedule),
        ceil.to(torch.long),
        torch.tensor(-1, device=device)
    )
    ceil_beta = schedule[ceil_idx]
    floor_idx = torch.where(
        floor < len(schedule),
        floor.to(torch.long),
        torch.tensor(-1, device=device)
    )
    floor_beta = schedule[floor_idx]
    if decreasing:
        beta = ceil_beta + (diffusion_time - floor) * (floor_beta - ceil_beta)
    else:
        beta = floor_beta + (diffusion_time - floor) * (ceil_beta - floor_beta)
    return beta

def cosine_beta_schedule(timesteps):
    """
    Get a pre-defined beta schedule for the given name.
    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.
    """
    return betas_for_alpha_bar(
        timesteps,
        lambda t: torch.cos(torch.tensor(t + 0.008) / 1.008 * torch.pi / 2) ** 2,
    )

def linear_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)

def quadratic_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start**0.5, beta_end**0.5, timesteps) ** 2

def sigmoid_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    betas = torch.linspace(-6, 6, timesteps)
    return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start

def continuous_linear_integral(t, beta0, beta1):
    return beta0 * t + (1/2) * t**2 * (beta1 - beta0)
def continuous_quadratic_integral(t, beta0, beta1):
    return beta0 * t + (1/3) * t**3 * (beta1 - beta0)
def continuous_cosine_integral(t, beta0, beta1):
    # return beta0 + (1 - torch.cos(torch.pi * t / 2)) * (beta1 - beta0)
    return beta0 * t + (t - (2/torch.pi) * torch.sin(torch.pi * t / 2)) * (beta1 - beta0)
def continuous_sigmoid_integral(t, beta0, beta1, k, t0):
    """ k defines steepness, t0 defines midpoint """
    # return beta0 * t + (beta1 - beta0) / (1 + torch.exp(-k * (t - t0)))
    return beta0 * t + (beta1 - beta0) * torch.exp(-k * t0) * torch.log(torch.exp(k * t) + 1) / k

def get_continuous_beta_integral(beta_schedule: BetaSchedule):
    if beta_schedule == BetaSchedule.LinearSchedule:
        return continuous_linear_integral
    elif beta_schedule == BetaSchedule.CosineSchedule:
        return continuous_cosine_integral
    elif beta_schedule == BetaSchedule.QuadraticSchedule:
        return continuous_quadratic_integral
    elif beta_schedule == BetaSchedule.SigmoidSchedule:
        return lambda t, beta0, beta1: continuous_sigmoid_integral(
            t,
            beta0,
            beta1,
            torch.tensor(10.),
            torch.tensor(0.5)
        )
    else:
        raise NotImplementedError


ForwardSample = namedtuple('ForwardSample', 'xt t noise to_predict')


class AbstractSampler:
    def __init__(
        self,
        diffusion_timesteps: int,
        guidance_coef: float,
    ):
        self.diffusion_timesteps = diffusion_timesteps
        self.guidance_coef = guidance_coef

    def prior_logp(self, z):
        raise NotImplementedError

    def forward_sample(self, x_start, extras=None):
        raise NotImplementedError

    def reverse_sample(self, xt, t, conditional_mean):
        raise NotImplementedError

    def get_classifier_free_mean(self, xt, unconditional_output, t, conditional_output):
        raise NotImplementedError

    def classifier_free_reverse_sample(self, xt, unconditional_output, t, conditional_output):
        conditional_mean = self.get_classifier_free_mean(xt, unconditional_output, torch.tensor([t]), conditional_output)
        return self.reverse_sample(xt, t, conditional_mean)

    def combine_eps(self, unconditional_eps, conditional_eps):
        return (1 + self.guidance_coef) * conditional_eps - self.guidance_coef * unconditional_eps

    def get_sf_estimator(self, model_output, xt, t):
        raise NotImplementedError

    def get_ground_truth(self, eps, xt, x0, t, extras=None):
        raise NotImplementedError

#######################
# Continuous Samplers #
#######################


class AbstractContinuousSampler(AbstractSampler):
    def __init__(
        self,
        diffusion_timesteps: int,
        guidance_coef: float,
        t_eps: float,
    ):
        super().__init__(diffusion_timesteps, guidance_coef)
        self.t_eps = t_eps

    def reverse_sde(
            self,
            x: torch.Tensor,
            t: torch.Tensor,
            score: torch.Tensor,
            steps: torch.Tensor=torch.tensor(1000)
    ):
        dt = -1. / steps
        z = torch.randn_like(x)
        drift, diffusion = self.sde(x, t)
        x_mean = x + (drift - diffusion ** 2 * score) * dt
        x = x_mean + diffusion * torch.sqrt(-dt) * z
        return x, x_mean

    def probability_flow_ode(
            self,
            x: torch.Tensor,
            t: torch.Tensor,
            score: torch.Tensor
    ):
        drift, diffusion = self.sde(x, t)
        dx_dt = drift - 0.5 * diffusion ** 2 * score
        return dx_dt

    # forward diffusion (using the nice property)
    def forward_sample(self, x_start, extras=None):
        lower = torch.tensor(self.t_eps, device=x_start.device)
        upper = torch.tensor(1., device=x_start.device)
        t = torch.distributions.Uniform(lower, upper).sample([x_start.shape[0]])
        noise = torch.randn_like(x_start)

        mean, log_mean_coeff, std = self.marginal_prob(x=x_start, t=t)
        xt = mean + std * noise
        return ForwardSample(
            xt=xt,
            t=t,
            noise=noise,
            to_predict=self.get_ground_truth(
                eps=noise,
                xt=xt,
                x0=x_start,
                t=t,
                extras=extras
            )
        )

    def reverse_sample(self, xt, t, conditional_mean):
        '''
        reverse probability flow ODE sampler
        provenance: score_sde/sampling.py
        '''
        noise = torch.randn_like(xt)
        nonzero_mask = torch.tensor(t != 0, dtype=self.betas.dtype)  # no noise when t == 0
        posterior_variance = self.posterior_variance[t]
        var = nonzero_mask * posterior_variance.sqrt() * noise
        out = conditional_mean + var
        return out

    def prior_logp(self, z: torch.Tensor, device):
        return self.prior_sampling(device).log_prob(z)

    ##################
    # VPSDE Samplers #
    ##################


class VPSDESampler(AbstractContinuousSampler):
    # heavily inspired by score_sde/sde_lib.py from Song et al. Score-Based Generative Modeling Through SDE
    def __init__(
        self,
        diffusion_timesteps: int,
        guidance_coef: float,
        t_eps: float,
        beta_schedule: BetaSchedule,
        beta0: float,
        beta1: float
    ):
        super().__init__(diffusion_timesteps, guidance_coef, t_eps)
        self.beta_schedule = beta_schedule
        continuous_beta_integral = get_continuous_beta_integral(beta_schedule)
        self.beta0 = torch.tensor(beta0)
        self.beta1 = torch.tensor(beta1)
        self.continuous_beta_integral = lambda t: continuous_beta_integral(
            t,
            self.beta0,
            self.beta1,
        )

    def sde(self, x: torch.Tensor, t: torch.Tensor):
        # dx_t = -0.5*beta(t)*x*dt + g(t)*dw
        is_grad = torch.is_grad_enabled()
        torch.set_grad_enabled(True)
        t_ = t.requires_grad_()
        int_beta_t = self.continuous_beta_integral(t_)  # int_0^t beta(s) ds
        beta_t = torch.autograd.grad(int_beta_t.sum(), t_)[0]  # beta(t)
        torch.set_grad_enabled(is_grad)

        drift = -0.5 * beta_t * x
        diffusion = beta_t.sqrt()
        return drift, diffusion

    def log_mean_coeff(self, x_shape: torch.Size, t: torch.Tensor):
        beta = self.continuous_beta_integral(t)  # int_0^t beta(s) ds
        log_mean_coeff = -0.5 * beta
        return log_mean_coeff.repeat(x_shape[1:] + (1,)).movedim(2, 0)

    def marginal_prob(self, x: torch.Tensor, t: torch.Tensor):
        log_mean_coeff = self.log_mean_coeff(x_shape=x.shape, t=t)
        mean = log_mean_coeff.exp() * x
        std = (1 - (2. * log_mean_coeff).exp()).sqrt()
        return mean, log_mean_coeff, std

    def analytical_marginal_prob(self, t: torch.Tensor, example):
        mu = torch.tensor([[[example.mu]]])
        log_mean_coeff = self.log_mean_coeff(x_shape=mu.shape, t=t)
        mean = log_mean_coeff.exp() * mu
        first_var_term = example.sigma ** 2 * (2. * log_mean_coeff).exp()
        second_var_term = (1 - (2. * log_mean_coeff).exp())
        std = (first_var_term + second_var_term).sqrt()
        return mean, log_mean_coeff, std

    def prior_analytic_logp(self, example, device, latent):
        mean, _, std = self.analytical_marginal_prob(
            t=torch.tensor(1.),
            example=example,
        )
        return torch.distributions.Normal(mean, std, device).log_prob(latent)

    def prior_sampling(self, device):
        return torch.distributions.Normal(0., 1., device)

    def reverse_sample(self, xt, t, conditional_mean):
        '''
        reverse probability flow ODE sampler
        provenance: score_sde/sampling.py
        '''
        noise = torch.randn_like(xt)
        nonzero_mask = torch.tensor(t != 0, dtype=self.betas.dtype)  # no noise when t == 0
        posterior_variance = self.posterior_variance[t]
        var = nonzero_mask * posterior_variance.sqrt() * noise
        out = conditional_mean + var
        return out


class VPSDEEpsilonSampler(VPSDESampler):
    def get_ground_truth(self, eps, xt, x0, t, extras):
        return eps

    def get_classifier_free_mean(self, xt, unconditional_output, t, conditional_output):
        eps = self.combine_eps(unconditional_output, conditional_output)
        return self.get_posterior_mean(xt, eps, t)

    def get_sf_estimator(self, eps_pred, xt, t):
        _, _, sigma_t = self.marginal_prob(torch.zeros_like(xt), t)
        return -eps_pred / sigma_t.reshape((-1,) + (1,) * (len(eps_pred.shape) - 1))


class VPSDEVelocitySampler(VPSDESampler):
    def get_ground_truth(self, eps, xt, x0, t, extras):
        _, log_mean_coeff, sigma_t = self.marginal_prob(x=x0, t=t)
        return log_mean_coeff.exp() * eps - sigma_t * x0

    def get_classifier_free_sf_estimator(
        self,
        xt,
        unconditional_output,
        t,
        conditional_output
    ):
        _, log_mean_coeff, sigma_t = self.marginal_prob(x=xt, t=t)
        alpha_t = log_mean_coeff.exp()
        unconditional_eps = sigma_t * xt + alpha_t * unconditional_output
        conditional_eps = sigma_t * xt + alpha_t * conditional_output
        eps_pred = self.combine_eps(unconditional_eps, conditional_eps)
        return -eps_pred / sigma_t

    def get_sf_estimator(self, v_pred, xt, t):
        _, log_mean_coeff, sigma_t = self.marginal_prob(x=xt, t=t)
        alpha_t = log_mean_coeff.exp()
        eps_pred = sigma_t * xt + alpha_t * v_pred
        return -eps_pred / sigma_t


class VPSDEScoreFunctionSampler(VPSDESampler):
    def get_ground_truth(self, eps, xt, x0, t, extras):
        """
        Note that this returns the *conditional* score function:
        \nabla log p_t(x_t|x_0)
        where x_t = lmc.exp()x_0 + sigma_t * epsilon so
        E[x_t|x_0] = lmc.exp()x_0 and Var[x_t|x_0] = sigma_t^2(t)
        """
        mean, _, sigma_t = self.marginal_prob(x=x0, t=t)
        var = sigma_t ** 2
        score = (mean - xt) / var
        return score

    def get_sf_estimator(self, sf_pred, xt, t):
        return sf_pred


class VPSDEGaussianScoreFunctionSampler(VPSDESampler):
    def get_ground_truth(self, eps, xt, x0, t, extras):
        """
        Note that this returns the *marginal* score function:
        \nabla log p_t(x_t|x_0).
        This assumes that the *ground truth distribution is a standard gaussian*
        """
        _, log_mean_coeff, sigma_t = self.marginal_prob(x=x0, t=t)
        f = log_mean_coeff.exp()
        var = extras['sigma'] ** 2 * f ** 2 + sigma_t ** 2
        score = (f * extras['mu'] - xt) / var
        return score

    def get_sf_estimator(self, sf_pred, xt, t):
        _, _, sigma_t = self.marginal_prob(
            torch.zeros_like(xt),
            t,
        )
        sf_estimate = -sf_pred * sigma_t
        return sf_estimate

class VPSDEMultivariateGaussianScoreFunctionSampler(VPSDESampler):
    def get_ground_truth(self, eps, xt, x0, t, extras):
        """
        Note that this returns the *marginal* score function:
        \nabla log p_t(x_t|x_0).
        This assumes that the *ground truth distribution is a standard gaussian*
        """
        _, log_mean_coeff, sigma_t = self.marginal_prob(x=x0, t=t)
        f = log_mean_coeff.exp()
        d = len(extras['mu'])
        mu = torch.tensor(extras['mu']).reshape(-1, d)
        sigma = torch.tensor(extras['sigmas']).reshape(-1, d, d)
        var =  torch.matmul(sigma, sigma.T) * f ** 2 + sigma_t ** 2 * torch.eye(d)
        score = torch.matmul(f * mu - xt, var.pinverse())
        return score

    def get_sf_estimator(self, sf_pred, xt, t):
        _, _, sigma_t = self.marginal_prob(
            torch.zeros_like(xt),
            t,
        )
        sf_estimate = -sf_pred * sigma_t
        return sf_estimate

    ##################
    # VESDE Samplers #
    ##################


class VESDESampler(AbstractContinuousSampler):
    # heavily inspired by score_sde/sde_lib.py from Song et al. Score-Based Generative Modeling Through SDE
    def __init__(self, diffusion_timesteps: int, guidance_coef: float, t_eps: float, sigma_min: float, sigma_max: float):
        super().__init__(diffusion_timesteps, guidance_coef, t_eps)
        self.sigma_min = torch.tensor(sigma_min)
        self.sigma_max = torch.tensor(sigma_max)

    def sigma(self, t: torch.Tensor):
        return self.sigma_min * (self.sigma_max / self.sigma_min) ** t

    def sde(self, x: torch.Tensor, t: torch.Tensor):
        drift = torch.zeros_like(x)
        sigma = self.sigma(t)
        diffusion = sigma * torch.sqrt(2 * (torch.log(self.sigma_max) - torch.log(self.sigma_min)))
        diffusion = diffusion.repeat((t.reshape(-1).shape[0],) + x.shape[1:])
        return drift, diffusion

    def marginal_prob(self, x: torch.Tensor, t: torch.Tensor):
        # first two arguments correspond to mean and log_mean
        std = self.sigma(t)
        mean = x
        return mean, torch.zeros_like(x), std

    def prior_sampling(self, device):
        return torch.distributions.Normal(0., torch.tensor(self.sigma_max, device=device))


class VESDEEpsilonSampler(VESDESampler):
    def get_ground_truth(self, eps, xt, x0, t, extras):
        return eps

    def get_classifier_free_mean(self, xt, unconditional_output, t, conditional_output):
        eps = self.combine_eps(unconditional_output, conditional_output)
        return self.get_posterior_mean(xt, eps, t)

    def get_sf_estimator(self, eps_pred, xt, t):
        _, _, sigma_t = self.marginal_prob(torch.zeros_like(xt), t)
        return -eps_pred / sigma_t.reshape((-1,) + (1,) * (len(eps_pred.shape) - 1))


class VESDEVelocitySampler(VESDESampler):
    def get_ground_truth(self, eps, xt, x0, t, extras):
        _, log_mean_coeff, sigma_t = self.marginal_prob(x=x0, t=t)
        return log_mean_coeff.exp() * eps - sigma_t * x0

    def get_sf_estimator(self, v_pred, xt, t):
        _, log_mean_coeff, sigma_t = self.marginal_prob(x=xt, t=t)
        alpha_t = log_mean_coeff.exp()
        eps_pred = sigma_t * xt + alpha_t * v_pred
        return -eps_pred / sigma_t.reshape((-1,) + (1,) * (len(eps_pred.shape) - 1))


class VESDEScoreFunctionSampler(VESDESampler):
    def get_ground_truth(self, eps, xt, x0, t, extras):
        """
        Note that this returns the *conditional* score function:
        \nabla log p_t(x_t|x_0)
        """
        mean, log_mean_coeff, sigma_t = self.marginal_prob(x=x0, t=t)
        var = sigma_t ** 2
        score = (mean - xt) / var
        return score

    def get_sf_estimator(self, sf_pred, xt, t):
        return sf_pred


class VESDEGaussianScoreFunctionSampler(VESDESampler):
    def get_ground_truth(self, eps, xt, x0, t, extras):
        """
        Note that this returns the *conditional* score function:
        \nabla log p_t(x_t|x_0).
        This assumes that the *ground truth distribution is a standard gaussian*
        """
        _, log_mean_coeff, sigma_t = self.marginal_prob(x=x0, t=t)
        var = log_mean_coeff.exp() ** 2 + sigma_t ** 2
        score = -xt / var
        return score

    def get_sf_estimator(self, sf_pred, xt, t):
        return sf_pred


#####################
# Discrete Samplers #
#####################

class AbstractDiscreteSampler(AbstractSampler):
    def __init__(
        self,
        diffusion_timesteps: int,
        guidance_coef: float,
        beta_schedule: BetaSchedule,
    ):
        super().__init__(beta_schedule, diffusion_timesteps, guidance_coef)
        # define beta schedule
        beta_schedule_fn = get_beta_schedule(beta_schedule)
        self.betas = beta_schedule_fn(diffusion_timesteps)

        # define alphas
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)

    def extract(self, a, t, x_shape):
        batch_size = t.shape[0]
        out = a.gather(-1, t)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

    def bad_predict_xstart(self, xt, t, nsamples):
        shape = (nsamples,) + tuple(xt.shape)
        return (
            self.extract(self.sqrt_recip_alphas_cumprod, t, shape) * xt.repeat(nsamples, *(1,) * len(xt.shape))
            - self.extract(self.sqrt_recipm1_alphas_cumprod, t, shape) * torch.randn(shape)
        )

    # forward diffusion (using the nice property)
    def forward_sample(self, x_start, extras=None):
        t = dist.Categorical(
            torch.ones(
                self.diffusion_timesteps,
                device=x_start.device
            )
        ).sample([
            x_start.shape[0]
        ])
        noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = self.extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self.extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )
        xt = sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
        return ForwardSample(
            xt,
            t,
            noise,
            self.get_ground_truth(eps=noise, xt=xt, x0=x_start, t=t, extras=extras)
        )

    def reverse_sample(self, xt, t, conditional_mean):
        noise = torch.randn_like(xt)
        nonzero_mask = torch.tensor(t != 0, dtype=self.betas.dtype)  # no noise when t == 0
        posterior_variance = self.posterior_variance[t]
        var = nonzero_mask * posterior_variance.sqrt() * noise
        out = conditional_mean + var
        return out

    def get_posterior_mean(self, xt, val, t):
        raise NotImplementedError

    def classifier_guided_reverse_sample(self, xt, unconditional_output, t, grad_log_lik=0.):
        mean = self.get_posterior_mean(xt, unconditional_output, t=torch.tensor([t]))
        conditional_mean = mean + self.guidance_coef * self.posterior_variance[t] * grad_log_lik
        return self.reverse_sample(xt, t, conditional_mean)

    def prior_sampling(self, device):
        return torch.distributions.Normal(0, self.sqrt_one_minus_alphas_cumprod[-1])

    def prior_logp(self, z, device):
        return self.prior_sampling(device).log_prob(z)

###################
# PREDICTION TYPE #
###################

class EpsilonSampler(AbstractDiscreteSampler):
    def predict_xstart(self, xt, eps, t):
        return self.sqrt_recip_alphas_cumprod[t] * xt - self.sqrt_recipm1_alphas_cumprod[t] * eps

    def get_posterior_mean(self, xt, eps, t):
        return self.sqrt_recip_alphas[t] * (xt - (self.betas / self.sqrt_one_minus_alphas_cumprod)[t] * eps)

    def get_ground_truth(self, eps, xt, x0, t, extras=None):
        return eps

    def get_classifier_free_mean(self, xt, unconditional_output, t, conditional_output):
        eps = self.combine_eps(unconditional_output, conditional_output)
        return self.get_posterior_mean(xt, eps, t)

    def get_sf_estimator(self, eps_pred, xt, t):
        sigma_t = interpolate_schedule(t, self.sqrt_one_minus_alphas_cumprod, decreasing=True)
        return -eps_pred / sigma_t.reshape((-1,) + (1,) * (len(eps_pred.shape) - 1))


class MuSampler(AbstractDiscreteSampler):
    def get_posterior_mean(self, xt, mean, t):
        return mean

    def get_ground_truth(self, eps, xt, x0, t, extras=None):
        alphas = self.extract(self.alphas.sqrt(), t, xt.shape)
        alphas_cumprod_prev = self.extract(self.alphas_cumprod_prev, t, xt.shape)
        betas = self.extract(self.betas, t, xt.shape)
        alphas_cumprod = self.extract(self.alphas_cumprod, t, xt.shape)
        return (alphas.sqrt() * (1 - alphas_cumprod_prev) * xt + alphas_cumprod_prev.sqrt() * betas * x0) / (1 - alphas_cumprod)


class XstartSampler(AbstractDiscreteSampler):
    def set_schedule_from_snr(self, snr, snr_prev):
        self.alphas_cumprod = torch.fill_(torch.ones(self.diffusion_timesteps), torch.sigmoid(torch.log(snr)))
        self.alphas_cumprod_prev = torch.sigmoid(torch.log(snr_prev)) if snr_prev is not None else torch.tensor(1.0)
        self.alphas_cumprod_prev = torch.fill_(torch.ones(self.diffusion_timesteps), self.alphas_cumprod_prev)
        self.sqrt_alphas_cumprod = torch.fill_(torch.ones(self.diffusion_timesteps), torch.sqrt(alphas_cumprod))
        self.sqrt_one_minus_alphas_cumprod = torch.fill_(torch.ones(self.diffusion_timesteps), torch.sqrt(1. - alphas_cumprod))
        self.alphas = torch.fill_(torch.ones(self.diffusion_timesteps), self.alphas_cumprod / self.alphas_cumprod_prev)
        self.betas = 1 - self.alphas

    def get_posterior_mean(self, xt, xhat, t):
        return (self.alphas[t].sqrt() * (1 - self.alphas_cumprod_prev[t]) * xt + self.alphas_cumprod_prev[t].sqrt() * self.betas[t] * xhat) / (1 - self.alphas_cumprod[t])

    def get_ground_truth(self, eps, xt, x0, t, extras=None):
        return x0

    def get_classifier_free_mean(self, xt, unconditional_xhat, t, conditional_xhat):
        unconditional_eps = (xt - self.sqrt_alphas_cumprod * unconditional_xhat) / self.sqrt_one_minus_alphas_cumprod[t]
        conditional_eps = (xt - self.sqrt_alphas_cumprod * conditional_xhat) / self.sqrt_one_minus_alphas_cumprod[t]
        eps = self.combine_eps(unconditional_eps, conditional_eps)
        classifier_free_xt = (xt - self.sqrt_one_minus_alphas_cumprod[t] * eps) / self.sqrt_alphas_cumprod[t]
        return self.get_posterior_mean(xt, classifier_free_xt, t)


class ScoreFunctionSampler(AbstractDiscreteSampler):
    def get_posterior_mean(self, xt, score_fun_est, t):
        return self.sqrt_recip_alphas[t] * (xt + self.betas[t] * score_fun_est)

    def get_ground_truth(self, eps, xt, x0, t, extras=None):
        alphas_cumprod = self.extract(self.alphas_cumprod, t, xt.shape)
        return (alphas_cumprod.sqrt() * x0 - xt) / (1 - alphas_cumprod)


class VelocitySampler(AbstractDiscreteSampler):
    def predict_xstart(self, xt, vt, t):
        return self.sqrt_alphas_cumprod[t] * xt - self.sqrt_one_minus_alphas_cumprod[t] * vt

    def get_posterior_mean(self, xt, vt, t):
        x0_hat = self.predict_xstart(xt, vt, t)
        return (self.alphas[t].sqrt() * (1 - self.alphas_cumprod_prev[t]) * xt + self.alphas_cumprod_prev[t].sqrt() * self.betas[t] * x0_hat) / (1 - self.alphas_cumprod[t])

    # def get_ground_truth(self, eps, xt, x0, t, extras=None):
    #     sqrt_alphas_cumprod_t = self.extract(self.sqrt_alphas_cumprod, t, xt.shape)
    #     sqrt_one_minus_alphas_cumprod_t = self.extract(
    #         self.sqrt_one_minus_alphas_cumprod, t, xt.shape
    #     )
    #     return (sqrt_alphas_cumprod_t * xt - x0) / sqrt_one_minus_alphas_cumprod_t

    def get_ground_truth(self, eps, xt, x0, t, extras=None):
        sqrt_alphas_cumprod_t = self.extract(self.sqrt_alphas_cumprod, t, xt.shape)
        sqrt_one_minus_alphas_cumprod_t = self.extract(
            self.sqrt_one_minus_alphas_cumprod, t, xt.shape
        )
        return sqrt_alphas_cumprod_t * eps - sqrt_one_minus_alphas_cumprod_t * x0

    def get_classifier_free_mean(self, xt, unconditional_output, t, conditional_output):
        unconditional_eps = self.sqrt_one_minus_alphas_cumprod[t] * xt + self.sqrt_alphas_cumprod[t] * unconditional_output
        conditional_eps = self.sqrt_one_minus_alphas_cumprod[t] * xt + self.sqrt_alphas_cumprod[t] * conditional_vt
        eps = self.combine_eps(unconditional_eps, conditional_eps)
        classifier_free_vt = (eps - self.sqrt_one_minus_alphas_cumprod[t] * xt) / self.sqrt_alphas_cumprod[t]
        return self.get_posterior_mean(xt, classifier_free_vt, t)

    def get_sf_estimator(self, v_pred, xt, t):
        sigma_t = interpolate_schedule(t, self.sqrt_one_minus_alphas_cumprod)
        alpha_t = interpolate_schedule(t, self.sqrt_alphas_cumprod)
        eps_pred = sigma_t * xt + alpha_t * v_pred
        return -eps_pred / sigma_t.reshape((-1,) + (1,) * (len(eps_pred.shape) - 1))
