#!/usr/bin/env python3

import torch
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


class AbstractSampler:
    def __init__(self, beta_schedule, diffusion_timesteps: int, guidance_coef: float):
        self.beta_schedule = beta_schedule
        self.diffusion_timesteps = diffusion_timesteps
        self.guidance_coef = guidance_coef

        # define beta schedule
        beta_schedule_fn = self.get_beta_schedule(beta_schedule)
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

    def get_beta_schedule(self, beta_schedule: BetaSchedule):
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

    def extract(self, a, t, x_shape):
        batch_size = t.shape[0]
        out = a.gather(-1, t)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

    def interpolate_beta(self, diffusion_time):
        floor = torch.floor(diffusion_time)
        ceil = torch.ceil(diffusion_time)
        if ceil < len(self.betas):
            ceil_beta = self.betas[ceil.to(torch.long)]
        else:
            ceil_beta = self.betas[-1]
        if floor < len(self.betas):
            floor_beta = self.betas[floor.to(torch.long)]
        else:
            floor_beta = self.betas[-1]
        return floor_beta + (diffusion_time - floor) * (ceil_beta - floor_beta)

    def get_ground_truth(self, eps, xt, x0, t):
        raise NotImplementedError

    def bad_predict_xstart(self, xt, t, nsamples):
        shape = (nsamples,) + tuple(xt.shape)
        return (
            self.extract(self.sqrt_recip_alphas_cumprod, t, shape) * xt.repeat(nsamples, *(1,) * len(xt.shape))
            - self.extract(self.sqrt_recipm1_alphas_cumprod, t, shape) * torch.randn(shape)
        )

    # forward diffusion (using the nice property)
    def forward_sample(self, x_start, t, noise):
        sqrt_alphas_cumprod_t = self.extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self.extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )
        xt = sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
        return xt, self.get_ground_truth(eps=noise, xt=xt, x0=x_start, t=t)

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

    def classifier_free_reverse_sample(self, xt, unconditional_output, t, conditional_output):
        conditional_mean = self.get_classifier_free_mean(xt, unconditional_output, torch.tensor([t]), conditional_output)
        return self.reverse_sample(xt, t, conditional_mean)

    def combine_eps(self, unconditional_eps, conditional_eps):
        return (1 + self.guidance_coef) * conditional_eps - self.guidance_coef * unconditional_eps


###################
# PREDICTION TYPE #
###################

class Sampler(AbstractSampler):
    def get_ground_truth(self, eps, xt, x0, t):
        raise NotImplementedError


class EpsilonSampler(Sampler):
    def get_posterior_mean(self, xt, eps, t):
        return self.sqrt_recip_alphas[t] * (xt - (self.betas / self.sqrt_one_minus_alphas_cumprod)[t] * eps)

    def get_ground_truth(self, eps, xt, x0, t):
        return eps

    def get_classifier_free_mean(self, xt, unconditional_output, t, conditional_output):
        eps = self.combine_eps(unconditional_output, conditional_output)
        return self.get_posterior_mean(xt, eps, t)


class MuSampler(Sampler):
    def get_posterior_mean(self, xt, mean, t):
        return mean

    def get_ground_truth(self, eps, xt, x0, t):
        alphas = self.extract(self.alphas.sqrt(), t, xt.shape)
        alphas_cumprod_prev = self.extract(self.alphas_cumprod_prev, t, xt.shape)
        betas = self.extract(self.betas, t, xt.shape)
        alphas_cumprod = self.extract(self.alphas_cumprod, t, xt.shape)
        return (alphas.sqrt() * (1 - alphas_cumprod_prev) * xt + alphas_cumprod_prev.sqrt() * betas * x0) / (1 - alphas_cumprod)


class XstartSampler(Sampler):
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

    def get_ground_truth(self, eps, xt, x0, t):
        return x0

    def get_classifier_free_mean(self, xt, unconditional_xhat, t, conditional_xhat):
        unconditional_eps = (xt - self.sqrt_alphas_cumprod * unconditional_xhat) / self.sqrt_one_minus_alphas_cumprod[t]
        conditional_eps = (xt - self.sqrt_alphas_cumprod * conditional_xhat) / self.sqrt_one_minus_alphas_cumprod[t]
        eps = self.combine_eps(unconditional_eps, conditional_eps)
        classifier_free_xt = (xt - self.sqrt_one_minus_alphas_cumprod[t] * eps) / self.sqrt_alphas_cumprod[t]
        return self.get_posterior_mean(xt, classifier_free_xt, t)


class ScoreFunctionSampler(Sampler):
    def get_posterior_mean(self, xt, score_fun_est, t):
        return self.sqrt_recip_alphas[t] * (xt + self.betas[t] * score_fun_est)

    def get_ground_truth(self, eps, xt, x0, t):
        alphas_cumprod = self.extract(self.alphas_cumprod, t, xt.shape)
        return (alphas_cumprod.sqrt() * x0 - xt) / (1 - alphas_cumprod)


class VelocitySampler(Sampler):
    def predict_xstart(self, xt, vt, t):
        return self.sqrt_alphas_cumprod[t] * xt - self.sqrt_one_minus_alphas_cumprod[t] * vt

    def get_posterior_mean(self, xt, vt, t):
        x0_hat = self.predict_xstart(xt, vt, t)
        return (self.alphas[t].sqrt() * (1 - self.alphas_cumprod_prev[t]) * xt + self.alphas_cumprod_prev[t].sqrt() * self.betas[t] * x0_hat) / (1 - self.alphas_cumprod[t])

    def get_ground_truth(self, eps, xt, x0, t):
        sqrt_alphas_cumprod_t = self.extract(self.sqrt_alphas_cumprod, t, xt.shape)
        sqrt_one_minus_alphas_cumprod_t = self.extract(
            self.sqrt_one_minus_alphas_cumprod, t, xt.shape
        )
        return (sqrt_alphas_cumprod_t * xt - x0) / sqrt_one_minus_alphas_cumprod_t

    def get_classifier_free_mean(self, xt, unconditional_vt, t, conditional_vt):
        unconditional_eps = self.sqrt_one_minus_alphas_cumprod[t] * xt + self.sqrt_alphas_cumprod[t] * unconditional_vt
        conditional_eps = self.sqrt_one_minus_alphas_cumprod[t] * xt + self.sqrt_alphas_cumprod[t] * conditional_vt
        eps = self.combine_eps(unconditional_eps, conditional_eps)
        classifier_free_vt = (eps - self.sqrt_one_minus_alphas_cumprod[t] * xt) / self.sqrt_alphas_cumprod[t]
        return self.get_posterior_mean(xt, classifier_free_vt, t)
