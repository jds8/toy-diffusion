#!/usr/bin/env python3
import os
import warnings
import logging

from collections import namedtuple

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
import scipy.stats as stats
import numpy as np
import torch
import torch.distributions as dist
import torch.nn as nn
import matplotlib.pyplot as plt

from torchdiffeq import odeint

from toy_plot import SDE, analytical_log_likelihood, plot_ode_trajectories
from toy_configs import register_configs
from toy_train_config import SampleConfig, SMCSampleConfig, \
    get_model_path, get_classifier_path, ExampleConfig, \
    GaussianExampleConfig, BrownianMotionDiffExampleConfig, \
    UniformExampleConfig, StudentTExampleConfig, TestType, IntegratorType
from models.toy_sampler import AbstractSampler, interpolate_schedule
from toy_likelihoods import Likelihood, ClassifierLikelihood, GeneralDistLikelihood
from models.toy_temporal import TemporalTransformerUnet, TemporalClassifier, TemporalNNet, DiffusionModel
from models.toy_diffusion_models_config import GuidanceType, DiscreteSamplerConfig, ContinuousSamplerConfig


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


SampleOutput = namedtuple('SampleOutput', 'samples fevals')

SDEConfig = namedtuple('SDEConfig', 'drift diffusion sde_steps end_time')
DiffusionConfig = namedtuple('DiffusionConfig', 'f g')


#########################
#########################
def suppresswarning():
    warnings.warn("user", UserWarning)


class ToyEvaluator:
    def __init__(self, cfg: SampleConfig):
        self.cfg = cfg

        d_model = torch.tensor(1)
        self.sampler = hydra.utils.instantiate(cfg.sampler)
        self.diffusion_model = hydra.utils.instantiate(
            cfg.diffusion,
            d_model=d_model,
            device=device
        ).to(device)

        self.diffusion_model.eval()
        self.likelihood = hydra.utils.instantiate(cfg.likelihood)
        self.example = OmegaConf.to_object(cfg.example)

        self.cond = torch.tensor([self.cfg.cond], device=device) if self.cfg.cond is not None and self.cfg.cond >= 0. else None

        self.load_model()

    def load_model(self):
        model_path = get_model_path(self.cfg)
        try:
            # load softmax model
            print('attempting to load diffusion model: {}'.format(model_path))
            self.diffusion_model.load_state_dict(torch.load('{}'.format(model_path)))
            print('successfully loaded diffusion model')
        except Exception as e:
            try:
                self.diffusion_model.load_state_dict(
                    torch.load(
                        '{}'.format(model_path),
                        map_location='cpu'
                    )
                )
            except Exception as e:
                print('FAILED to load model: {} because {}\ncreating it...'.format(model_path, e))

    def grad_log_lik(self, xt, t, cond, model_output, cond_traj):
        x0_hat = self.sampler.predict_xstart(xt, model_output, t)
        if type(self.diffusion_model) == TemporalTransformerUnet:
            return self.likelihood.grad_log_lik(cond, xt, x0_hat, cond_traj)
        elif type(self.likelihood) in [ClassifierLikelihood, GeneralDistLikelihood]:
            return self.likelihood.grad_log_lik(cond, xt, x0_hat, t)
        else:
            return self.likelihood.grad_log_lik(cond, xt, x0_hat)

    def viz_trajs(self, traj, end_time, idx, clf=True, clr='green'):
        full_state_pred = traj.detach().squeeze(0).cpu().numpy()

        plt.plot(torch.linspace(0, end_time, full_state_pred.shape[0]), full_state_pred, color=clr)

        plt.savefig('figs/sample_{}.pdf'.format(idx))

        if clf:
            plt.clf()

    def get_x_min(self):
        if type(self.example) == GaussianExampleConfig:
            pseudo_example = self.cfg.example.copy()
            pseudo_example['mu'] = 0.
            pseudo_example['sigma'] = 1.
            mean, _, std = self.sampler.analytical_marginal_prob(
                t=torch.tensor(1.),
                example=pseudo_example,
            )
            x_min = dist.Normal(
                mean.item(),
                std.item(),
                device
            ).sample([
                self.cfg.num_samples,
                1,
                1
            ])
        elif type(self.example) == BrownianMotionDiffExampleConfig:
            x_min = self.sampler.prior_sampling(device).sample([
                self.cfg.num_samples,
                self.cfg.example.sde_steps-1,
                1,
            ])
        elif type(self.example) == UniformExampleConfig:
            x_min = dist.Normal(0, 1, device).sample([
                self.cfg.num_samples, 1, 1
            ])
        else:
            x_min = dist.Normal(0, 1, device).sample([
                self.cfg.num_samples, 1, 1
            ])
        return x_min.to(device)


class DiscreteEvaluator(ToyEvaluator):
    def sample_trajectories(self, cond_traj=None):
        x = self.get_x_min().reshape(-1, 1)

        samples = [x]
        for t in torch.arange(self.sampler.diffusion_timesteps-1, -1, -1, device=device):
            if t % 100 == 0:
                print(x[0, 0])
            time = t.reshape(-1)
            if type(self.diffusion_model) == TemporalTransformerUnet:
                unconditional_output = self.diffusion_model(x, time, None, None)
            else:
                # unconditional_output = self.diffusion_model(x, time, None)

                # TODO: Remove
                unconditional_output = self.diffusion_model(x.reshape(-1, 1), time.repeat(x.shape[0], 1), None)
            if self.cfg.guidance == GuidanceType.Classifier:
                if self.cond is not None:
                    with torch.enable_grad():
                        xt = x.detach().clone().requires_grad_(True)
                        grad_log_lik = self.grad_log_lik(xt, time, self.cond, unconditional_output, cond_traj)
                else:
                    grad_log_lik = torch.tensor(0.)
                x = self.sampler.classifier_guided_reverse_sample(
                    xt=x.reshape(unconditional_output.shape),
                    unconditional_output=unconditional_output,
                    t=t.item(),
                    grad_log_lik=grad_log_lik
                )
            elif self.cfg.guidance == GuidanceType.ClassifierFree:
                if self.cond is None or self.cond < 0.:
                    conditional_output = unconditional_output
                else:
                    if type(self.diffusion_model) == TemporalTransformerUnet:
                        conditional_output = self.diffusion_model(x, time, cond_traj, self.cond)
                    else:
                        conditional_output = self.diffusion_model(x, time, self.cond)
                x = self.sampler.classifier_free_reverse_sample(
                    xt=x, unconditional_output=unconditional_output,
                    conditional_output=conditional_output, t=t.item()
                )
            else:
                print('Unknown guidance: {}... defaulting to unconditional sampling'.format(self.cfg.guidance))
                posterior_mean = self.sampler.get_posterior_mean(x, unconditional_output, time)
                x = self.sampler.reverse_sample(
                    x, t.item(), posterior_mean,
                )
            samples.append(x)
        return SampleOutput(samples=samples, fevals=-1)

    @torch.no_grad()
    def ode_log_likelihood(self, x, extras=None, atol=1e-4, rtol=1e-4):
        """ THIS PROBABLY SHOULDN'T BE USED """
        extras = {} if extras is None else extras
        # hutchinson's trick
        v = torch.randint_like(x, 2) * 2 - 1
        fevals = 0
        def ode_fn(t, x):
            nonlocal fevals
            with torch.enable_grad():
                x = x[0].detach().requires_grad_()
                diffusion_time = t.reshape(-1) * self.sampler.diffusion_timesteps
                model_output = self.diffusion_model(x, diffusion_time, **extra_args)
                sf_est = self.sampler.get_sf_estimator(model_output, xt=x, t=diffusion_time)
                coef = -0.5 * interpolate_schedule(diffusion_time, self.sampler.betas)
                dx_dt = coef * (x + sf_est)
                fevals += 1
                grad = torch.autograd.grad((dx_dt * v).sum(), x)[0]
                d_ll = (v * grad).flatten(1).sum(1)
            return torch.cat([dx_dt.reshape(-1), d_ll.reshape(-1)])
        x_min = x, x.new_zeros([x.shape[0]])
        times = torch.tensor([self.sampler.t_eps, 1.], device=x.device)
        sol = odeint(ode_fn, x_min, times, atol=atol, rtol=rtol, method='dopri5')
        latent, delta_ll = sol[0][-1], sol[1][-1]
        ll_prior = self.sampler.prior_logp(latent, device=device).flatten(1).sum(1)
        # compute log(p(0)) = log(p(T)) + Tr(df/dx) where dx/dt = f
        return ll_prior + delta_ll, {'fevals': fevals}


class ContinuousEvaluator(ToyEvaluator):
    def get_score_function(self, t, x, evaluate_likelihood, **kwargs):
        if self.cfg.test == TestType.Gaussian:
            return self.analytical_gaussian_score(t=t, x=x)
        elif self.cfg.test == TestType.BrownianMotionDiff:
            return self.analytical_brownian_motion_diff_score(t=t, x=x)
        elif self.cfg.test == TestType.Test:
            # uncond_sf_est = self.sampler.get_sf_estimator(
            #     unconditional_output,
            #     xt=x.to(device),
            #     t=t.to(device)
            # )
            if self.cfg.guidance == GuidanceType.ClassifierFree:
                unconditional_output = torch.zeros_like(x)
                if self.cfg.sampler.guidance_coef != 0.:
                    unconditional_output = self.diffusion_model(
                        x=x.to(device),
                        time=t.to(device),
                    )
                conditional_output = self.diffusion_model(
                    x=x.to(device),
                    time=t.to(device),
                    cond=kwargs['cond'],
                    alpha=kwargs['alpha'],
                )
                cond_sf_est = self.sampler.get_classifier_free_sf_estimator(
                    xt=x.to(device),
                    unconditional_output=unconditional_output,
                    t=t.to(device),
                    conditional_output=conditional_output,
                )
                # if evaluate_likelihood:
                #     return torch.stack([uncond_sf_est, cond_sf_est], dim=0)
                return cond_sf_est
            else:
                unconditional_output = self.diffusion_model(
                    x=x.to(device),
                    time=t.to(device),
                )
                return self.sampler.get_sf_estimator(
                    unconditional_output,
                    xt=x.to(device),
                    t=t.to(device)
                )
        else:
            raise NotImplementedError

    def set_no_guidance(self):
        old_guidance = self.cfg.guidance
        self.cfg.guidance = GuidanceType.NoGuidance
        return old_guidance

    def get_dx_dt(self, t, x, evaluate_likelihood, **kwargs):
        time = t.reshape(-1)
        sf_est = self.get_score_function(
            t=time,
            x=x,
            evaluate_likelihood=evaluate_likelihood,
            **kwargs,
        )
        dx_dt = self.sampler.probability_flow_ode(
            x,
            time,
            sf_est,
        )
        return dx_dt

    def analytical_gaussian_score(self, t, x):
        """
        Compute the analytical marginal score of p_t for t in (0, 1)
        given the SDE formulation from Song et al. in the case that
        p_0 = N(mu_0, sigma_0) and p_1 = N(0, 1)
        """
        pseudo_example = self.cfg.example.copy()
        pseudo_example['mu'] = 0.
        pseudo_example['sigma'] = 1.
        mean, lmc, std = self.sampler.analytical_marginal_prob(
            t=t,
            example=pseudo_example
        )
        var = std ** 2
        score = (mean - x) / var
        return score

    def analytical_brownian_motion_diff_score(self, t, x):
        """
        Compute the analytical score p_t for t in (0, 1)
        given the SDE formulation from Song et al. in the case that
        p_0(x, s) = N(0, d(s)sqrt(s)) and p_1(x, s) = N(0, 1)
        where we consider sequential differences dX_t equiv X_t - X_{t-1}
        and where X_t is Brownian Motion so X_t sim N(X_{t-1}, sqrt{dt})
        """
        f = self.sampler.marginal_prob(x, t)[1].exp()[:, 0, :]
        g = self.sampler.marginal_prob(x, t)[2][:, 0, :]

        dt = 1. / (self.cfg.example.sde_steps-1)

        var = f ** 2 * dt + g ** 2

        score = -x / var

        return score

    def analytical_student_t_score(self, t, x):
        """
        Compute the analytical marginal score of p_t for t in (0, 1)
        given the SDE formulation from Song et al. in the case that
        p_0 = T(degress_of_freedom=nu, loc=0, scale=1) and p_1 = N(0, 1)
        https://chatgpt.com/c/0e26fe1f-e8e7-4fce-8b89-719fa0479589
        nabla log p(x) = -(nu + 1)(x - mu) / sigma^2(nu + (x-mu)^2 / sigma^2)
        """

        pseudo_example = self.cfg.example.copy()
        pseudo_example['mu'] = 0.
        pseudo_example['sigma'] = 1.
        mean, lmc, std = self.sampler.analytical_marginal_prob(
            t=t,
            example=pseudo_example
        )
        num = -(self.cfg.example.nu + 1) * (x - mean)
        var = std ** 2
        denom = var * (self.cfg.example.nu + (x - mean) ** 2 / var)
        return num / denom

    def sample_trajectories_euler_maruyama(self, steps=torch.tensor(1000), **kwargs):
        x_min = self.get_x_min()
        x = x_min.clone()

        steps = steps.to(x.device)
        for time in torch.linspace(1., self.sampler.t_eps, steps, device=x.device):
            time = time.reshape(-1)
            sf_est = self.get_score_function(t=time, x=x)
            x, _ = self.sampler.reverse_sde(x=x, t=time, score=sf_est, steps=steps)

        return SampleOutput(samples=torch.stack([x_min, x]), fevals=steps)

    def sample_trajectories_probability_flow(self, atol=1e-5, rtol=1e-5, **kwargs):
        x_min = self.get_x_min()

        fevals = 0
        def ode_fn(t, x):
            nonlocal fevals
            fevals += 1
            dx_dt = self.get_dx_dt(t, x, evaluate_likelihood=False, **kwargs)
            return dx_dt

        times = torch.linspace(
            1.,
            self.sampler.t_eps,
            self.sampler.diffusion_timesteps,
            device=x_min.device
        )
        sol = odeint(ode_fn, x_min, times, atol=atol, rtol=rtol, method='rk4')
        return SampleOutput(samples=sol, fevals=fevals)

    def sample_trajectories(self, **kwargs):
        print('sampling trajectories...')
        if self.cfg.integrator_type == IntegratorType.ProbabilityFlow:
            sample_out = self.sample_trajectories_probability_flow(**kwargs)
        elif self.cfg.integrator_type == IntegratorType.EulerMaruyama:
            sample_out = self.sample_trajectories_euler_maruyama(**kwargs)
        else:
            raise NotImplementedError
        return sample_out

    @torch.no_grad()
    def ode_log_likelihood(self, x, atol=1e-5, rtol=1e-5, **kwargs):
        print('evaluating likelihood...')
        # hutchinson's trick
        # if self.cfg.guidance == GuidanceType.ClassifierFree:
        #     v = torch.randint_like(x.tile(2, 1, 1, 1), 2) * 2 - 1
        # else:
        #     v = torch.randint_like(x, 2) * 2 - 1
        v = torch.randint_like(x, 2) * 2 - 1
        fevals = 0
        def ode_fn(t, x):
            nonlocal fevals
            fevals += 1
            with torch.enable_grad():
                x = x[0].detach().requires_grad_()
                dx_dt = self.get_dx_dt(t, x, evaluate_likelihood=True, **kwargs)
                # if self.cfg.guidance == GuidanceType.ClassifierFree:
                #     grad_uncond = torch.autograd.grad(
                #         (dx_dt[0] * v[0]).sum(),
                #         x,
                #         retain_graph=True
                #     )[0]
                #     grad_cond = torch.autograd.grad(
                #         (dx_dt[1] * v[1]).sum(),
                #         x
                #     )[0]
                #     grad = torch.stack([grad_uncond, grad_cond])
                #     dx_dt = dx_dt[1]  # update x according to conditional gradient
                # else:
                #     grad = torch.autograd.grad((dx_dt * v).sum(), x)[0]
                grad = torch.autograd.grad((dx_dt * v).sum(), x)[0]
                d_ll = (v * grad).sum([-1, -2])
            return torch.cat([dx_dt.reshape(-1), d_ll.reshape(-1)])
        # if self.cfg.guidance == GuidanceType.ClassifierFree:
        #     ll = x.new_zeros([2*x.shape[0]])
        # else:
        #     ll = x.new_zeros([x.shape[0]])
        ll = x.new_zeros([x.shape[0]])
        x_min = x, ll
        times = torch.linspace(
            self.sampler.t_eps,
            1.-self.sampler.t_eps,
            self.sampler.diffusion_timesteps,
            device=x.device
        )
        sol = odeint(ode_fn, x_min, times, atol=atol, rtol=rtol, method='rk4')
        latent, delta_ll = sol[0][-1], sol[1][-1]
        # if self.cfg.test == TestType.Gaussian:
        #     pseudo_example = self.cfg.example.copy()
        #     pseudo_example['mu'] = 0.
        #     pseudo_example['sigma'] = 1.
        #     ll_prior = self.sampler.prior_analytic_logp(
        #         # self.cfg.example,
        #         pseudo_example,
        #         device,
        #         latent,
        #     ).flatten(1).sum(1)
        # else:
        #     ll_prior = self.sampler.prior_logp(latent, device=device).flatten(1).sum(1)
        ll_prior = self.sampler.prior_logp(latent, device=device).flatten(1).sum(1)
        # compute log(p(0)) = log(p(T)) + \int_0^T Tr(df(x,t)/dx)dt where dx/dt = f
        # if self.cfg.guidance == GuidanceType.ClassifierFree:
        #     ll_output = ll_prior.repeat(2) + delta_ll
        # else:
        #     ll_output = ll_prior + delta_ll
        ll_output = ll_prior + delta_ll
        return ll_output, {'fevals': fevals}


def plt_llk(traj, lik, plot_type='scatter', ax=None):
    full_state_pred = traj.detach().squeeze().cpu().numpy()
    full_state_lik = lik.detach().squeeze().cpu().numpy()

    if plot_type == 'scatter':
        plt.scatter(full_state_pred, full_state_lik, color='blue')
    elif plot_type == 'line':
        idx = full_state_pred.argsort()
        sorted_state = full_state_pred[idx]
        sorted_lik = full_state_lik[idx]
        plt.plot(sorted_state, sorted_lik, color='red')
    elif plot_type == '3d_scatter':
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
        for i, x in enumerate(torch.linspace(0., 1., full_state_pred.shape[-1])):
            xs = x.repeat(full_state_pred.shape[0]) if len(full_state_pred.shape) > 1 else x
            ys = full_state_pred[:, i] if len(full_state_pred.shape) > 1 else full_state_pred[i]
            zs = full_state_lik
            ax.scatter(xs=xs, ys=ys, zs=zs, color='blue')
        return ax

    elif plot_type == '3d_line':
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
        idx = full_state_pred.argsort(0)
        xs = torch.linspace(0., 1., full_state_pred.shape[-1])
        for i, lik in enumerate(full_state_lik):
            ys = full_state_pred[i, :]
            zs = np.array(lik).repeat(full_state_pred.shape[1])
            ax.plot(xs=xs, ys=ys, zs=zs, color='red')

    plt.savefig('figs/scatter.pdf')

def test_gaussian(end_time, cfg, sample_trajs, std):
    cond = std.cond if std.cond else torch.tensor([-1.])
    traj = sample_trajs * cfg.example.sigma + cfg.example.mu
    alpha = torch.tensor([std.likelihood.alpha]) if cond == 1. else torch.tensor([0.])
    datapoints_left = torch.linspace(
        cfg.example.mu-6*cfg.example.sigma,
        cfg.example.mu-alpha.item()*cfg.example.sigma,
        500
    )
    datapoints_right = torch.linspace(
        cfg.example.mu+alpha.item()*cfg.example.sigma,
        cfg.example.mu+6*cfg.example.sigma,
        500
    )
    datapoints_center = torch.tensor([
        cfg.example.mu-alpha.item()*cfg.example.sigma + 1/500,
        cfg.example.mu+alpha.item()*cfg.example.sigma - 1/500,
    ])
    if alpha > 0:
        datapoints = torch.hstack([datapoints_left, datapoints_center, datapoints_right])
    else:
        datapoints = torch.hstack([datapoints_left, torch.tensor([-1/500]), datapoints_right])
    datapoints = torch.hstack([datapoints_left, datapoints_center, datapoints_right])
    datapoints = datapoints.sort().values.unique()
    datapoint_dist = torch.distributions.Normal(
        cfg.example.mu, cfg.example.sigma
    )
    tail = 2 * datapoint_dist.cdf(cfg.example.mu-alpha*cfg.example.sigma)
    datapoint_left_llk = datapoint_dist.log_prob(datapoints_left) - tail.log()
    datapoint_right_llk = datapoint_dist.log_prob(datapoints_right) - tail.log()
    datapoint_center_llk = -torch.ones(2) * torch.inf
    datapoint_llk = torch.hstack([datapoint_left_llk, datapoint_center_llk, datapoint_right_llk])
    analytical_llk = datapoint_dist.log_prob(traj) - tail.log()
    a_lk = analytical_llk.exp().squeeze()
    print('analytical_llk: {}'.format(a_lk))
    ode_llk = std.ode_log_likelihood(sample_trajs, cond=cond, alpha=alpha)
    ode_lk = ode_llk[0].exp() / cfg.example.sigma
    print('\node_llk: {}\node evals: {}'.format(ode_lk, ode_llk[1]))
    mse_llk = torch.nn.MSELoss()(
        a_lk,
        ode_lk,
    )
    print('\nmse_llk: {}'.format(mse_llk))

    plt.clf()
    plt_llk(traj, ode_lk, plot_type='scatter')
    plt_llk(datapoints, datapoint_llk.exp(), plot_type='line')
    import pdb; pdb.set_trace()

def test_brownian_motion(end_time, cfg, sample_trajs, std):
    dt = end_time / (cfg.example.sde_steps-1)
    analytical_trajs = torch.cat([
        torch.zeros(sample_trajs.shape[0], 1, 1, device=sample_trajs.device),
        sample_trajs
    ], dim=1)

    analytical_llk = analytical_log_likelihood(
        analytical_trajs,
        SDE(cfg.example.sde_drift, cfg.example.sde_diffusion),
        dt
    )
    print('analytical_llk: {}'.format(analytical_llk))

    ode_llk = std.ode_log_likelihood(sample_trajs)
    print('\node_llk: {}'.format(ode_llk))

    mse_llk = torch.nn.MSELoss()(analytical_llk.squeeze(), ode_llk[0])
    print('\nmse_llk: {}'.format(mse_llk))

    plt.clf()
    if sample_trajs.shape[1] > 1:
        ax = plt_llk(sample_trajs, ode_llk[0].exp(), plot_type='3d_scatter')
        plt_llk(sample_trajs, analytical_llk.exp(), plot_type='3d_line', ax=ax)
    else:
        plt_llk(sample_trajs, ode_llk[0].exp(), plot_type='scatter')
        plt_llk(sample_trajs, analytical_llk.exp(), plot_type='line')
    import pdb; pdb.set_trace()

def test_brownian_motion_diff(end_time, cfg, sample_trajs, std):
    dt = end_time / (cfg.example.sde_steps-1)
    # de-standardize data
    trajs = sample_trajs * dt.sqrt()

    # make histogram
    data = sample_trajs.reshape(-1).numpy()
    plt.clf()
    plt.hist(data, bins=30, edgecolor='black')
    plt.title('Histogram of brownian motion state diffs')
    save_dir = 'figs/{}'.format(cfg.model_name)
    os.makedirs(save_dir, exist_ok=True)
    alpha = torch.tensor([std.likelihood.alpha])
    alpha_str = '%.1f' % alpha.item()
    plt.savefig('{}/alpha={}_brownian_motion_diff_hist.pdf'.format(
        save_dir,
        alpha_str,
    ))

    # turn state diffs into Brownian motion
    bm_trajs = torch.cat([
        torch.zeros(trajs.shape[0], 1, 1, device=trajs.device),
        trajs.cumsum(dim=-2)
    ], dim=1)

    # plot trajectories
    plt.clf()
    times = torch.linspace(0., 1., bm_trajs.shape[1])
    plt.plot(times.numpy(), bm_trajs[..., 0].numpy().T)
    plt.savefig('{}/alpha={}_brownian_motion_diff_samples.pdf'.format(
        save_dir,
        alpha_str,
    ))

    # plot cut off trajectories
    exit_idx = (bm_trajs.abs() > alpha).to(float).argmax(dim=1)
    plt.clf()
    times = torch.linspace(0., 1., bm_trajs.shape[1])
    dtimes = exit_idx * dt
    states = bm_trajs[torch.arange(bm_trajs.shape[0]), exit_idx.squeeze()]
    plt.plot(times.numpy(), bm_trajs[..., 0].numpy().T, alpha=0.2)
    plt.scatter(dtimes.numpy(), states, marker='o', color='red')
    plt.savefig('{}/alpha={}_exit_brownian_motion_diff_samples.pdf'.format(
        save_dir,
        alpha_str,
    ))
    exited = (bm_trajs.abs() > std.likelihood.alpha).any(dim=1).to(float)
    prop_exited = exited.mean() * 100
    num_exited = exited.mean()
    print('{}% of {} trajectories exited [-{}, {}]'.format(
        prop_exited,
        num_exited,
        std.likelihood.alpha,
        std.likelihood.alpha,
    ))

    # compute (discretized) "analytical" log likelihood
    analytical_llk = (dist.Normal(0, 1).log_prob(sample_trajs) - dt.sqrt().log()).sum(1).squeeze()
    print('analytical_llk: {}'.format(analytical_llk))

    # compute log likelihood under diffusion model
    ode_llk = std.ode_log_likelihood(sample_trajs, cond=std.cond, alpha=alpha)
    scaled_ode_llk = ode_llk[0] - dt.sqrt().log() * (cfg.example.sde_steps-1)
    print('\node_llk: {}'.format(scaled_ode_llk))

    # compare log likelihoods by MSE
    mse_llk = torch.nn.MSELoss()(analytical_llk, scaled_ode_llk)
    sse_llk = ((analytical_llk - scaled_ode_llk) ** 2).std()
    print('\nmse_llk: {}\nsse_llk: {}'.format(mse_llk, sse_llk))

    llk_stats = torch.stack([mse_llk, sse_llk])
    torch.save(llk_stats, '{}/alpha={}_llk_stats.pt'.format(
        save_dir,
        std.cond
    ))
    import pdb; pdb.set_trace()

def test_uniform(end_time, cfg, sample_trajs, std):
    scale = torch.pi / torch.tensor(3).sqrt()
    traj = sample_trajs * scale
    traj = traj.sigmoid() * (cfg.example.upper - cfg.example.lower) + cfg.example.lower
    analytical_llk = torch.distributions.Uniform(
        cfg.example.lower, cfg.example.upper
    ).log_prob(traj)
    a_lk = analytical_llk.exp().squeeze()
    print('analytical_llk: {}'.format(a_lk))
    ode_llk = std.ode_log_likelihood(sample_trajs, cond=std.cond)
    # derivative of sigmoid inverse is derivative of logit
    std_unif_traj = (traj - cfg.example.lower) / (cfg.example.upper - cfg.example.lower)
    logit_derivative = 1 / (std_unif_traj * (1 - std_unif_traj))
    ode_lk = ode_llk[0].exp() * logit_derivative.squeeze() / (scale * (cfg.example.upper - cfg.example.lower))
    print('\node_llk: {}\node evals: {}'.format(ode_lk, ode_llk[1]))
    mse_llk = torch.nn.MSELoss()(
        a_lk,
        ode_lk,
    )
    print('\nmse_llk: {}'.format(mse_llk))

    plt.clf()
    plt_llk(traj, ode_lk, plot_type='scatter')
    plt_llk(traj, a_lk, plot_type='line')
    import pdb; pdb.set_trace()

def test_student_t(end_time, cfg, sample_trajs, std):
    cond = std.cond if std.cond else torch.tensor([-1.])
    sigma = torch.tensor(35.9865)  # from dist.StudentT(1.5).sample([100000000]).std()
    if std.cfg.example.nu > 2.:
        sigma = torch.tensor(std.cfg.example.nu / (std.cfg.example.nu - 2)).sqrt()
    traj = sample_trajs * sigma
    alpha = std.likelihood.alpha if cond == 1. else torch.tensor([0.])
    datapoints_left = torch.linspace(
        -6.5*sigma,
        -alpha.item()*sigma,
        500
    )
    datapoints_right = torch.linspace(
        alpha.item()*sigma,
        6.5*sigma,
        500
    )
    datapoints_center = torch.tensor([
        -alpha.item()*sigma + 1/500,
        alpha.item()*sigma - 1/500,
    ])
    if alpha > 0:
        datapoints = torch.hstack([datapoints_left, datapoints_center, datapoints_right])
    else:
        datapoints = torch.hstack([datapoints_left, torch.tensor([-1/500]), datapoints_right])
    datapoints = datapoints.sort().values.unique()
    datapoint_dist = torch.distributions.StudentT(std.cfg.example.nu)
    tail = 2 * stats.t.cdf(-alpha*sigma, std.cfg.example.nu)
    tail_log = torch.tensor(np.log(tail))
    datapoint_left_llk = datapoint_dist.log_prob(datapoints_left) - tail_log
    datapoint_right_llk = datapoint_dist.log_prob(datapoints_right) - tail_log
    datapoint_center_llk = -torch.ones(2) * torch.inf if tail_log else torch.empty(0)
    datapoint_llk = torch.hstack([datapoint_left_llk, datapoint_center_llk, datapoint_right_llk])
    analytical_llk = datapoint_dist.log_prob(traj) - tail_log
    a_lk = analytical_llk.exp().squeeze()
    print('analytical_llk: {}'.format(a_lk))
    ode_llk = std.ode_log_likelihood(sample_trajs, cond=cond, alpha=alpha)
    ode_lk = ode_llk[0].exp() / sigma
    print('\node_llk: {}\node evals: {}'.format(ode_lk, ode_llk[1]))
    mse_llk = torch.nn.MSELoss()(
        a_lk,
        ode_lk,
    )
    print('\nmse_llk: {}'.format(mse_llk))

    plt.clf()
    plt_llk(traj, ode_lk, plot_type='scatter')
    plt_llk(datapoints, datapoint_llk.exp(), plot_type='line')
    import pdb; pdb.set_trace()

def test_student_t_diff(end_time, cfg, sample_trajs, std):
    dt = end_time / (cfg.example.sde_steps-1)
    # de-standardize data
    trajs = sample_trajs * dt.sqrt()

    # make histogram
    data = sample_trajs.reshape(-1).numpy()
    plt.clf()
    plt.hist(data, bins=30, edgecolor='black')
    plt.title('Histogram of Student T state diffs')
    save_dir = 'figs/{}'.format(cfg.model_name)
    alpha = std.likelihood.alpha
    alpha_str = '%.1f' % alpha.item()
    plt.savefig('{}/alpha={}_brownian_motion_diff_hist.pdf'.format(
        save_dir,
        alpha_str,
    ))

    # turn state diffs into Brownian motion
    st_trajs = torch.cat([
        torch.zeros(trajs.shape[0], 1, 1, device=trajs.device),
        trajs.cumsum(dim=-2)
    ], dim=1)

    # plot trajectories
    plt.clf()
    times = torch.linspace(0., 1., st_trajs.shape[1])
    plt.plot(times.numpy(), st_trajs[..., 0].numpy().T)
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig('{}/alpha={}_brownian_motion_diff_samples.pdf'.format(
        save_dir,
        alpha_str,
    ))

    # plot cut off trajectories
    exit_idx = (st_trajs.abs() > alpha).to(float).argmax(dim=1)
    plt.clf()
    times = torch.linspace(0., 1., st_trajs.shape[1])
    dtimes = exit_idx * dt
    states = st_trajs[torch.arange(st_trajs.shape[0]), exit_idx.squeeze()]
    plt.plot(times.numpy(), st_trajs[..., 0].numpy().T, alpha=0.2)
    plt.scatter(dtimes.numpy(), states, marker='o', color='red')
    plt.savefig('{}/alpha={}_exit_brownian_motion_diff_samples.pdf'.format(
        save_dir,
        alpha_str,
    ))
    exited = (st_trajs.abs() > std.likelihood.alpha).any(dim=1).to(float)
    prop_exited = exited.mean() * 100
    num_exited = exited.mean()
    print('{}% of {} trajectories exited [-{}, {}]'.format(
        prop_exited,
        num_exited,
        std.likelihood.alpha,
        std.likelihood.alpha,
    ))

    # compute (discretized) "analytical" log likelihood
    scale = torch.tensor(cfg.example.nu / (cfg.example.nu - 2)).sqrt()
    scaled_trajs = sample_trajs * scale
    analytical_llk = (
        dist.StudentT(cfg.example.nu).log_prob(scaled_trajs) \
        + scale.log() \
        - dt.sqrt().log()
    ).sum(1).squeeze()
    print('analytical_llk: {}'.format(analytical_llk))

    # compute log likelihood under diffusion model
    ode_llk = std.ode_log_likelihood(sample_trajs, cond=std.cond, alpha=alpha)
    scaled_ode_llk = ode_llk[0] - dt.sqrt().log() * (cfg.example.sde_steps-1)
    print('\node_llk: {}'.format(scaled_ode_llk))

    # compare log likelihoods by MSE
    mse_llk = torch.nn.MSELoss()(analytical_llk, scaled_ode_llk)
    sse_llk = ((analytical_llk - scaled_ode_llk) ** 2).std()
    print('\nmse_llk: {}\nsse_llk: {}'.format(mse_llk, sse_llk))

    llk_stats = torch.stack([mse_llk, sse_llk])
    torch.save(llk_stats, '{}/alpha={}_llk_stats.pt'.format(
        save_dir,
        std.cond
    ))
    import pdb; pdb.set_trace()

def test_transformer_bm(end_time, std):
    all_bm_trajs = []
    alphas = []
    cond = None
    alpha = None
    for _ in range(std.cfg.time_steps):
        alphas.append(alpha)
        sample_trajs = std.sample_trajectories(
            cond=cond,
            alpha=alpha,
        ).samples[-1]
        import pdb; pdb.set_trace()

        # # make histogram
        # data = sample_trajs.reshape(-1).numpy()
        # plt.clf()
        # plt.hist(data, bins=30, edgecolor='black')
        # plt.title('Histogram of brownian motion state diffs')
        save_dir = 'figs/{}'.format(std.cfg.model_name)
        # alpha = '%.1f' % std.likelihood.alpha.item()
        # plt.savefig('{}/alpha={}_brownian_motion_diff_hist.pdf'.format(
        #     save_dir,
        #     alpha,
        # ))

        # de-standardize data for visualization purposes
        dt = end_time / (std.cfg.example.sde_steps-1)
        destandardized_trajs = sample_trajs * dt.sqrt()
        # turn state diffs into Brownian motion
        bm_trajs = torch.cat([
            torch.zeros(sample_trajs.shape[0], 1, 1, device=sample_trajs.device),
            destandardized_trajs.cumsum(dim=-2)
        ], dim=1)
        all_bm_trajs.append(bm_trajs)

        # set new cond and alpha
        cond = sample_trajs
        alpha = bm_trajs.max(dim=1).values + 1.

    import pdb; pdb.set_trace()
    # create colors
    cmap = plt.get_cmap('seismic')
    num_colors = 10  # Number of colors in the gradient
    colors = [cmap(i / (num_colors - 1)) for i in range(num_colors)]

    times = torch.linspace(0., 1., bm_trajs.shape[1])
    alphas[0] = torch.zeros(std.cfg.num_samples)
    bm_trajs_tensor = torch.stack(all_bm_trajs)

    for j, bm_trajs_vec in enumerate(bm_trajs_tensor.split(dim=1, split_size=1)):
        trajs_vec = bm_trajs_vec.squeeze(1)
        plt.clf()
        for i, traj in enumerate(trajs_vec):
            alpha = alphas[i][j]
            plt.plot(times.numpy(), traj.numpy().T, label=f'alpha={alpha}', color=colors[i])
        plt.legend()
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig('{}/brownian_motion_smc_samples_{}.pdf'.format(
            save_dir,
            j,
        ))
    import pdb; pdb.set_trace()

def test(end_time, cfg, out_trajs, std):
    if type(std.example) == GaussianExampleConfig:
        test_gaussian(end_time, cfg, out_trajs, std)
    elif type(std.example) == BrownianMotionDiffExampleConfig:
        test_brownian_motion_diff(end_time, cfg, out_trajs, std)
    elif type(std.example) == UniformExampleConfig:
        test_uniform(end_time, cfg, out_trajs, std)
    elif type(std.example) == StudentTExampleConfig:
        test_student_t(end_time, cfg, out_trajs, std)
    else:
        raise NotImplementedError

def viz_trajs(cfg, std, out_trajs, end_time):
    if type(cfg.example) == BrownianMotionDiffExampleConfig:
        undiffed_trajs = out_trajs.cumsum(dim=-2)
        out_trajs = torch.cat([
            torch.zeros(undiffed_trajs.shape[0], 1, 1, device=undiffed_trajs.device),
            undiffed_trajs
        ], dim=1)
    for idx, out_traj in enumerate(out_trajs):
        std.viz_trajs(out_traj, end_time, idx, clf=False)


@hydra.main(version_base=None, config_path="conf", config_name="continuous_smc_sample_config")
def sample(cfg):
    logger = logging.getLogger("main")
    logger.info(f"CONFIG\n{OmegaConf.to_yaml(cfg)}")

    os.system('echo git commit: $(git rev-parse HEAD)')

    omega_sampler = OmegaConf.to_object(cfg.sampler)
    if isinstance(omega_sampler, DiscreteSamplerConfig):
        std = DiscreteEvaluator(cfg=cfg)
    elif isinstance(omega_sampler, ContinuousSamplerConfig):
        std = ContinuousEvaluator(cfg=cfg)
    else:
        raise NotImplementedError

    end_time = torch.tensor(1., device=device)

    with torch.no_grad():
        if isinstance(std.diffusion_model, TemporalTransformerUnet):
            test_transformer_bm(end_time, std)
            exit()

        sample_traj_out = std.sample_trajectories(
            cond=std.cond,
            alpha=std.likelihood.alpha.reshape(-1, 1),
        )

        # ode_trajs = (sample_traj_out.samples).reshape(-1, cfg.num_samples)
        # plot_ode_trajectories(ode_trajs)

        print('fevals: {}'.format(sample_traj_out.fevals))
        sample_trajs = sample_traj_out.samples
        trajs = sample_trajs[-1]
        out_trajs = trajs

        # viz_trajs(cfg, std, out_trajs, end_time)

        test(end_time, cfg, out_trajs, std)
        import pdb; pdb.set_trace()


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        suppresswarning()

    cs = ConfigStore.instance()
    cs.store(name="vpsde_sample_config", node=SampleConfig)
    cs.store(name="vpsde_smc_sample_config", node=SMCSampleConfig)
    register_configs()

    with torch.no_grad():
        sample()
