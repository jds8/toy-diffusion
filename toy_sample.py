#!/usr/bin/env python3
import os
import warnings
import logging

from pathlib import Path

from collections import namedtuple

import hydra
from hydra.core.config_store import ConfigStore
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
import numpy as np
import einops
import torch
import torch.distributions as dist
import torch.nn as nn
import matplotlib.pyplot as plt
import scipy.stats as stats
import scipy

from torchdiffeq import odeint, odeint_adjoint

from toy_plot import SDE, analytical_log_likelihood, plot_ode_trajectories
from toy_configs import register_configs
from toy_train_config import SampleConfig, SMCSampleConfig, \
    get_model_path, get_classifier_path, ExampleConfig, \
    GaussianExampleConfig, BrownianMotionDiffExampleConfig, \
    UniformExampleConfig, StudentTExampleConfig, TestType, IntegratorType, \
    get_target, MultivariateGaussianExampleConfig
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
        self.likelihood = hydra.utils.instantiate(cfg.likelihood, sampler=self.sampler)
        self.example = OmegaConf.to_object(cfg.example)

        self.cond = torch.tensor([self.cfg.cond], device=device) if self.cfg.cond is not None and self.cfg.cond >= 0. else None

        self.num_params = self.get_num_params()
        self.load_model()

    def get_num_params(self):
        model_parameters = filter(
            lambda p: p.requires_grad,
            self.diffusion_model.parameters()
        )
        num_params = sum([np.prod(p.size()) for p in model_parameters])
        return num_params

    def load_model_state_dict(self, model_path, map_location):
        model = torch.load(
            '{}'.format(model_path),
            map_location=map_location,
            weights_only=False
        )
        if 'model_state_dict' in model:
            self.diffusion_model.load_state_dict(model['model_state_dict'])
            model_parameters = filter(lambda p: p.requires_grad, self.diffusion_model.parameters())
            num_params = sum([np.prod(p.size()) for p in model_parameters])
            logger = logging.getLogger("main")
            logger.info('loaded model with {} parameters'.format(num_params))
        else:
            self.diffusion_model.load_state_dict(model)

    def load_model(self):
        model_path = get_model_path(self.cfg, self.cfg.diffusion.dim)
        path = Path(model_path)
        if not os.path.isfile(model_path):
            # scp from ubcml
            os.system('ssh -t jsefas@remote.cs.ubc.ca "scp submit-ml:/ubc/cs/research/ubc_ml/jsefas/toy-diffusion/diffusion_models/{} ~"'.format(path.name))
            os.system('scp jsefas@remote.cs.ubc.ca:~/{} {}'.format(path.name, model_path))
            if not os.path.isfile(model_path):
                raise Exception('cannot find file: {}'.format(model_path))
        try:
            # load softmax model
            print('attempting to load diffusion model: {}'.format(model_path))
            self.load_model_state_dict(model_path, map_location='cuda')
        except Exception as e:
            try:
                self.load_model_state_dict(model_path, map_location='cpu')
            except Exception as e:
                print('FAILED to load model: {} because {}'.format(model_path, e))
                raise e
        print('successfully loaded diffusion model')

    def grad_log_lik(self, xt, t, cond, model_output, cond_traj):
        x0_hat = self.sampler.predict_xstart(xt, model_output, t)
        if type(self.diffusion_model) == TemporalTransformerUnet:
            return self.likelihood.grad_log_lik(cond, xt, x0_hat, cond_traj)
        elif type(self.likelihood) in [ClassifierLikelihood, GeneralDistLikelihood]:
            return self.likelihood.grad_log_lik(cond, xt, x0_hat, t)
        else:
            return self.likelihood.grad_log_lik(cond, xt, x0_hat)

    def viz_trajs(self, traj, end_time, idx, figs_dir, clf=True, clr='green'):
        full_state_pred = traj.detach().squeeze(0).cpu().numpy()

        plt.plot(torch.linspace(0, end_time, full_state_pred.shape[0]), full_state_pred, color=clr)

        plt.savefig('{}/sample_{}.pdf'.format(figs_dir, idx))

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
        elif type(self.example) == MultivariateGaussianExampleConfig:
            d = self.cfg.example.d
            x_min = dist.MultivariateNormal(
                torch.zeros(d),
                torch.eye(d),
            ).sample([self.cfg.num_samples]).to(device).unsqueeze(-1)
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
        sol = odeint(ode_fn, x_min, times, atol=atol, rtol=rtol, method='rk4')
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
                if self.cfg.guidance == GuidanceType.Classifier:
                    if self.cond is not None:
                        with torch.enable_grad():
                            xt = x.detach().clone().requires_grad_(True)
                            grad_log_lik = self.likelihood.grad_log_lik(
                                xt,
                                t.item(),
                            )
                    else:
                        grad_log_lik = torch.tensor(0.)
                    cond_sf_est = self.sampler.get_classifier_guided_sf_estimator(
                        xt=x.reshape(unconditional_output.shape),
                        unconditional_output=unconditional_output,
                        t=t.item(),
                        cond_score=grad_log_lik
                    )

                    return cond_sf_est
                elif self.cfg.guidance == GuidanceType.NoGuidance:
                    uncond_sf_est = self.sampler.get_sf_estimator(
                        unconditional_output,
                        xt=x.to(device),
                        t=t.to(device)
                    )
                    return uncond_sf_est
                else:
                    raise NotImplementedError
        else:
            raise NotImplementedError

    def set_guidance(self, guidance: GuidanceType):
        old_guidance = self.cfg.guidance
        self.cfg.guidance = guidance
        return old_guidance

    def set_no_guidance(self):
        no_guidance = GuidanceType.NoGuidance
        return self.set_guidance(no_guidance)

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

    class ODESampleFunc(nn.Module):
        def __init__(self, ce, **kwargs):
            self.ce = ce
            self.kwargs = kwargs
            self.fevals = 0
            super().__init__()

        def forward(self, t, x):
            self.fevals += 1
            if 'observed_idx' in self.kwargs:
                x[:, self.kwargs['observed_idx']] = self.kwargs['observed_values']
            dx_dt = self.ce.get_dx_dt(t, x, evaluate_likelihood=False, **self.kwargs)
            return dx_dt

    def sample_trajectories_probability_flow(self, atol=1e-5, rtol=1e-5, **kwargs):
        x_min = self.get_x_min()

        fevals = 0
        def ode_fn(t, x):
            nonlocal fevals
            fevals += 1
            if 'observed_idx' in kwargs:
                x[:, kwargs['observed_idx']] = kwargs['observed_values']
            dx_dt = self.get_dx_dt(t, x, evaluate_likelihood=False, **kwargs)
            return dx_dt

        times = torch.linspace(
            1.,
            self.sampler.t_eps,
            self.sampler.diffusion_timesteps,
            device=x_min.device
        )
        # ode_fn = ContinuousEvaluator.ODESampleFunc(self, **kwargs)
        sol = odeint(
            ode_fn,
            x_min,
            times,
            atol=atol,
            rtol=rtol,
            method='rk4',
            # method='scipy_solver',
            # options={'solver': 'LSODA'},
        )
        # import matplotlib.pyplot as plt
        # dt = 1 / torch.tensor(self.cfg.example.sde_steps-1)
        # bm_trajs = torch.cat([
        #     torch.zeros(sol.shape[0], sol.shape[1], 1, 1, device=sol.device),
        #     (sol * dt.sqrt()).cumsum(dim=-2)
        # ], dim=2)
        # plt.plot(torch.arange(104), bm_trajs[-10].squeeze().cpu(), color='blue')
        # gt = torch.load('bm_dataset.pt', map_location=device, weights_only=True)
        # plt.plot(torch.arange(104), gt[0].squeeze().cpu(), color='red')
        # plt.show()

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

    class ODELLKFunc(nn.Module):
        def __init__(self, ce, x_shape, **kwargs):
            self.ce = ce
            self.kwargs = kwargs
            self.v = torch.randint(2, x_shape) * 2 - 1
            self.fevals = 0
            super().__init__()

        def forward(self, t, x):
            self.fevals += 1
            with torch.enable_grad():
                x = x[0].detach().requires_grad_()
                dx_dt = self.ce.get_dx_dt(t, x, evaluate_likelihood=True, **self.kwargs)
                grad = torch.autograd.grad((dx_dt * self.v).sum(), x)[0]
                d_ll = (self.v * grad).sum([-1, -2])
            out = torch.cat([dx_dt.reshape(-1), d_ll.reshape(-1)])
            return out

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
                top_dx = dx_dt.norm(dim=[1, 2]).topk(k=5)
                top_x = x[top_dx.indices]
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
        # ode_fn = ContinuousEvaluator.ODELLKFunc(self, x.shape, **kwargs)
        sol = odeint(
            ode_fn,
            x_min,
            times,
            atol=atol,
            rtol=rtol,
            method='rk4',
            # method='scipy_solver',
            # options={'solver': 'LSODA'},
        )
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


def plt_llk(traj, lik, figs_dir, plot_type='scatter', ax=None):
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

    plt.savefig('{}/scatter.pdf'.format(figs_dir))

def test_gaussian(end_time, cfg, sample_trajs, std):
    exited = (sample_trajs.abs() > std.likelihood.alpha).any(dim=1).to(float)
    prop_exited = exited.mean() * 100
    print('{}% of {} samples outside [-{}, {}]'.format(
        prop_exited,
        sample_trajs.shape[0],
        std.likelihood.alpha,
        std.likelihood.alpha,
    ))

    # plot_histogram of sample_trajs under analytical distribution
    plt.clf()
    plt.hist(sample_trajs.squeeze().numpy(), bins=100, edgecolor='black', density=True)
    x = np.linspace(-sample_trajs.max().item(), sample_trajs.max().item(), 1000)
    pdf = stats.norm.pdf(x) / (1 - stats.norm.cdf(std.likelihood.alpha))
    pdf[np.abs(x) < std.likelihood.alpha.numpy()] = 0
    plt.plot(x, pdf, color='red', label='Analytical PDF')
    plt.savefig('{}/gaussian_hist_w_analytical.pdf'.format(HydraConfig.get().run.dir))

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
    analytical_llk_w_nan = torch.where(
        torch.abs(traj - cfg.example.mu) > alpha * cfg.example.sigma,
        datapoint_dist.log_prob(traj) - tail.log(),
        torch.nan
    )
    non_nan_idx = ~torch.any(analytical_llk_w_nan.isnan(), dim=1)
    non_nan_analytical_llk = analytical_llk_w_nan[non_nan_idx]
    non_nan_a_lk = non_nan_analytical_llk.exp().squeeze()
    print('analytical_llk: {}'.format(non_nan_a_lk))
    ode_llk = std.ode_log_likelihood(sample_trajs, cond=cond, alpha=alpha)
    ode_lk = ode_llk[0].exp() / cfg.example.sigma
    non_nan_ode_lk = ode_lk[non_nan_idx.squeeze()].squeeze()
    print('\node_llk: {}\node evals: {}'.format(non_nan_ode_lk, ode_llk[1]))
    mse_llk = torch.nn.MSELoss()(non_nan_a_lk, non_nan_ode_lk)
    print('\nmse_llk: {}'.format(mse_llk))

    plt.clf()
    try:
        plt_llk(traj, ode_lk, HydraConfig.get().run.dir, plot_type='scatter')
        plt_llk(datapoints, datapoint_llk.exp(), HydraConfig.get().run.dir, plot_type='line')
    except Exception as e:
        print(f'error: {e}')

    import pdb; pdb.set_trace()

def plot_theta_from_sample_trajs(end_time, cfg, sample_trajs, std):
    theta = torch.atan2(sample_trajs[..., 1, :], sample_trajs[..., 0, :])
    plt.clf()
    plt.hist(
        theta.numpy(),
        bins=sample_trajs.shape[0] // 10,
        edgecolor='black',
        density=True
    )
    plt.savefig('{}/theta_hist.pdf'.format(HydraConfig.get().run.dir))
    plt.clf()

def plot_rayleigh_from_sample_trajs(cfg, sample_trajs, std, ode_llk):
    plt.clf()
    # dd = dist.MultivariateNormal(torch.zeros(2), torch.eye(2))
    # sample_trajs = dd.sample([10*sample_trajs.shape[0], 1]).movedim(1,2)
    # sample_trajs = sample_trajs[(sample_trajs.norm(dim=[1,2]) > std.likelihood.alpha.sqrt())]
    sample_levels = sample_trajs.norm(dim=[1, 2])  # [B]
    num_bins = sample_trajs.shape[0] // 10
    plt.hist(
        sample_levels.numpy(),
        bins=num_bins,
        edgecolor='black',
        density=True
    )

    # Plot analytical Rayleigh distribution using scipy
    x = np.linspace(0, sample_levels.max().item(), 1000)  # [1000]
    alpha = std.likelihood.alpha.item() if std.cond == 1. else 0.
    if alpha > 0:
        # For conditional distribution, need to normalize by P(X > alpha)
        pdf = stats.rayleigh.pdf(x) / (1 - stats.rayleigh.cdf(alpha))
        # Zero out values below alpha
        pdf[x < alpha] = 0
    else:
        pdf = stats.rayleigh.pdf(x)
    plt.plot(x, pdf, 'r-', label='Analytical PDF')
    plt.legend()
    plt.savefig('{}/rayleigh_hist.pdf'.format(HydraConfig.get().run.dir))

    # plot points (ode_llk, sample_trajs) against analytical rayleigh
    plt.clf()
    rayleigh_ode_lk = ode_llk.exp() * torch.tensor(2*np.pi) * sample_levels
    y = einops.repeat(x, 'x_len -> batch x_len', batch=sample_levels.shape[0])  # [B 1000]
    levels = einops.repeat(sample_levels, 'batch -> batch x_len', x_len=x.shape[0])  # [B 1000]
    idx = (levels > torch.tensor(y)).sum(dim=1)
    true_rayleigh_ode_lk = torch.tensor(pdf[idx])
    nonzero_idx = true_rayleigh_ode_lk.to(bool).nonzero()
    ratios = rayleigh_ode_lk[nonzero_idx] / true_rayleigh_ode_lk[nonzero_idx]
    plt.hist(ratios, bins=num_bins, edgecolor='black')
    plt.xlabel('Likelihood Ratio (Estimate / True)')
    plt.ylabel('Count')
    plt.savefig('{}/rayleigh_ratios.pdf'.format(HydraConfig.get().run.dir))

    # split sample_levels into 100 bins and compute the variance of the corresponding
    # rayleigh_ode_lk values
    plt.clf()
    level_bins = torch.linspace(
        sample_levels.min(),
        sample_levels.max(),
        num_bins
    )
    var = einops.reduce(
        rayleigh_ode_lk,
        '(n_bins bin_size) -> n_bins',
        torch.var,
        n_bins=num_bins
    )
    plt.scatter(level_bins, var)
    plt.ylabel('Variance')
    plt.xlabel('Radius')
    plt.savefig('{}/rayleigh_variances.pdf'.format(HydraConfig.get().run.dir))

    plt.clf()
    gamma = std.likelihood.gamma
    # plot sigmoid-adjusted conditional distribution
    cond = std.cond if std.cond is not None else -1
    if cond > 0:
        sigmoid = torch.sigmoid(gamma * (torch.tensor(x) - alpha))
        # plot sigmoid function as approximation to indicator
        plt.plot(x, sigmoid, label=f'Sigmoid({gamma} * (x-{alpha}))', color='violet')

        # For unnormalized conditional distribution
        unnormed_pdf = stats.rayleigh.pdf(x) * sigmoid.numpy()
        # trapezoid integration
        normalization_factor = scipy.integrate.trapezoid(y=unnormed_pdf, x=x)
        # normalized conditional
        normed_pdf = unnormed_pdf / normalization_factor
        plt.plot(x, normed_pdf, label='Sigmoid-Adjusted PDF', color='orange')

    plt.scatter(sample_levels, rayleigh_ode_lk, label='Density Estimates')
    plt.plot(x, pdf, 'r-', label='Analytical PDF')
    plt.legend()
    plt.savefig('{}/rayleigh_scatter.pdf'.format(HydraConfig.get().run.dir))

    plt.clf()
    trunc_idx = (ratios < 2).nonzero()
    plt.scatter(sample_levels[trunc_idx], rayleigh_ode_lk[trunc_idx])
    plt.plot(x, pdf, 'r-', label='Analytical PDF')
    plt.legend()
    plt.savefig('{}/truncated_rayleigh_scatter.pdf'.format(HydraConfig.get().run.dir))

def generate_diffusion_video(ode_llk, all_trajs, cfg):
    samples = all_trajs.squeeze()
    # generate gif of samples where each index in samples represents a frame
    import matplotlib.animation as animation
    import matplotlib.colors as colors

    # Create figure and axis
    fig, ax = plt.subplots()
    
    # Set up plot limits based on data range
    x_min, x_max = samples[..., 0].min(), samples[..., 0].max()
    y_min, y_max = samples[..., 1].min(), samples[..., 1].max()
    margin = 0.1 * max(x_max - x_min, y_max - y_min)
    ax.set_xlim(x_min - margin, x_max + margin)
    ax.set_ylim(y_min - margin, y_max + margin)

    # Create color normalization based on likelihood values
    norm = colors.Normalize(vmin=ode_llk.min(), vmax=ode_llk.max())
    
    # Initialize scatter plot
    scat = ax.scatter([], [], c=[], cmap='viridis', norm=norm)
    fig.colorbar(scat, label='Log Likelihood')
    
    def update(frame):
        # Update positions and colors for current frame
        scat.set_offsets(samples[frame, :, :2])
        scat.set_array(ode_llk)
        return scat,
    
    # Create animation
    frames = len(samples)
    anim = animation.FuncAnimation(
        fig,
        update,
        frames=frames,
        interval=100,  # 100ms between frames
        blit=True
    )

    # Save animation
    cond = 'conditional' if cfg.cond else 'unconditional'
    anim.save(f'{HydraConfig.get().run.dir}/{cond}_mvn_diffusion.gif', writer='pillow')
    plt.close()

def plot_likelihood_heat_maps(ode_llk, sample_trajs, cfg):
    """Create 2D heatmaps of mean and variance of log-likelihoods."""
    # Reshape inputs
    points = sample_trajs.squeeze(-1)  # Shape: (B, 2)
    llk = ode_llk.exp()  # Shape: (B,)

    # Calculate analytical likelihoods
    mu = torch.tensor(cfg.example.mu).squeeze(-1)
    sigma = torch.tensor(cfg.example.sigma)
    analytical_dist = torch.distributions.MultivariateNormal(mu, sigma)
    analytical_lk = analytical_dist.log_prob(points).exp()
    
    # Calculate estimated likelihoods and relative errors
    estimated_lk = ode_llk.exp()
    relative_errors = torch.abs(estimated_lk - analytical_lk) / analytical_lk

    # Create grid for interpolation
    margin = 0.1
    x_min, x_max = points[:, 0].min() - margin, points[:, 0].max() + margin
    y_min, y_max = points[:, 1].min() - margin, points[:, 1].max() + margin
    
    grid_size = 50
    x_grid = torch.linspace(x_min, x_max, grid_size)
    y_grid = torch.linspace(y_min, y_max, grid_size)
    X, Y = torch.meshgrid(x_grid, y_grid, indexing='ij')
    
    # Calculate grid cell assignments for each point
    x_bins = torch.bucketize(points[:, 0].contiguous(), x_grid) - 1
    y_bins = torch.bucketize(points[:, 1].contiguous(), y_grid) - 1
    
    # Create mean heatmap
    mean_Z = torch.zeros_like(X)
    count_Z = torch.zeros_like(X)
    
    # Create variance heatmap 
    squared_sum_Z = torch.zeros_like(X)
    
    # Create relative error heatmap 
    err_Z = torch.zeros_like(X)

    # Accumulate values in grid cells
    for i in range(len(points)):
        x_idx = x_bins[i]
        y_idx = y_bins[i]
        if 0 <= x_idx < grid_size and 0 <= y_idx < grid_size:
            mean_Z[x_idx, y_idx] += llk[i]
            squared_sum_Z[x_idx, y_idx] += llk[i] ** 2
            err_Z[x_idx, y_idx] += relative_errors[i]
            count_Z[x_idx, y_idx] += 1
    
    # Calculate mean and variance
    mask = count_Z > 0
    mean_Z[mask] /= count_Z[mask]
    var_Z = torch.zeros_like(mean_Z)
    var_Z[mask] = (squared_sum_Z[mask] / count_Z[mask]) - (mean_Z[mask] ** 2)
    
    err_Z[mask] /= count_Z[mask]
    
    # Plot heatmap
    plt.figure(figsize=(10, 8))
    plt.pcolormesh(X.numpy(), Y.numpy(), err_Z.numpy(), shading='auto', cmap='viridis')
    plt.colorbar(label='Relative Error')
    plt.title('Relative Error Heatmap')
    plt.xlabel('x')
    plt.ylabel('y')
    
    plt.savefig(f'{HydraConfig.get().run.dir}/relative_error_heatmap.pdf')
    plt.close()

    # Plot mean heatmap
    plt.figure(figsize=(10, 8))
    plt.subplot(1, 2, 1)
    plt.pcolormesh(X.numpy(), Y.numpy(), mean_Z.numpy(), shading='auto', cmap='viridis')
    plt.colorbar(label='Mean Density')
    plt.title('Mean Density Heatmap')
    plt.xlabel('x')
    plt.ylabel('y')
    
    # Plot variance heatmap
    plt.subplot(1, 2, 2)
    plt.pcolormesh(X.numpy(), Y.numpy(), var_Z.numpy(), shading='auto', cmap='viridis')
    plt.colorbar(label='Variance of Density')
    plt.title('Density Variance Heatmap')
    plt.xlabel('x')
    plt.ylabel('y')
    
    plt.tight_layout()
    plt.savefig(f'{HydraConfig.get().run.dir}/likelihood_heatmaps.pdf')
    plt.close()

    # Create scatter plot overlay
    plt.figure(figsize=(10, 8))
    plt.subplot(1, 2, 1)
    plt.pcolormesh(X.numpy(), Y.numpy(), mean_Z.numpy(), shading='auto', cmap='viridis', alpha=0.7)
    plt.scatter(points[:, 0], points[:, 1], c=llk, cmap='viridis', alpha=0.5, s=1)
    plt.colorbar(label='Density')
    plt.title('Mean Density with Sample Points')
    plt.xlabel('x')
    plt.ylabel('y')
    
    plt.subplot(1, 2, 2)
    plt.pcolormesh(X.numpy(), Y.numpy(), var_Z.numpy(), shading='auto', cmap='viridis', alpha=0.7)
    plt.scatter(points[:, 0], points[:, 1], c=llk, cmap='viridis', alpha=0.5, s=1)
    plt.colorbar(label='Density')
    plt.title('Variance with Sample Points')
    plt.xlabel('x')
    plt.ylabel('y')
    
    plt.tight_layout()
    plt.savefig(f'{HydraConfig.get().run.dir}/likelihood_heatmaps_with_points.pdf')
    plt.close()

def plot_likelihood_surface(sample_trajs, ode_llk, cfg):
    """
    Creates a 3D surface plot of likelihood values with optional analytical Gaussian overlay.
    
    Args:
        sample_trajs: Tensor of shape (B, 2, 1) containing x,y coordinates
        ode_llk: Tensor of shape (B,) containing likelihood values
        show_analytical: Boolean to toggle analytical Gaussian overlay
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    # Convert tensors to numpy arrays if needed
    if torch.is_tensor(sample_trajs):
        sample_trajs = sample_trajs.cpu().numpy()
    if torch.is_tensor(ode_llk):
        ode_llk = ode_llk.cpu().numpy()
    
    # Extract x and y coordinates
    x = sample_trajs[:, 0, 0]  # First dimension values
    y = sample_trajs[:, 1, 0]  # Second dimension values
    z = np.exp(ode_llk)  # Likelihood values
    
    # Create the 3D plot
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create the scatter plot
    scatter = ax.scatter(x, y, z, c=z, cmap='viridis', alpha=0.6, label='Samples')
    
    # Create a grid of points for the analytical surface
    x_grid = np.linspace(min(x)-0.5, max(x)+0.5, 100)
    y_grid = np.linspace(min(y)-0.5, max(y)+0.5, 100)
    X, Y = np.meshgrid(x_grid, y_grid)
    
    # Bivariate Gaussian parameters
    mean = np.array([3./5., 3./5.])
    variance = 4/5

    # Calculate the Gaussian PDF
    Z = np.zeros_like(X)
    for i in range(len(x_grid)):
        for j in range(len(y_grid)):
            point = np.array([X[i, j], Y[i, j]])
            Z[i, j] = (1/(2*np.pi*variance)) * np.exp(
                -0.5 * np.sum((point - mean)**2) / variance
            )

    # Plot the analytical surface
    surf = ax.plot_surface(X, Y, Z, cmap='Reds', alpha=0.3, label='Analytical')
    
    # Add labels and colorbar
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Likelihood')
    plt.colorbar(scatter)
    
    # Add title and legend
    ax.set_title('Likelihood Surface: Samples vs Analytical')
    
    # Hack to add legend for surface plot
    fake_surf = plt.Rectangle((0, 0), 1, 1, fc='r', alpha=0.3)
    ax.legend([scatter, fake_surf], ['Samples', 'Analytical Gaussian'])
    plt.savefig(f'{HydraConfig.get().run.dir}/p(x_0|y)_3d_samples.pdf')
    
    return fig, ax

def test_multivariate_gaussian(end_time, cfg, sample_trajs, std, all_trajs):
    plt.clf()
    alpha = torch.tensor([std.likelihood.alpha]) if std.cond == 1. else torch.tensor([0.])
    sample_levels = sample_trajs.norm(dim=[1, 2])
    exited = (sample_levels > std.likelihood.alpha).to(float)
    prop_exited = exited.mean() * 100
    print('{}% of {} samples outside Level {}'.format(
        prop_exited,
        sample_trajs.shape[0],
        std.likelihood.alpha,
    ))

    cond = std.cond if std.cond else torch.tensor([-1.])
    mu = torch.tensor(cfg.example.mu)
    sigma = torch.tensor(cfg.example.sigma)
    L = torch.linalg.cholesky(sigma)
    traj = torch.matmul(L, sample_trajs) + mu  # Shape: (N, d, 1)

    # Find the largest level curve value for the samples
    max_level = sample_levels.max().ceil().item()
    levels = torch.arange(0, max_level + 1)

    # Calculate the radius needed to contain the largest level curve
    # For a given level value k, points (x,y) on the curve satisfy:
    # (x-μ)^T Σ^(-1) (x-μ) = k
    # For a 2D Gaussian, this forms an ellipse
    # Get eigenvalues of covariance matrix to find major/minor axes
    eigenvals = torch.linalg.eigvalsh(sigma)
    # Maximum distance from mean = sqrt(max_level * largest_eigenvalue)
    max_radius = torch.sqrt(max_level * eigenvals.max())
    
    # Add buffer and round up to nearest integer
    buffer = 2
    plot_radius = torch.ceil(max_radius + buffer).item()
    
    # Calculate plot limits centered on mean
    x_min = mu[0].item() - plot_radius
    x_max = mu[0].item() + plot_radius
    y_min = mu[1].item() - plot_radius 
    y_max = mu[1].item() + plot_radius

    # Generate grid points
    x = torch.linspace(x_min, x_max, 500)
    y = torch.linspace(y_min, y_max, 500)
    X, Y = torch.meshgrid(x, y, indexing="ij")
    points = torch.stack([X, Y], dim=-1)  # Shape: (500, 500, 2)

    # Compute the quadratic form (x - mean)^T P (x - mean)
    diff = points - mu[:2, 0]
    precision_matrix = sigma.pinverse()[:2, :2]
    Z = torch.einsum('...i,ij,...j->...', diff, precision_matrix, diff)  # Shape: (500, 500)

    # Convert to NumPy for plotting
    Z = Z.numpy()

    # Plot the level curves
    plt.figure(figsize=(8, 6))
    contour = plt.contour(
        X.numpy(),
        Y.numpy(),
        Z,
        levels=levels,
        colors='red'
    )
    plt.clabel(contour, inline=True, fontsize=8)

    # Add labels, title, and styling
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Level Curves of Gaussian and Diffusion Samples')
    plt.axhline(0, color='black', linewidth=0.5, linestyle='--')  # x-axis
    plt.axvline(0, color='black', linewidth=0.5, linestyle='--')  # y-axis
    plt.grid(alpha=0.3)
    plt.gca().set_aspect('equal', adjustable='box')  # Equal aspect ratio
    plt.scatter(traj[:, 0], traj[:, 1], color='blue')

    datapoint_dist = torch.distributions.MultivariateNormal(mu.squeeze(-1), sigma)
    # alpha == sqrt(x^2 + y^2) as indicated by how the `exited` variable is defined
    # at the top of this function
    tail = torch.exp(-alpha**2 / 2)
    non_nan_analytical_llk = datapoint_dist.log_prob(traj.squeeze(-1)) - tail.log()
    non_nan_a_lk = non_nan_analytical_llk.exp().squeeze()
    print('analytical_llk: {}'.format(non_nan_a_lk))
    ode_llk = std.ode_log_likelihood(sample_trajs, cond=cond, alpha=alpha.unsqueeze(-1))
    non_nan_ode_lk = (ode_llk[0] - L.det().abs().log()).exp()
    print('\node_llk: {}\node evals: {}'.format(non_nan_ode_lk, ode_llk[1]))

    avg_rel_error = torch.expm1(non_nan_a_lk - non_nan_ode_lk).abs().mean()
    print('\naverage relative error: {}'.format(avg_rel_error))

    plt.savefig('{}/ellipsoid_scatter.pdf'.format(HydraConfig.get().run.dir))

    if cfg.example.d == 2:
        plot_theta_from_sample_trajs(end_time, cfg, sample_trajs, std)
        plot_rayleigh_from_sample_trajs(cfg, sample_trajs, std, ode_llk[0])
        generate_diffusion_video(ode_llk[0], all_trajs, cfg)
        plot_likelihood_heat_maps(ode_llk[0], sample_trajs, cfg)
        plot_likelihood_surface(sample_trajs, ode_llk[0], cfg)
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
        ax = plt_llk(sample_trajs, ode_llk[0].exp(), HydraConfig.get().run.dir, plot_type='3d_scatter')
        plt_llk(sample_trajs, analytical_llk.exp(), HydraConfig.get().run.dir, plot_type='3d_line', ax=ax)
    else:
        plt_llk(sample_trajs, ode_llk[0].exp(), HydraConfig.get().run.dir, plot_type='scatter')
        plt_llk(sample_trajs, analytical_llk.exp(), HydraConfig.get().run.dir, plot_type='line')
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
    save_dir = '{}/{}'.format(HydraConfig.get().run.dir, cfg.model_name)
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
    print('{}% of {} trajectories exited [-{}, {}]'.format(
        prop_exited,
        bm_trajs.shape[0],
        std.likelihood.alpha,
        std.likelihood.alpha,
    ))

    # compute (discretized) "analytical" log likelihood
    uncond_analytical_llk = (
        dist.Normal(0, 1).log_prob(sample_trajs) - dt.sqrt().log()
    ).sum(1).squeeze()
    tail = get_target(std).analytical_prob(alpha) if alpha.numpy() else torch.tensor(1.)
    analytical_llk = uncond_analytical_llk - np.log(tail.item())
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
    plt_llk(traj, ode_lk, HydraConfig.get().run.dir, plot_type='scatter')
    plt_llk(traj, a_lk, HydraConfig.get().run.dir, plot_type='line')
    import pdb; pdb.set_trace()

def test_student_t(end_time, cfg, sample_trajs, std):
    cond = std.cond if std.cond else torch.tensor([-1.])
    sigma = torch.tensor(35.9865)  # from dist.StudentT(1.5).sample([100000000]).std()
    if std.cfg.example.nu > 2.:
        sigma = torch.tensor(std.cfg.example.nu / (std.cfg.example.nu - 2)).sqrt()
    traj = sample_trajs * sigma
    alpha = torch.tensor([std.likelihood.alpha]) if cond == 1. else torch.tensor([0.])
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
    plt_llk(traj, ode_lk, HydraConfig.get().run.dir, plot_type='scatter')
    plt_llk(datapoints, datapoint_llk.exp(), HydraConfig.get().run.dir, plot_type='line')
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
    save_dir = '{}/{}'.format(HydraConfig.get().run.dir, cfg.model_name)
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
        save_dir = '{}/{}'.format(std.HydraConfig.get().run.dir, std.cfg.model_name)
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

def test(end_time, cfg, out_trajs, std, all_trajs):
    if type(std.example) == GaussianExampleConfig:
        test_gaussian(end_time, cfg, out_trajs, std)
    elif type(std.example) == MultivariateGaussianExampleConfig:
        test_multivariate_gaussian(end_time, cfg, out_trajs, std, all_trajs)
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
        std.viz_trajs(out_traj, end_time, idx, HydraConfig.get().run.dir, clf=False)

@hydra.main(version_base=None, config_path="conf", config_name="continuous_sample_config")
def sample(cfg):
    logger = logging.getLogger("main")
    logger.info('run type: sample')
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

    logger.info(f'Num model params: {std.get_num_params()}')

    with torch.no_grad():
        if isinstance(std.diffusion_model, TemporalTransformerUnet):
            test_transformer_bm(end_time, std)
            exit()

        # dt = 1 / torch.tensor(cfg.example.sde_steps-1)
        # data = torch.load('bm_dataset.pt', map_location=device, weights_only=True)
        # trajs = data[:cfg.num_samples].diff(dim=1) / dt.sqrt()
        # max_idx = 0
        # observed_values = trajs[:, :max_idx]
        # observed_idx = torch.arange(max_idx)
        # sample_traj_out = std.sample_trajectories(
        #     cond=std.cond,
        #     alpha=std.likelihood.alpha.reshape(-1, 1),
        #     observed_values=observed_values,
        #     observed_idx=observed_idx,
        # )

        # TODO: delete comments
        # guidance = GuidanceType.NoGuidance
        # if std.cond == 1:
        #     guidance = GuidanceType.ClassifierFree
        # old_guidance = std.set_guidance(guidance)
        sample_traj_out = std.sample_trajectories(
            cond=std.cond,
            alpha=std.likelihood.alpha.reshape(-1, 1),
        )
        # std.set_guidance(old_guidance)

        # ode_trajs = (sample_traj_out.samples).reshape(-1, cfg.num_samples)
        # plot_ode_trajectories(ode_trajs)

        # TODO: uncomment
        print('fevals: {}'.format(sample_traj_out.fevals))
        sample_trajs = sample_traj_out.samples
        trajs = sample_trajs[-1]
        out_trajs = trajs

        # viz_trajs(cfg, std, out_trajs, end_time)

        # TODO: Remove
        # sample_trajs = torch.randn(20, 1000, 2, 1)
        # out_trajs = sample_trajs[-1]
        test(end_time, cfg, out_trajs, std, sample_trajs)
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
