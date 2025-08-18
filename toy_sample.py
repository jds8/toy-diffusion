#!/usr/bin/env python3
import os
import warnings
import logging
import time as timer
from typing_extensions import Callable, Optional, List, Union, Tuple
import math

from pathlib import Path

import matplotlib.animation as animation
import matplotlib.colors as colors
from matplotlib.ticker import MaxNLocator

from collections import namedtuple

import einops
import hydra
from hydra.core.config_store import ConfigStore
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
import scipy
import scipy.stats as stats
import numpy as np
import torch
import torch.distributions as dist
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd

from torchdiffeq import odeint

from toy_plot import SDE, analytical_log_likelihood, plot_ode_trajectories
from toy_configs import register_configs
from toy_train_config import SampleConfig, SMCSampleConfig, \
    get_model_path, get_classifier_path, ExampleConfig, \
    GaussianExampleConfig, BrownianMotionDiffExampleConfig, \
    UniformExampleConfig, StudentTExampleConfig, TestType, IntegratorType, \
    get_target, MultivariateGaussianExampleConfig, Integrator
from models.toy_sampler import AbstractSampler, interpolate_schedule
from toy_likelihoods import Likelihood, ClassifierLikelihood, GeneralDistLikelihood
from models.toy_temporal import TemporalTransformerUnet, TemporalClassifier, TemporalNNet, DiffusionModel
from models.toy_diffusion_models_config import GuidanceType, DiscreteSamplerConfig, ContinuousSamplerConfig
from histogram_error_utils import get_bm_pdf_map, \
    AnalyticalCalculator, DensityCalculator, TrapezoidCalculator, \
    SimpsonsRuleCalculator, MISE, SimpsonsMISE, \
    ErrorMeasure#, SumError, MaxError, ForwardKLError, ReverseKLError
from compute_quadratures import pdf_2d_quadrature_bm, pdf_3d_quadrature_bm


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

SampleOutput = namedtuple('SampleOutput', 'samples fevals x_min')

SDEConfig = namedtuple('SDEConfig', 'drift diffusion sde_steps end_time')
DiffusionConfig = namedtuple('DiffusionConfig', 'f g')
HistogramErrorsOutput = namedtuple(
    'HistogramErrorsOutput',
    'errors subsample_sizes all_num_bins'
)
AllHistogramOutputs = namedtuple(
    'AllHistogramOutputs',
    'errors all_subsamples all_props all_bins all_num_bins subsample_sizes'
)
HistogramData = namedtuple('HistogramData', 'histogram_errors_output label')
HistogramErrorBarOutput = namedtuple(
    'HistogramErrorBarOutput',
    'error_mu error_pct_5 error_pct_95 '\
    'best_line best_line_label subsample_sizes all_props_list all_bins_list all_num_bins other_title_prefix'
)

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

    def load_model_state_dict(self, model_path, map_location):
        model = torch.load(
            '{}'.format(model_path),
            map_location=map_location,
            weights_only=False
        )
        if 'model_state_dict' in model:
            self.diffusion_model.load_state_dict(model['model_state_dict'])
            num_params = self.diffusion_model.get_num_params()
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
        total_samples = self.cfg.num_samples*self.cfg.num_sample_batches
        if type(self.example) == GaussianExampleConfig:
            pseudo_example = self.cfg.example.copy()
            pseudo_example['mu'] = 0.
            pseudo_example['sigma'] = 1.
            mean, _, std = self.sampler.analytical_marginal_prob(
                t=torch.tensor(1.),
                example=pseudo_example,
            )
            std = var.sqrt()
            x_min = dist.Normal(
                mean.item(),
                std.item(),
                device
            ).sample([
                total_samples,
                1,
                1
            ])
        elif type(self.example) == MultivariateGaussianExampleConfig:
            d = self.cfg.example.d
            x_min = dist.MultivariateNormal(
                torch.zeros(d),
                torch.eye(d),
            ).sample([total_samples]).to(device).unsqueeze(-1)
        elif type(self.example) == BrownianMotionDiffExampleConfig:
            x_min = self.sampler.prior_sampling(device).sample([
                total_samples,
                self.cfg.example.sde_steps-1,
                1,
            ])
        elif type(self.example) == UniformExampleConfig:
            x_min = dist.Normal(0, 1, device).sample([
                total_samples, 1, 1
            ])
        else:
            x_min = dist.Normal(0, 1, device).sample([
                total_samples, 1, 1
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
        return SampleOutput(samples=samples, fevals=-1, x_min=x)

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
        sol = odeint(ode_fn, x_min, times, atol=atol, rtol=rtol, method=self.cfg.density_integrator)
        latent, delta_ll = sol[0][-1], sol[1][-1]
        ll_prior = self.sampler.prior_logp(latent, device=device).flatten(1).sum(1)
        # compute log(p(0)) = log(p(T)) + Tr(df/dx) where dx/dt = f
        return ll_prior + delta_ll, {'fevals': fevals}


class ContinuousEvaluator(ToyEvaluator):
    def get_score_function(self, t, x, evaluate_likelihood, **kwargs):
        if self.cfg.test == TestType.Gaussian:
            return self.analytical_gaussian_score(t=t, x=x)
        elif self.cfg.test == TestType.MultivariateGaussian:
            return self.analytical_multivariate_gaussian_score(t=t, x=x)
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
                cond = kwargs['cond'].to(device) if kwargs['cond'] is not None else None
                conditional_output = self.diffusion_model(
                    x=x.to(device),
                    time=t.to(device),
                    cond=cond,
                    alpha=kwargs['alpha'].to(device),
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

    def get_gaussian_marginal_prob_from_sampler(self, t, x):
        pseudo_example = self.cfg.example.copy()
        pseudo_example['mu'] = 0.
        pseudo_example['sigma'] = 1.
        mean, lmc, std = self.sampler.analytical_marginal_prob(
            t=t,
            example=pseudo_example
        )
        var = std ** 2
        return mean, lmc, var

    def get_multivariate_gaussian_marginal_prob_from_sampler(self, t, x):
        mu = torch.zeros(1, self.cfg.example.d, 1, device=device)
        sigma = torch.eye(self.cfg.example.d, device=device)
        beta = self.sampler.continuous_beta_integral(t)  # int_0^t beta(s) ds
        lmc = -0.5 * beta
        mean = lmc.exp() * mu
        first_var_term = sigma * (2. * lmc).exp()
        second_var_term = (1 - (2. * lmc).exp())
        var = first_var_term + second_var_term
        return mean, lmc, var

    def get_analytical_brownian_motion_diff_variance(self, t, x):
        beta = self.sampler.continuous_beta_integral(t)
        log_mean_coeff = -0.5 * beta[:, None, None]
        f = log_mean_coeff.exp()
        g = 1 - (2. * log_mean_coeff).exp()

        dt = 1. / (self.cfg.example.sde_steps-1)
        ds = (dt * torch.arange(1, self.cfg.example.sde_steps, device=f.device)).diag()

        var = f ** 2 * ds + g * torch.eye(2, 2, device=g.device)
        return var

    def get_variance_from_sampler(self, t, x):
        if isinstance(self.example, GaussianExampleConfig):
            return self.get_gaussian_marginal_prob_from_sampler(t, x)[2]
        elif isinstance(self.example, MultivariateGaussianExampleConfig):
            return self.get_multivariate_gaussian_marginal_prob_from_sampler(t, x)[2]
        elif isinstance(self.example, BrownianMotionDiffExampleConfig):
            return self.get_analytical_brownian_motion_diff_variance(t, x)
        else:
            raise NotImplementedError

    def analytical_gaussian_score(self, t, x):
        """
        Compute the analytical marginal score of p_t for t in (0, 1)
        given the SDE formulation from Song et al. in the case that
        p_0 = N(mu_0, sigma_0) and p_1 = N(0, 1)
        """
        mean, lmc, std = self.get_gaussian_marginal_prob_from_sampler(t, x)
        var = std ** 2
        score = (mean - x) / var
        return score

    def analytical_multivariate_gaussian_score(self, t, x):
        """
        Compute the analytical marginal score of p_t for t in (0, 1)
        given the SDE formulation from Song et al. in the case that
        p_0 = N(mu_0, sigma_0) and p_1 = N(0, 1)
        """
        mean, lmc, std = self.get_multivariate_gaussian_marginal_prob_from_sampler(t, x)
        var = std ** 2
        precision = var.pinverse()
        score = torch.matmul(precision, mean - x)
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

        d = self.cfg.example.sde_steps - 1
        var = (f ** 2 + g ** 2) * torch.eye(d, device=g.device)
        assert var.det() != 0
        precision = var.pinverse()
        score = -torch.matmul(precision, x)

        return score

    def analytical_student_t_score(self, t, x):
        """
        Compute the analytical marginal score of p_t for t in (0, 1)
        given the SDE formulation from Song et al. in the case that
        p_0 = T(degress_of_freedom=nu, loc=0, scale=1) and p_1 = N(0, 1)
        https://chatgpt.com/c/0e26fe1f-e8e7-4fce-8b89-719fa0479589
        nabla log p(x) = -(nu + 1)(x - mu) / sigma^2(nu + (x-mu)^2 / sigma^2)
        """

        mean, lmc, std = self.get_gaussian_marginal_prob_from_sampler(t, x)
        var = std ** 2
        num = -(self.cfg.example.nu + 1) * (x - mean)
        denom = var * (self.cfg.example.nu + (x - mean) ** 2 / var)
        return num / denom

    def sample_trajectories_euler_maruyama(self, **kwargs):
        x_min = self.get_x_min()
        x = x_min.clone()

        steps = torch.tensor(self.sampler.diffusion_timesteps).to(x.device)
        for time in torch.linspace(1., self.sampler.t_eps, steps, device=x.device):
            time = time.reshape(-1)
            sf_est = self.get_score_function(
                t=time,
                x=x,
                evaluate_likelihood=False,
                **kwargs,
            )
            x, _ = self.sampler.reverse_sde(x=x, t=time, score=sf_est, steps=steps)

        return SampleOutput(samples=torch.stack([x_min, x]), fevals=steps, x_min=x_min)

    def sample_trajectories_probability_flow(self, atol=1e-7, rtol=1e-7, **kwargs):
        if 'x_min' in kwargs:
            x_min = kwargs['x_min']
        else:
            x_min = self.get_x_min()

        fevals = 0
        def ode_fn(t, x):
            nonlocal fevals
            fevals += 1
            if 'observed_idx' in kwargs:
                x[:, kwargs['observed_idx']] = kwargs['observed_values']
            dx_dt = self.get_dx_dt(t, x, evaluate_likelihood=False, **kwargs)
            return dx_dt

        end_time = self.sampler.t_eps if self.cfg.test == TestType.Test else 0.
        times = torch.linspace(
            1.,
            end_time,
            self.sampler.diffusion_timesteps,
            device=x_min.device
        )
        if self.cfg.sample_integrator == Integrator.EULER:
            sol = self.euler_integrate(ode_fn, x_min, times)
        elif self.cfg.sample_integrator == Integrator.HEUN:
            sol = self.heun_integrate(ode_fn, x_min, times)
        elif self.cfg.sample_integrator == Integrator.RK4:
            sol = self.rk4_integrate(ode_fn, x_min, times)
        else:
            sol = odeint(ode_fn, x_min, times, atol=atol, rtol=rtol, method='rk4')

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

        return SampleOutput(samples=sol, fevals=fevals, x_min=x_min)

    def compute_bins(
            self,
            alpha: np.ndarray,
            samples: torch.tensor,
            subsample_sizes: np.ndarray
    ):
        all_props = []
        all_bins = []
        all_num_bins = []
        all_subsamples = []
        max_sample = samples.max().cpu().numpy()
        for subsample_size in subsample_sizes:
            subsap = samples[:subsample_size.astype(int)].cpu().numpy()
            # bin_width = subsample_size ** (-1/3)  # Optimal
            # bin_width = 1 / np.log2(subsample_size)  # Sturges Rule
            bin_width = 2. * scipy.stats.iqr(subsap) * subsap.shape[0] ** (-1/3)  # Freedman-Diaconis
            # num_bins = int((max_sample - alpha.item()) / bin_width)
            num_bins = int((subsap.max() - alpha.item()) / bin_width)
            subsamples, bins = np.histogram(
                subsap,
                bins=num_bins,
                density=True,
                # range=(alpha.item(), max_sample)
            )
            # props = subsamples / (subsample_size * bin_width)
            props = subsamples
            all_subsamples.append(subsamples)
            all_props.append(props)
            all_bins.append(bins)
            all_num_bins.append(num_bins)
        return all_subsamples, all_props, all_bins, all_num_bins

    def compute_all_sample_errors(
        self,
        total_raw_samples: torch.tensor,  # [B, D, 1]
        alpha: np.ndarray,
        subsample_sizes: np.ndarray,
        analytical_calculator: AnalyticalCalculator,
        error_measure: ErrorMeasure,
    ):
        dim = total_raw_samples.shape[1]
        total_samples = total_raw_samples.norm(dim=[1, 2])  # [B]
        all_subsamples, all_props, all_bins, all_num_bins = self.compute_bins(
            alpha,
            total_samples,
            subsample_sizes
        )
        errors = []
        for empirical_props, bins in zip(all_props, all_bins):
            error = error_measure(
                analytical_calculator,
                empirical_props,
                bins,
                alpha,
            )
            errors.append(error)
        return errors, all_subsamples, all_props, all_bins, all_num_bins

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
    def heun_integrate(self, ode_fn: Callable, x: Tuple, times: torch.Tensor) -> List:
        if type(x) == tuple:
            return self.heun_integrate_tuple(ode_fn, x, times)
        return self.heun_integrate_tensor(ode_fn, x, times)

    @torch.no_grad()
    def heun_integrate_tensor(self, ode_fn: Callable, x: torch.Tensor, times: torch.Tensor) -> List:
        sol = torch.zeros(times.shape[:1] + x.shape, device=x.device)
        for i, (t1, t2) in enumerate(zip(times[:-1], times[1:])):
            sol[i] = x
            x = self.heun_step_tensor(ode_fn, x, t1, t2)
        sol[-1] = x
        return sol

    @torch.no_grad()
    def heun_step_tensor(
            self,
            ode_fn: Callable,
            x: torch.Tensor,
            t1: torch.Tensor,
            t2: torch.Tensor,
    ):
        dt = t2-t1
        x1, dx1_dt = self.euler_step_tensor(ode_fn, x, t1, dt)
        _, dx2_dt = self.euler_step_tensor(ode_fn, x1, t2, dt)
        return x + (t2 - t1) * (dx2_dt + dx1_dt) / 2  # trapezoid rule

    @torch.no_grad()
    def heun_integrate_tuple(self, ode_fn: Callable, x: Tuple, times: torch.Tensor) -> List:
        sol0 = torch.zeros(times.shape[:1] + x[0].shape, device=x[0].device)
        sol1 = torch.zeros(times.shape[:1] + x[1].shape, device=x[1].device)
        for i, (t1, t2) in enumerate(zip(times[:-1], times[1:])):
            sol0[i] = x[0]
            sol1[i] = x[1]
            x = self.heun_step_tuple(ode_fn, x, t1, t2)
        sol0[-1] = x[0]
        sol1[-1] = x[1]
        return (sol0, sol1)

    @torch.no_grad()
    def heun_step_tuple(
            self,
            ode_fn: Callable,
            x: Tuple,
            t1: torch.Tensor,
            t2: torch.Tensor
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor],
               Tuple[torch.Tensor, torch.Tensor]]:
        dt = t2-t1
        x1, dx1_dt = self.euler_step_tuple(ode_fn, x, t1, dt)
        _, dx2_dt = self.euler_step_tuple(ode_fn, x1, t2, dt)
        y0 = x[0] + (t2 - t1) * (dx2_dt[0] + dx1_dt[0]) / 2  # trapezoid rule
        y1 = x[1] + (t2 - t1) * (dx2_dt[1] + dx1_dt[1]) / 2  # trapezoid rule
        y = (y0, y1)
        return y

    @torch.no_grad()
    def euler_integrate(self, ode_fn: Callable, x: Tuple, times: torch.Tensor) -> List:
        if type(x) == tuple:
            return self.euler_integrate_tuple(ode_fn, x, times)
        return self.euler_integrate_tensor(ode_fn, x, times)

    @torch.no_grad()
    def euler_integrate_tensor(self, ode_fn: Callable, x: torch.Tensor, times: torch.Tensor) -> List:
        sol = torch.zeros(times.shape[:1] + x.shape, device=x.device)
        for i, (t, dt) in enumerate(zip(times[:-1], times.diff())):
            sol[i] = x
            x, _ = self.euler_step_tensor(ode_fn, x, t, dt)
        sol[-1] = x
        return sol

    @torch.no_grad()
    def euler_integrate_tuple(self, ode_fn: Callable, x: Tuple, times: torch.Tensor) -> List:
        sol0 = torch.zeros(times.shape[:1] + x[0].shape, device=x[0].device)
        sol1 = torch.zeros(times.shape[:1] + x[1].shape, device=x[1].device)
        for i, (t, dt) in enumerate(zip(times[:-1], times.diff())):
            sol0[i] = x[0]
            sol1[i] = x[1]
            x, _ = self.euler_step_tuple(ode_fn, x, t, dt)
        sol0[-1] = x[0]
        sol1[-1] = x[1]
        return (sol0, sol1)

    @torch.no_grad()
    def euler_step_tensor(
            self,
            ode_fn: Callable,
            x: Tuple,
            t: torch.Tensor,
            dt: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        dx_dt = ode_fn(t, x)
        return x + dx_dt * dt, dx_dt

    @torch.no_grad()
    def euler_step_tuple(
            self,
            ode_fn: Callable,
            x: Tuple,
            t: torch.Tensor,
            dt: torch.Tensor
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor],
               Tuple[torch.Tensor, torch.Tensor]]:
        reshaped_x0 = x[0].reshape(-1)
        x0_len = len(reshaped_x0)
        dx_dt = ode_fn(t, x)
        dx_dt0 = dx_dt[:x0_len].reshape(x[0].shape)
        dx_dt1 = dx_dt[x0_len:].reshape(x[1].shape)
        x0 = x[0] + dx_dt0 * dt
        x1 = x[1] + dx_dt1 * dt
        x = (x0, x1), (dx_dt0, dx_dt1)
        return x

    @torch.no_grad()
    def rk4_integrate(self, ode_fn: Callable, x: Tuple, times: torch.Tensor) -> List:
        if type(x) == tuple:
            return self.rk4_integrate_tuple(ode_fn, x, times)
        return self.rk4_integrate_tensor(ode_fn, x, times)

    @torch.no_grad()
    def rk4_integrate_tensor(self, ode_fn: Callable, x: torch.Tensor, times: torch.Tensor) -> List:
        sol = torch.zeros(times.shape[:1] + x.shape, device=x.device)
        for i, (t, dt) in enumerate(zip(times[:-1], times.diff())):
            sol[i] = x
            x = self.rk4_step_tensor(ode_fn, x, t, dt)
        sol[-1] = x
        return sol

    @torch.no_grad()
    def rk4_integrate_tuple(self, ode_fn: Callable, x: Tuple, times: torch.Tensor) -> List:
        sol0 = torch.zeros(times.shape[:1] + x[0].shape, device=x[0].device)
        sol1 = torch.zeros(times.shape[:1] + x[1].shape, device=x[1].device)
        for i, (t, dt) in enumerate(zip(times[:-1], times.diff())):
            sol0[i] = x[0]
            sol1[i] = x[1]
            x = self.rk4_step_tuple(ode_fn, x, t, dt)
        sol0[-1] = x[0]
        sol1[-1] = x[1]
        return (sol0, sol1)

    @torch.no_grad()
    def rk4_step_tensor(
            self,
            ode_fn: Callable,
            x: torch.Tensor,
            t: torch.Tensor,
            dt: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        k1 = ode_fn(t, x)
        k2 = ode_fn(t+dt/2, x+dt*k1/2)
        k3 = ode_fn(t+dt/2, x+dt*k2/2)
        k4 = ode_fn(t+dt, x+dt*k3)
        y = x + dt*(k1+2*k2+2*k3+k4)/6
        return y

    @torch.no_grad()
    def tuple_call(self, fn: Callable, t_in: torch.Tensor, x_in: Tuple, x0_len: int):
        out = fn(t_in, x_in)
        return out[:x0_len].reshape(x_in[0].shape), out[x0_len:].reshape(x_in[1].shape)

    @torch.no_grad()
    def tuple_add(self, t1: Tuple, t2: Tuple):
        return (t1[0] + t2[0], t1[1] + t2[1])

    @torch.no_grad()
    def tuple_multiply(self, t: Tuple, k):
        return (t[0] * k, t[1] * k)

    @torch.no_grad()
    def rk4_step_tuple(
            self,
            ode_fn: Callable,
            x: Tuple,
            t: torch.Tensor,
            dt: torch.Tensor
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor],
               Tuple[torch.Tensor, torch.Tensor]]:
        reshaped_x0 = x[0].reshape(-1)
        x0_len = len(reshaped_x0)
        k1 = self.tuple_call(ode_fn, t, x, x0_len)
        k2 = self.tuple_call(ode_fn, t+dt/2, self.tuple_add(x, self.tuple_multiply(k1, dt/2)), x0_len)
        k3 = self.tuple_call(ode_fn, t+dt/2, self.tuple_add(x, self.tuple_multiply(k2, dt/2)), x0_len)
        k4 = self.tuple_call(ode_fn, t+dt, self.tuple_add(x, self.tuple_multiply(k3, dt)), x0_len)
        k12 = self.tuple_add(k1, self.tuple_multiply(k2, 2))
        k34 = self.tuple_add(self.tuple_multiply(k3, 2), k4)
        k1234 = self.tuple_add(k12, k34)
        delta_x = self.tuple_multiply(k1234, dt/6)
        y = self.tuple_add(x, delta_x)
        return y

    @torch.no_grad()
    def ode_log_likelihood(self, x, **kwargs):
        print('evaluating likelihood...')
        fevals = 0
        if 'num_hutchinson_samples' not in kwargs:
            kwargs['num_hutchinson_samples'] = 1
        def exact_ode_fn(t, x):
            nonlocal fevals
            fevals += 1
            with torch.enable_grad():
                # x is a tuple and x[0].shape is BxDx1 where B is batch size and D is dimension
                x = x[0].detach().requires_grad_()
                dx_dt = lambda y: self.get_dx_dt(t, y, evaluate_likelihood=True, **kwargs)
                d_ll = torch.zeros(x.shape[0], device=x.device)
                for i in range(x.shape[1]):
                    v = torch.zeros_like(x)
                    v[:, i, 0] = 1.0
                    dx, vjp = torch.autograd.functional.vjp(dx_dt, x, v)
                    d_ll += (vjp * v).sum([-1, -2])
                out = torch.cat([dx.reshape(-1), d_ll.reshape(-1)])
                return out
        def hutchinson_ode_fn(t, x):
            nonlocal fevals
            fevals += 1
            with torch.enable_grad():
                x = x[0].detach().requires_grad_()
                dx_dt = lambda y: self.get_dx_dt(t, y, evaluate_likelihood=True, **kwargs)
                d_ll = torch.zeros(x.shape[0], device=x.device)
                for _ in range(kwargs['num_hutchinson_samples']):
                    # hutchinson's trick
                    v = torch.randint_like(x, 2) * 2 - 1
                    dx, vjp = torch.autograd.functional.vjp(dx_dt, x, v)
                    d_ll += (vjp * v).sum([-1, -2]) / kwargs['num_hutchinson_samples']
                out = torch.cat([dx.reshape(-1), d_ll.reshape(-1)])
            return out
        ll = x.new_zeros([x.shape[0]])
        x_min = x, ll
        start_time = self.sampler.t_eps if self.cfg.test == TestType.Test else 0.
        times = torch.linspace(
            start_time,
            1.,
            self.sampler.diffusion_timesteps,
            device=x.device
        )
        # times = torch.linspace(
        #     self.sampler.t_eps,
        #     1.-self.sampler.t_eps,
        #     self.sampler.diffusion_timesteps,
        #     device=x.device
        # )
        ode_fn = exact_ode_fn if 'exact' in kwargs and kwargs['exact'] else hutchinson_ode_fn
        if self.cfg.density_integrator == Integrator.EULER:
            sol = self.euler_integrate(ode_fn, x_min, times)
        elif self.cfg.density_integrator == Integrator.HEUN:
            sol = self.heun_integrate(ode_fn, x_min, times)
        elif self.cfg.density_integrator == Integrator.RK4:
            sol = self.rk4_integrate(ode_fn, x_min, times)
        else:
            sol = odeint(ode_fn, x_min, times, atol=self.cfg.atol, rtol=self.cfg.rtol, method='dopri5')
        delta_ll = sol[1].diff(dim=0).flip(dims=[0]).cumsum(dim=0)
        delta_ll = torch.concat([
            torch.zeros(1, delta_ll.shape[-1], device=delta_ll.device),
            delta_ll
        ])
        # summed_ll = sol[1].diff(dim=0).flip(dims=[0]).cumsum(dim=0)
        # delta_ll = torch.concat([summed_ll, summed_ll[-1:]])
        latent = sol[0][-1]
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
        return ll_output, {'fevals': fevals}, sol, latent, delta_ll


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

def compute_ode_log_likelihood(
        analytical_llk: torch.Tensor,
        hist_llks: List[torch.Tensor],
        sample_trajs: torch.Tensor,
        std: ToyEvaluator,
        cfg: SampleConfig,
        alpha: torch.Tensor,
        scale_fn: Callable,
):
    analytical_llk = analytical_llk.cpu()
    print('analytical_llk: {}'.format(analytical_llk))

    # compute log likelihood under diffusion model
    tm = timer.time()
    ode_llk = std.ode_log_likelihood(
        sample_trajs,
        cond=std.cond,
        alpha=alpha,
        exact=cfg.compute_exact_trace,
        num_hutchinson_samples=cfg.num_hutchinson_samples,
    )
    eval_time = timer.time() - tm
    print(f'pfode time: {eval_time}')
    ode_llk_val = ode_llk[0].cpu()
    scaled_ode_llk = scale_fn(ode_llk_val[-1]).cpu()
    print('\node_llk: {}'.format(scaled_ode_llk))

    # compare log likelihoods by MSE
    avg_rel_error = torch.expm1(analytical_llk - scaled_ode_llk).abs().mean()
    print('\naverage relative error: {}'.format(avg_rel_error))

    rkl_pfode = (scaled_ode_llk - analytical_llk).mean()
    print('\nreverse KL divergence btwn diffusion and analytical: {}'.format(rkl_pfode))

    if hist_llks:
        sample_trajs_list = einops.rearrange(
            sample_trajs,
            '(n c) b 1 -> n c b 1',
            n=cfg.num_sample_batches
        )
        unsorted_sample_levels = sample_trajs_list.norm(dim=[2, 3])[:, :hist_llks[0].shape[0]]
        sample_levels = unsorted_sample_levels.sort(dim=1).values
        dim = sample_trajs.shape[1]
        dd = scipy.stats.chi(dim)
        samples_cpu = sample_levels.cpu().numpy()
        log_pdfs = np.log(dd.pdf(samples_cpu)) - np.log(1 - dd.cdf(samples_cpu))
        log_pdfs_tensor = torch.tensor(log_pdfs)
        rkl_hists = torch.tensor([
            (hist_llk - pdf).mean() for hist_llk, pdf in zip(hist_llks, log_pdfs_tensor)
        ])
        quantiles = rkl_hists.quantile(
            torch.tensor(
                [0.05, 0.5, 0.95],
                dtype=rkl_hists.dtype
            )
        )
        print(f'\n reverse KL divergence quantiles (0.05, 0.5, 0.95) btwn hist of diffusion and analytical: {quantiles}')
    else:
        quantiles = torch.tensor([-1, -1, -1])

    torch.save(
        ode_llk_val,
        f'{HydraConfig.get().run.dir}/{cfg.model_name}_ode_llk.pt'
    )
    headers = [
        'Avg. Rel. Error',
        'Reverse KL (PFODE)',
        'Reverse KL (Histogram) quantiles (0.05, 0.5, 0.95)',
        'Time',
        'Model',
        'Diffusion Timesteps',
        'is_exact',
        'num hutchinson trace samples'
    ]
    data = [[
        avg_rel_error.cpu().numpy(),
        rkl_pfode.cpu().numpy(),
        quantiles.cpu().numpy(),
        eval_time,
        cfg.model_name,
        cfg.sampler.diffusion_timesteps,
        cfg.compute_exact_trace,
        cfg.num_hutchinson_samples if not cfg.compute_exact_trace else 0
    ]]
    df = pd.DataFrame(data, columns=headers)
    df.to_csv(
        f'{HydraConfig.get().run.dir}/{cfg.model_name}_ode_eval' \
        f'_exact?_{cfg.compute_exact_trace}_HS_samples_' \
        f'{cfg.num_hutchinson_samples}.csv',
        index=False
    )
    return ode_llk, scaled_ode_llk

def compute_histogram_errors(
        sample_trajs: torch.Tensor,
        alpha: np.ndarray,
        std: ToyEvaluator,
        analytical_calculator: AnalyticalCalculator,
        error_measure: ErrorMeasure,
):
    max_samples = max(sample_trajs.shape[0], 50000)
    all_subsample_sizes = np.linspace(50, max_samples, 100)
    subsample_sizes = all_subsample_sizes[
        (all_subsample_sizes <= sample_trajs.shape[0]).nonzero()[0]
    ]
    errors, all_subsamples, all_props, all_bins, all_num_bins = std.compute_all_sample_errors(
        sample_trajs,
        alpha,
        subsample_sizes=subsample_sizes,
        analytical_calculator=analytical_calculator,
        error_measure=error_measure,
    )
    error_measure.clean_up()
    return AllHistogramOutputs(
        errors,
        all_subsamples,
        all_props,
        all_bins,
        all_num_bins,
        subsample_sizes,
    )

def compute_multiple_histogram_errors(
        sample_trajs_list: torch.Tensor,
        alpha_np: np.ndarray,
        std: ToyEvaluator,
        analytical_calculator: AnalyticalCalculator,
        error_measure: ErrorMeasure,
        title_prefix: str,
):
    subsample_sizes_list = []
    errors_list = []
    all_props_list = []
    all_bins_list = []
    for sample_trajs in sample_trajs_list:
        all_hist_outputs = compute_histogram_errors(
            sample_trajs,
            alpha_np,
            std,
            analytical_calculator=analytical_calculator,
            error_measure=error_measure
        )
        errors = all_hist_outputs.errors
        errors_list.append(errors)
        subsample_sizes = all_hist_outputs.subsample_sizes
        subsample_sizes_list.append(subsample_sizes)
        all_props = all_hist_outputs.all_props
        all_props_list.append(all_props)
        all_bins = all_hist_outputs.all_bins
        all_bins_list.append(all_bins)
        all_num_bins = all_hist_outputs.all_num_bins

    all_subsample_sizes = np.array(subsample_sizes_list)
    subsaps = einops.rearrange(all_subsample_sizes, 'n l -> (l n)')
    all_errors = np.array(errors_list)
    errs = einops.rearrange(all_errors, 'n l -> (l n)')
    m, b = np.polyfit(np.log(subsaps), np.log(errs), 1)
    best_line = np.exp(m*np.log(subsample_sizes) + b)
    best_line_label = f"{title_prefix}: y = {m:.2f}x + {b:.2f}"
    error_mu = np.array(errors_list).mean(0)
    error_pct_5 = np.percentile(np.array(errors_list), 5, 0)
    error_pct_95 = np.percentile(np.array(errors_list), 95, 0)
    return HistogramErrorBarOutput(
        error_mu,
        error_pct_5,
        error_pct_95,
        best_line,
        best_line_label,
        subsample_sizes,
        all_props_list,
        all_bins_list,
        all_num_bins,
        title_prefix,
    )

def plot_histogram_errors(
        alpha: np.ndarray,
        dim: int,
        histogram_error_bar_output: HistogramErrorBarOutput,
        error_measure_label: str,
        title_prefix: str,
        cfg: SampleConfig,
        other_histogram_data: Optional[List[HistogramErrorBarOutput]]=None,
):
    plt.clf()

    fig, ((ax1, ax3), (ax2, ax4)) = plt.subplots(2, 2)

    subsample_sizes = torch.tensor(histogram_error_bar_output.subsample_sizes).unique()

    repeat_suffix = f' ({cfg.num_sample_batches} runs)'
    if other_histogram_data is not None:
        for other_histogram_datum in other_histogram_data:
            other_error_mu = other_histogram_datum.error_mu
            other_error_pct_5 = other_histogram_datum.error_pct_5
            other_error_pct_95 = other_histogram_datum.error_pct_95
            other_best_line = other_histogram_datum.best_line
            other_best_line_label = other_histogram_datum.best_line_label
            other_title_prefix = other_histogram_datum.other_title_prefix + repeat_suffix
            other_subsample_sizes = other_histogram_datum.subsample_sizes
            other_lwr = other_error_mu - other_error_pct_5
            other_upr = other_error_pct_95 - other_error_mu
            other_yerr = np.array([other_lwr, other_upr])
            ax1.errorbar(
                other_subsample_sizes,
                other_histogram_datum.error_mu,
                yerr=other_yerr,
                label=other_title_prefix,
                color='black',
            )
            ax1.plot(
                other_subsample_sizes,
                other_best_line,
                label=other_best_line_label,
                color='gray'
            )

    increment_size = int(np.diff(subsample_sizes)[0])
    ax1.set_xlabel(f'Sample Size (increments of {increment_size})')
    ax1.set_ylabel('Error')
    ax1.set_title(f'{error_measure_label} vs. Sample Size')

    error_mu = histogram_error_bar_output.error_mu
    error_pct_5 = histogram_error_bar_output.error_pct_5
    error_pct_95 = histogram_error_bar_output.error_pct_95
    best_line = histogram_error_bar_output.best_line
    best_line_label = histogram_error_bar_output.best_line_label

    lwr = error_mu - error_pct_5
    upr = error_pct_95 - error_mu
    yerr = np.array([lwr, upr])

    ax1.errorbar(
        subsample_sizes,
        error_mu,
        yerr=yerr,
        label=title_prefix + repeat_suffix,
        color='b',
    )

    # Compute theoretical rate
    bin_width = subsample_sizes ** (-1/3)  # Optimal
    # bin_width = 1 / np.log2(subsample_sizes)  # Sturges Rule
    # theoretical_rate = 1 / (subsample_sizes * bin_width)
    one_half_rate = 1 / (subsample_sizes ** (1/2))
    two_thirds_rate = 1 / (subsample_sizes ** (2/3))
    four_fifths_rate = 1 / (subsample_sizes ** (4/5))
    # Normalize to match MISE scale
    one_half_rate *= best_line[0] / one_half_rate[0]
    two_thirds_rate *= best_line[0] / two_thirds_rate[0]
    four_fifths_rate *= best_line[0] / four_fifths_rate[0]

    ax1.plot(
        subsample_sizes,
        best_line,
        label=best_line_label,
        color='g'
    )
    ax1.plot(
        subsample_sizes,
        one_half_rate,
        label='x^(-0.5)',
        linestyle='--',
        color='r'
    )
    ax1.plot(
        subsample_sizes,
        two_thirds_rate,
        label='x^(-0.667) (theoretical rate)',
        linestyle='-',
        color='r'
    )
    ax1.plot(
        subsample_sizes,
        four_fifths_rate,
        label='x^(-0.8)',
        linestyle=':',
        color='r'
    )

    if other_histogram_data is not None:
        for other_histogram_datum in other_histogram_data:
            other_error_mu = other_histogram_datum.error_mu
            other_error_pct_5 = other_histogram_datum.error_pct_5
            other_error_pct_95 = other_histogram_datum.error_pct_95
            other_best_line = other_histogram_datum.best_line
            other_best_line_label = other_histogram_datum.best_line_label
            other_title_prefix = other_histogram_datum.other_title_prefix + repeat_suffix
            other_subsample_sizes = other_histogram_datum.subsample_sizes
            other_lwr = other_error_mu - other_error_pct_5
            other_upr = other_error_pct_95 - other_error_mu
            other_yerr = np.array([other_lwr, other_upr])
            ax3.errorbar(
                other_subsample_sizes,
                other_histogram_datum.error_mu,
                yerr=other_yerr,
                label=other_title_prefix,
                color='black',
            )
            ax3.plot(
                other_subsample_sizes,
                other_best_line,
                label=other_best_line_label,
                color='gray'
            )

    ax3.errorbar(
        subsample_sizes,
        error_mu,
        yerr=yerr,
        label=title_prefix + repeat_suffix,
        color='b',
    )
    ax3.set_xlabel(f'Log Sample Size')
    ax3.set_ylabel('Log Error')
    ax3.set_title(f'Log Error vs. Log Sample Size')

    # Plot the best fit line
    # Compute theoretical rate
    bin_width = subsample_sizes ** (-1/3)  # Optimal
    one_half_rate = 1 / (subsample_sizes ** (1/2))
    two_thirds_rate = 1 / (subsample_sizes ** (2/3))
    four_fifths_rate = 1 / (subsample_sizes ** (4/5))
    one_half_rate *= best_line[0] / one_half_rate[0]
    two_thirds_rate *= best_line[0] / two_thirds_rate[0]
    four_fifths_rate *= best_line[0] / four_fifths_rate[0]
    m, b = np.polyfit(np.log(subsample_sizes), np.log(error_mu), 1)
    best_line = np.exp(m*np.log(subsample_sizes) + b)

    ax3.plot(
        subsample_sizes,
        best_line,
        label=best_line_label,
        color="g"
    )
    ax3.plot(
        subsample_sizes,
        one_half_rate,
        label='x^(-0.5)',
        linestyle='--',
        color='r'
    )
    ax3.plot(
        subsample_sizes,
        two_thirds_rate,
        label='x^(-0.667) (theoretical rate)',
        linestyle='-',
        color='r'
    )
    ax3.plot(
        subsample_sizes,
        four_fifths_rate,
        label='x^(-0.8)',
        linestyle=':',
        color='r'
    )

    ax3.set_yscale('log')
    ax3.set_xscale('log')

    all_num_bins = histogram_error_bar_output.all_num_bins
    ax2.plot(subsample_sizes, all_num_bins, alpha=0.2, color='r', label='Num Bins')
    ax2.set_xlabel('Sample Size')
    ax2.set_ylabel('Num Bins')
    fig.tight_layout()
    ax2.set_title('Num Bins vs. Sample Size')

    ax4.set_xticks([])
    ax4.set_yticks([])
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)
    ax4.spines['bottom'].set_visible(False)
    ax4.spines['left'].set_visible(False)

    handles, labels = ax3.get_legend_handles_labels()
    ax4.legend(
        handles,
        labels,
        loc='center',
    )

    plt.savefig('{}/{}D_{}_{}_histogram_approx_error.pdf'.format(
        HydraConfig.get().run.dir,
        dim,
        title_prefix,
        alpha.item()
    ))

def plot_hist_w_analytical(
    title_prefix: str,
    sample_trajs: torch.Tensor,
    alpha: np.ndarray,
    std: ToyEvaluator,
    pdf_map: Union[dict, Callable],
) -> Tuple[np.ndarray, np.ndarray]:
    sample_levels = sample_trajs.norm(dim=[1, 2]).cpu()  # [B]
    # bin_width = sample_trajs.shape[0] ** (-1/3)  # Optimal
    # bin_width = 1 / np.log2(sample_trajs.shape[0])  # Sturge's Rule
    bin_width = 2. * scipy.stats.iqr(sample_levels) * sample_trajs.shape[0] ** (-1/3)  # Freedman-Diaconis
    sample_max = sample_levels.max()
    sample_min = sample_levels.min()
    num_bins = int((sample_max - sample_min) / bin_width)
    dim = int(sample_trajs.shape[1])

    sample_max = sample_levels.max().item()
    plt.clf()
    plt.hist(
        sample_levels.numpy(),
        bins=num_bins,
        edgecolor='black',
        density=True,
        label=f'{num_bins} bins',
        # range=(alpha.item(), sample_max)
    )

    # Plot analytical Chi distribution using scipy
    if isinstance(pdf_map, dict):
        x = np.array(list(pdf_map.keys()))
        pdf = np.array(list(pdf_map.values()))
    elif isinstance(pdf_map, Callable):
        x = np.linspace(alpha.item(), sample_max)
        pdf = np.zeros_like(x)
        for i, p in enumerate(x):
            pdf[i] = pdf_map(p, alpha.item())
    plt.plot(x, pdf, 'r-', label='Analytical PDF')
    plt.xlabel('Radius')
    plt.ylabel('Probability Density')
    plt.title(f'Histogram of {sample_levels.shape[0]} Samples with Analytical Tail Density')
    plt.legend()
    plt.savefig('{}/{}D_{}_{}_analytical_w_hist.pdf'.format(
        HydraConfig.get().run.dir,
        dim,
        title_prefix,
        alpha
    ))

    return x, pdf

def plot_chi_hist(
    title_prefix: str,
    sample_trajs: np.ndarray,
    alpha: np.ndarray,
    std: ToyEvaluator,
):
    dim = int(sample_trajs.shape[1])

    sample_max = sample_trajs.norm(dim=[1,2]).max()
    xs = np.linspace(alpha.item(), sample_max.item())
    if alpha > 0:
        # For conditional distribution, need to normalize by P(X > alpha)
        pdf = stats.chi(dim).pdf(xs) / (1 - stats.chi(dim).cdf(alpha))
        # Zero out values below alpha
        pdf[xs < alpha] = 0
    else:
        pdf = stats.chi(dim).pdf(xs)

    pdf_map = {x: p for x, p in zip(xs, pdf)}

    return plot_hist_w_analytical(title_prefix, sample_trajs, alpha, std, pdf_map)

def test_gaussian(end_time, cfg, sample_trajs, std):
    torch.save(sample_trajs, f'{HydraConfig.get().run.dir}/gaussian_sample_trajs.pt')
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
    plt.hist(sample_trajs.squeeze().cpu().numpy(), bins=100, edgecolor='black', density=True)
    x = np.linspace(-sample_trajs.max().item(), sample_trajs.max().item(), 1000)
    pdf = stats.norm.pdf(x) / (1 - stats.norm.cdf(std.likelihood.alpha))
    pdf[np.abs(x) < std.likelihood.alpha.cpu().numpy()] = 0
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
    non_nan_a_llk = non_nan_analytical_llk.squeeze()

    scale_fn = lambda ode: (
        ode - torch.tensor(cfg.example.example.sigma).log()
    )[non_nan_idx.squeeze()]
    hist_llks = []
    if cfg.run_histogram_convergence:
        hist_llks = get_hist_llks(sample_trajs, hebo)
    ode_llk, scaled_ode_llk = compute_ode_log_likelihood(
        non_nan_a_llk,
        hist_llks,
        sample_trajs,
        std,
        cfg,
        alpha,
        scale_fn,
    )

    plt.clf()
    try:
        plt_llk(traj, ode_llk.exp(), HydraConfig.get().run.dir, plot_type='scatter')
        plt_llk(datapoints, datapoint_llk.exp(), HydraConfig.get().run.dir, plot_type='line')
    except Exception as e:
        print(f'error: {e}')

    import pdb; pdb.set_trace()

def plot_theta_from_sample_trajs(end_time, cfg, sample_trajs, std):
    theta = torch.atan2(sample_trajs[..., 1, :], sample_trajs[..., 0, :])
    plt.clf()
    plt.hist(
        theta.cpu().numpy(),
        bins=sample_trajs.shape[0] // 10,
        edgecolor='black',
        density=True
    )
    plt.savefig('{}/theta_hist.pdf'.format(HydraConfig.get().run.dir))
    plt.clf()

def plot_chi_from_sample_trajs(
        title_prefix: str,
        cfg: SampleConfig,
        sample_trajs: np.ndarray,
        std: ToyEvaluator,
        ode_llk: torch.Tensor,
        alpha: np.ndarray,
) -> torch.Tensor:
    x, pdf = plot_chi_hist(
        title_prefix,
        sample_trajs,
        alpha,
        std,
    )

    sample_levels = sample_trajs.norm(dim=[1, 2]).cpu()  # [B]
    dim = int(sample_trajs.shape[1])

    # plot points (ode_llk, sample_trajs) against analytical chi
    # ode_llk represents the log product (sum of logs) density of
    # cfg.example.d IID standard Gaussians
    plt.clf()
    chi_ode_llk = ode_llk + (cfg.example.d / 2) * torch.tensor(2 * np.pi).log() + \
                 (cfg.example.d - 1) * sample_levels.log() - (cfg.example.d / 2 - 1) * \
                 torch.tensor(2.).log() - scipy.special.loggamma(cfg.example.d / 2)
    chi_ode = chi_ode_llk.exp()

    plt.scatter(sample_levels, chi_ode, label='Density Estimates')
    plt.plot(x, pdf, 'r-', label='Analytical PDF')
    plt.legend()
    plt.xlabel('Radius')
    plt.ylabel('Probability Density')
    plt.title(f'Density Estimate with Analytical {cfg.example.d}D Tail Density')
    num_hutchinson_samples = -1 if cfg.compute_exact_trace else cfg.num_hutchinson_samples
    plt.ylim((0., pdf.max()*(1.2)))
    plt.savefig('{}/{}D_{}_chi_scatter_{}.pdf'.format(
        HydraConfig.get().run.dir,
        dim,
        title_prefix,
        num_hutchinson_samples
    ))

    return chi_ode

def generate_diffusion_video(ode_llk, all_trajs, cfg):
    samples = all_trajs.squeeze().cpu()
    # generate gif of samples where each index in samples represents a frame

    # Create figure and axis
    fig, ax = plt.subplots()
    
    # Set up plot limits based on data range
    y_min, y_max = samples[..., 1].min(), samples[..., 1].max()
    x_min, x_max = samples[..., 0].min(), samples[..., 0].max()
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
        scat.set_array(ode_llk[frame])
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

def plot_ellipsoid(end_time, cfg, sample_trajs, std):
    plt.clf()

    sample_levels = sample_trajs.norm(dim=[1,2])
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
    Z = Z.sqrt().cpu().numpy()

    # Plot the level curves
    plt.figure(figsize=(8, 6))
    contour = plt.contour(
        X.cpu().numpy(),
        Y.cpu().numpy(),
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
    plt.savefig('{}/ellipsoid_scatter.pdf'.format(HydraConfig.get().run.dir))

def test_multivariate_gaussian(
        end_time: torch.Tensor,
        cfg: SampleConfig,
        sample_trajs: torch.Tensor,
        std: ToyEvaluator,
        all_trajs: torch.Tensor,
        other_histogram_data: List[HistogramErrorBarOutput],
):
    torch.save(sample_trajs, f'{HydraConfig.get().run.dir}/{cfg.example.d}_dim_sample_trajs.pt')
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
    title_prefix = 'MVN_diffusion'
    alpha_np = alpha.numpy().squeeze()
    dd = stats.chi(cfg.example.d)
    pdf = lambda x, alpha: dd.pdf(x) / (1 - dd.cdf(alpha))
    calculator = SimpsonsRuleCalculator(pdf, cfg.pdf_values_dir)
    error = MISE(title_prefix)
    sample_trajs_list = einops.rearrange(
        sample_trajs,
        '(n c) b 1 -> n c b 1',
        n=cfg.num_sample_batches
    )
    if cfg.run_histogram_convergence:
        hebo = compute_multiple_histogram_errors(
            sample_trajs_list,
            alpha_np,
            std,
            calculator,
            error,
            title_prefix
        )
        plot_histogram_errors(
            alpha_np,
            cfg.example.d,
            hebo,
            error.label(),
            title_prefix,
            cfg,
            other_histogram_data
        )
    plot_chi_hist(title_prefix, sample_trajs, alpha_np, std)

    cond = std.cond if std.cond else torch.tensor([-1.])
    mu = torch.tensor(cfg.example.mu).to(sample_trajs.device)
    sigma = torch.tensor(cfg.example.sigma).to(sample_trajs.device)
    L = torch.linalg.cholesky(sigma).to(sample_trajs.device)
    traj = torch.matmul(L, sample_trajs) + mu  # Shape: (N, d, 1)

    try:
        datapoint_dist = torch.distributions.MultivariateNormal(mu.squeeze(-1), sigma)
    except:
        datapoint_dist = torch.distributions.MultivariateNormal(
            mu.squeeze(-1),
            torch.matmul(L, L.T)
        )
    dim = traj.shape[1]
    tail = 1 - scipy.stats.chi2.cdf(alpha_np, df=dim)  # tail for dim-dimensional standard normal
    print(f'tail proability: {tail}')
    non_nan_analytical_llk = datapoint_dist.log_prob(traj.squeeze(-1)).cpu() - np.log(tail)
    non_nan_a_llk = non_nan_analytical_llk.squeeze()

    scale_fn = lambda ode: ode - L.to(ode.device).logdet()
    hist_llks = []
    if cfg.run_histogram_convergence:
        hist_llks = get_hist_llks(sample_trajs_list, hebo)
    ode_llk, scaled_ode_llk = compute_ode_log_likelihood(
        non_nan_a_llk,
        hist_llks,
        sample_trajs,
        std,
        cfg,
        alpha,
        scale_fn,
    )
    cpu_sample_trajs = sample_trajs.cpu()
    chi_ode = plot_chi_from_sample_trajs(
        title_prefix,
        cfg,
        cpu_sample_trajs,
        std,
        ode_llk[0][-1].cpu(),
        alpha_np,
    )
    sample_levels_list = einops.rearrange(
        sample_levels,
        '(n c) -> n c',
        n=cfg.num_sample_batches
    )
    if cfg.run_histogram_convergence:
        pfode_prefix = 'PFODE'
        pfode_error = MISE(pfode_prefix)
        pfode_hebo = generate_pfode_hebo(
            cfg,
            sample_levels_list,
            chi_ode,
            alpha_np,
            pfode_error,
            calculator,
            pfode_prefix,
            hebo
        )
        more_histogram_data = other_histogram_data + [hebo]
        plot_histogram_errors(
            alpha_np,
            cfg.example.d,
            pfode_hebo,
            pfode_error.label(),
            pfode_prefix,
            cfg,
            more_histogram_data
        )
    if cfg.example.d == 2:
        plot_ellipsoid(end_time, cfg, cpu_sample_trajs, std)
        plot_theta_from_sample_trajs(end_time, cfg, cpu_sample_trajs, std)
        generate_diffusion_video(ode_llk[0].cpu(), all_trajs, cfg)

    idx = sample_levels < sample_levels.min() + 0.2
    edge_trajs = sample_trajs[idx].cpu()
    edge_lk = chi_ode[idx.cpu()].numpy()
    prob_edge = dd.pdf(alpha) / (1-dd.cdf(alpha))
    low_idx = edge_lk < prob_edge
    low_edges = edge_trajs[low_idx]
    high_edges = edge_trajs[~low_idx]
    rng = edge_lk.max() - edge_lk.min()
    lk_color = (edge_lk - edge_lk.min()) / rng
    plt.scatter(edge_trajs[:, 0], edge_trajs[:, 1], c=lk_color, cmap='viridis')
    plt.colorbar(label='Likelihood')
    plt.savefig('{}/edge_points.pdf'.format(
        HydraConfig.get().run.dir,
    ))
    import pdb; pdb.set_trace()

def generate_pfode_hebo(
        cfg: SampleConfig,
        sample_levels_list: torch.Tensor,
        chi_ode: torch.Tensor,
        alpha_np: np.array,
        error: ErrorMeasure,
        calculator: AnalyticalCalculator,
        title_prefix: str,
        hebo: HistogramErrorBarOutput,
) -> HistogramErrorBarOutput:
    batched_ode = einops.rearrange(
        chi_ode,
        '(n c) -> n c',
        n=cfg.num_sample_batches
    )
    batch_errors = []
    sample_levels_list_cpu = sample_levels_list.cpu()
    for sap, ode in zip(sample_levels_list_cpu, batched_ode):
        errors = []
        subsample_sizes_list = []
        for subsample_size in hebo.subsample_sizes:
            subsample_size_idx = int(subsample_size)
            subsap_values, sap_idx = sap[:subsample_size_idx].sort()
            pfode_props = ode[sap_idx]
            pfode_bins = np.append(alpha_np, subsap_values.numpy())
            imse = error.error(calculator, pfode_props, pfode_bins, alpha_np)
            errors.append(imse)
        error.clean_up()
        batch_errors.append(np.array(errors))
    subsaps = einops.repeat(hebo.subsample_sizes, 'l -> (l n)', n=sample_levels_list_cpu.shape[0])
    batch_errors_np = np.array(batch_errors)
    errs = einops.rearrange(batch_errors_np, 'n l -> (l n)')
    m, b = np.polyfit(np.log(subsaps), np.log(errs), 1)
    best_line = np.exp(m*np.log(hebo.subsample_sizes) + b)
    best_line_label = f"{title_prefix}: y = {m:.2f}x + {b:.2f}"
    error_mu = batch_errors_np.mean(0)
    error_pct_5 = np.percentile(batch_errors_np, 5, 0)
    error_pct_95 = np.percentile(batch_errors_np, 95, 0)
    return HistogramErrorBarOutput(
        error_mu,
        error_pct_5,
        error_pct_95,
        best_line,
        best_line_label,
        subsaps,
        hebo.all_props_list,
        hebo.all_bins_list,
        hebo.all_num_bins,
        title_prefix,
    )

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

def plot_bm_pdf_histogram_estimate(
    sample_trajs: np.ndarray,
    calculator: SimpsonsRuleCalculator,
    alpha: np.ndarray,
):
    plt.clf()
    sample_levels = sample_trajs.norm(dim=[1, 2]).cpu().numpy()  # [B]
    # bin_width = sample_trajs.shape[0] ** (-1/3)  # optimal
    # bin_width = 1 / np.log2(sample_trajs.shape[0])  # Sturge's Rule
    bin_width = 2. * scipy.stats.iqr(sample_levels) * sample_trajs.shape[0] ** (-1/3)  # Freedman-Diaconis
    max_sample = sample_levels.max()
    num_bins = int((max_sample - alpha) / bin_width)
    plt.hist(
        sample_levels,
        bins=num_bins,
        edgecolor='black',
        density=True,
        # range=(alpha, max_sample)
    )

    pdf_map = calculator.get_pdf_map(num_bins)
    # Plot analytical Chi distribution using scipy
    pdf = pdf_map.values()
    x = pdf_map.keys()
    plt.scatter(x, pdf, color='r', label='Analytical PDF')
    plt.plot(x, pdf, color='r', linestyle='-')
    plt.xlabel('Radius')
    plt.ylabel('Probability Density')
    plt.title(f'Histogram of {sample_trajs.shape[0]} Samples with Analytical Tail Density')
    plt.legend()
    plt.savefig('{}/chi_histogram_estimate.pdf'.format(HydraConfig.get().run.dir))
    return sample_levels, x, pdf

def line_circle_intersection(r, alpha, dt):
    """
    Computes the intersection points of the line y = alpha/dt - x
    with the circle x^2 + y^2 = r^2

    Inputs:
        r     : scalar or tensor, radius of circle
        alpha : scalar or tensor
        dt    : scalar or tensor

    Returns:
        A tensor of shape (N, 2) where N is the number of real intersection points (0, 1, or 2).
    """
    C = alpha / dt  # constant term in y = C - x

    # Substitute y = C - x into x^2 + y^2 = r^2
    # Gives: x^2 + (C - x)^2 = r^2
    # => x^2 + C^2 - 2Cx + x^2 = r^2
    # => 2x^2 - 2C x + C^2 - r^2 = 0

    a = 2.0
    b = -2.0 * C
    c = C**2 - r**2

    discriminant = b**2 - 4 * a * c

    if discriminant < 0:
        return torch.empty(0, 2)  # No real intersection
    elif discriminant == 0:
        x = -b / (2 * a)
        y = C - x
        return torch.stack([x, y], dim=-1).unsqueeze(0)  # One intersection
    else:
        sqrt_disc = torch.sqrt(discriminant)
        x1 = (-b + sqrt_disc) / (2 * a)
        x2 = (-b - sqrt_disc) / (2 * a)
        y1 = C - x1
        y2 = C - x2
        return torch.stack([[x1, y1], [x2, y2]])

def shortest_arc_length(p1: torch.Tensor, p2: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    """
    Computes the shortest arc length between two points on a circle.

    Args:
        p1: Tensor of shape (..., 2), first point on the circle
        p2: Tensor of shape (..., 2), second point on the circle
        r: Radius of the circle (scalar or tensor broadcastable to p1[..., 0])

    Returns:
        Arc length along the circle (same batch shape as input)
    """
    # Ensure both points lie on the circle by normalizing (optional, if not exact)
    p1 = p1 / p1.norm(dim=-1, keepdim=True)
    p2 = p2 / p2.norm(dim=-1, keepdim=True)

    # Compute cosine of angle between p1 and p2
    dot = (p1 * p2).sum(dim=-1)
    cos_theta = torch.clamp(dot, -1.0, 1.0)  # Numerical stability

    theta = torch.acos(cos_theta)

    # Shortest arc (always <= πr)
    arc_len = r * theta
    return arc_len.cpu(), theta.cpu()

def vertical_line_circle_intersection(
        r: torch.Tensor,
        alpha: torch.Tensor,
        dt_sqrt: torch.Tensor
):
    """
    Computes the intersection points of the vertical line x = alpha/dt_sqrt
    with the circle x^2 + y^2 = r^2

    Inputs:
        r     : scalar or tensor, radius of circle
        alpha : scalar or tensor
        dt_sqrt    : scalar or tensor

    Returns:
        A tensor of shape (N, 2) where N is the number of real intersection points (0, 1, or 2).
    """
    x = alpha / dt_sqrt

    # x^2 + y^2 = r^2 => y^2 = r^2 - x^2
    rhs = r**2 - x**2

    if rhs < 0:
        output = torch.empty(0, 2)  # No real intersection
    elif rhs == 0:
        y = torch.tensor(0.0)
        output = torch.stack([x, y], dim=-1).unsqueeze(0)  # One intersection (tangent)
    else:
        y_pos = torch.sqrt(rhs)
        y_neg = -y_pos
        output = torch.stack([torch.tensor([x, y_pos]), torch.tensor([x, y_neg])])
    return output.cpu()

def line_circle_intersection(
        r: torch.Tensor,
        alpha: torch.Tensor,
        dt_sqrt: torch.Tensor
):
    """
    Computes the intersection points of the line y = alpha/dt_sqrt - x
    with the circle x^2 + y^2 = r^2

    Inputs:
        r     : scalar or tensor, radius of circle
        alpha : scalar or tensor
        dt_sqrt    : scalar or tensor

    Returns:
        A tensor of shape (N, 2) where N is the number of real intersection points (0, 1, or 2).
    """
    C = alpha / dt_sqrt  # constant term in y = C - x

    # Substitute y = C - x into x^2 + y^2 = r^2
    # Gives: x^2 + (C - x)^2 = r^2
    # => x^2 + C^2 - 2Cx + x^2 = r^2
    # => 2x^2 - 2C x + C^2 - r^2 = 0

    a = 2.0
    b = -2.0 * C
    c = C**2 - r**2

    discriminant = b**2 - 4 * a * c

    if discriminant < 0:
        output = torch.empty(0, 2)  # No real intersection
    elif discriminant == 0:
        x = -b / (2 * a)
        y = C - x
        output = torch.stack([x, y], dim=-1).unsqueeze(0)  # One intersection
    else:
        sqrt_disc = torch.sqrt(discriminant)
        x1 = (-b + sqrt_disc) / (2 * a)
        x2 = (-b - sqrt_disc) / (2 * a)
        y1 = C - x1
        y2 = C - x2
        output = torch.stack([torch.tensor([x1, y1]), torch.tensor([x2, y2])])
    return output.cpu()

def compute_lengths(r, top_points, bottom_points, right_points, left_points) -> Tuple[torch.Tensor, torch.Tensor]:
    total_perimeter = torch.tensor([0.], device='cpu')
    top_angle = torch.tensor(0., device='cpu')
    bottom_angle = torch.tensor(0., device='cpu')
    right_angle = torch.tensor(0., device='cpu')
    left_angle = torch.tensor(0., device='cpu')
    top_angle_points = torch.zeros(2,2)
    bottom_angle_points = torch.zeros(2,2)
    right_angle_points = torch.zeros(2,2)
    left_angle_points = torch.zeros(2,2)
    if top_points.nelement():
        top_length, top_angle = shortest_arc_length(top_points[0], top_points[1], r)
        bottom_length, bottom_angle = shortest_arc_length(bottom_points[0], bottom_points[1], r)
        total_perimeter += top_length + bottom_length
        top_top_point = top_points[top_points[:, 1].sort().indices[-1]]
        top_bottom_point = top_points[top_points[:, 1].sort().indices[0]]
        bottom_top_point = bottom_points[bottom_points[:, 1].sort().indices[-1]]
        bottom_bottom_point = bottom_points[bottom_points[:, 1].sort().indices[0]]
        top_angle_points = torch.stack([top_points[1], top_points[0]])
        bottom_angle_points = torch.stack([bottom_points[1], bottom_points[0]])
    if right_points.nelement():
        right_bottom_point = right_points[right_points[:, 1].sort().indices[0]]
        top_bottom_point = top_points[top_points[:, 1].sort().indices[0]]
        right_length, right_angle = shortest_arc_length(top_bottom_point, right_bottom_point, r)
        left_top_point = left_points[left_points[:, 1].sort().indices[-1]]
        bottom_top_point = bottom_points[bottom_points[:, 1].sort().indices[-1]]
        left_length, left_angle = shortest_arc_length(left_top_point, bottom_top_point, r)
        total_perimeter += left_length + right_length
        right_angle_points = torch.stack([right_bottom_point, top_bottom_point])
        left_angle_points = torch.stack([bottom_top_point, left_top_point])
    angles = torch.stack([top_angle, bottom_angle, right_angle, left_angle])
    angle_points = torch.stack([top_angle_points, bottom_angle_points, right_angle_points, left_angle_points])
    return total_perimeter, angles, angle_points

def compute_perimeter(
        r: torch.Tensor,
        alpha: torch.Tensor,
        dt_sqrt: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    angles = torch.zeros(4)
    angle_points = torch.zeros(4,2,2)
    if r <= alpha:
        output = torch.tensor([0.])
    elif r >= torch.tensor(5.).sqrt()*alpha/dt_sqrt:
        output = torch.tensor([2 * torch.pi * r])
    else:
        top_points = line_circle_intersection(r, alpha, dt_sqrt)
        bottom_points = line_circle_intersection(r, -alpha, dt_sqrt)
        right_points = vertical_line_circle_intersection(r, alpha, dt_sqrt)
        left_points = vertical_line_circle_intersection(r, -alpha, dt_sqrt)
        output, angles, angle_points = compute_lengths(r, top_points, bottom_points, right_points, left_points)
    return output.cpu(), angles.cpu(), angle_points.cpu()

def compute_transformed_ode(
        sample_levels: torch.Tensor,
        ode_llk: torch.Tensor,
        alpha: float,
        dt: torch.Tensor,
) -> torch.Tensor:
    perimeters = torch.stack([
        compute_perimeter(r.cpu(), alpha, dt.sqrt().cpu())[0] for r in sample_levels
    ])
    transformed_ode = perimeters.squeeze() * ode_llk.exp().cpu()
    return transformed_ode

def plot_bm_pdf_pfode_estimate(sample_trajs, ode_llk, cfg, tail, alpha, dt, x, pdf):
    # plot points (ode_llk, sample_levels) against analytical
    sample_levels = sample_trajs.norm(dim=[1, 2]).cpu()  # [B]
    plt.clf()
    dim = cfg.example.sde_steps - 1

    transformed_ode = compute_transformed_ode(sample_levels, ode_llk, alpha, dt)

    plt.scatter(sample_levels, transformed_ode, label='Density Estimates')
    plt.scatter(x, pdf, color='r', label='Analytical PDF')
    plt.plot(x, pdf, color='r', linestyle='-')
    plt.legend()
    plt.xlabel('Radius')
    plt.ylabel('Probability Density')
    plt.title(f'Density Estimate with Analytical {cfg.example.sde_steps} Step BM Tail Density')
    num_hutchinson_samples = -1 if cfg.compute_exact_trace else cfg.num_hutchinson_samples
    plt.ylim((0., pdf.max()*1.2))
    plt.savefig('{}/bm_pfode_estimate_{}.pdf'.format(
        HydraConfig.get().run.dir,
        num_hutchinson_samples
    ))

    return transformed_ode

def get_hist_llks(
        sample_trajs: torch.Tensor,
        hebo: HistogramErrorBarOutput
) -> List[torch.Tensor]:
    """
    sample_levels has shape b
    all_bins has shape n
    all_props has shape n-1
    """
    hist_llks = []
    max_subsample_size = hebo.subsample_sizes[-1].astype(int)
    for idx, (all_bins, all_props) in enumerate(zip(
            hebo.all_bins_list,
            hebo.all_props_list
    )):
        unsorted_sample_levels = sample_trajs[idx].norm(dim=[1, 2])[:max_subsample_size]
        sample_levels = unsorted_sample_levels.cpu().sort().values  # b
        all_bins_tensor = torch.tensor(all_bins[-1])  # n
        all_bins_tensor[-1] += 1e-4
        repeated_all_bins = einops.repeat(all_bins_tensor, 'n -> n b', b=sample_levels.shape[0])
        repeated_sample_levels = einops.repeat(sample_levels, 'b -> n b', n=all_bins[-1].shape[0])
        bin_idx = (repeated_all_bins <= repeated_sample_levels).sum(dim=0) - 1
        hist_llk = torch.tensor(all_props[-1][bin_idx]).log()
        hist_llks.append(hist_llk)
    return hist_llks

def test_brownian_motion_diff(
        end_time,
        cfg,
        sample_trajs,
        std,
        other_histogram_data,
):
    torch.save(sample_trajs, f'{HydraConfig.get().run.dir}/bm_sample_trajs.pt')
    dt = end_time.cpu() / (cfg.example.sde_steps-1)
    # de-standardize data
    trajs = sample_trajs * dt.sqrt()

    alpha = torch.tensor([std.likelihood.alpha]) if std.cond == 1. else torch.tensor([0.])
    alpha = alpha.cpu()

    title_prefix = 'BM_diffusion'
    alpha_np = alpha.numpy().squeeze()
    if cfg.example.sde_steps == 3:
        quadrature = pdf_2d_quadrature_bm
    elif cfg.example.sde_steps == 4:
        quadrature = pdf_3d_quadrature_bm
    else:
        raise NotImplementedError
    calculator = SimpsonsRuleCalculator(
        quadrature,
        cfg.pdf_values_dir
    )
    sample_trajs_list = einops.rearrange(
        sample_trajs,
        '(n c) b 1 -> n c b 1',
        n=cfg.num_sample_batches
    )
    if cfg.run_histogram_convergence:
        # error = SimpsonsMISE()
        pfode_error = MISE(title_prefix)
        hebo = compute_multiple_histogram_errors(
            sample_trajs_list,
            alpha_np,
            std,
            calculator,
            pfode_error,
            title_prefix
        )
        plot_histogram_errors(
            alpha_np,
            cfg.example.sde_steps-1,
            hebo,
            pfode_error.label(),
            title_prefix,
            cfg,
            other_histogram_data
        )
    x, pdf = plot_hist_w_analytical(
        title_prefix,
        sample_trajs,
        alpha_np,
        std,
        quadrature,
    )

    # make histogram
    data = sample_trajs.reshape(-1).cpu().numpy()
    plt.clf()
    plt.hist(data, bins=30, edgecolor='black')
    plt.title('Histogram of brownian motion state diffs')
    save_dir = '{}/{}'.format(HydraConfig.get().run.dir, cfg.model_name)
    os.makedirs(save_dir, exist_ok=True)
    alpha_str = '%.1f' % alpha.item()
    plt.savefig('{}/alpha={}_brownian_motion_diff_hist.pdf'.format(
        save_dir,
        alpha_str,
    ))

    # turn state diffs into Brownian motion
    bm_trajs = torch.cat([
        torch.zeros(trajs.shape[0], 1, 1, device='cpu'),
        trajs.cumsum(dim=-2).cpu()
    ], dim=1)

    # plot trajectories
    plt.clf()
    times = torch.linspace(0., 1., bm_trajs.shape[1]).cpu()
    plt.plot(times.numpy(), bm_trajs[..., 0].numpy().T)
    plt.savefig('{}/alpha={}_brownian_motion_diff_samples.pdf'.format(
        save_dir,
        alpha_str,
    ))

    # plot cut off trajectories
    exit_idx = (bm_trajs.abs() > alpha).to(float).argmax(dim=1)
    plt.clf()
    dtimes = exit_idx * dt
    states = bm_trajs[torch.arange(bm_trajs.shape[0]), exit_idx.squeeze()]
    plt.plot(times.numpy(), bm_trajs[..., 0].numpy().T, alpha=0.2)
    plt.scatter(dtimes.cpu().numpy(), states, marker='o', color='red')
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
    tail = get_target(std).analytical_prob(alpha) if alpha.cpu().numpy() else torch.tensor(1.)
    print(f'true tail prob: {tail}')
    analytical_llk = uncond_analytical_llk - np.log(tail.item())

    scale_fn = lambda ode: ode - dt.sqrt().log() * (cfg.example.sde_steps-1)
    hist_llks = []
    if cfg.run_histogram_convergence:
        hist_llks = get_hist_llks(sample_trajs_list, hebo)
    ode_llk, scaled_ode_llk = compute_ode_log_likelihood(
        analytical_llk,
        hist_llks,
        sample_trajs,
        std,
        cfg,
        alpha,
        scale_fn,
    )
    if alpha in [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]:
        chi_ode = plot_bm_pdf_pfode_estimate(
            sample_trajs=sample_trajs,
            ode_llk=ode_llk[0][-1],
            cfg=cfg,
            tail=tail,
            alpha=alpha,
            dt=dt,
            x=x,
            pdf=pdf
        )

        # if cfg.run_histogram_convergence:
        #     pfode_prefix = 'PFODE'
        #     pfode_error = MISE(pfode_prefix)
        #     pfode_hebo = generate_pfode_hebo(
        #         cfg,
        #         sample_levels_list,
        #         chi_ode,
        #         alpha_np,
        #         pfode_error,
        #         calculator,
        #         pfode_prefix,
        #         hebo
        #     )
        #     more_histogram_data = other_histogram_data + [hebo]
        #     plot_histogram_errors(
        #         alpha_np,
        #         cfg.example.d,
        #         pfode_hebo,
        #         pfode_error.label(),
        #         pfode_prefix,
        #         cfg,
        #         more_histogram_data
        #     )

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
    data = sample_trajs.reshape(-1).cpu().numpy()
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
    plt.plot(times.cpu().numpy(), st_trajs[..., 0].cpu().numpy().T)
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
    plt.plot(times.cpu().numpy(), st_trajs[..., 0].cpu().numpy().T, alpha=0.2)
    plt.scatter(dtimes.cpu().numpy(), states, marker='o', color='red')
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
    scaled_ode_llk = ode_llk[0][-1] - dt.sqrt().log() * (cfg.example.sde_steps-1)
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
        # data = sample_trajs.reshape(-1).cpu().numpy()
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
            plt.plot(times.cpu().numpy(), traj.cpu().numpy().T, label=f'alpha={alpha}', color=colors[i])
        plt.legend()
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig('{}/brownian_motion_smc_samples_{}.pdf'.format(
            save_dir,
            j,
        ))
    import pdb; pdb.set_trace()

def test(end_time, cfg, out_trajs, std, all_trajs, other_histogram_data):
    if type(std.example) == GaussianExampleConfig:
        test_gaussian(
            end_time,
            cfg,
            out_trajs,
            std,
        )
    elif type(std.example) == MultivariateGaussianExampleConfig:
        test_multivariate_gaussian(
            end_time,
            cfg,
            out_trajs,
            std,
            all_trajs,
            other_histogram_data,
        )
    elif type(std.example) == BrownianMotionDiffExampleConfig:
        test_brownian_motion_diff(
            end_time,
            cfg,
            out_trajs,
            std,
            other_histogram_data,
        )
    elif type(std.example) == UniformExampleConfig:
        test_uniform(
            end_time,
            cfg,
            out_trajs,
            std
        )
    elif type(std.example) == StudentTExampleConfig:
        test_student_t(
            end_time,
            cfg,
            out_trajs,
            std
        )
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

def get_dim(std):
    if isinstance(std.example, GaussianExampleConfig):
        return std.example.d
    elif isinstance(std.example, MultivariateGaussianExampleConfig):
        return std.example.d
    elif isinstance(std.example, BrownianMotionDiffExampleConfig):
        return std.example.sde_steps-1
    else:
        raise NotImplementedError

def compute_df_dx(cfg, std, x_min):
    # compute
    # $-\dfrac{1}{2}\beta(t)(I-\sigma(t)^{-1})$
    ts = torch.linspace(0., 1., std.sampler.diffusion_timesteps, device=device)

    is_grad = torch.is_grad_enabled()
    torch.set_grad_enabled(True)
    t_ = ts.requires_grad_()
    int_beta_t = std.sampler.continuous_beta_integral(t_)  # int_0^t beta(s) ds
    beta_t = torch.autograd.grad(int_beta_t.sum(), t_)[0]  # beta(t)
    torch.set_grad_enabled(is_grad)

    # note that the sigma output from the following function does NOT depend on x_min in the
    # Gaussian, BM, nor Student T cases
    d = get_dim(std)
    eye = torch.eye(d, device=device)
    sigma = std.get_variance_from_sampler(ts, x_min)
    sigma_inv = 1 / sigma if sigma[0].shape == torch.Size([1, 1]) else sigma.pinverse()
    df_dx = -beta_t[:, None, None] / 2 * (eye - sigma_inv)
    return df_dx, ts

def compute_analytical_ode_traj(cfg, std, x_min):
    df_dx, ts = compute_df_dx(cfg, std, x_min)
    lnx = (df_dx[:-1] * ts.diff()[0]).flip(dims=[0]).cumsum(dim=0)
    d = get_dim(std)
    lnx_exp = lnx.exp() if d == 1 else lnx.matrix_exp()
    assert (lnx_exp.det() > 0).all()
    coeff = lnx_exp.pinverse()
    coeff = torch.concat([torch.eye(d, device=coeff.device).unsqueeze(0), coeff])
    return torch.matmul(coeff[:, None], x_min[None])

def compute_analytical_icov(cfg, std, x_min):
    df_dx, ts = compute_df_dx(cfg, std, x_min)
    lnx = -einops.einsum(df_dx, 'b n n -> b')[:-1]  # compute traces
    lnp = (lnx * ts.diff()).flip(dims=[0]).cumsum(dim=0)  # compute integral
    lnp = torch.concat([torch.zeros(1, device=lnp.device), lnp])
    logp0 = std.sampler.prior_logp(x_min, device=device).flatten(1).sum(1)
    logpt = logp0[None] - lnp[:, None]
    return logpt.exp()

def legend_without_duplicate_labels(ax):
    handles, labels = ax.get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    ax.legend(*zip(*unique))

def get_raw(cfg, data):
    if isinstance(cfg.example, MultivariateGaussianExampleConfig):
        d = cfg.example.d
        mu = torch.tensor(cfg.example.mu, device=data.device)
        sigma = torch.tensor(cfg.example.sigma, device=data.device)
        L = torch.linalg.cholesky(sigma)
        raw_data = torch.matmul(L, data) + mu
        return raw_data
    elif isinstance(cfg.example, BrownianMotionDiffExampleConfig):
        dt = torch.tensor(1. / (cfg.example.sde_steps-1))
        scaled_data = data * dt.sqrt()  # standardize data
        raw_data = torch.cat([
            torch.zeros(scaled_data.shape[0], 1, 1, device=scaled_data.device),
            scaled_data.cumsum(dim=1)
        ], dim=1)
        return raw_data
    else:
        print(cfg.example)
        raise NotImplementedError

def rejection_sample_x_final(cfg, std):
    """ Rejection sampling to find a conditional sample. """
    num_samples = 10000
    cond = torch.zeros(num_samples, dtype=bool)
    d = get_dim(std)
    while not cond.any():
        samples = torch.distributions.Normal(0., 1.).sample([num_samples, d, 1]).to(device)
        raw_samples = get_raw(cfg, samples)
        cond = std.likelihood.get_condition(raw_samples, samples).to(bool).squeeze()
    sample = samples[cond][0]
    return sample

def get_x_min(std, x_final):
    start_time = std.sampler.t_eps if std.cfg.test == TestType.Test else 0.
    times = torch.linspace(start_time, 1., std.sampler.diffusion_timesteps, device=x_final.device)
    def ode_fn(t, x):
        dx_dt = std.get_dx_dt(
            t,
            x,
            evaluate_likelihood=False,
            cond=std.cond,
            alpha=std.likelihood.alpha.reshape(-1, 1),
        )
        return dx_dt
    unsqueezed_x_final = x_final.unsqueeze(0)
    if std.cfg.sample_integrator == Integrator.EULER:
        sol = std.euler_integrate(ode_fn, unsqueezed_x_final, times)
    elif std.cfg.sample_integrator == Integrator.HEUN:
        sol = std.heun_integrate(ode_fn, unsqueezed_x_final, times)
    elif std.cfg.sample_integrator == Integrator.RK4:
        sol = std.rk4_integrate(ode_fn, unsqueezed_x_final, times)
    return sol[-1]

def compute_ode_error(
        std,
        diffusion_timesteps: torch.Tensor,
        integrator: Integrator,
        x_final: torch.Tensor
):
    std.cfg.sample_integrator = integrator
    errors = []
    dst = std.sampler.diffusion_timesteps
    for time in diffusion_timesteps:
        std.sampler.diffusion_timesteps = time
        x_min = get_x_min(std, x_final)
        trajs = std.sample_trajectories(
            cond=std.cond,
            alpha=std.likelihood.alpha.reshape(-1, 1),
            x_min=x_min,
        ).samples
        error = (trajs[-1] - x_final).norm()
        errors.append(error.cpu())
    std.sampler.diffusion_timesteps = dst
    return errors

def make_ode_error_plot(cfg, std, test_type: TestType):
    plt.clf()
    std.cfg.test = test_type if std.cond == -1 else TestType.Test
    diffusion_timesteps = torch.tensor([10, 100, 1000], device='cpu')
    x_final = rejection_sample_x_final(cfg, std)
    euler_errors = compute_ode_error(std, diffusion_timesteps, Integrator.EULER, x_final)
    heun_errors = compute_ode_error(std, diffusion_timesteps, Integrator.HEUN, x_final)
    rk4_errors = compute_ode_error(std, diffusion_timesteps, Integrator.RK4, x_final)
    dopri5_errors = compute_ode_error(std, diffusion_timesteps, Integrator.PYTORCH, x_final)
    plt.scatter(diffusion_timesteps, euler_errors, label='Euler', color='blue')
    plt.plot(diffusion_timesteps, euler_errors, color='blue', alpha=0.4)
    plt.scatter(diffusion_timesteps, heun_errors, label='Heun', color='red')
    plt.plot(diffusion_timesteps, heun_errors, color='red', alpha=0.4)
    plt.scatter(diffusion_timesteps, rk4_errors, label='RK4', color='yellow')
    plt.plot(diffusion_timesteps, rk4_errors, color='yellow', alpha=0.4)
    plt.scatter(diffusion_timesteps, dopri5_errors, label='Dopri5', color='green')
    plt.plot(diffusion_timesteps, dopri5_errors, color='green', alpha=0.4)
    plt.yscale('log')
    plt.xscale('log')
    plt.ylabel('L^2 Error')
    plt.xlabel('Timesteps')
    plt.title('Convergence Error vs. Timesteps')
    plt.legend()
    plt.savefig('{}/ode_error.pdf'.format(
        HydraConfig.get().run.dir,
    ))
    plt.clf()

def plot_ode_trajs(cfg, std, sample_trajs):
    x_min = sample_trajs.x_min
    alpha = std.likelihood.alpha if std.cond == 1 else torch.tensor(0.)
    cfg_obj = OmegaConf.to_object(cfg)
    if isinstance(cfg_obj.example, MultivariateGaussianExampleConfig):
        std.likelihood.set_alpha(torch.ones_like(x_min) * alpha)
    else:
        std.likelihood.set_alpha(alpha)
    diffusion_trajs = sample_trajs.samples
    test = cfg.test
    if isinstance(cfg_obj.example, MultivariateGaussianExampleConfig):
        test_type = TestType.MultivariateGaussian
    else:
        test_type = TestType.BrownianMotionDiff
    cfg.test = test_type
    cfg.sample_integrator = Integrator.EULER
    euler_trajs = std.sample_trajectories(
        cond=std.cond,
        alpha=std.likelihood.alpha.reshape(-1, 1),
        x_min=x_min,
    ).samples
    cfg.sample_integrator = Integrator.HEUN
    heun_trajs = std.sample_trajectories(
        cond=std.cond,
        alpha=std.likelihood.alpha.reshape(-1, 1),
        x_min=x_min,
    ).samples
    cfg.sample_integrator = Integrator.RK4
    rk4_trajs = std.sample_trajectories(
        cond=std.cond,
        alpha=std.likelihood.alpha.reshape(-1, 1),
        x_min=x_min,
    ).samples

    euler_raw_traj = get_raw(cfg_obj, euler_trajs[-1])
    euler_cond = std.likelihood.get_condition(euler_raw_traj, euler_trajs[-1]).to(bool).squeeze()
    good_euler_trajs = euler_trajs[:, euler_cond]

    heun_raw_traj = get_raw(cfg_obj, heun_trajs[-1])
    heun_cond = std.likelihood.get_condition(heun_raw_traj, heun_trajs[-1]).to(bool).squeeze()
    good_heun_trajs = heun_trajs[:, heun_cond]

    rk4_raw_traj = get_raw(cfg_obj, rk4_trajs[-1])
    rk4_cond = std.likelihood.get_condition(rk4_raw_traj, rk4_trajs[-1]).to(bool).squeeze()
    good_rk4_trajs = rk4_trajs[:, rk4_cond]

    diffusion_samples = diffusion_trajs.squeeze()[:, euler_cond].cpu()
    analytical_euler_samples = good_euler_trajs.squeeze(-1).cpu()
    analytical_heun_samples = good_heun_trajs.squeeze(-1).cpu()
    analytical_rk4_samples = good_rk4_trajs.squeeze(-1).cpu()

    if not ((analytical_euler_samples[0] - analytical_heun_samples[0]).abs() < 1e-3).all() or \
       not ((analytical_euler_samples[0] - analytical_rk4_samples[0]).abs() < 1e-3).all() or \
       not ((analytical_euler_samples[0] - diffusion_samples[0]).abs() < 1e-3).all():
        print('uh oh analytical samples not matching!')

    d = get_dim(std)

    # Create figure and axis
    fig, axs = plt.subplots(d+1, 1)
    if isinstance(cfg_obj.example, MultivariateGaussianExampleConfig):
        plot_type = 'Gaussian'
    else:
        plot_type = 'Brownian_Motion'
    plt.suptitle(f'{plot_type}\nODE Trajectories Analytical vs. Diffusion')
    single_times = torch.arange(std.sampler.diffusion_timesteps, dtype=int)
    times = einops.repeat(
        single_times,
        't -> t b',
        b=euler_cond.sum(),
    )
    # plot samples
    for i, ax in enumerate(axs[:-1]):
        ax.plot(times, analytical_euler_samples[..., i], color='red', dashes=[5, 3], alpha=0.6, label='Analytical (euler)')
        ax.plot(times, analytical_heun_samples[..., i], color='green', dashes=[3, 1], alpha=0.6, label='Analytical (heun)')
        ax.plot(times, analytical_rk4_samples[..., i], color='yellow', dashes=[2, 7], alpha=0.8, label='Analytical (rk4)')
        ax.plot(times, diffusion_samples[..., i], color='blue', dashes=[4, 4], label='Diffusion')
        ax.set_ylabel(f'State (dim {i})')
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.scatter(times[-1], analytical_rk4_samples[-1, :, i], marker='o', color='red')

    # legend_without_duplicate_labels(ax)
    # compute likelihoods
    cfg.test = test
    cfg.density_integrator = Integrator.HEUN
    diffusion_pdf = std.ode_log_likelihood(
        diffusion_trajs[-1, euler_cond].to(device),
        cond=std.cond,
        alpha=std.likelihood.alpha.reshape(-1, 1),
        exact=cfg.compute_exact_trace,
    )[0].exp().cpu()
    cfg.test = test_type
    cfg.density_integrator = Integrator.EULER
    analytical_euler_pdf = std.ode_log_likelihood(
        euler_trajs[-1, euler_cond].to(device),
        cond=std.cond,
        alpha=std.likelihood.alpha.reshape(-1, 1),
        exact=cfg.compute_exact_trace,
    )[0].exp().cpu()
    cfg.density_integrator = Integrator.HEUN
    analytical_heun_pdf = std.ode_log_likelihood(
        heun_trajs[-1, heun_cond].to(device),
        cond=std.cond,
        alpha=std.likelihood.alpha.reshape(-1, 1),
        exact=cfg.compute_exact_trace,
    )[0].exp().cpu()
    cfg.density_integrator = Integrator.RK4
    analytical_rk4_pdf = std.ode_log_likelihood(
        rk4_trajs[-1, rk4_cond].to(device),
        cond=std.cond,
        alpha=std.likelihood.alpha.reshape(-1, 1),
        exact=cfg.compute_exact_trace,
    )[0].exp().cpu()

    if not ((analytical_euler_pdf[0] - analytical_heun_pdf[0]).abs() < 1e-3).all() or \
       not ((analytical_euler_pdf[0] - analytical_rk4_pdf[0]).abs() < 1e-3).all() or \
       not ((analytical_euler_pdf[0] - diffusion_pdf[0]).abs() < 1e-3).all():
        print('uh oh analytical pdfs not matching!')

    # plot likelihoods
    axs[-1].plot(times, analytical_euler_pdf, color='red', dashes=[5, 3], alpha=0.6, label='Analytical (euler)')
    axs[-1].plot(times, analytical_heun_pdf, color='green', dashes=[3, 1], alpha=0.6, label='Analytical (heun)')
    axs[-1].plot(times, analytical_rk4_pdf, color='yellow', dashes=[2, 7], alpha=0.8, label='Analytical (rk4)')
    axs[-1].plot(times, diffusion_pdf, color='blue', dashes=[4, 4], label='Diffusion')
    axs[-1].scatter(times[-1], analytical_rk4_pdf[-1], marker='o', color='red')

    axs[-1].set_xlabel('(Reverse) Diffusion Timestep')
    axs[-1].set_ylabel('PDF')
    axs[-1].xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.savefig('{}/{}_ode_trajs_eps={}.pdf'.format(
        HydraConfig.get().run.dir,
        plot_type,
        std.sampler.t_eps,
    ))
    plt.clf()

def plot_divergences(std):
    t = torch.tensor([1.], device=device)
    spacing = 100
    xx = torch.linspace(-5, 5, spacing, device=device)
    yy = torch.linspace(-5, 5, spacing, device=device)
    X, Y = torch.meshgrid(xx, yy, indexing='xy')
    x = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=1).unsqueeze(-1)  # Nx2x1
    mask = x.norm(dim=[1, 2]) < std.likelihood.alpha
    dx_dt = lambda y: std.get_dx_dt(t, y, evaluate_likelihood=True, cond=std.cond, alpha=std.likelihood.alpha.reshape(-1, 1))
    div = torch.zeros(x.shape[0], device=x.device)
    for i in range(2):
        v = torch.zeros_like(x)
        v[:, i, 0] = 1.0
        dx, vjp = torch.autograd.functional.vjp(dx_dt, x, v)
        div += (vjp * v).sum([-1, -2])
    plt.xlabel("x")
    plt.ylabel("y")
    # Plot heatmap
    plt.figure(figsize=(6, 6))
    div[mask] = torch.nan
    div_np = div.reshape(spacing, spacing).cpu()
    plt.imshow(div_np, extent=[
        xx.min().cpu(),
        xx.max().cpu(),
        yy.min().cpu(),
        yy.max().cpu()
    ], origin='lower', cmap='hot', interpolation='bilinear')
    plt.colorbar(label='Magnitude')
    plt.title("Divergence")
    plt.axis("equal")
    plt.savefig('{}/divergence.pdf'.format(
        HydraConfig.get().run.dir,
    ))
    plt.clf()

def plot_boundary(std, cfg_obj, ax):
    if isinstance(cfg_obj.example, MultivariateGaussianExampleConfig):
        plot_gaussian_boundary(std, ax)
    else:
        plot_brownian_motion_boundary(std, ax)

def plot_gaussian_boundary(std, ax):
    circle = plt.Circle((0,0), std.alpha, color='black', fill=False)
    ax.add_patch(circle)

def plot_brownian_motion_boundary(std, ax):
    x = torch.linspace(-5, 5, 500)
    t = np.sqrt(1/(std.cfg.example.sde_steps-1))
    ax.plot(x, std.likelihood.alpha/t - x, color='black')
    ax.plot(x, -std.likelihood.alpha/t - x, color='black')
    ax.axvline(std.likelihood.alpha/t, color='black')
    ax.axvline(-std.likelihood.alpha/t, color='black')

def plot_circles(std, cfg_obj):
    num_radii = 6
    radii = torch.linspace(0.2, 2., num_radii, device=device)
    num_angles = 20
    pi = torch.linspace(0., 2*torch.pi, num_angles, device=device)
    unit = torch.stack([torch.cos(pi), torch.sin(pi)], dim=1).unsqueeze(-1)
    x_min = (unit * radii).movedim(1, 2).reshape(-1, 2, 1)
    roygbiv_rgb = [
        (255,   0,   0),
        (255, 165,   0),
        (255, 255,   0),
        (  0, 128,   0),
        (  0,   0, 255),
        ( 75,   0, 130),
        (148,   0, 211),
    ]
    roygbiv = np.array([[r/255, g/255, b/255] for r, g, b in roygbiv_rgb])
    clr = einops.repeat(np.arange(6), 'b -> (c b)', c=num_radii)

    # Build colormap + normalization
    cmap = colors.ListedColormap(roygbiv, name="ROYGBIV")

    std.cfg.sample_integrator = Integrator.RK4
    samples = std.sample_trajectories(
        cond=std.cond,
        alpha=std.likelihood.alpha.reshape(-1, 1),
        x_min=x_min,
    ).samples.squeeze().cpu()

    # generate gif of samples where each index in samples represents a frame
    # Create figure and axis
    fig, ax = plt.subplots()
    
    # Set up plot limits based on data range
    y_min, y_max = samples[..., 1].min(), samples[..., 1].max()
    x_min, x_max = samples[..., 0].min(), samples[..., 0].max()
    margin = 0.1 * max(x_max - x_min, y_max - y_min)
    ax.set_xlim(x_min - margin, x_max + margin)
    ax.set_ylim(y_min - margin, y_max + margin)
    plot_boundary(std, cfg_obj, ax)

    # Create color normalization based on likelihood values
    norm = colors.Normalize(vmin=0, vmax=6)

    # Initialize scatter plot
    scat = ax.scatter([], [], c=[], cmap=cmap, norm=norm)
    fig.colorbar(scat, label='Meaningless')

    def update(frame):
        # Update positions and colors for current frame
        scat.set_offsets(samples[frame, :, :2])
        scat.set_array(clr)
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
    cond = 'conditional' if std.cond else 'unconditional'
    anim.save(f'{HydraConfig.get().run.dir}/{cond}_circle_diffusion.gif', writer='pillow')
    plt.savefig(f'{HydraConfig.get().run.dir}/{cond}_circle_diffusion_final_frame.pdf'.format(
        HydraConfig.get().run.dir,
    ))
    plt.close()

@hydra.main(version_base=None, config_path="conf", config_name="continuous_sample_config")
def sample(cfg):
    logger = logging.getLogger("main")
    logger.info('run type: sample')
    logger.info(f"CONFIG\n{OmegaConf.to_yaml(cfg)}\n")
    logger.info(f'OUTPUT\n{HydraConfig.get().run.dir}\n')

    os.system('echo git commit: $(git rev-parse HEAD)')

    torch.manual_seed(cfg.random_seed)

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

        # plot divergences
        # plot_divergences(std)

        # plot circles
        cfg_obj = OmegaConf.to_object(cfg)
        plot_circles(std, cfg_obj)

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

        # Plot convergence of histogram from analytical samples
        # Use these samples to debug if the debug flag is set
        class A:
            def __init__(self, samples):
                self.samples = samples
        if cfg.run_histogram_convergence:
            saps = None
            alpha = torch.tensor([std.likelihood.alpha]) if std.cond == 1. else torch.tensor([0.])
            alpha_np = alpha.cpu().numpy().squeeze()
            if type(std.example) == MultivariateGaussianExampleConfig:
                dim = cfg.example.d
                dist_type = 'MVN'
                dd = stats.chi(dim)
                pdf = lambda x, alpha: dd.pdf(x) / (1 - dd.cdf(alpha))
                calculator = SimpsonsRuleCalculator(pdf, cfg.pdf_values_dir)
            elif type(std.example) == BrownianMotionDiffExampleConfig:
                dim = cfg.example.sde_steps - 1
                dist_type = 'BM'
                if cfg.example.sde_steps == 3:
                    quadrature = pdf_2d_quadrature_bm
                elif cfg.example.sde_steps == 4:
                    quadrature = pdf_3d_quadrature_bm
                else:
                    raise NotImplementedError
                calculator = SimpsonsRuleCalculator(
                    quadrature,
                    cfg.pdf_values_dir
                )
                if cfg.sample_file:
                    saps = torch.load(cfg.sample_file)
            title_prefix = f'{dist_type}_analytical'
            error = MISE(title_prefix)
            sample_trajs_list = []
            for i in range(cfg.num_sample_batches):
                if saps is None:
                    sample_traj_out = A(torch.randn(100*cfg.num_samples, dim, 1))
                    if type(std.example) == MultivariateGaussianExampleConfig:
                        cond_idx = (sample_traj_out.samples.norm(dim=[1, 2]) > alpha)
                    elif type(std.example) == BrownianMotionDiffExampleConfig:
                        dt = torch.tensor(1. / dim)
                        scaled_x0 = sample_traj_out.samples * dt.sqrt()  # standardize data
                        bm = torch.cat([
                            torch.zeros(scaled_x0.shape[0], 1, 1),
                            scaled_x0.cumsum(dim=1)
                        ], dim=1)
                        cond_idx = (bm.abs() > alpha).any(dim=[1,2])
                    else:
                        raise NotImplementedError
                    cond_samples = sample_traj_out.samples[cond_idx]
                else:
                    sample_traj_out = A(saps)
                if std.cond == 1:
                    sample_traj_out.samples = cond_samples

                sample_traj_out.samples = sample_traj_out.samples[:cfg.num_samples]
                sample_traj_out.samples = sample_traj_out.samples.unsqueeze(0)
                sample_trajs_list.append(sample_traj_out.samples[-1])
            hebo = compute_multiple_histogram_errors(
                torch.stack(sample_trajs_list),
                alpha_np,
                std,
                analytical_calculator=calculator,
                error_measure=error,
                title_prefix=title_prefix,
            )
            plot_histogram_errors(
                alpha_np,
                dim,
                hebo,
                error.label(),
                title_prefix,
                cfg,
            )
            # hist_llks = get_hist_llks(sample_trajs_list, hebo)
            # rkls = []
            # for hist_llk, sample_trajs in zip(hist_llks, sample_trajs_list):
            #     sorted_sample_levels = sample_trajs.norm(dim=[1,2]).sort()
            #     sample_levels = sorted_sample_levels.values
            #     max_sample_size = hist_llk.shape[0]
            #     samples_cpu = sample_levels.cpu().numpy()[:max_sample_size]
            #     log_pdfs = np.log(dd.pdf(samples_cpu)) - np.log(1 - dd.cdf(samples_cpu))
            #     analytical_llk = torch.tensor(log_pdfs)
            #     rkls.append((hist_llk-torch.tensor(analytical_llk)).mean())
            #     # plt.clf()
            #     # plt.plot(sample_levels, np.exp(analytical_llk))
            #     # plt.plot(sample_levels, hist_llk.exp())
            # rkl_tensor = torch.tensor(rkls)
            # quantiles = torch.tensor([0.05, 0.5, 0.95], dtype=rkl_tensor.dtype)
            # quantile_values = rkl_tensor.quantile(quantiles)
            # print(f'\nanalytical KL quantiles (0.05, 0.5, 0.95): {quantile_values}\n')
            # torch.save({
            #         'quantiles': quantiles.cpu().numpy(),
            #         'values': quantile_values.cpu().numpy(),
            #     },
            #     f'{HydraConfig.get().run.dir}/analytical_rkl_quantiles.pt'
            # )
            if type(std.example) == MultivariateGaussianExampleConfig:
                plot_chi_hist(
                    title_prefix,
                    sample_traj_out.samples[-1],
                    alpha_np,
                    std
                )
        else:
            hebo = None

        if cfg.debug:
            print('DEBUG is on!')
            # if type(std.example) == BrownianMotionDiffExampleConfig:
            #     cfg.pdf_values_dir = calculator.pdf_values_dir
        else:
            tm = timer.time()
            sample_traj_out = std.sample_trajectories(
                cond=std.cond,
                alpha=std.likelihood.alpha.reshape(-1, 1),
            )
            eval_time = timer.time() - tm
            print(f'sample time: {eval_time}')

            headers = [
                'Time',
            ]
            data = [[
                eval_time,
            ]]
            df = pd.DataFrame(data, columns=headers)
            df.to_csv(
                f'{HydraConfig.get().run.dir}/{cfg.model_name}_sample_time.csv',
                index=False
            )

        # --IGNORE--
        # ode_trajs = (sample_traj_out.samples).reshape(-1, cfg.num_samples)
        # plot_ode_trajectories(ode_trajs)
        # ----------

        print('fevals: {}'.format(sample_traj_out.fevals))
        sample_trajs = sample_traj_out.samples
        trajs = sample_trajs[-1]
        out_trajs = trajs

        # viz_trajs(cfg, std, out_trajs, end_time)

        # ode integration error plot
        if isinstance(cfg_obj.example, MultivariateGaussianExampleConfig):
            test_type = TestType.MultivariateGaussian
        else:
            test_type = TestType.BrownianMotionDiff
        make_ode_error_plot(cfg_obj, std, test_type)

        # make unconditional ODE convergence plots
        # plot_ode_trajs(cfg, std, sample_traj_out)

        test(end_time, cfg, out_trajs, std, sample_trajs, [hebo])
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
