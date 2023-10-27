#!/usr/bin/env python3
import warnings

from collections import namedtuple

import hydra
from hydra.core.config_store import ConfigStore
import numpy as np
import torch
import torch.distributions as dist
import torch.nn as nn
import matplotlib.pyplot as plt

from torchdiffeq import odeint

from toy_plot import SDE, analytical_log_likelihood
from toy_configs import register_configs
from toy_train_config import SampleConfig, get_model_path, get_classifier_path, ExampleType, TestType
from models.toy_sampler import AbstractSampler, interpolate_schedule
from toy_likelihoods import Likelihood, ClassifierLikelihood, GeneralDistLikelihood
from models.toy_temporal import TemporalTransformerUnet, TemporalClassifier, TemporalIDK
from models.toy_diffusion_models_config import GuidanceType


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


SampleOutput = namedtuple('SampleOutput', 'samples fevals')

# TODO: FIX THIS UP
from generative_model import generate_trajectory
from linear_gaussian_prob_prog import get_linear_gaussian_variables, JointVariables


SDEConfig = namedtuple('SDEConfig', 'drift diffusion sde_steps end_time')
DiffusionConfig = namedtuple('DiffusionConfig', 'f g')


def create_table(A, Q, C, R, mu_0, Q_0, dim):
    table = {}
    table[dim] = {}
    table[dim]['A'] = A
    table[dim]['Q'] = Q
    table[dim]['C'] = C
    table[dim]['R'] = R
    table[dim]['mu_0'] = mu_0
    table[dim]['Q_0'] = Q_0
    return table

def compute_diffusion_step(
        sde: SDEConfig,
        diffusion: DiffusionConfig,
        diffusion_time: torch.tensor
):
    A = torch.tensor([[1.]], device=device)
    Q = sde.diffusion * torch.tensor(1. / sde.sde_steps, device=device).reshape(1, 1)
    C = diffusion.f(diffusion_time).reshape(1, -1)
    R = diffusion.g(diffusion_time).reshape(1, -1) ** 2
    mu_0 = torch.tensor([[0.]], device=device)
    Q_0 = Q

    table = create_table(
        A=A, Q=Q, C=C, R=R, mu_0=mu_0, Q_0=Q_0, dim=1
    )

    ys, _, _, _ = generate_trajectory(
        num_steps=sde.sde_steps, A=A, Q=Q, C=C, R=R, mu_0=mu_0, Q_0=Q_0
    )

    lgv = get_linear_gaussian_variables(dim=1, num_obs=sde.sde_steps, table=table)
    jvs = JointVariables(lgv.ys)
    return jvs

#########################
#########################
def suppresswarning():
    warnings.warn("user", UserWarning)


class ToyEvaluator:
    def __init__(
        self,
        cfg: SampleConfig,
        sampler: AbstractSampler,
        diffusion_model: nn.Module,
        likelihood: Likelihood,
    ):
        self.cfg = cfg
        self.cond = torch.tensor([self.cfg.cond], device=device) if self.cfg.cond is not None and self.cfg.cond >= 0. else None
        self.sampler = sampler
        self.diffusion_model = diffusion_model.to(device)
        self.diffusion_model.eval()
        self.likelihood = likelihood

        self.load_model()

    def load_model(self):
        model_path = get_model_path(self.cfg)
        try:
            # load softmax model
            print('attempting to load diffusion model: {}'.format(model_path))
            self.diffusion_model.load_state_dict(torch.load('{}'.format(model_path)))
            print('successfully loaded diffusion model')
        except Exception as e:
            print('FAILED to load model: {} because {}\ncreating it...'.format(model_path, e))

    def grad_log_lik(self, xt, t, cond, model_output, cond_traj):
        x0_hat = self.sampler.predict_xstart(xt, model_output, t)
        if isinstance(self.diffusion_model, TemporalTransformerUnet):
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


class DiscreteEvaluator(ToyEvaluator):
    def sample_trajectories(self, cond_traj=None):
        x_min = self.get_x_min()

        samples = [x]
        for t in torch.arange(self.sampler.diffusion_timesteps-1, -1, -1, device=device):
            if t % 100 == 0:
                print(x[0, 0])
            time = t.reshape(-1)
            if isinstance(self.diffusion_model, TemporalTransformerUnet):
                unconditional_output = self.diffusion_model(x, time, None, None)
            else:
                unconditional_output = self.diffusion_model(x, time, None)
            if self.cfg.guidance == GuidanceType.Classifier:
                if self.cond is not None:
                    with torch.enable_grad():
                        xt = x.detach().clone().requires_grad_(True)
                        grad_log_lik = self.grad_log_lik(xt, time, self.cond, unconditional_output, cond_traj)
                else:
                    grad_log_lik = torch.tensor(0.)
                x = self.sampler.classifier_guided_reverse_sample(
                    xt=x, unconditional_output=unconditional_output,
                    t=t.item(), grad_log_lik=grad_log_lik
                )
            elif self.cfg.guidance == GuidanceType.ClassifierFree:
                if self.cond is None or self.cond < 0.:
                    conditional_output = unconditional_output
                else:
                    if isinstance(self.diffusion_model, TemporalTransformerUnet):
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
        ll_prior = self.sampler.prior_logp(latent).flatten(1).sum(1)
        # compute log(p(0)) = log(p(T)) + Tr(df/dx) where dx/dt = f
        return ll_prior + delta_ll, {'fevals': fevals}


class ContinuousEvaluator(ToyEvaluator):
    def get_score_function(self, t, x, extras=None):
        if self.cfg.test == TestType.Gaussian:
            return self.analytical_gaussian_score(t=t, x=x, mu_0=extras['mu_0'], sigma_0=extras['sigma_0'])
        elif self.cfg.test == TestType.BrownianMotion:
            return self.analytical_brownian_motion_score(t=t, x=x)
        elif self.cfg.test == TestType.Test:
            model_output = self.diffusion_model(x, t, None)
            return self.sampler.get_sf_estimator(model_output, xt=x, t=t)
        else:
            raise NotImplementedError

    def get_dx_dt(self, t, x, extras=None):
        time = t.reshape(-1)
        sf_est = self.get_score_function(t=time, x=x, extras=extras)
        # sf_est = self.analytical_gaussian_score(t=time, x=x, mu_0=extras['mu_0'], sigma_0=extras['sigma_0'])
        # sf_est = self.analytical_brownian_motion_score(t=time, x=x)
        dx_dt = self.sampler.probability_flow_ode(x, time, sf_est)
        return dx_dt

    def get_gaussian_dx_dt(self, t, x, extras=None):
        time = t.reshape(-1)
        sf_est = self.analytical_gaussian_score(t=t, x=x, mu_0=extras['mu_0'], sigma_0=extras['sigma_0'])
        dx_dt = self.sampler.probability_flow_ode(x, time, sf_est)
        return dx_dt

    def get_brownian_dx_dt(self, t, x, extras=None):
        time = t.reshape(-1)
        sf_est = self.analytical_brownian_motion_score(t=t, x=x)
        dx_dt = self.sampler.probability_flow_ode(x, time, sf_est)
        return dx_dt

    def analytical_gaussian_score(self, t, x, mu_0, sigma_0):
        '''
        Compute the analytical score p_t for t \in (0, 1)
        given the SDE formulation from Song et al. in the case that
        p_0 = N(mu_0, sigma_0) and p_1 = N(0, 1)
        '''
        lmc = self.sampler.log_mean_coeff(x_shape=x.shape, t=t)
        f = lmc.exp()
        g2 = (1 - (2. * lmc).exp())
        var = sigma_0 ** 2 * f ** 2 + g2
        score = (f * mu_0 - x) / var
        return score

    # def analytical_brownian_motion_score(self, t, x):
    #     '''
    #     Compute the analytical score p_t for t \in (0, 1)
    #     given the SDE formulation from Song et al. in the case that
    #     p_0(x, s) = N(0, d(s)\sqrt(s)) and p_1(x, s) = N(0, 1)
    #     '''
    #     print("WARNING: Using analytical Brownian motion score function")
    #     x_s = x.clone().squeeze(-1)
    #     rolled_x = x_s.roll(1)
    #     rolled_x[:, 0] = 0.
    #     lmc = self.sampler.log_mean_coeff(x_shape=x.shape, t=t)
    #     f = lmc.exp().squeeze(-1)
    #     mu = f * (rolled_x + self.cfg.sde_drift / self.cfg.sde_steps)
    #     g2 = (1 - (2. * lmc).exp()).squeeze(-1)
    #     bm_var_s = torch.linspace(0., 1., self.cfg.sde_steps, device=f.device)[self.cfg.sde_steps-f.shape[1]:]
    #     var = f ** 2 * self.cfg.sde_diffusion ** 2 * bm_var_s + g2
    #     score = ((mu - x_s) / var).unsqueeze(-1)
    #     return score

    def analytical_brownian_motion_score(self, t, x):
        '''
        Compute the analytical score p_t for t \in (0, 1)
        given the SDE formulation from Song et al. in the case that
        p_0(x, s) = N(0, d(s)\sqrt(s)) and p_1(x, s) = N(0, 1)
        '''
        sde = SDEConfig(self.cfg.sde_drift, self.cfg.sde_diffusion, self.cfg.sde_steps-1, 1.)

        f = lambda t : self.sampler.marginal_prob(x, t)[1].exp()[:, 0, :]
        g = lambda t : self.sampler.marginal_prob(x, t)[2][:, 0, :]
        diffusion = DiffusionConfig(f, g)

        jvs = compute_diffusion_step(sde, diffusion, diffusion_time=t)
        y_dist = jvs.dist.dist
        score = torch.linalg.solve(y_dist.covariance_matrix, (y_dist.loc.reshape(-1, 1) - x))

        # dt = 1. / (self.cfg.sde_steps-1)
        # cov = f(t) ** 2 * dt
        # y1 = f(t) ** 2 * dt + g(t) ** 2
        # y2 = f(t) ** 2 * 2 * dt + g(t) ** 2
        # cov_mat = torch.tensor([[y1, cov], [cov, y2]], device=device)
        # mean = torch.zeros(x.shape[1], 1, device=device)
        # score = torch.linalg.solve(cov_mat, (mean.reshape(-1, 1) - x))

        return score

    def get_x_min(self):
        if self.cfg.example == ExampleType.Gaussian:
            return torch.distributions.Normal(0, torch.tensor(1., device=device)).sample([
                    self.cfg.num_samples, 1, 1
            ])
        elif self.cfg.example == ExampleType.BrownianMotion:
            return torch.distributions.Normal(0, torch.tensor(1., device=device)).sample([
                self.cfg.num_samples,
                self.cfg.sde_steps-1,
                1,
            ])
        else:
            raise NotImplementedError

    def sample_trajectories(self, extras, atol=1e-4, rtol=1e-4):
        x_min = self.get_x_min()

        fevals = 0
        def ode_fn(t, x):
            nonlocal fevals
            fevals += 1
            dx_dt = self.get_dx_dt(t, x, extras)
            return dx_dt

        times = torch.tensor([1., self.sampler.t_eps], device=x_min.device)
        # times = torch.arange(1., self.sampler.t_eps, -0.01, device=x_min.device)
        sol = odeint(ode_fn, x_min, times, atol=atol, rtol=rtol, method='rk4')
        return SampleOutput(samples=sol, fevals=fevals)

    @torch.no_grad()
    def ode_log_likelihood(self, x, extras=None, atol=1e-4, rtol=1e-4):
        extras = {} if extras is None else extras
        # hutchinson's trick
        v = torch.randint_like(x, 2) * 2 - 1
        fevals = 0
        def ode_fn(t, x):
            nonlocal fevals
            fevals += 1
            with torch.enable_grad():
                x = x[0].detach().requires_grad_()
                dx_dt = self.get_dx_dt(t, x, extras)
                grad = torch.autograd.grad((dx_dt * v).sum(), x)[0]
                d_ll = (v * grad).flatten(1).sum(1)
            return torch.cat([dx_dt.reshape(-1), d_ll.reshape(-1)])
        x_min = x, x.new_zeros([x.shape[0]])
        times = torch.tensor([self.sampler.t_eps, 1.], device=x.device)
        sol = odeint(ode_fn, x_min, times, atol=atol, rtol=rtol, method='rk4')
        latent, delta_ll = sol[0][-1], sol[1][-1]
        ll_prior = self.sampler.prior_logp(latent).flatten(1).sum(1)
        # compute log(p(0)) = log(p(T)) + Tr(df/dx) where dx/dt = f
        return ll_prior + delta_ll, {'fevals': fevals}


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

def test_gaussian(end_time, cfg, sample_trajs, std, extras=None):
    dt = end_time / cfg.sde_steps
    analytical_llk = torch.distributions.Normal(
        extras['mu_0'], extras['sigma_0']
    ).log_prob(sample_trajs)
    print('analytical_llk: {}'.format(analytical_llk.squeeze()))
    ode_llk = std.ode_log_likelihood(sample_trajs, extras=extras)
    print('\node_llk: {}'.format(ode_llk))
    mse_llk = torch.nn.MSELoss()(analytical_llk.squeeze(), ode_llk[0])
    print('\nmse_llk: {}'.format(mse_llk))

    plt.clf()
    plt_llk(sample_trajs, ode_llk[0].exp(), plot_type='scatter')
    plt_llk(sample_trajs, analytical_llk.exp(), plot_type='line')
    import pdb; pdb.set_trace()

def test_brownian_motion(end_time, cfg, sample_trajs, std, extras=None):
    dt = end_time / (cfg.sde_steps-1)
    analytical_trajs = torch.cat([
        torch.zeros(sample_trajs.shape[0], 1, 1, device=sample_trajs.device),
        sample_trajs
    ], dim=1)

    analytical_llk = analytical_log_likelihood(analytical_trajs, SDE(cfg.sde_drift, cfg.sde_diffusion), dt)
    print('analytical_llk: {}'.format(analytical_llk))

    ode_llk = std.ode_log_likelihood(sample_trajs, extras=extras)
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

def test(end_time, cfg, out_trajs, std, extras=None):
    if cfg.example == ExampleType.Gaussian:
        test_gaussian(end_time, cfg, out_trajs, std, extras)
    elif cfg.example == ExampleType.BrownianMotion:
        test_brownian_motion(end_time, cfg, out_trajs, std, extras)
    else:
        raise NotImplementedError


@hydra.main(version_base=None, config_path="conf", config_name="continuous_sample_config")
def sample(cfg):
    d_model = torch.tensor(1)
    sampler = hydra.utils.instantiate(cfg.sampler)
    diffusion_model = hydra.utils.instantiate(cfg.diffusion, d_model=d_model, device=device)
    diffusion_model = TemporalIDK()
    likelihood = hydra.utils.instantiate(cfg.likelihood)

    std = ContinuousEvaluator(
        cfg=cfg,
        sampler=sampler,
        diffusion_model=diffusion_model,
        likelihood=likelihood,
    )
    end_time = torch.tensor(1., device=device)


    # TODO: delete
    mu_0 = torch.tensor(0., device=device)
    sigma_0 = torch.tensor(1., device=device)
    extras = {'mu_0': mu_0, 'sigma_0': sigma_0}


    # cond_traj = None
    # rare_traj_file = 'rare_traj.pt'
    # rare_traj = torch.load(rare_traj_file).to(device)
    # std.viz_trajs(rare_traj, end_time, 100, clf=False, clr='red')
    # cond_traj = rare_traj.diff(dim=-1).reshape(1, -1, 1)


    sample_out = std.sample_trajectories(extras)
    print('fevals: {}'.format(sample_out.fevals))
    sample_traj_out = sample_out.samples
    trajs = sample_traj_out[-1]
    out_trajs = trajs
    # undiffed_trajs = trajs.cumsum(dim=-2)
    # out_trajs = torch.cat([
    #     torch.zeros(undiffed_trajs.shape[0], 1, 1, device=undiffed_trajs.device),
    #     undiffed_trajs
    # ], dim=1)
    for idx, out_traj in enumerate(out_trajs):
        std.viz_trajs(out_traj, end_time, idx, clf=False)

    test(end_time, cfg, out_trajs, std, extras)


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        suppresswarning()

    cs = ConfigStore.instance()
    cs.store(name="vpsde_sample_config", node=SampleConfig)
    register_configs()

    with torch.no_grad():
        sample()
