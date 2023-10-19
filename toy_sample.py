#!/usr/bin/env python3
import warnings

from collections import namedtuple

import hydra
from hydra.core.config_store import ConfigStore
import torch
import torch.distributions as dist
import torch.nn as nn
import matplotlib.pyplot as plt

from torchdiffeq import odeint

from toy_plot import SDE, analytical_log_likelihood
from toy_configs import register_configs
from toy_train_config import SampleConfig, get_model_path, get_classifier_path
from models.toy_sampler import AbstractSampler, interpolate_schedule
from toy_likelihoods import Likelihood, ClassifierLikelihood, GeneralDistLikelihood
from models.toy_temporal import TemporalTransformerUnet, TemporalClassifier, TemporalIDK
from models.toy_diffusion_models_config import GuidanceType


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


SampleOutput = namedtuple('SampleOutput', 'samples fevals')


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
        # x = torch.distributions.Normal(0, self.sampler.sqrt_one_minus_alphas_cumprod[-1]).sample([
        #     self.cfg.num_samples, 1, 1
        # ])
        x = torch.distributions.Normal(0, self.sampler.sqrt_one_minus_alphas_cumprod[-1]).sample([
            self.cfg.num_samples,
            self.cfg.sde_steps,
            self.diffusion_model.d_model,
        ])
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
    def ode_log_likelihood(self, x, extra_args=None, atol=1e-4, rtol=1e-4):
        """ THIS PROBABLY SHOULDN'T BE USED """
        extra_args = {} if extra_args is None else extra_args
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
    def get_dx_dt(self, t, x, cond_traj=None):
        time = t.reshape(-1)
        model_output = self.diffusion_model(x, time, cond_traj)
        sf_est = self.sampler.get_sf_estimator(model_output, xt=x, t=time)
        dx_dt = self.sampler.probability_flow_ode(x, time, sf_est)
        return dx_dt

    def analytical_gaussian_score(self, t, x):
        '''
        Compute the analytical score p_t for t \in (0, 1)
        given the SDE formulation from Song et al. in the case that
        p_0 = N(1, 2) and p_1 = N(0, 1)
        '''
        lmc = self.sampler.log_mean_coeff(x_shape=x.shape, t=t)
        f = lmc.exp()
        g = (1 - (2. * lmc).exp())
        var = 4 * f ** 2 + g
        return (f - x) / var

    def sample_trajectories(self, cond_traj=None, atol=1e-4, rtol=1e-4):
        x_min = torch.distributions.Normal(0, torch.tensor(1., device=device)).sample([
                self.cfg.num_samples, 1, 1
        ])

        fevals = 0
        def ode_fn(t, x):
            nonlocal fevals
            fevals += 1
            dx_dt = self.get_dx_dt(t, x, cond_traj)
            print('mean: {}'.format(x.mean()))
            print('std: {}'.format(x.std()))
            print('range: {}\n'.format(x.max()-x.min()))
            return dx_dt

        # times = torch.tensor([1., self.sampler.t_eps], device=x_min.device)
        times = torch.range(1., self.sampler.t_eps, -0.01, device=x_min.device)
        sol = odeint(ode_fn, x_min, times, atol=atol, rtol=rtol, method='rk4')
        return SampleOutput(samples=sol, fevals=fevals)

    @torch.no_grad()
    def ode_log_likelihood(self, x, extra_args=None, atol=1e-4, rtol=1e-4):
        extra_args = {} if extra_args is None else extra_args
        # hutchinson's trick
        v = torch.randint_like(x, 2) * 2 - 1
        fevals = 0
        def ode_fn(t, x):
            nonlocal fevals
            fevals += 1
            with torch.enable_grad():
                x = x[0].detach().requires_grad_()
                dx_dt = self.get_dx_dt(t, x)
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


def plt_llk(traj, lik, plot_type='scatter'):
    full_state_pred = traj.detach().squeeze().cpu().numpy()
    full_state_lik = lik.detach().squeeze().cpu().numpy()

    if plot_type == 'scatter':
        plt.scatter(full_state_pred, full_state_lik, color='blue')
    else:
        idx = full_state_pred.argsort()
        sorted_state = full_state_pred[idx]
        sorted_lik = full_state_lik[idx]
        plt.plot(sorted_state, sorted_lik, color='red')

    plt.savefig('figs/scatter.pdf')


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

    cond_traj = None
    # rare_traj_file = 'rare_traj.pt'
    # rare_traj = torch.load(rare_traj_file).to(device)
    # std.viz_trajs(rare_traj, end_time, 100, clf=False, clr='red')
    # cond_traj = rare_traj.diff(dim=-1).reshape(1, -1, 1)

    sample_out = std.sample_trajectories(cond_traj)
    print('fevals: {}'.format(sample_out.fevals))
    sample_traj_out = sample_out.samples
    trajs = sample_traj_out[-1]
    undiffed_trajs = trajs.cumsum(dim=-2)
    out_trajs = torch.cat([
        torch.zeros(undiffed_trajs.shape[0], 1, 1, device=undiffed_trajs.device),
        undiffed_trajs
    ], dim=1)
    for idx, out_traj in enumerate(out_trajs):
        std.viz_trajs(out_traj, end_time, idx, clf=False)

    # TODO: remove
    dt = end_time / cfg.sde_steps
    sample_trajs = out_trajs[:, 1].reshape(-1, 1, 1)
    analytical_llk = torch.distributions.Normal(1, 2).log_prob(sample_trajs)
    print('analytical_llk: {}'.format(analytical_llk.squeeze()))
    ode_llk = std.ode_log_likelihood(sample_trajs, extra_args={'cond': None})
    print('\node_llk: {}'.format(ode_llk))
    mse_llk = torch.nn.MSELoss()(analytical_llk.squeeze(), ode_llk[0])
    print('\nmse_llk: {}'.format(mse_llk))

    plt.clf()
    plt_llk(sample_trajs, ode_llk[0].exp(), plot_type='scatter')
    plt_llk(sample_trajs, analytical_llk.exp(), plot_type='line')
    import pdb; pdb.set_trace()

    # dt = end_time / cfg.sde_steps
    # analytical_llk = analytical_log_likelihood(out_trajs, SDE(cfg.sde_drift, cfg.sde_diffusion), dt)
    # print('analytical_llk: {}'.format(analytical_llk))
    # ode_llk = std.ode_log_likelihood(trajs, extra_args={'cond': None})
    # print('\node_llk: {}'.format(ode_llk))
    # mse_llk = torch.nn.MSELoss()(analytical_llk.squeeze(), ode_llk[0])
    # print('\nmse_llk: {}'.format(mse_llk))

    # # TODO: remove
    # std_trajs = torch.randn(10, 1000, 1, device=device)
    # undiffed_trajs = std_trajs.cumsum(dim=-2)
    # out_trajs = torch.cat([
    #     torch.zeros(undiffed_trajs.shape[0], 1, 1, device=undiffed_trajs.device),
    #     undiffed_trajs
    # ], dim=1)
    # sde = SDE(cfg.sde_drift, cfg.sde_diffusion)
    # dt = torch.tensor(1., device=device) / cfg.sde_steps
    # analytical_llk = analytical_log_likelihood(out_trajs, sde, dt)
    # print('analytical_llk: {}'.format(analytical_llk))
    # ode_llk = std.ode_log_likelihood(std_trajs, extra_args={'cond': None})
    # print('\node_llk: {}'.format(ode_llk))
    # mse_llk = torch.nn.MSELoss()(analytical_llk.squeeze(), ode_llk[0])
    # print('\nmse_llk: {}'.format(mse_llk))
    # import pdb; pdb.set_trace()


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        suppresswarning()

    cs = ConfigStore.instance()
    cs.store(name="vpsde_sample_config", node=SampleConfig)
    register_configs()

    with torch.no_grad():
        sample()
