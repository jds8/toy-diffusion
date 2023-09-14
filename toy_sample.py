#!/usr/bin/env python3
import warnings

import hydra
from hydra.core.config_store import ConfigStore
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from toy_configs import register_configs
from toy_train_config import SampleConfig, get_model_path
from models.toy_sampler import Sampler


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def suppresswarning():
    warnings.warn("user", UserWarning)


class ToyEvaluator:
    def __init__(
        self,
        cfg: SampleConfig,
        sampler: Sampler,
        diffusion_model: nn.Module,
    ):
        self.cfg = cfg
        # self.cond = torch.tensor([self.cfg.cond], device=device) if self.cfg.cond > 0. else None
        self.sampler = sampler
        self.diffusion_model = diffusion_model.to(device)
        self.diffusion_model.eval()

        self.load_model()

    def load_model(self):
        model_path = get_model_path(self.cfg)
        try:
            # load softmax model
            print('attempting to load model: {}'.format(model_path))
            self.diffusion_model.load_state_dict(torch.load('{}'.format(model_path)))
            print('successfully loaded')
        except Exception as e:
            print('FAILED to load model: {} because {}\ncreating it...'.format(model_path, e))

    # def grad_log_lik(self, xt, t, cond, model_output, cond_traj):
    #     x0_hat = self.sampler.predict_xstart(xt, model_output, t)
    #     if isinstance(self.diffusion_model, TemporalTransformerUnet):
    #         return self.likelihood.grad_log_lik(cond, xt, x0_hat, cond_traj)
    #     else:
    #         return self.likelihood.grad_log_lik(cond, xt, x0_hat)

    def sample_trajectories(self, cond_traj=None):
        x = torch.randn(
            self.cfg.num_samples,
            self.cfg.sde_steps,
            self.diffusion_model.d_model,
            device=device
        )
        for t in torch.arange(self.sampler.diffusion_timesteps-1, -1, -1, device=device):
            if t % 100 == 0:
                print(x[0, 0])
            time = t.reshape(-1)
            # if isinstance(self.diffusion_model, TemporalTransformerUnet):
            #     unconditional_output = self.diffusion_model(x, time, None, None)
            # else:
            #     unconditional_output = self.diffusion_model(x, time, None)
            unconditional_output = self.diffusion_model(x, time, None)
            # if self.cfg.guidance == GuidanceType.Classifier:
            #     with torch.enable_grad():
            #         xt = x.detach().clone().requires_grad_(True)
            #         grad_log_lik = self.grad_log_lik(xt, time, self.cond, unconditional_output, cond_traj)
            #     x = self.sampler.classifier_guided_reverse_sample(
            #         xt=x, unconditional_output=unconditional_output,
            #         t=t.item(), grad_log_lik=grad_log_lik
            #     )
            # elif self.cfg.guidance == GuidanceType.ClassifierFree:
            #     if self.cond is None or self.cond <= 0.:
            #         conditional_output = unconditional_output
            #     else:
            #         if isinstance(self.diffusion_model, TemporalTransformerUnet):
            #             conditional_output = self.diffusion_model(x, time, cond_traj, self.cond)
            #         else:
            #             conditional_output = self.diffusion_model(x, time, self.cond)
            #     x = self.sampler.classifier_free_reverse_sample(
            #         xt=x, unconditional_output=unconditional_output,
            #         conditional_output=conditional_output, t=t.item()
            #     )
            # else:
            #     raise NotImplementedError
            x = self.sampler.reverse_sample(
                x, unconditional_output, t.item(), unconditional_output,
            )
            return x

    def viz_trajs(self, traj, end_time, idx, clf=True):
        full_state_pred = traj.detach().squeeze(0).cpu().numpy()

        plt.plot(torch.linspace(0, end_time, full_state_pred.shape[0]), full_state_pred, color='green')

        plt.savefig('figs/sample_{}.pdf'.format(idx))

        if clf:
            plt.clf()

    @torch.no_grad()
    def log_likelihood(self, x, extra_args=None, atol=1e-4, rtol=1e-4):
        extra_args = {} if extra_args is None else extra_args
        s_in = x.new_ones([x.shape[0]])
        # hutchinson's trick
        v = torch.randint_like(x, 2) * 2 - 1
        fevals = 0
        def ode_fn(t, x):
            nonlocal fevals
            with torch.enable_grad():
                x = x[0].detach().requires_grad_()
                diffusion_time = t.reshape(-1) * self.sampler.diffusion_timesteps
                sf_est = self.diffusion_model(x, diffusion_time, **extra_args)
                coef = -0.5 * self.sampler.interpolate_beta(diffusion_time)
                dx_dt = coef * (x + sf_est)
                fevals += 1
                grad = torch.autograd.grad((dx_dt * v).sum(), x)[0]
                d_ll = (v * grad).flatten(1).sum(1)
            return torch.cat([dx_dt.reshape(-1), d_ll.reshape(-1)])
        x_min = x, x.new_zeros([x.shape[0]])
        times = torch.tensor([0., 1.], device=x.device)
        sol = odeint(ode_fn, x_min, times, atol=atol, rtol=rtol, method='dopri5')
        latent, delta_ll = sol[0][-1], sol[1][-1]
        ll_prior = torch.distributions.Normal(0, self.sampler.betas[-1]).log_prob(latent).flatten(1).sum(1)
        return ll_prior + delta_ll, {'fevals': fevals}


@hydra.main(version_base=None, config_path="conf", config_name="sample_config")
def sample(cfg):
    d_model = torch.tensor(1)
    sampler = hydra.utils.instantiate(cfg.sampler)
    diffusion_model = hydra.utils.instantiate(cfg.diffusion, d_model=d_model, device=device)
    # likelihood = hydra.utils.instantiate(cfg.likelihood)

    std = ToyEvaluator(
        cfg=cfg,
        sampler=sampler,
        diffusion_model=diffusion_model,
        # likelihood=likelihood,
    )
    trajs = std.sample_trajectories()
    # undiffed_trajs = trajs.cumsum(dim=-2)
    # out_trajs = torch.cat([
    #     torch.zeros(undiffed_trajs.shape[0], 1, 1, device=undiffed_trajs.device),
    #     undiffed_trajs
    # ], dim=1)
    end_time = torch.tensor(1.)
    for idx, out_traj in enumerate(out_trajs):
        std.viz_trajs(out_traj, end_time, idx, clf=False)
    # TODO: remove
    # std_trajs = torch.rand(20, 39, 4, device=device)
    # log_lik = std.log_likelihood(std_trajs, extra_args={'cond': None})


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        suppresswarning()

    cs = ConfigStore.instance()
    cs.store(name="base_sample_config", node=SampleConfig)
    register_configs()

    with torch.no_grad():
        sample()
