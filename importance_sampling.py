#!/usr/bin/env python3

from typing import Callable
import torch
import torch.distributions as dist
import numpy as np
import scipy.stats as stats
import scipy.integrate as integrate

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Proposal:
    def __init__(self, std):
        self.std = std
        self.trajs = None
        self.ode_llk = None


class GaussianProposal(Proposal):
    def sample(self):
        sample_traj_out = self.std.sample_trajectories(
            cond=self.std.cond.to(device),
            alpha=self.std.likelihood.alpha.reshape(-1, 1).to(device),
        )
        self.sample_trajs = sample_traj_out.samples[-1]
        self.trajs = self.sample_trajs * self.std.cfg.example.sigma + self.std.cfg.example.mu
        return self.trajs, self.sample_trajs

    def log_prob(self, samples: torch.Tensor):
        raw_ode_llk = self.std.ode_log_likelihood(
            samples,
            cond=self.std.cond.to(device),
            alpha=self.std.likelihood.alpha.reshape(-1, 1).to(device),
            exact=self.std.cfg.compute_exact_trace,
            num_hutchinson_samples=self.std.cfg.num_hutchinson_samples,
        )[0][-1]
        scale_factor = torch.tensor(self.std.cfg.example.sigma).log()
        self.ode_llk = raw_ode_llk - scale_factor
        return self.ode_llk


class MultivariateGaussianProposal(Proposal):
    def sample(self):
        cond = torch.tensor([self.std.cfg.cond]).to(device)
        sample_traj_out = self.std.sample_trajectories(
            cond=cond,
            alpha=self.std.likelihood.alpha.reshape(-1, 1).to(device),
        )
        self.sample_trajs = sample_traj_out.samples[-1]
        mu = torch.tensor(self.std.cfg.example.mu, device=self.sample_trajs.device)
        sigma = torch.tensor(self.std.cfg.example.sigma, device=self.sample_trajs.device)
        L = torch.linalg.cholesky(sigma)
        self.trajs = torch.matmul(L, self.sample_trajs) + mu
        return self.trajs, self.sample_trajs

    def log_prob(self, samples: torch.Tensor):
        cond = torch.tensor([self.std.cfg.cond]).to(device)
        raw_ode_llk = self.std.ode_log_likelihood(
            samples,
            cond=cond,
            alpha=self.std.likelihood.alpha.reshape(-1, 1).to(device),
            exact=self.std.cfg.compute_exact_trace,
            num_hutchinson_samples=self.std.cfg.num_hutchinson_samples,
        )[0][-1]
        L = torch.linalg.cholesky(torch.tensor(self.std.cfg.example.sigma))
        scale_factor = L.det().abs().log()
        self.ode_llk = raw_ode_llk - scale_factor
        return self.ode_llk


class BrownianMotionDiffProposal(Proposal):
    def __init__(self, std):
        super().__init__(std)
        self.dt = torch.tensor(1/(self.std.example.sde_steps-1))

    def sample(self):
        sample_traj_out = self.std.sample_trajectories(
            cond=self.std.cond.to(device),
            alpha=self.std.likelihood.alpha.reshape(-1, 1).to(device)
        )
        self.sample_trajs = sample_traj_out.samples[-1]
        scaled_trajs = self.sample_trajs * self.dt.sqrt()
        self.trajs = torch.cat([
            torch.zeros(
                scaled_trajs.shape[0], 1, 1,
                device=scaled_trajs.device
            ),
            scaled_trajs.cumsum(dim=-2)
        ], dim=1)
        return self.trajs, self.sample_trajs

    def log_prob(self, samples: torch.Tensor):
        raw_ode_llk = self.std.ode_log_likelihood(
            samples,
            cond=self.std.cond.to(device),
            alpha=self.std.likelihood.alpha.reshape(-1, 1).to(device),
            exact=self.std.cfg.compute_exact_trace,
            num_hutchinson_samples=self.std.cfg.num_hutchinson_samples,
        )[0][-1]
        scale_factor = self.dt.sqrt().log() * (self.std.cfg.example.sde_steps-1)
        self.ode_llk = raw_ode_llk - scale_factor
        return self.ode_llk


class StudentTProposal(Proposal):
    def __init__(self, std):
        super().__init__(std)
        self.sigma = torch.tensor(35.9865)  # from dist.StudentT(1.5).sample([100000000]).std()
        if self.std.cfg.example.nu > 2.:
            self.sigma = torch.tensor(self.std.cfg.example.nu / (self.std.cfg.example.nu - 2)).sqrt()
    def sample(self):
        sample_traj_out = self.std.sample_trajectories(
            cond=self.std.cond.to(device),
            alpha=self.std.likelihood.alpha.reshape(-1, 1).to(device),
        )
        self.sample_trajs = sample_traj_out.samples[-1]
        self.trajs = self.sample_trajs * self.sigma
        return self.trajs, self.sample_trajs

    def log_prob(self, samples: torch.Tensor):
        raw_ode_llk = self.std.ode_log_likelihood(
            samples,
            cond=self.std.cond.to(device),
            alpha=self.std.likelihood.alpha.reshape(-1, 1).to(device),
            exact=self.std.cfg.compute_exact_trace,
            num_hutchinson_samples=self.std.cfg.num_hutchinson_samples,
        )[0][-1]
        self.ode_llk = raw_ode_llk - self.sigma.log()
        return self.ode_llk


class Target:
    def __init__(self, cfg):
        self.cfg = cfg
    def log_prob(self, saps: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
    def analytical_prob(self) -> torch.Tensor:
        return torch.tensor(-1.)
    def analytical_conditional_log_prob(
        self, 
        saps: torch.Tensor,
        alpha: torch.Tensor
    ) -> torch.Tensor:
        return self.log_prob(saps) - self.analytical_prob(alpha).log()


class GaussianTarget(Target):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.dist = dist.Normal(
            cfg.example.mu,
            cfg.example.sigma
        )
    def log_prob(self, saps: torch.Tensor) -> torch.Tensor:
        return self.dist.log_prob(saps)
    def analytical_prob(self, alpha: torch.Tensor) -> torch.Tensor:
        # 0.0027 for self.cfg.cond=3
        return 2*self.dist.cdf(
            self.cfg.example.mu - alpha * self.cfg.example.sigma
        )


class MultivariateGaussianTarget(Target):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.dist = dist.MultivariateNormal(
            torch.tensor(cfg.example.mu, device=device).squeeze(-1),
            torch.tensor(cfg.example.sigma, device=device)
        )
    def log_prob(self, saps: torch.Tensor) -> torch.Tensor:
        return self.dist.log_prob(saps.squeeze(-1))
    def analytical_prob(self, alpha: torch.Tensor) -> torch.Tensor  :
        return torch.tensor(1 - stats.chi(df=self.cfg.example.d).cdf(alpha), dtype=alpha.dtype)


class BrownianMotionDiffTarget(Target):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.dt = torch.tensor(1. / (cfg.example.sde_steps-1))
        self.dist = dist.Normal(
            0.,
            self.dt.sqrt(),
        )
        self.quadrature_values = {
            3: {
                1.: 0.37064871,
                1.5: 0.146098491,
                2.0: 0.047321563,
                2.5: 0.012584247,
                3.0: 0.00270902,
                3.5: 0.0004655756,
                4.0: 6.334919e-05,
                4.5: 6.79543301e-06,
                5.0: 5.7330383e-07,
                5.5: 3.7979128e-08,
                6.0: 1.97317518e-09,
            },
            5: {
                1.: 0.4216649,
                1.5: 0.1655791,
                2.0: 0.0531212,
                2.5: 0.0138067,
                3.0: 0.0028901723,
                3.5: 0.00048466001,
                4.0: 6.4789885e-05,
            }
        }
    def log_prob(self, saps: torch.Tensor) -> torch.Tensor:
        # the elements of saps are dependent, but those of
        # saps.diff(dim=1) are independent increments
        return self.dist.log_prob(saps.diff(dim=1)).sum(dim=1)
    def empirical_prob(self, alpha: torch.Tensor) -> torch.Tensor:
        batch_size = 1690000
        x0 = torch.randn(
            batch_size,
            self.cfg.example.sde_steps-1,
            1,
        )
        dt = torch.tensor(1. / (self.cfg.example.sde_steps-1))
        scaled_x0 = x0 * dt.sqrt()  # standardize data
        sample_trajs = torch.cat([
            torch.zeros(batch_size, 1, 1),
            scaled_x0.cumsum(dim=1)
        ], dim=1)
        return (sample_trajs > alpha).any(dim=1).to(sample_trajs.dtype).mean()
    def quadrature(self, alpha):
        def normal_pdf(x, mean, var):
            """Probability density function of a normal distribution."""
            return stats.norm.pdf(x, loc=mean, scale=np.sqrt(var))
        """Computes the probability of leaving [-alpha, alpha] using quadrature."""
        # First integral over X1 ~ N(0, 1/4)
        def inner_integral_x1(x1):
            # Second integral over X2 ~ N(X1, 1/4)
            def inner_integral_x2(x2):
                # Third integral over X3 ~ N(X2, 1/4)
                def inner_integral_x3(x3):
                    # Fourth integral over X4 ~ N(X3, 1/4)
                    def inner_integral_x4(x4):
                        return normal_pdf(x4, x3, 1/4)
                    result_x4, _ = integrate.quad(inner_integral_x4, -alpha, alpha)
                    return normal_pdf(x3, x2, 1/4) * result_x4
                result_x3, _ = integrate.quad(inner_integral_x3, -alpha, alpha)
                return normal_pdf(x2, x1, 1/4) * result_x3
            result_x2, _ = integrate.quad(inner_integral_x2, -alpha, alpha)
            return normal_pdf(x1, 0, 1/4) * result_x2
        result_x1, _ = integrate.quad(inner_integral_x1, -alpha, alpha)
        return 1 - result_x1
    def analytical_prob(self, alpha: torch.Tensor) -> torch.Tensor:
        # no better solution known
        if self.cfg.example.sde_steps == 104:
            if alpha == 2.5:
                return torch.tensor(0.0108)
            if alpha == 3.0:
                return torch.tensor(0.00225)
        elif self.cfg.example.sde_steps in [3, 5]:
            value_dct = self.quadrature_values[self.cfg.example.sde_steps]
            alpha_float = alpha.item()
            if alpha_float in value_dct:
                return torch.tensor(value_dct[alpha_float])
            raise NotImplementedError
        else:
            raise NotImplementedError
    def analytical_upper_bound(self, alpha: torch.Tensor) -> torch.Tensor:
        # 0.0059 for alpha=3
        # https://math.stackexchange.com/questions/2336266/exit-probability-on-a-brownian-motion-from-an-interval
        val = 2 * np.sqrt(2)/(alpha * np.sqrt(np.pi)) * np.exp(-alpha**2/2)
        return torch.tensor(val)


class StudentTTarget(Target):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.sigma = torch.tensor(35.9865)  # from dist.StudentT(1.5).sample([100000000]).std()
        if self.cfg.example.nu > 2.:
            self.sigma = torch.tensor(self.cfg.example.nu / (self.cfg.example.nu - 2)).sqrt()
    def log_prob(self, saps):
        return torch.tensor(
            stats.t.pdf(saps.cpu(), df=self.cfg.example.nu),
            device=saps.device
        ).log()
    def analytical_prob(self, alpha: torch.Tensor):
        # 0.0027 for self.cfg.cond=3
        return torch.tensor(2*stats.t.cdf(
            -alpha * self.sigma,
            df=self.cfg.example.nu
        ))


class ImportanceSampler:
    def __init__(self, target, proposal):
        self.target = target
        self.proposal = proposal

    def estimate(self, test_fn: Callable):
        saps = self.proposal.sample()
        log_qrobs = self.proposal.log_prob(saps).squeeze()
        log_probs = self.target.log_prob(saps).squeeze()
        log_ws = log_probs - log_qrobs
        max_log_w = log_ws.max()
        w_bars = (log_ws - max_log_w).exp()
        phis = test_fn(saps)
        return ((phis * w_bars).mean().log() + max_log_w).exp()
