#!/usr/bin/env python3

from typing import Callable
import torch
import numpy as np
import scipy.stats as stats


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

    def log_prob(self, samples):
        raw_ode_llk = self.std.ode_log_likelihood(
            samples,
            cond=self.std.cond.to(device),
            alpha=self.std.likelihood.alpha.reshape(-1, 1).to(device)
        )[0]
        scale_factor = torch.tensor(self.std.cfg.example.sigma).log()
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

    def log_prob(self, samples):
        raw_ode_llk = self.std.ode_log_likelihood(
            samples,
            cond=self.std.cond.to(device),
            alpha=self.std.likelihood.alpha.reshape(-1, 1).to(device)
        )[0]
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

    def log_prob(self, samples):
        raw_ode_llk = self.std.ode_log_likelihood(
            samples,
            cond=self.std.cond.to(device),
            alpha=self.std.likelihood.alpha.reshape(-1, 1).to(device)
        )[0]
        self.ode_llk = raw_ode_llk - self.sigma.log()
        return self.ode_llk


class Target:
    def __init__(self, cfg):
        self.cfg = cfg
    def log_prob(self, saps):
        raise NotImplementedError
    def analytical_prob(self):
        return -1.


class GaussianTarget(Target):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.dist = torch.distributions.Normal(
            cfg.example.mu,
            cfg.example.sigma
        )
    def log_prob(self, saps):
        return self.dist.log_prob(saps)
    def analytical_prob(self, alpha):
        # 0.0027 for self.cfg.cond=3
        return 2*self.dist.cdf(
            self.cfg.example.mu - alpha * self.cfg.example.sigma
        )


class BrownianMotionDiffTarget(Target):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.dt = torch.tensor(1. / (cfg.example.sde_steps-1))
        self.dist = torch.distributions.Normal(
            0.,
            self.dt.sqrt(),
        )
    def log_prob(self, saps):
        # the elements of saps are dependent, but those of
        # saps.diff(dim=1) are independent increments
        return self.dist.log_prob(saps.diff(dim=1)).sum(dim=1)
    def analytical_prob(self, alpha):
        # 0.0059 for alpha=3
        # https://math.stackexchange.com/questions/2336266/exit-probability-on-a-brownian-motion-from-an-interval
        return 2 * np.sqrt(2)/(alpha * np.sqrt(np.pi)) * np.exp(-alpha**2/2)


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
    def analytical_prob(self, alpha):
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
