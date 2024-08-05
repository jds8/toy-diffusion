#!/usr/bin/env python3

from typing import Callable
import torch


class Proposal:
    def __init__(self, std):
        self.std = std
        self.trajs = None
        self.ode_llk = None


class GaussianProposal(Proposal):
    def sample(self):
        sample_traj_out = self.std.sample_trajectories()
        sample_trajs = sample_traj_out.samples
        self.trajs = sample_trajs[-1] * self.std.cfg.example.sigma + self.std.cfg.example.mu
        return self.trajs

    def log_prob(self, samples):
        sample_trajs = (samples - self.std.cfg.example.mu) / self.std.cfg.example.sigma
        raw_ode_llk = self.std.ode_log_likelihood(sample_trajs)[0]
        scale_factor = torch.tensor(self.std.cfg.example.sigma).log()
        self.ode_llk = raw_ode_llk - scale_factor
        return self.ode_llk


class BrownianMotionDiffProposal(Proposal):
    def __init__(self, std):
        super().__init__(std)
        self.dt = torch.tensor(1/(self.std.example.sde_steps-1))

    def sample(self):
        sample_traj_out = self.std.sample_trajectories()
        sample_trajs = sample_traj_out.samples
        self.trajs = sample_trajs[-1] * self.dt.sqrt()
        return self.trajs

    def log_prob(self, samples):
        sample_trajs = samples / self.dt.sqrt()
        raw_ode_llk = self.std.ode_log_likelihood(sample_trajs)[0]
        scale_factor = self.dt.sqrt().log() * self.std.cfg.example.sde_steps
        self.ode_llk = raw_ode_llk - scale_factor
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
        # 0.3351 for alpha=3
        if alpha == 3.:
            return 0.005909131215917344
        else:
            raise NotImplementedError


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
