#!/usr/bin/env python3

from typing import Callable
import torch

from toy_sample import ToyEvaluator


class Proposal:
    def __init__(self, std: ToyEvaluator):
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
