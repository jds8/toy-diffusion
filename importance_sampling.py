#!/usr/bin/env python3

from typing import Callable


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
