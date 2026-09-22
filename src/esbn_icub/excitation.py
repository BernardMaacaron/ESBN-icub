"""Excitation process for the additive-input robot formulation."""

from __future__ import annotations
import numpy as np


class FilteredTorque:
    """Bounded torque process: tau_dot = -alpha * tau + xi(t)."""

    def __init__(self, limits, dt, *, alpha=4.0, noise_std=1.0, seed=0):
        self.limits = np.asarray(limits, dtype=float)
        self.dt = dt
        self.alpha = alpha
        self.noise_std = noise_std
        self.rng = np.random.default_rng(seed)
        self.tau = np.zeros_like(self.limits)

    def reset(self):
        self.tau.fill(0.0)

    def step(self):
        xi = self.rng.normal(scale=self.noise_std, size=self.tau.shape)
        self.tau += self.dt * (-self.alpha * self.tau + xi)
        self.tau = np.clip(self.tau, -self.limits, self.limits)
        return self.tau.copy(), xi
