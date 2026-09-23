"""Excitation process for the additive-input robot formulation."""

from __future__ import annotations

import numpy as np


class FilteredTorque:
    """Bounded, band-limited torque process.

    The augmented state contains torque and obeys

        tau_dot = -alpha * tau + xi(t)

    where xi(t) is itself a low-pass-filtered random signal. noise_std is
    dimensionless: values near 1 drive the equilibrium torque over most of the
    configured torque range; smaller values produce proportionally weaker
    excitation. The returned xi is the exact additive command supplied to the
    Alemi network.
    """

    def __init__(
        self,
        limits,
        dt,
        *,
        alpha=4.0,
        noise_std=1.0,
        driver_beta=20.0,
        resample_time=0.05,
        seed=0,
    ):
        self.limits = np.asarray(limits, dtype=float)
        self.dt = float(dt)
        self.alpha = float(alpha)
        self.noise_std = float(noise_std)
        self.driver_beta = float(driver_beta)
        self.resample_steps = max(1, int(round(resample_time / self.dt)))
        self.rng = np.random.default_rng(seed)

        self.tau = np.zeros_like(self.limits)
        self.xi = np.zeros_like(self.limits)
        self.target_xi = np.zeros_like(self.limits)
        self.step_index = 0

    def reset(self):
        self.tau.fill(0.0)
        self.xi.fill(0.0)
        self.target_xi.fill(0.0)
        self.step_index = 0

    def step(self):
        if self.step_index % self.resample_steps == 0:
            # If xi = alpha * tau_target, the stationary torque would be
            # tau_target. This makes noise_std interpretable relative to limits.
            self.target_xi = (
                self.alpha
                * self.limits
                * self.noise_std
                * self.rng.uniform(-1.0, 1.0, size=self.tau.shape)
            )

        self.xi += (
            self.dt
            * self.driver_beta
            * (self.target_xi - self.xi)
        )
        self.tau += self.dt * (-self.alpha * self.tau + self.xi)
        self.tau = np.clip(self.tau, -self.limits, self.limits)
        self.step_index += 1
        return self.tau.copy(), self.xi.copy()
