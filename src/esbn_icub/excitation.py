"""Excitation process for the additive-input robot formulation."""

from __future__ import annotations

import numpy as np


class FilteredTorque:
    """Bounded, band-limited torque process.

    The augmented state contains the *total* applied torque and obeys

        tau_dot = -alpha * tau + c_tau(t)

    with

        c_tau(t) = alpha * bias + xi(t).

    bias is fixed within an episode (typically the gravity-compensation torque
    at the initial pose), while xi(t) is a low-pass-filtered random signal.
    This keeps the plant near the sampled pose without changing Alemi's
    additive-input formulation.
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
        if not (0.0 <= self.noise_std <= 1.0):
            raise ValueError(
                "noise_std must lie in [0, 1] so the exact augmented torque "
                "dynamics remain inside the configured bounds without clipping"
            )
        self.driver_beta = float(driver_beta)
        self.resample_steps = max(1, int(round(resample_time / self.dt)))
        self.rng = np.random.default_rng(seed)

        self.bias = np.zeros_like(self.limits)
        self.tau = np.zeros_like(self.limits)
        self.xi = np.zeros_like(self.limits)
        self.target_xi = np.zeros_like(self.limits)
        self.step_index = 0

    def reset(self, *, seed=None, bias=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        if bias is None:
            self.bias.fill(0.0)
        else:
            bias = np.asarray(bias, dtype=float)
            if bias.shape != self.limits.shape:
                raise ValueError(f"bias must have shape {self.limits.shape}")
            self.bias = bias.copy()
        self.tau = self.bias.copy()
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
        additive_command = self.alpha * self.bias + self.xi
        self.tau += self.dt * (-self.alpha * self.tau + additive_command)
        self.step_index += 1
        return self.tau.copy(), additive_command.copy()
