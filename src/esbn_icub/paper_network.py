"""Efficient balanced spiking network from Alemi et al. (2018).

This file contains only the network used in the paper experiment:
    x_hat = D r
    du/dt = -lambda u + D^T c - W_fast s + W_slow Psi(r) + k D^T e
    W_fast = D^T D + mu I
    Psi(r) = tanh(M r + theta)
    dW_slow/dt = eta (D^T e) Psi(r)^T

D, M and theta are fixed. Only W_slow is learned.
"""

from __future__ import annotations

import numpy as np


class PaperEBN:
    """Efficient balanced network with the paper's local slow-weight rule."""

    def __init__(
        self,
        state_dim: int,
        n_neurons: int,
        dt: float,
        *,
        lam: float = 20.0,
        mu: float = 1e-3,
        nu: float = 1e-3,
        eta: float = 0.05,
        feedback_gain: float = 40.0,
        seed: int = 0,
    ):
        self.state_dim = int(state_dim)
        self.n_neurons = int(n_neurons)
        self.dt = float(dt)
        self.lam = float(lam)
        self.mu = float(mu)
        self.nu = float(nu)
        self.eta = float(eta)
        self.feedback_gain = float(feedback_gain)

        rng = np.random.default_rng(seed)

        # Natural random decoder scale: O(1/sqrt(N)).
        self.D = rng.normal(
            scale=1.0 / np.sqrt(self.n_neurons),
            size=(self.state_dim, self.n_neurons),
        )

        self.W_fast = self.D.T @ self.D + self.mu * np.eye(self.n_neurons)

        # Random nonlinear dendritic basis used in the general EBN formulation.
        self.M = rng.normal(
            scale=1.0 / np.sqrt(self.n_neurons),
            size=(self.n_neurons, self.n_neurons),
        )
        self.theta = rng.uniform(-1.0, 1.0, size=self.n_neurons)

        self.W_slow = np.zeros((self.n_neurons, self.n_neurons))

        self.threshold = 0.5 * (
            np.sum(self.D * self.D, axis=0) + self.mu + self.nu
        )

        self.u = np.zeros(self.n_neurons)
        self.r = np.zeros(self.n_neurons)
        self.spikes = np.zeros(self.n_neurons)

    @property
    def decoded_state(self) -> np.ndarray:
        return self.D @ self.r

    def reset_state(self) -> None:
        """Reset neural state while preserving learned slow weights."""
        self.u.fill(0.0)
        self.r.fill(0.0)
        self.spikes.fill(0.0)

    def step(
        self,
        command: np.ndarray,
        teacher_state: np.ndarray | None = None,
        *,
        learn: bool = True,
    ) -> np.ndarray:
        command = np.asarray(command, dtype=float)
        if command.shape != (self.state_dim,):
            raise ValueError(f"command must have shape {(self.state_dim,)}")

        estimate = self.decoded_state

        if teacher_state is None:
            error = np.zeros(self.state_dim)
        else:
            teacher_state = np.asarray(teacher_state, dtype=float)
            if teacher_state.shape != (self.state_dim,):
                raise ValueError(
                    f"teacher_state must have shape {(self.state_dim,)}"
                )
            error = teacher_state - estimate

        psi = np.tanh(self.M @ self.r + self.theta)
        projected_error = self.D.T @ error

        self.u += self.dt * (
            -self.lam * self.u
            + self.D.T @ command
            + self.W_slow @ psi
            + self.feedback_gain * projected_error
        )

        # Fast recurrent spike/reset term. Resolve all threshold crossings
        # within the current discrete timestep.
        self.spikes.fill(0.0)
        for _ in range(4 * self.n_neurons):
            excess = self.u - self.threshold
            neuron = int(np.argmax(excess))
            if excess[neuron] <= 0.0:
                break
            self.spikes[neuron] += 1.0
            self.u -= self.W_fast[:, neuron]
        else:
            if np.any(self.u > self.threshold):
                raise RuntimeError(
                    "Spike resolution did not converge; reduce dt."
                )

        self.r += self.dt * (-self.lam * self.r)
        self.r += self.spikes

        if learn and teacher_state is not None:
            self.W_slow += (
                self.eta
                * self.dt
                * np.outer(projected_error, psi)
            )

        return self.decoded_state.copy()
