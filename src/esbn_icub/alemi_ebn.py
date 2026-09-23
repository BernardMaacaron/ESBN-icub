"""Minimal Alemi et al. (2018) efficient balanced spiking network."""

from __future__ import annotations
import numpy as np


class AlemiEBN:
    """EBN with fixed fast weights and locally learned slow weights."""

    def __init__(
        self,
        state_dim,
        n_neurons,
        dt,
        *,
        lam=20.0,
        mu=1e-3,
        nu=1e-3,
        eta=1e-3,
        feedback_gain=10.0,
        decoder_scale=None,
        basis_mode="decoded",
        seed=0,
    ):
        self.state_dim = state_dim
        self.n_neurons = n_neurons
        self.dt = dt
        self.lam = lam
        self.mu = mu
        self.nu = nu
        self.eta = eta
        self.feedback_gain = feedback_gain
        if basis_mode not in ("decoded", "random"):
            raise ValueError("basis_mode must be 'decoded' or 'random'")
        self.basis_mode = basis_mode

        rng = np.random.default_rng(seed)
        if decoder_scale is None:
            decoder_scale = 1.0 / np.sqrt(n_neurons)
        self.decoder_scale = float(decoder_scale)
        self.D = rng.normal(
            scale=self.decoder_scale,
            size=(state_dim, n_neurons),
        )

        self.W_fast = self.D.T @ self.D + mu * np.eye(n_neurons)
        if basis_mode == "decoded":
            # Low-rank dendritic basis: Psi_i(r)=tanh(m_i^T x_hat+theta_i),
            # x_hat=Dr. This is the control-theoretically direct construction
            # discussed by Alemi et al.; it is equivalent to M = M_state D.
            self.M_state = rng.normal(size=(n_neurons, state_dim))
            self.M = self.M_state @ self.D
        else:
            self.M_state = None
            self.M = rng.normal(
                scale=1.0 / np.sqrt(n_neurons),
                size=(n_neurons, n_neurons),
            )
        self.theta = rng.uniform(-1.0, 1.0, size=n_neurons)
        self.W_slow = np.zeros((n_neurons, n_neurons))

        self.threshold = 0.5 * (
            np.sum(self.D * self.D, axis=0) + mu + nu
        )
        self.u = np.zeros(n_neurons)
        self.r = np.zeros(n_neurons)
        self.spikes = np.zeros(n_neurons)

    @property
    def decoded_state(self):
        return self.D @ self.r

    def reset(self):
        self.u.fill(0.0)
        self.r.fill(0.0)
        self.spikes.fill(0.0)

    def basis(self):
        if self.M_state is not None:
            drive = self.M_state @ self.decoded_state
        else:
            drive = self.M @ self.r
        return np.tanh(drive + self.theta)

    def step(self, command, target_state=None, *, learn=True):
        command = np.asarray(command, dtype=float)
        if command.shape != (self.state_dim,):
            raise ValueError(f"command must have shape {(self.state_dim,)}")

        x_hat = self.decoded_state
        if target_state is None:
            error = np.zeros(self.state_dim)
        else:
            target_state = np.asarray(target_state, dtype=float)
            if target_state.shape != (self.state_dim,):
                raise ValueError(f"target_state must have shape {(self.state_dim,)}")
            error = target_state - x_hat

        psi = self.basis()
        projected_error = self.D.T @ error

        drive = (
            -self.lam * self.u
            + self.D.T @ command
            + self.W_slow @ psi
            + self.feedback_gain * projected_error
        )
        self.u += self.dt * drive

        # Greedy EBN spike selection: emit one spike, apply the fast
        # recurrent reset/inhibition immediately, then re-evaluate. This is
        # the discrete event analogue of the -W_fast s impulse term.
        self.spikes.fill(0.0)
        for _ in range(2 * self.n_neurons):
            excess = self.u - self.threshold
            i = int(np.argmax(excess))
            if excess[i] <= 0.0:
                break
            self.spikes[i] += 1.0
            self.u -= self.W_fast[:, i]

        self.r += self.dt * (-self.lam * self.r)
        self.r += self.spikes

        if learn and target_state is not None:
            self.W_slow += self.eta * self.dt * np.outer(projected_error, psi)

        return self.decoded_state.copy()
