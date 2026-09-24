"""Fixed physical normalization for the augmented robot state."""

from __future__ import annotations

import numpy as np


class RobotStateNormalizer:
    """Normalize x=[q,qdot,tau] using fixed robot limits.

    Positions are centered at the middle of each legal joint range and scaled
    by half-range. Velocities and torques are scaled by fixed positive limits.
    The additive command c=[0,0,xi] is transformed with the same torque scale.
    """

    def __init__(self, lower, upper, velocity_scale, torque_scale):
        self.lower = np.asarray(lower, dtype=float)
        self.upper = np.asarray(upper, dtype=float)
        self.velocity_scale = np.asarray(velocity_scale, dtype=float)
        self.torque_scale = np.asarray(torque_scale, dtype=float)

        self.q_center = 0.5 * (self.lower + self.upper)
        self.q_scale = 0.5 * (self.upper - self.lower)

        if np.any(self.q_scale <= 0):
            raise ValueError("all joint position ranges must be positive")
        if np.any(self.velocity_scale <= 0):
            raise ValueError("all velocity scales must be positive")
        if np.any(self.torque_scale <= 0):
            raise ValueError("all torque scales must be positive")

        n = len(self.lower)
        shapes = {
            self.upper.shape,
            self.velocity_scale.shape,
            self.torque_scale.shape,
        }
        if shapes != {(n,)} or self.lower.shape != (n,):
            raise ValueError("all normalization vectors must have the same shape")

        self.n_dof = n

    @property
    def state_dim(self):
        return 3 * self.n_dof

    def encode_state(self, x):
        x = np.asarray(x, dtype=float)
        if x.shape != (self.state_dim,):
            raise ValueError(f"x must have shape {(self.state_dim,)}")

        n = self.n_dof
        q = (x[:n] - self.q_center) / self.q_scale
        qdot = x[n:2*n] / self.velocity_scale
        tau = x[2*n:] / self.torque_scale
        return np.concatenate([q, qdot, tau])

    def decode_state(self, x_norm):
        x_norm = np.asarray(x_norm, dtype=float)
        if x_norm.shape != (self.state_dim,):
            raise ValueError(f"x_norm must have shape {(self.state_dim,)}")

        n = self.n_dof
        q = self.q_center + self.q_scale * x_norm[:n]
        qdot = self.velocity_scale * x_norm[n:2*n]
        tau = self.torque_scale * x_norm[2*n:]
        return np.concatenate([q, qdot, tau])

    def encode_command(self, c):
        c = np.asarray(c, dtype=float)
        if c.shape != (self.state_dim,):
            raise ValueError(f"c must have shape {(self.state_dim,)}")

        n = self.n_dof
        out = np.zeros_like(c)
        # Only the torque-driving process is an additive command.
        out[2*n:] = c[2*n:] / self.torque_scale
        return out
