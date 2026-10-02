"""Small nonlinear teacher systems used to validate the Alemi EBN independently."""

from __future__ import annotations

import numpy as np


class BistableTeacher:
    """One-dimensional driven bistable system from Alemi et al. Eq. 13."""

    def __init__(self, dt: float, x0: float = 0.0):
        self.dt = float(dt)
        self.x = float(x0)

    @staticmethod
    def vector_field(x):
        x = np.asarray(x, dtype=float)
        return x * (0.5 - x) * (0.5 + x)

    def reset(self, x0: float = 0.0):
        self.x = float(x0)
        return np.array([self.x])

    def state(self):
        return np.array([self.x])

    def step(self, command):
        c = float(np.asarray(command).reshape(-1)[0])
        self.x += self.dt * (self.x * (0.5 - self.x) * (0.5 + self.x) + c)
        return np.array([self.x])
