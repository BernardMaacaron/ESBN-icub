"""Small nonlinear teacher systems used to validate the Alemi EBN independently."""

from __future__ import annotations

import numpy as np


class BistableTeacher:
    """One-dimensional driven bistable system: x_dot = x - x^3 + c(t)."""

    def __init__(self, dt: float, x0: float = 0.0):
        self.dt = float(dt)
        self.x = float(x0)

    @staticmethod
    def vector_field(x):
        x = np.asarray(x, dtype=float)
        return x - x**3

    def reset(self, x0: float = 0.0):
        self.x = float(x0)
        return np.array([self.x])

    def step(self, command):
        c = float(np.asarray(command).reshape(-1)[0])
        self.x += self.dt * (self.x - self.x**3 + c)
        return np.array([self.x])
