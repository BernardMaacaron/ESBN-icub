"""Map the torque-driven iCub teacher into Alemi's additive-input form."""

from __future__ import annotations

import numpy as np

from .excitation import FilteredTorque
from .icub_teacher import ICubTeacher


class RobotExperiment:
    """Stream x=[q, qdot, tau] and c=[0, 0, xi] from the PyBullet teacher."""

    def __init__(
        self,
        teacher: ICubTeacher,
        *,
        alpha=4.0,
        noise_std=1.0,
        torque_fraction=0.2,
        seed=0,
    ):
        self.teacher = teacher
        self.torque_limits = torque_fraction * np.maximum(teacher.effort, 1e-6)
        self.excitation = FilteredTorque(
            self.torque_limits,
            teacher.dt,
            alpha=alpha,
            noise_std=noise_std,
            seed=seed,
        )

    @property
    def state_dim(self):
        return 3 * self.teacher.n_dof

    def reset(self, q=None, qdot=None, *, excitation_seed=None):
        if q is None:
            q = 0.5 * (self.teacher.lower + self.teacher.upper)
        if qdot is None:
            qdot = np.zeros(self.teacher.n_dof)
        self.teacher.reset(q, qdot)
        self.excitation.reset(seed=excitation_seed)
        return self.state()

    def state(self):
        q, qdot = self.teacher.state()
        return np.concatenate([q, qdot, self.excitation.tau])

    def step(self):
        tau, xi = self.excitation.step()
        q, qdot = self.teacher.step(tau)
        x = np.concatenate([q, qdot, tau])
        c = np.concatenate([
            np.zeros(self.teacher.n_dof),
            np.zeros(self.teacher.n_dof),
            xi,
        ])
        return x, c
