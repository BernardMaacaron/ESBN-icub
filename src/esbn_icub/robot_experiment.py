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
        torque_reference=None,
        seed=0,
    ):
        self.teacher = teacher
        if torque_reference is None:
            if np.any(np.asarray(teacher.effort) >= 1e3):
                raise ValueError(
                    "The physical iCub URDF uses placeholder effort limits; "
                    "pass an explicit model-based torque_reference."
                )
            torque_reference = np.maximum(teacher.effort, 1e-6)
        torque_reference = np.asarray(torque_reference, dtype=float)
        if torque_reference.shape != (teacher.n_dof,):
            raise ValueError(
                f"torque_reference must have shape {(teacher.n_dof,)}"
            )
        if np.any(torque_reference <= 0.0):
            raise ValueError("torque_reference must be strictly positive")
        self.torque_reference = torque_reference
        self.torque_limits = float(torque_fraction) * torque_reference
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

    def transition(self):
        """Advance one interval and return (x_t, c_t, x_{t+1}).

        This keeps the teacher and EBN on the same discrete-time interval:
        feedback is computed from x_t, while the returned network state is
        compared against x_{t+1}.
        """
        x_before = self.state()
        tau, xi = self.excitation.step()
        q, qdot = self.teacher.step(tau)
        x_after = np.concatenate([q, qdot, tau])
        command = np.concatenate([
            np.zeros(self.teacher.n_dof),
            np.zeros(self.teacher.n_dof),
            xi,
        ])
        return x_before, command, x_after

    def step(self):
        """Compatibility wrapper returning the post-step state and command."""
        _, command, x_after = self.transition()
        return x_after, command
