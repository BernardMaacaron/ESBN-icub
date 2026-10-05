"""Rigid-body iCubGenova11 teacher using the official full URDF.

The complete physical URDF is loaded with Pinocchio, then all joints except the
seven right-arm/wrist joints are locked at a legal reference configuration.
Pinocchio composes the inertias of locked subtrees into the reduced model, so
we do not cut links out of the URDF or invent inertial parameters.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    import pinocchio as pin
except ImportError:  # pragma: no cover
    pin = None

try:
    import icub_models
except ImportError:  # pragma: no cover
    icub_models = None

from .icub_genova11_config import (
    HARDWARE_POSITION_MAX,
    HARDWARE_POSITION_MIN,
    RIGHT_ARM_JOINTS,
)


class ICubTeacher:
    """Fixed-base seven-DOF right arm derived from full iCubGenova11."""

    def __init__(self, dt=1e-3, gui=False, integration_substeps=4):
        if pin is None:
            raise ImportError("Pinocchio is required: pip install -e '.[robot]'")
        if icub_models is None:
            raise ImportError("icub-models is required: pip install -e '.[robot]'")
        if gui:
            raise ValueError("GUI mode is not supported by the Pinocchio teacher.")

        self.dt = float(dt)
        self.integration_substeps = int(integration_substeps)
        if self.integration_substeps < 1:
            raise ValueError("integration_substeps must be >= 1")
        self.integration_dt = self.dt / self.integration_substeps
        self.urdf_path = Path(icub_models.get_model_file("iCubGenova11"))

        full_model = pin.buildModelFromUrdf(str(self.urdf_path))
        self.full_model_nq = int(full_model.nq)
        self.full_model_nv = int(full_model.nv)

        active_ids = []
        for name in RIGHT_ARM_JOINTS:
            jid = int(full_model.getJointId(name))
            if jid == 0:
                raise RuntimeError(f"Missing expected Genova11 joint: {name}")
            active_ids.append(jid)

        # Legal reference configuration for every joint. Zero is used whenever
        # it is legal; otherwise the nearest position limit is used.
        q_ref = pin.neutral(full_model)
        lower = np.asarray(full_model.lowerPositionLimit, dtype=float)
        upper = np.asarray(full_model.upperPositionLimit, dtype=float)
        finite_lower = np.where(np.isfinite(lower), lower, -np.inf)
        finite_upper = np.where(np.isfinite(upper), upper, np.inf)
        q_ref = np.minimum(np.maximum(q_ref, finite_lower), finite_upper)

        locked_ids = [
            jid
            for jid in range(1, full_model.njoints)
            if jid not in active_ids
        ]

        self.model = pin.buildReducedModel(full_model, locked_ids, q_ref)
        self.data = self.model.createData()

        if self.model.nq != 7 or self.model.nv != 7:
            raise RuntimeError(
                f"Expected 7-DOF reduced arm, got nq={self.model.nq}, nv={self.model.nv}"
            )

        reduced_names = tuple(self.model.names[1:])
        if reduced_names != RIGHT_ARM_JOINTS:
            raise RuntimeError(
                "Unexpected reduced-model joint order: "
                f"{reduced_names}; expected {RIGHT_ARM_JOINTS}"
            )

        self.lower = HARDWARE_POSITION_MIN.copy()
        self.upper = HARDWARE_POSITION_MAX.copy()

        # Keep raw URDF values visible only for diagnostics. The official
        # physical URDF uses placeholder limits here and they are never used
        # for excitation or normalization.
        self.effort = np.asarray(self.model.effortLimit, dtype=float).copy()
        self.velocity = np.asarray(self.model.velocityLimit, dtype=float).copy()

        self.q = np.zeros(self.n_dof)
        self.qdot = np.zeros(self.n_dof)
        self._tau = np.zeros(self.n_dof)
        self.reset(0.5 * (self.lower + self.upper), np.zeros(self.n_dof))

    @property
    def n_dof(self):
        return int(self.model.nv)

    @property
    def active_joint_names(self):
        return RIGHT_ARM_JOINTS

    def close(self):
        """Compatibility no-op; Pinocchio owns no external simulator process."""
        return None

    def reset(self, q, qdot):
        q = np.asarray(q, dtype=float)
        qdot = np.asarray(qdot, dtype=float)
        expected = (self.n_dof,)
        if q.shape != expected or qdot.shape != expected:
            raise ValueError(f"q and qdot must have shape {expected}")
        if np.any(q < self.lower) or np.any(q > self.upper):
            raise ValueError("q must lie inside the Genova11 hardware joint limits")

        self.q = q.copy()
        self.qdot = qdot.copy()
        self._tau.fill(0.0)

    def state(self):
        return self.q.copy(), self.qdot.copy()

    def set_torque(self, tau):
        tau = np.asarray(tau, dtype=float)
        if tau.shape != (self.n_dof,):
            raise ValueError(f"tau must have shape {(self.n_dof,)}")
        self._tau = tau.copy()

    def mass_matrix(self, q=None):
        if q is None:
            q = self.q
        q = np.asarray(q, dtype=float)
        if q.shape != (self.n_dof,):
            raise ValueError(f"q must have shape {(self.n_dof,)}")

        M = np.asarray(pin.crba(self.model, self.data, q), dtype=float)
        # CRBA may expose only one triangle depending on bindings/version.
        return 0.5 * (M + M.T)

    def inverse_dynamics(self, q, qdot, qddot):
        q = np.asarray(q, dtype=float)
        qdot = np.asarray(qdot, dtype=float)
        qddot = np.asarray(qddot, dtype=float)
        expected = (self.n_dof,)
        if q.shape != expected or qdot.shape != expected or qddot.shape != expected:
            raise ValueError(f"q, qdot and qddot must have shape {expected}")

        return np.asarray(
            pin.rnea(self.model, self.data, q, qdot, qddot),
            dtype=float,
        ).copy()

    def acceleration(self, q, qdot, tau):
        """Forward rigid-body acceleration from Pinocchio ABA."""
        q = np.asarray(q, dtype=float)
        qdot = np.asarray(qdot, dtype=float)
        tau = np.asarray(tau, dtype=float)
        expected = (self.n_dof,)
        if q.shape != expected or qdot.shape != expected or tau.shape != expected:
            raise ValueError(f"q, qdot and tau must have shape {expected}")

        return np.asarray(
            pin.aba(self.model, self.data, q, qdot, tau),
            dtype=float,
        ).copy()

    def _rk4_substep(self, q, qdot, tau, h):
        """One fourth-order integration substep on the configuration manifold."""
        a1 = self.acceleration(q, qdot, tau)

        v2 = qdot + 0.5 * h * a1
        q2 = np.asarray(
            pin.integrate(self.model, q, 0.5 * h * qdot),
            dtype=float,
        )
        a2 = self.acceleration(q2, v2, tau)

        v3 = qdot + 0.5 * h * a2
        q3 = np.asarray(
            pin.integrate(self.model, q, 0.5 * h * v2),
            dtype=float,
        )
        a3 = self.acceleration(q3, v3, tau)

        v4 = qdot + h * a3
        q4 = np.asarray(
            pin.integrate(self.model, q, h * v3),
            dtype=float,
        )
        a4 = self.acceleration(q4, v4, tau)

        qdot_next = qdot + (h / 6.0) * (a1 + 2.0 * a2 + 2.0 * a3 + a4)
        tangent_step = (h / 6.0) * (qdot + 2.0 * v2 + 2.0 * v3 + v4)
        q_next = np.asarray(
            pin.integrate(self.model, q, tangent_step),
            dtype=float,
        )
        return q_next, qdot_next

    def step(self, tau=None):
        """Advance one external timestep using RK4 with internal substeps."""
        if tau is not None:
            self.set_torque(tau)

        q = self.q
        qdot = self.qdot
        for _ in range(self.integration_substeps):
            q, qdot = self._rk4_substep(
                q,
                qdot,
                self._tau,
                self.integration_dt,
            )
            if np.any(~np.isfinite(q)) or np.any(~np.isfinite(qdot)):
                raise FloatingPointError(
                    "Non-finite state in iCub rigid-body integration"
                )

        self.q = q
        self.qdot = qdot
        return self.state()

    def total_energy(self, q=None, qdot=None):
        """Mechanical energy for numerical diagnostics."""
        if q is None:
            q = self.q
        if qdot is None:
            qdot = self.qdot
        q = np.asarray(q, dtype=float)
        qdot = np.asarray(qdot, dtype=float)
        kinetic = float(pin.computeKineticEnergy(self.model, self.data, q, qdot))
        potential = float(pin.computePotentialEnergy(self.model, self.data, q))
        return kinetic + potential
