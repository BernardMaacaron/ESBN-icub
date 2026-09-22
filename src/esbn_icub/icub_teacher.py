"""PyBullet teacher for the official iCubGenova11 rigid-body model."""

from __future__ import annotations

from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET

import numpy as np

try:
    import pybullet as p
except ImportError:
    p = None

try:
    import icub_models
except ImportError:
    icub_models = None


RIGHT_ARM_JOINTS = (
    "r_shoulder_pitch",
    "r_shoulder_roll",
    "r_shoulder_yaw",
    "r_elbow",
    "r_wrist_prosup",
    "r_wrist_pitch",
    "r_wrist_yaw",
)


def _resolved_genova11_urdf():
    """Resolve package paths and zero-mass fixed auxiliary links for Bullet.

    The official model contains many massless sensor/skin/frame links with no
    inertial element. URDF semantics permit these fixed auxiliary links, but
    PyBullet otherwise assigns them a default mass/inertia, which would corrupt
    the robot dynamics. We explicitly give only such fixed auxiliary links zero
    mass and reject any movable link that lacks inertial data.
    """
    if icub_models is None:
        raise ImportError("icub-models is required for the robot teacher")

    model_path = Path(icub_models.get_model_file("iCubGenova11"))
    models_path = Path(icub_models.get_models_path())

    tree = ET.parse(model_path)
    root = tree.getroot()

    joint_by_child = {}
    for joint in root.findall("joint"):
        child = joint.find("child")
        if child is not None:
            joint_by_child[child.attrib["link"]] = joint

    for link in root.findall("link"):
        if link.find("inertial") is not None:
            continue

        name = link.attrib["name"]
        parent_joint = joint_by_child.get(name)

        # The root is fixed by p.loadURDF(useFixedBase=True), so its inertial
        # data is irrelevant; all other inertial-less links must be fixed
        # auxiliaries or the model is unsafe for a dynamics experiment.
        if parent_joint is not None and parent_joint.attrib.get("type") != "fixed":
            raise RuntimeError(
                f"Movable link {name} has no inertial data in iCubGenova11"
            )

        inertial = ET.Element("inertial")
        ET.SubElement(inertial, "origin", xyz="0 0 0", rpy="0 0 0")
        ET.SubElement(inertial, "mass", value="0")
        ET.SubElement(
            inertial,
            "inertia",
            ixx="0",
            ixy="0",
            ixz="0",
            iyy="0",
            iyz="0",
            izz="0",
        )
        link.insert(0, inertial)

    for elem in root.iter():
        mesh = elem.find("mesh")
        if mesh is not None:
            filename = mesh.attrib.get("filename", "")
            if filename.startswith("package://iCub/"):
                mesh.attrib["filename"] = filename.replace(
                    "package://iCub/",
                    f"{models_path.parent.as_posix()}/",
                    1,
                )

    tmp = tempfile.NamedTemporaryFile(
        mode="wb", suffix=".urdf", prefix="icub_genova11_", delete=False
    )
    tree.write(tmp, encoding="utf-8", xml_declaration=True)
    tmp.close()
    return Path(tmp.name)


class ICubTeacher:
    """Fixed-base iCubGenova11 with direct torque control of the right arm."""

    def __init__(self, dt=1e-3, gui=False):
        if p is None:
            raise ImportError("pybullet is required for the robot teacher")

        self.dt = dt
        self.client = p.connect(p.GUI if gui else p.DIRECT)
        p.setTimeStep(dt, physicsClientId=self.client)
        p.setGravity(0.0, 0.0, -9.81, physicsClientId=self.client)

        self.urdf_path = _resolved_genova11_urdf()
        self.body = p.loadURDF(
            str(self.urdf_path),
            useFixedBase=True,
            flags=p.URDF_USE_INERTIA_FROM_FILE,
            physicsClientId=self.client,
        )

        self.joint_name_to_id = {}
        scalar_dofs = []
        for jid in range(p.getNumJoints(self.body, physicsClientId=self.client)):
            info = p.getJointInfo(self.body, jid, physicsClientId=self.client)
            name = info[1].decode()
            self.joint_name_to_id[name] = jid
            if info[3] >= 0:
                if info[2] not in (p.JOINT_REVOLUTE, p.JOINT_PRISMATIC):
                    raise RuntimeError(
                        f"Unsupported non-scalar joint {name} with Bullet type {info[2]}"
                    )
                scalar_dofs.append((info[3], jid))

        # Bullet's generalized-coordinate vectors are ordered by qIndex, not
        # by raw URDF joint index.
        self.dof_joint_ids = [jid for _, jid in sorted(scalar_dofs)]

        missing = [name for name in RIGHT_ARM_JOINTS if name not in self.joint_name_to_id]
        if missing:
            raise RuntimeError(f"Missing expected Genova11 joints: {missing}")

        self.active_joint_ids = [self.joint_name_to_id[name] for name in RIGHT_ARM_JOINTS]
        self.active_dof_indices = [
            self.dof_joint_ids.index(jid) for jid in self.active_joint_ids
        ]
        self.inactive_joint_ids = [
            jid for jid in self.dof_joint_ids if jid not in self.active_joint_ids
        ]

        self.lower = np.array([
            p.getJointInfo(self.body, jid, physicsClientId=self.client)[8]
            for jid in self.active_joint_ids
        ])
        self.upper = np.array([
            p.getJointInfo(self.body, jid, physicsClientId=self.client)[9]
            for jid in self.active_joint_ids
        ])
        self.effort = np.array([
            p.getJointInfo(self.body, jid, physicsClientId=self.client)[10]
            for jid in self.active_joint_ids
        ])

        self._held_positions = {}
        self.reset(np.zeros(self.n_dof), np.zeros(self.n_dof))

    @property
    def n_dof(self):
        return len(self.active_joint_ids)

    def close(self):
        if p.isConnected(self.client):
            p.disconnect(self.client)
        try:
            self.urdf_path.unlink()
        except FileNotFoundError:
            pass

    def reset(self, q, qdot):
        q = np.asarray(q, dtype=float)
        qdot = np.asarray(qdot, dtype=float)
        if q.shape != (self.n_dof,) or qdot.shape != (self.n_dof,):
            raise ValueError(f"q and qdot must have shape {(self.n_dof,)}")

        for jid in self.dof_joint_ids:
            p.resetJointState(self.body, jid, 0.0, 0.0, physicsClientId=self.client)

        for jid, qi, vi in zip(self.active_joint_ids, q, qdot):
            p.resetJointState(
                self.body, jid, float(qi), float(vi), physicsClientId=self.client
            )

        # Every inactive movable joint was reset to zero above. Keeping the
        # target explicitly avoids querying joint-state APIs for non-scalar
        # joint types that may be present in future URDF revisions.
        self._held_positions = {jid: 0.0 for jid in self.inactive_joint_ids}

        p.setJointMotorControlArray(
            self.body,
            self.active_joint_ids,
            p.VELOCITY_CONTROL,
            forces=[0.0] * self.n_dof,
            physicsClientId=self.client,
        )
        self._hold_inactive_joints()

    def _hold_inactive_joints(self):
        if not self.inactive_joint_ids:
            return
        p.setJointMotorControlArray(
            self.body,
            self.inactive_joint_ids,
            p.POSITION_CONTROL,
            targetPositions=[self._held_positions[jid] for jid in self.inactive_joint_ids],
            forces=[
                max(1.0, float(p.getJointInfo(
                    self.body, jid, physicsClientId=self.client
                )[10]))
                for jid in self.inactive_joint_ids
            ],
            physicsClientId=self.client,
        )

    def state(self):
        # PyBullet's batched getJointStates is unreliable for this large URDF
        # on some builds; scalar queries are robust for the seven active DOFs.
        states = [
            p.getJointState(self.body, jid, physicsClientId=self.client)
            for jid in self.active_joint_ids
        ]
        q = np.array([state[0] for state in states])
        qdot = np.array([state[1] for state in states])
        return q, qdot

    def set_torque(self, tau):
        tau = np.asarray(tau, dtype=float)
        if tau.shape != (self.n_dof,):
            raise ValueError(f"tau must have shape {(self.n_dof,)}")
        p.setJointMotorControlArray(
            self.body,
            self.active_joint_ids,
            p.TORQUE_CONTROL,
            forces=tau.tolist(),
            physicsClientId=self.client,
        )

    def step(self, tau=None):
        if tau is not None:
            self.set_torque(tau)
        self._hold_inactive_joints()
        p.stepSimulation(physicsClientId=self.client)
        return self.state()

    def _full_state_vectors(self):
        # Inactive DOFs are held at zero throughout this experiment.
        # Constructing the vectors directly also avoids relying on Bullet
        # state-query behavior for auxiliary joints in the full iCub URDF.
        return np.zeros(len(self.dof_joint_ids)), np.zeros(len(self.dof_joint_ids))

    def mass_matrix(self, q=None):
        q_all, _ = self._full_state_vectors()
        if q is not None:
            q = np.asarray(q, dtype=float)
            if q.shape != (self.n_dof,):
                raise ValueError(f"q must have shape {(self.n_dof,)}")
            q_all[self.active_dof_indices] = q

        M = np.asarray(p.calculateMassMatrix(
            self.body, q_all.tolist(), physicsClientId=self.client
        ))
        idx = np.ix_(self.active_dof_indices, self.active_dof_indices)
        return M[idx]

    def inverse_dynamics(self, q, qdot, qddot):
        q = np.asarray(q, dtype=float)
        qdot = np.asarray(qdot, dtype=float)
        qddot = np.asarray(qddot, dtype=float)
        expected = (self.n_dof,)
        if q.shape != expected or qdot.shape != expected or qddot.shape != expected:
            raise ValueError(f"q, qdot and qddot must have shape {expected}")

        q_all, qdot_all = self._full_state_vectors()
        qddot_all = np.zeros_like(q_all)
        q_all[self.active_dof_indices] = q
        qdot_all[self.active_dof_indices] = qdot
        qddot_all[self.active_dof_indices] = qddot

        tau_all = np.asarray(p.calculateInverseDynamics(
            self.body,
            q_all.tolist(),
            qdot_all.tolist(),
            qddot_all.tolist(),
            physicsClientId=self.client,
        ))
        return tau_all[self.active_dof_indices]
