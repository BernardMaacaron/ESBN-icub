import numpy as np
import pytest

pytest.importorskip("pybullet")
pytest.importorskip("icub_models")

from esbn_icub.icub_teacher import ICubTeacher, RIGHT_ARM_JOINTS


@pytest.fixture
def teacher():
    robot = ICubTeacher(dt=1e-3, gui=False)
    yield robot
    robot.close()


def test_expected_arm_joint_set(teacher):
    assert teacher.n_dof == 7
    assert tuple(RIGHT_ARM_JOINTS) == (
        "r_shoulder_pitch",
        "r_shoulder_roll",
        "r_shoulder_yaw",
        "r_elbow",
        "r_wrist_prosup",
        "r_wrist_pitch",
        "r_wrist_yaw",
    )


def test_mass_matrix_is_symmetric_positive_definite(teacher):
    q = 0.5 * (teacher.lower + teacher.upper)
    M = teacher.mass_matrix(q)
    np.testing.assert_allclose(M, M.T, atol=1e-10, rtol=1e-10)
    assert np.linalg.eigvalsh(M).min() > 0.0


def test_inverse_dynamics_mass_matrix_identity_at_zero_velocity(teacher):
    q = 0.5 * (teacher.lower + teacher.upper)
    qdot = np.zeros(teacher.n_dof)
    qddot = np.linspace(-0.5, 0.5, teacher.n_dof)

    h = teacher.inverse_dynamics(q, qdot, np.zeros(teacher.n_dof))
    tau = teacher.inverse_dynamics(q, qdot, qddot)
    M = teacher.mass_matrix(q)

    np.testing.assert_allclose(tau, M @ qddot + h, rtol=1e-5, atol=1e-7)


def test_mass_matrix_changes_with_configuration(teacher):
    q_mid = 0.5 * (teacher.lower + teacher.upper)
    span = teacher.upper - teacher.lower
    q_a = q_mid - 0.15 * span
    q_b = q_mid + 0.15 * span

    M_a = teacher.mass_matrix(q_a)
    M_b = teacher.mass_matrix(q_b)
    assert np.linalg.norm(M_a - M_b) > 1e-6
