import numpy as np
import pytest

pytest.importorskip("pinocchio")
pytest.importorskip("icub_models")

from esbn_icub.icub_genova11_config import (
    HARDWARE_POSITION_MAX,
    HARDWARE_POSITION_MIN,
    RIGHT_ARM_JOINTS,
)
from esbn_icub.icub_teacher import ICubTeacher


@pytest.fixture
def teacher():
    robot = ICubTeacher(dt=1e-3, gui=False)
    yield robot
    robot.close()


def test_full_model_is_loaded_then_reduced_to_seven_dof(teacher):
    assert teacher.full_model_nq > 7
    assert teacher.full_model_nv > 7
    assert teacher.model.nq == 7
    assert teacher.model.nv == 7
    assert tuple(teacher.model.names[1:]) == RIGHT_ARM_JOINTS


def test_hardware_position_limits_are_used(teacher):
    np.testing.assert_allclose(teacher.lower, HARDWARE_POSITION_MIN)
    np.testing.assert_allclose(teacher.upper, HARDWARE_POSITION_MAX)


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

    np.testing.assert_allclose(tau, M @ qddot + h, rtol=1e-7, atol=1e-9)


def test_acceleration_inverts_rigid_body_dynamics(teacher):
    q = 0.5 * (teacher.lower + teacher.upper)
    qdot = np.linspace(-0.1, 0.1, teacher.n_dof)
    desired_qddot = np.linspace(-0.5, 0.5, teacher.n_dof)
    tau = teacher.inverse_dynamics(q, qdot, desired_qddot)

    recovered = teacher.acceleration(q, qdot, tau)
    np.testing.assert_allclose(recovered, desired_qddot, rtol=1e-7, atol=1e-9)


def test_mass_matrix_changes_with_configuration(teacher):
    q_mid = 0.5 * (teacher.lower + teacher.upper)
    span = teacher.upper - teacher.lower
    q_a = q_mid - 0.15 * span
    q_b = q_mid + 0.15 * span

    M_a = teacher.mass_matrix(q_a)
    M_b = teacher.mass_matrix(q_b)
    assert np.linalg.norm(M_a - M_b) > 1e-6


def test_mass_matrix_random_configuration_sweep(teacher):
    rng = np.random.default_rng(0)
    for _ in range(25):
        q = rng.uniform(teacher.lower, teacher.upper)
        M = teacher.mass_matrix(q)
        np.testing.assert_allclose(M, M.T, atol=1e-10, rtol=1e-10)
        assert np.linalg.eigvalsh(M).min() > 0.0


def test_gravity_term_changes_with_configuration(teacher):
    q_mid = 0.5 * (teacher.lower + teacher.upper)
    span = teacher.upper - teacher.lower
    q_a = q_mid - 0.2 * span
    q_b = q_mid + 0.2 * span
    zeros = np.zeros(teacher.n_dof)

    g_a = teacher.inverse_dynamics(q_a, zeros, zeros)
    g_b = teacher.inverse_dynamics(q_b, zeros, zeros)
    assert np.linalg.norm(g_a - g_b) > 1e-6


def test_one_step_is_finite(teacher):
    q0 = 0.5 * (teacher.lower + teacher.upper)
    teacher.reset(q0, np.zeros(teacher.n_dof))
    q1, qdot1 = teacher.step(np.zeros(teacher.n_dof))
    assert np.all(np.isfinite(q1))
    assert np.all(np.isfinite(qdot1))
