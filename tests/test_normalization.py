import numpy as np

from esbn_icub.normalization import RobotStateNormalizer


def test_robot_state_normalization_roundtrip():
    norm = RobotStateNormalizer(
        lower=np.array([-1.0, -2.0]),
        upper=np.array([3.0, 2.0]),
        velocity_scale=np.array([4.0, 5.0]),
        torque_scale=np.array([10.0, 20.0]),
    )

    x = np.array([1.0, 0.0, 2.0, -2.5, 5.0, -10.0])
    x_norm = norm.encode_state(x)

    np.testing.assert_allclose(x_norm, [0.0, 0.0, 0.5, -0.5, 0.5, -0.5])
    np.testing.assert_allclose(norm.decode_state(x_norm), x)


def test_command_uses_torque_scale_only():
    norm = RobotStateNormalizer(
        lower=np.array([-1.0]),
        upper=np.array([1.0]),
        velocity_scale=np.array([2.0]),
        torque_scale=np.array([5.0]),
    )

    c = np.array([0.0, 0.0, 10.0])
    np.testing.assert_allclose(norm.encode_command(c), [0.0, 0.0, 2.0])
