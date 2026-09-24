import numpy as np
import pytest

pytest.importorskip("pybullet")
pytest.importorskip("icub_models")

from esbn_icub.icub_teacher import ICubTeacher
from esbn_icub.robot_experiment import RobotExperiment


def test_augmented_state_and_command_shapes():
    teacher = ICubTeacher(dt=1e-3, gui=False)
    try:
        experiment = RobotExperiment(
            teacher,
            noise_std=0.1,
            torque_fraction=0.01,
            torque_reference=np.ones(teacher.n_dof),
            seed=0,
        )
        x0 = experiment.reset()
        x1, c = experiment.step()

        assert experiment.state_dim == 21
        assert x0.shape == (21,)
        assert x1.shape == (21,)
        assert c.shape == (21,)
        np.testing.assert_array_equal(c[:14], np.zeros(14))
    finally:
        teacher.close()
