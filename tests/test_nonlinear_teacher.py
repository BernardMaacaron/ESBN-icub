import numpy as np

from esbn_icub.nonlinear_teacher import BistableTeacher


def test_bistable_vector_field_fixed_points():
    x = np.array([-1.0, 0.0, 1.0])
    np.testing.assert_allclose(BistableTeacher.vector_field(x), 0.0)


def test_bistable_teacher_euler_step():
    teacher = BistableTeacher(dt=1e-3, x0=0.5)
    x1 = teacher.step([0.2])
    expected = 0.5 + 1e-3 * (0.5 - 0.5**3 + 0.2)
    np.testing.assert_allclose(x1, [expected])
