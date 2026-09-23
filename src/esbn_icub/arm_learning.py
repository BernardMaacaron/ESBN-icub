"""Training/evaluation harness for the seven-DOF iCub arm."""

from __future__ import annotations

import numpy as np

from .alemi_ebn import AlemiEBN
from .icub_teacher import ICubTeacher
from .normalization import RobotStateNormalizer
from .robot_experiment import RobotExperiment


def build_arm_experiment(
    *,
    dt=1e-3,
    n_neurons=256,
    seed=0,
    torque_fraction=0.1,
    noise_std=1.0,
):
    """Construct teacher, normalized experiment stream, and Alemi network."""
    teacher = ICubTeacher(dt=dt, gui=False)
    experiment = RobotExperiment(
        teacher,
        noise_std=noise_std,
        torque_fraction=torque_fraction,
        seed=seed,
    )

    velocity_scale = np.maximum(teacher.velocity, 1e-6)
    normalizer = RobotStateNormalizer(
        teacher.lower,
        teacher.upper,
        velocity_scale,
        experiment.torque_limits,
    )

    net = AlemiEBN(
        state_dim=experiment.state_dim,
        n_neurons=n_neurons,
        dt=dt,
        feedback_gain=40.0,
        eta=0.05,
        seed=seed,
    )
    return teacher, experiment, normalizer, net


def train_steps(experiment, normalizer, net, n_steps):
    """Run online teacher-forced Alemi learning for n_steps."""
    errors = np.empty(n_steps)
    for i in range(n_steps):
        x, c = experiment.step()
        x_n = normalizer.encode_state(x)
        c_n = normalizer.encode_command(c)
        x_hat = net.step(c_n, x_n, learn=True)
        errors[i] = np.sqrt(np.mean((x_n - x_hat) ** 2))
    return errors


def autonomous_steps(experiment, normalizer, net, n_steps):
    """Evaluate with teacher feedback and learning disabled."""
    feedback = net.feedback_gain
    net.feedback_gain = 0.0

    errors = np.empty(n_steps)
    targets = np.empty((n_steps, experiment.state_dim))
    estimates = np.empty_like(targets)

    try:
        for i in range(n_steps):
            x, c = experiment.step()
            x_n = normalizer.encode_state(x)
            c_n = normalizer.encode_command(c)
            x_hat = net.step(c_n, target_state=None, learn=False)

            targets[i] = x_n
            estimates[i] = x_hat
            errors[i] = np.sqrt(np.mean((x_n - x_hat) ** 2))
    finally:
        net.feedback_gain = feedback

    return {
        "rmse": float(np.sqrt(np.mean((targets - estimates) ** 2))),
        "step_rmse": errors,
        "targets": targets,
        "estimates": estimates,
    }
