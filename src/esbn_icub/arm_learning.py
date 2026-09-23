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
    eta=0.05,
    feedback_gain=40.0,
    decoder_scale=None,
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
        feedback_gain=feedback_gain,
        eta=eta,
        decoder_scale=decoder_scale,
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



def train_episodes(
    experiment,
    normalizer,
    net,
    *,
    n_episodes,
    steps_per_episode,
    seed=0,
    position_margin=0.2,
    velocity_fraction=0.05,
    final_feedback_gain=10.0,
):
    """Train across random legal initial states with feedback annealing."""
    rng = np.random.default_rng(seed)
    teacher = experiment.teacher

    lower = teacher.lower
    upper = teacher.upper
    span = upper - lower
    q_low = lower + position_margin * span
    q_high = upper - position_margin * span
    qdot_scale = velocity_fraction * np.maximum(teacher.velocity, 1e-6)

    initial_feedback_gain = float(net.feedback_gain)
    episode_rmse = np.empty(n_episodes)

    for episode in range(n_episodes):
        if n_episodes == 1:
            fraction = 1.0
        else:
            fraction = episode / (n_episodes - 1)
        net.feedback_gain = (
            (1.0 - fraction) * initial_feedback_gain
            + fraction * final_feedback_gain
        )

        q0 = rng.uniform(q_low, q_high)
        qdot0 = rng.uniform(-qdot_scale, qdot_scale)
        experiment.reset(q0, qdot0)
        net.reset()

        error = train_steps(
            experiment,
            normalizer,
            net,
            steps_per_episode,
        )
        episode_rmse[episode] = np.sqrt(np.mean(error**2))

    net.feedback_gain = initial_feedback_gain
    return episode_rmse


def evaluate_unseen_episode(
    experiment,
    normalizer,
    net,
    *,
    n_steps,
    sync_steps=200,
    seed=1,
    position_margin=0.2,
):
    """Evaluate k=0 on an unseen initial state after a short state-sync period.

    The sync period uses teacher error feedback with learning disabled only to
    initialize the network's represented state. Metrics are computed strictly
    after feedback is removed.
    """
    rng = np.random.default_rng(seed)
    teacher = experiment.teacher

    span = teacher.upper - teacher.lower
    q0 = rng.uniform(
        teacher.lower + position_margin * span,
        teacher.upper - position_margin * span,
    )
    qdot0 = np.zeros(teacher.n_dof)

    experiment.reset(q0, qdot0)
    net.reset()

    feedback = net.feedback_gain
    sync_error = np.empty(sync_steps)
    try:
        for i in range(sync_steps):
            x, c = experiment.step()
            x_n = normalizer.encode_state(x)
            c_n = normalizer.encode_command(c)
            x_hat = net.step(c_n, x_n, learn=False)
            sync_error[i] = np.sqrt(np.mean((x_n - x_hat) ** 2))

        net.feedback_gain = 0.0

        targets = np.empty((n_steps, experiment.state_dim))
        estimates = np.empty_like(targets)

        for i in range(n_steps):
            x, c = experiment.step()
            x_n = normalizer.encode_state(x)
            c_n = normalizer.encode_command(c)
            x_hat = net.step(c_n, target_state=None, learn=False)
            targets[i] = x_n
            estimates[i] = x_hat
    finally:
        net.feedback_gain = feedback

    error = targets - estimates
    n = teacher.n_dof
    return {
        "sync_rmse": float(
            np.mean(sync_error[-min(50, sync_steps):])
        ),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "q_rmse": float(np.sqrt(np.mean(error[:, :n]**2))),
        "qdot_rmse": float(np.sqrt(np.mean(error[:, n:2*n]**2))),
        "tau_rmse": float(np.sqrt(np.mean(error[:, 2*n:]**2))),
        "targets": targets,
        "estimates": estimates,
    }
