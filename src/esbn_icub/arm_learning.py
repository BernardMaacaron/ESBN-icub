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
        experiment.reset(q0, qdot0, excitation_seed=seed + episode)
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

    experiment.reset(q0, qdot0, excitation_seed=seed + 10000)
    net.reset()

    feedback = net.feedback_gain
    sync_error = np.empty(sync_steps)
    sync_component_error = np.empty((sync_steps, experiment.state_dim))
    try:
        for i in range(sync_steps):
            x, c = experiment.step()
            x_n = normalizer.encode_state(x)
            c_n = normalizer.encode_command(c)
            x_hat = net.step(c_n, x_n, learn=False)
            delta = x_n - x_hat
            sync_component_error[i] = delta
            sync_error[i] = np.sqrt(np.mean(delta ** 2))

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
    step_rmse = np.sqrt(np.mean(error**2, axis=1))
    n = teacher.n_dof
    sync_tail = sync_component_error[-min(50, sync_steps):]
    horizons = {}
    for horizon in (50, 100, 250, 500, 1000):
        if horizon <= n_steps:
            prefix = error[:horizon]
            horizons[horizon] = {
                "rmse": float(np.sqrt(np.mean(prefix**2))),
                "q_rmse": float(np.sqrt(np.mean(prefix[:, :n]**2))),
                "qdot_rmse": float(np.sqrt(np.mean(prefix[:, n:2*n]**2))),
                "tau_rmse": float(np.sqrt(np.mean(prefix[:, 2*n:]**2))),
            }

    return {
        "sync_rmse": float(np.sqrt(np.mean(sync_tail**2))),
        "sync_q_rmse": float(np.sqrt(np.mean(sync_tail[:, :n]**2))),
        "sync_qdot_rmse": float(np.sqrt(np.mean(sync_tail[:, n:2*n]**2))),
        "sync_tau_rmse": float(np.sqrt(np.mean(sync_tail[:, 2*n:]**2))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "early_rmse": float(np.sqrt(np.mean(error[:min(100, n_steps)]**2))),
        "late_rmse": float(np.sqrt(np.mean(error[-min(100, n_steps):]**2))),
        "horizon_rmse": horizons,
        "step_rmse": step_rmse,
        "q_rmse": float(np.sqrt(np.mean(error[:, :n]**2))),
        "qdot_rmse": float(np.sqrt(np.mean(error[:, n:2*n]**2))),
        "tau_rmse": float(np.sqrt(np.mean(error[:, 2*n:]**2))),
        "targets": targets,
        "estimates": estimates,
    }



def evaluate_untrained_baseline(
    *,
    dt=1e-3,
    n_neurons=128,
    seed=0,
    torque_fraction=0.10,
    noise_std=0.5,
    n_steps=1000,
    sync_steps=300,
):
    """Evaluate a fresh, untrained network using the same unseen-state protocol."""
    teacher, experiment, normalizer, net = build_arm_experiment(
        dt=dt,
        n_neurons=n_neurons,
        seed=seed,
        torque_fraction=torque_fraction,
        noise_std=noise_std,
    )
    try:
        return evaluate_unseen_episode(
            experiment,
            normalizer,
            net,
            n_steps=n_steps,
            sync_steps=sync_steps,
            seed=seed + 1000,
        )
    finally:
        teacher.close()


def sweep_arm_hyperparameters(
    *,
    etas=(0.1, 0.5, 1.0, 2.0),
    decoder_scales=(None, 0.05),
    feedback_gains=(20.0, 40.0),
    n_neurons=128,
    n_episodes=8,
    steps_per_episode=1000,
    eval_steps=1000,
    sync_steps=300,
    torque_fraction=0.10,
    noise_std=0.5,
    seed=0,
):
    """Run a deterministic compact sweep and return comparable metrics.

    Every candidate receives the same teacher/excitation seed and evaluation
    seed so differences reflect the network hyperparameters rather than a
    different input realization.
    """
    rows = []

    for eta in etas:
        for decoder_scale in decoder_scales:
            for feedback_gain in feedback_gains:
                teacher, experiment, normalizer, net = build_arm_experiment(
                    dt=1e-3,
                    n_neurons=n_neurons,
                    seed=seed,
                    torque_fraction=torque_fraction,
                    noise_std=noise_std,
                    eta=eta,
                    feedback_gain=feedback_gain,
                    decoder_scale=decoder_scale,
                )
                try:
                    episodes = train_episodes(
                        experiment,
                        normalizer,
                        net,
                        n_episodes=n_episodes,
                        steps_per_episode=steps_per_episode,
                        seed=seed,
                        final_feedback_gain=max(5.0, feedback_gain / 4.0),
                    )
                    result = evaluate_unseen_episode(
                        experiment,
                        normalizer,
                        net,
                        n_steps=eval_steps,
                        sync_steps=sync_steps,
                        seed=seed + 123,
                    )
                    rows.append({
                        "eta": float(eta),
                        "decoder_scale": (
                            None if decoder_scale is None else float(decoder_scale)
                        ),
                        "feedback_gain": float(feedback_gain),
                        "first_episode_rmse": float(episodes[0]),
                        "last_episode_rmse": float(episodes[-1]),
                        "sync_rmse": result["sync_rmse"],
                        "autonomous_rmse": result["rmse"],
                        "q_rmse": result["q_rmse"],
                        "qdot_rmse": result["qdot_rmse"],
                        "tau_rmse": result["tau_rmse"],
                        "slow_weight_norm": float(np.linalg.norm(net.W_slow)),
                    })
                finally:
                    teacher.close()

    return rows
