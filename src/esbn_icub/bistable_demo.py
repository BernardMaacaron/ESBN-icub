"""Standalone nonlinear-system experiment for validating the Alemi EBN."""

from __future__ import annotations

import numpy as np

from .alemi_ebn import AlemiEBN
from .nonlinear_teacher import BistableTeacher


def run_bistable_demo(
    *,
    dt=1e-3,
    train_steps=5000,
    test_steps=2000,
    n_neurons=64,
    seed=0,
):
    """Train on a driven bistable system, then evaluate with feedback disabled.

    Returns compact numerical diagnostics rather than plotting so the same
    experiment can be exercised in CI and notebooks.
    """
    rng = np.random.default_rng(seed)
    teacher = BistableTeacher(dt=dt, x0=0.2)
    net = AlemiEBN(
        state_dim=1,
        n_neurons=n_neurons,
        dt=dt,
        lam=20.0,
        mu=1e-3,
        nu=1e-3,
        eta=0.5,
        feedback_gain=40.0,
        seed=seed,
    )

    train_error = []
    spike_count = 0.0
    for t in range(train_steps):
        # Smooth persistent excitation plus a weak stochastic component.
        time = t * dt
        command = np.array([
            0.8 * np.sin(2.0 * np.pi * 0.7 * time)
            + 0.25 * np.sin(2.0 * np.pi * 1.9 * time)
            + 0.05 * rng.normal()
        ])
        target = teacher.step(command)
        estimate = net.step(command, target, learn=True)
        spike_count += float(np.sum(net.spikes))
        train_error.append(float(target[0] - estimate[0]))

    net.feedback_gain = 0.0

    target_trace = np.empty(test_steps)
    estimate_trace = np.empty(test_steps)
    for t in range(test_steps):
        time = (train_steps + t) * dt
        command = np.array([
            0.7 * np.sin(2.0 * np.pi * 0.9 * time)
            + 0.20 * np.sin(2.0 * np.pi * 2.3 * time)
        ])
        target = teacher.step(command)
        estimate = net.step(command, target_state=None, learn=False)
        spike_count += float(np.sum(net.spikes))
        target_trace[t] = target[0]
        estimate_trace[t] = estimate[0]

    train_error = np.asarray(train_error)
    test_error = target_trace - estimate_trace

    return {
        "train_rmse_tail": float(np.sqrt(np.mean(train_error[-1000:] ** 2))),
        "test_rmse": float(np.sqrt(np.mean(test_error ** 2))),
        "target_trace": target_trace,
        "estimate_trace": estimate_trace,
        "spike_rate_hz": float(
            spike_count / (n_neurons * (train_steps + test_steps) * dt)
        ),
        "slow_weight_norm": float(np.linalg.norm(net.W_slow)),
    }



def run_bistable_trials(
    *,
    dt=1e-3,
    train_trials=50,
    train_trial_steps=1000,
    command_steps=500,
    test_steps=3000,
    n_neurons=50,
    seed=0,
):
    """Paper-shaped bistable experiment with repeated random-input trials.

    Alemi et al. use N=50 neurons and repeated learning iterations. Each
    training trial here begins near the unstable fixed point, is driven by a
    smooth random command, and keeps learning while the command is removed.
    The test trial uses an unseen command with feedback and learning disabled.
    """
    rng = np.random.default_rng(seed)
    teacher = BistableTeacher(dt=dt)
    net = AlemiEBN(
        state_dim=1,
        n_neurons=n_neurons,
        dt=dt,
        lam=20.0,
        mu=1e-3,
        nu=1e-3,
        eta=0.1,
        feedback_gain=40.0,
        basis_mode="decoded",
        seed=seed,
    )

    train_error = []
    train_spikes = 0.0
    beta = 8.0
    sigma = 2.0

    for trial in range(train_trials):
        teacher.reset(rng.uniform(-0.05, 0.05))
        net.reset()
        command_value = 0.0

        for t in range(train_trial_steps):
            if t < command_steps:
                command_value += (
                    -beta * command_value * dt
                    + sigma * np.sqrt(dt) * rng.normal()
                )
            else:
                command_value = 0.0

            command = np.array([command_value])
            target = teacher.step(command)
            estimate = net.step(command, target, learn=True)

            train_spikes += float(np.sum(net.spikes))
            train_error.append(float(target[0] - estimate[0]))

    # Unseen trial: no error feedback and no plasticity.
    teacher.reset(rng.uniform(-0.05, 0.05))
    net.reset()
    net.feedback_gain = 0.0
    command_value = 0.0

    target_trace = np.empty(test_steps)
    estimate_trace = np.empty(test_steps)
    command_trace = np.empty(test_steps)
    test_spikes = 0.0

    for t in range(test_steps):
        if t < command_steps:
            command_value += (
                -beta * command_value * dt
                + sigma * np.sqrt(dt) * rng.normal()
            )
        else:
            command_value = 0.0

        command = np.array([command_value])
        target = teacher.step(command)
        estimate = net.step(command, target_state=None, learn=False)

        target_trace[t] = target[0]
        estimate_trace[t] = estimate[0]
        command_trace[t] = command_value
        test_spikes += float(np.sum(net.spikes))

    train_error = np.asarray(train_error)
    test_error = target_trace - estimate_trace

    return {
        "train_rmse_tail": float(
            np.sqrt(np.mean(train_error[-train_trial_steps:] ** 2))
        ),
        "test_rmse": float(np.sqrt(np.mean(test_error ** 2))),
        "final_target": float(target_trace[-1]),
        "final_estimate": float(estimate_trace[-1]),
        "final_attractor_error": float(
            abs(abs(estimate_trace[-1]) - 0.5)
        ),
        "train_rate_hz": float(
            train_spikes
            / (n_neurons * train_trials * train_trial_steps * dt)
        ),
        "test_rate_hz": float(
            test_spikes / (n_neurons * test_steps * dt)
        ),
        "slow_weight_norm": float(np.linalg.norm(net.W_slow)),
        "target_trace": target_trace,
        "estimate_trace": estimate_trace,
        "command_trace": command_trace,
    }
