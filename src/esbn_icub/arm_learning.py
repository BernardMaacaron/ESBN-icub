"""Training/evaluation harness for the seven-DOF iCub arm."""

from __future__ import annotations

import numpy as np

from .alemi_ebn import AlemiEBN
from .icub_teacher import ICubTeacher
from .normalization import RobotStateNormalizer
from .robot_experiment import RobotExperiment



def estimate_characteristic_torque_scale(
    teacher,
    *,
    seed=0,
    n_configurations=16,
    acceleration_scale=1.0,
):
    """Estimate a dynamics-based torque scale, not an actuator limit.

    The physical iCub URDF ships placeholder effort/velocity limits (50000),
    so those fields must not be used for normalization or excitation. This
    scale is derived from the actual rigid-body model by sampling gravity and
    unit-acceleration inverse dynamics over legal configurations.
    """
    rng = np.random.default_rng(seed)
    span = teacher.upper - teacher.lower
    q_low = teacher.lower + 0.2 * span
    q_high = teacher.upper - 0.2 * span
    zeros = np.zeros(teacher.n_dof)
    scale = np.zeros(teacher.n_dof)

    configurations = [0.5 * (teacher.lower + teacher.upper)]
    configurations.extend(
        rng.uniform(q_low, q_high) for _ in range(n_configurations)
    )

    for q in configurations:
        scale = np.maximum(
            scale,
            np.abs(teacher.inverse_dynamics(q, zeros, zeros)),
        )
        for j in range(teacher.n_dof):
            qddot = np.zeros(teacher.n_dof)
            qddot[j] = acceleration_scale
            scale = np.maximum(
                scale,
                np.abs(teacher.inverse_dynamics(q, zeros, qddot)),
            )
            qddot[j] = -acceleration_scale
            scale = np.maximum(
                scale,
                np.abs(teacher.inverse_dynamics(q, zeros, qddot)),
            )

    return np.maximum(scale, 0.1)


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
    basis_mode="random",
):
    """Construct teacher, normalized experiment stream, and Alemi network."""
    teacher = ICubTeacher(dt=dt, gui=False)
    torque_reference = estimate_characteristic_torque_scale(
        teacher,
        seed=seed,
    )
    experiment = RobotExperiment(
        teacher,
        noise_std=noise_std,
        torque_fraction=torque_fraction,
        torque_reference=torque_reference,
        seed=seed,
    )

    # Numerical normalization scale: roughly the velocity required to traverse
    # half of each legal joint range in one second. The URDF velocity fields
    # are placeholders and are deliberately ignored.
    velocity_scale = np.maximum(
        0.5 * (teacher.upper - teacher.lower),
        0.5,
    )
    normalizer = RobotStateNormalizer(
        teacher.lower,
        teacher.upper,
        velocity_scale,
        experiment.torque_reference,
    )

    net = AlemiEBN(
        state_dim=experiment.state_dim,
        n_neurons=n_neurons,
        dt=dt,
        feedback_gain=feedback_gain,
        eta=eta,
        decoder_scale=decoder_scale,
        basis_mode=basis_mode,
        seed=seed,
    )
    return teacher, experiment, normalizer, net


def _inside_position_guard(x, teacher, margin=0.02):
    q = np.asarray(x[:teacher.n_dof], dtype=float)
    span = teacher.upper - teacher.lower
    lower = teacher.lower + margin * span
    upper = teacher.upper - margin * span
    return bool(np.all(q > lower) and np.all(q < upper))


def train_steps(
    experiment,
    normalizer,
    net,
    n_steps,
    *,
    position_margin=0.02,
):
    """Run teacher-forced learning, stopping before hardware limits are crossed."""
    errors = []
    teacher = experiment.teacher
    span = teacher.upper - teacher.lower
    lower_guard = teacher.lower + position_margin * span
    upper_guard = teacher.upper - position_margin * span

    for _ in range(n_steps):
        x_t, c_t, x_next = experiment.transition()
        q_next = x_next[:teacher.n_dof]
        if np.any(q_next <= lower_guard) or np.any(q_next >= upper_guard):
            break

        x_t_n = normalizer.encode_state(x_t)
        c_t_n = normalizer.encode_command(c_t)
        x_next_n = normalizer.encode_state(x_next)
        x_hat_next = net.step(c_t_n, x_t_n, learn=True)
        errors.append(np.sqrt(np.mean((x_next_n - x_hat_next) ** 2)))

    return np.asarray(errors, dtype=float)


def autonomous_steps(
    experiment,
    normalizer,
    net,
    n_steps,
    *,
    position_margin=0.02,
):
    """Evaluate with feedback and learning disabled inside the legal state region."""
    feedback = net.feedback_gain
    net.feedback_gain = 0.0

    targets = []
    estimates = []

    try:
        for _ in range(n_steps):
            _, c_t, x_next = experiment.transition()
            if not _inside_position_guard(
                x_next,
                experiment.teacher,
                margin=position_margin,
            ):
                break

            c_t_n = normalizer.encode_command(c_t)
            x_next_n = normalizer.encode_state(x_next)
            x_hat_next = net.step(c_t_n, target_state=None, learn=False)

            targets.append(x_next_n)
            estimates.append(x_hat_next)
    finally:
        net.feedback_gain = feedback

    if not targets:
        raise RuntimeError("Autonomous rollout left the legal state region immediately")

    targets = np.asarray(targets)
    estimates = np.asarray(estimates)
    error = targets - estimates
    errors = np.sqrt(np.mean(error**2, axis=1))

    horizons = {}
    for horizon in (1, 5, 10, 25, 50, 100, 250, 500, 1000):
        if horizon <= len(error):
            prefix = error[:horizon]
            horizons[horizon] = float(np.sqrt(np.mean(prefix**2)))

    return {
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "horizon_rmse": horizons,
        "step_rmse": errors,
        "targets": targets,
        "estimates": estimates,
        "executed_steps": int(len(error)),
        "terminated_at_position_guard": bool(len(error) < n_steps),
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
    sync_steps=0,
):
    """Train across random legal initial states with feedback annealing.

    Each episode begins with a feedback-only synchronization period. Learning
    during the large reset transient would otherwise teach the slow weights to
    compensate for an artificial state-initialization error rather than the
    robot vector field.
    """
    rng = np.random.default_rng(seed)
    teacher = experiment.teacher

    lower = teacher.lower
    upper = teacher.upper
    span = upper - lower
    q_low = lower + position_margin * span
    q_high = upper - position_margin * span
    qdot_scale = velocity_fraction * normalizer.velocity_scale

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

        # Synchronize the represented state before adapting W_slow.
        for _ in range(sync_steps):
            x_t, c_t, x_next = experiment.transition()
            if not _inside_position_guard(x_next, teacher, margin=0.02):
                raise RuntimeError(
                    "Training synchronization left the legal state region"
                )
            x_t_n = normalizer.encode_state(x_t)
            c_t_n = normalizer.encode_command(c_t)
            net.step(c_t_n, x_t_n, learn=False)

        error = train_steps(
            experiment,
            normalizer,
            net,
            steps_per_episode,
        )
        if error.size == 0:
            episode_rmse[episode] = np.nan
        else:
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
            x_t, c_t, x_next = experiment.transition()
            if not _inside_position_guard(x_next, teacher, margin=0.02):
                raise RuntimeError(
                    "Synchronization trajectory left the legal state region"
                )
            x_t_n = normalizer.encode_state(x_t)
            c_t_n = normalizer.encode_command(c_t)
            x_next_n = normalizer.encode_state(x_next)
            x_hat_next = net.step(c_t_n, x_t_n, learn=False)
            delta = x_next_n - x_hat_next
            sync_component_error[i] = delta
            sync_error[i] = np.sqrt(np.mean(delta ** 2))

        net.feedback_gain = 0.0

        targets = []
        estimates = []

        for _ in range(n_steps):
            _, c_t, x_next = experiment.transition()
            if not _inside_position_guard(x_next, teacher, margin=0.02):
                break
            c_t_n = normalizer.encode_command(c_t)
            x_next_n = normalizer.encode_state(x_next)
            x_hat_next = net.step(c_t_n, target_state=None, learn=False)
            targets.append(x_next_n)
            estimates.append(x_hat_next)

        if not targets:
            raise RuntimeError(
                "Autonomous evaluation left the legal state region immediately"
            )

        targets = np.asarray(targets)
        estimates = np.asarray(estimates)
    finally:
        net.feedback_gain = feedback

    error = targets - estimates
    step_rmse = np.sqrt(np.mean(error**2, axis=1))
    n = teacher.n_dof
    sync_tail = sync_component_error[-min(50, sync_steps):]
    horizons = {}
    for horizon in (1, 5, 10, 25, 50, 100, 250, 500, 1000):
        if horizon <= len(error):
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
        "executed_steps": int(len(error)),
        "terminated_at_position_guard": bool(len(error) < n_steps),
    }



def evaluate_untrained_baseline(
    *,
    dt=1e-3,
    n_neurons=128,
    seed=0,
    torque_fraction=0.10,
    noise_std=0.5,
    n_steps=1000,
    sync_steps=150,
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
    basis_modes=("random",),
    n_neurons=128,
    n_episodes=8,
    steps_per_episode=400,
    eval_steps=400,
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
                for basis_mode in basis_modes:
                    teacher, experiment, normalizer, net = build_arm_experiment(
                        dt=1e-3,
                        n_neurons=n_neurons,
                        seed=seed,
                        torque_fraction=torque_fraction,
                        noise_std=noise_std,
                        eta=eta,
                        feedback_gain=feedback_gain,
                        decoder_scale=decoder_scale,
                        basis_mode=basis_mode,
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
                            "basis_mode": basis_mode,
                            "first_episode_rmse": float(episodes[0]),
                            "last_episode_rmse": float(episodes[-1]),
                            "sync_rmse": result["sync_rmse"],
                            "autonomous_rmse": result["rmse"],
                            "q_rmse": result["q_rmse"],
                            "qdot_rmse": result["qdot_rmse"],
                            "tau_rmse": result["tau_rmse"],
                            "executed_steps": result["executed_steps"],
                            "slow_weight_norm": float(np.linalg.norm(net.W_slow)),
                        })
                    finally:
                        teacher.close()

    return rows
