"""Run the paper's EBN learning rule on the iCubGenova11 right arm.

The teacher is the official iCubGenova11 rigid-body model reduced to the
seven right-arm joints with Pinocchio.

To keep the paper's additive-input form x_dot=f(x)+c(t), torque is included
in the learned state:

    x = [q, qdot, tau]
    tau_dot = -alpha tau + c_tau(t)

The network sees the teacher state only while error feedback is enabled
during training/synchronization. Testing sets k=0 and freezes learning.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .icub_teacher import ICubTeacher
from .paper_network import PaperEBN


class StateScaler:
    def __init__(self, teacher: ICubTeacher, torque_scale: np.ndarray):
        self.n = teacher.n_dof
        self.q_center = 0.5 * (teacher.lower + teacher.upper)
        self.q_scale = np.maximum(
            0.5 * (teacher.upper - teacher.lower),
            1e-6,
        )
        self.qdot_scale = np.maximum(self.q_scale, 0.5)
        self.torque_scale = np.asarray(torque_scale, dtype=float)

    def encode_state(self, x: np.ndarray) -> np.ndarray:
        n = self.n
        return np.concatenate([
            (x[:n] - self.q_center) / self.q_scale,
            x[n:2*n] / self.qdot_scale,
            x[2*n:] / self.torque_scale,
        ])

    def decode_state(self, x: np.ndarray) -> np.ndarray:
        n = self.n
        return np.concatenate([
            self.q_center + x[:n] * self.q_scale,
            x[n:2*n] * self.qdot_scale,
            x[2*n:] * self.torque_scale,
        ])

    def encode_command(self, c: np.ndarray) -> np.ndarray:
        n = self.n
        return np.concatenate([
            c[:n] / self.q_scale,
            c[n:2*n] / self.qdot_scale,
            c[2*n:] / self.torque_scale,
        ])


class TorqueDriver:
    """Smooth torque process with exact additive state dynamics."""

    def __init__(
        self,
        scale: np.ndarray,
        dt: float,
        *,
        fraction: float = 0.02,
        alpha: float = 4.0,
        beta: float = 20.0,
        seed: int = 0,
    ):
        self.scale = np.asarray(scale, dtype=float)
        self.dt = float(dt)
        self.fraction = float(fraction)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.rng = np.random.default_rng(seed)
        self.bias = np.zeros_like(self.scale)
        self.tau = np.zeros_like(self.scale)
        self.xi = np.zeros_like(self.scale)
        self.target_xi = np.zeros_like(self.scale)
        self.step_index = 0

    def reset(self, bias: np.ndarray, *, seed: int) -> None:
        self.rng = np.random.default_rng(seed)
        self.bias = np.asarray(bias, dtype=float).copy()
        self.tau = self.bias.copy()
        self.xi.fill(0.0)
        self.target_xi.fill(0.0)
        self.step_index = 0

    def step(self) -> tuple[np.ndarray, np.ndarray]:
        if self.step_index % 50 == 0:
            self.target_xi = (
                self.alpha
                * self.fraction
                * self.scale
                * self.rng.uniform(-1.0, 1.0, size=self.scale.shape)
            )

        self.xi += self.dt * self.beta * (self.target_xi - self.xi)
        command_tau = self.alpha * self.bias + self.xi
        self.tau += self.dt * (-self.alpha * self.tau + command_tau)
        self.step_index += 1
        return self.tau.copy(), command_tau.copy()


def characteristic_torque_scale(
    teacher: ICubTeacher,
    *,
    seed: int = 0,
    samples: int = 12,
) -> np.ndarray:
    """Get a numerical torque scale from the actual rigid-body model."""
    rng = np.random.default_rng(seed)
    span = teacher.upper - teacher.lower
    q_low = teacher.lower + 0.2 * span
    q_high = teacher.upper - 0.2 * span
    zeros = np.zeros(teacher.n_dof)
    scale = np.zeros(teacher.n_dof)

    poses = [0.5 * (teacher.lower + teacher.upper)]
    poses.extend(rng.uniform(q_low, q_high) for _ in range(samples))

    for q in poses:
        scale = np.maximum(
            scale,
            np.abs(teacher.inverse_dynamics(q, zeros, zeros)),
        )
        for j in range(teacher.n_dof):
            qddot = np.zeros(teacher.n_dof)
            qddot[j] = 1.0
            scale = np.maximum(
                scale,
                np.abs(teacher.inverse_dynamics(q, zeros, qddot)),
            )
            qddot[j] = -1.0
            scale = np.maximum(
                scale,
                np.abs(teacher.inverse_dynamics(q, zeros, qddot)),
            )

    return np.maximum(scale, 0.1)


def inside_limits(
    q: np.ndarray,
    teacher: ICubTeacher,
    *,
    margin: float = 0.02,
) -> bool:
    span = teacher.upper - teacher.lower
    return bool(
        np.all(q > teacher.lower + margin * span)
        and np.all(q < teacher.upper - margin * span)
    )


def sample_initial_state(
    teacher: ICubTeacher,
    scaler: StateScaler,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    span = teacher.upper - teacher.lower
    q = rng.uniform(
        teacher.lower + 0.2 * span,
        teacher.upper - 0.2 * span,
    )
    qdot = rng.uniform(
        -0.03 * scaler.qdot_scale,
        0.03 * scaler.qdot_scale,
    )
    return q, qdot


def one_teacher_step(
    teacher: ICubTeacher,
    driver: TorqueDriver,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    q, qdot = teacher.state()
    x_t = np.concatenate([q, qdot, driver.tau])

    tau, command_tau = driver.step()
    q_next, qdot_next = teacher.step(tau)

    x_next = np.concatenate([q_next, qdot_next, tau])
    command = np.concatenate([
        np.zeros(teacher.n_dof),
        np.zeros(teacher.n_dof),
        command_tau,
    ])
    return x_t, command, x_next


def evaluate_short_rollout(
    teacher: ICubTeacher,
    scaler: StateScaler,
    driver: TorqueDriver,
    network: PaperEBN,
    q0: np.ndarray,
    qdot0: np.ndarray,
    *,
    driver_seed: int,
    sync_steps: int,
    test_steps: int,
    sync_feedback_gain: float,
) -> dict | None:
    """Synchronize to a teacher trajectory, then evaluate a short k=0 rollout."""
    teacher.reset(q0, qdot0)
    gravity = teacher.inverse_dynamics(
        q0,
        np.zeros(teacher.n_dof),
        np.zeros(teacher.n_dof),
    )
    driver.reset(gravity, seed=driver_seed)
    network.reset_state()
    old_gain = network.feedback_gain
    network.feedback_gain = sync_feedback_gain

    try:
        for _ in range(sync_steps):
            x_t, command, x_next = one_teacher_step(teacher, driver)
            if not inside_limits(x_next[:teacher.n_dof], teacher):
                return None
            network.step(
                scaler.encode_command(command),
                scaler.encode_state(x_t),
                learn=False,
            )

        network.feedback_gain = 0.0
        errors = []
        for _ in range(test_steps):
            _, command, x_next = one_teacher_step(teacher, driver)
            if not inside_limits(x_next[:teacher.n_dof], teacher):
                break
            estimate = network.step(
                scaler.encode_command(command),
                teacher_state=None,
                learn=False,
            )
            errors.append(scaler.encode_state(x_next) - estimate)

        if not errors:
            return None

        error = np.asarray(errors)
        n = teacher.n_dof
        return {
            "steps": int(len(error)),
            "rmse": float(np.sqrt(np.mean(error**2))),
            "q_rmse": float(np.sqrt(np.mean(error[:, :n]**2))),
            "qdot_rmse": float(np.sqrt(np.mean(error[:, n:2*n]**2))),
            "tau_rmse": float(np.sqrt(np.mean(error[:, 2*n:]**2))),
        }
    finally:
        network.feedback_gain = old_gain


def run_neighborhood_diagnostic(
    teacher: ICubTeacher,
    scaler: StateScaler,
    driver: TorqueDriver,
    network: PaperEBN,
    q0: np.ndarray,
    qdot0: np.ndarray,
    *,
    driver_seed: int,
    sync_steps: int,
    test_steps: int = 25,
    sync_feedback_gain: float,
    epsilons=(0.005, 0.01, 0.02),
) -> dict:
    """Test k=0 prediction on nearby *real teacher trajectories*.

    Perturbations are expressed in normalized state units. Each perturbed
    initial condition is simulated by the real iCub teacher, the EBN is
    synchronized to that trajectory using the same protocol as the main test,
    and feedback is then removed. This avoids constructing artificial hidden
    EBN states.
    """
    baseline = evaluate_short_rollout(
        teacher,
        scaler,
        driver,
        network,
        q0,
        qdot0,
        driver_seed=driver_seed,
        sync_steps=sync_steps,
        test_steps=test_steps,
        sync_feedback_gain=sync_feedback_gain,
    )
    if baseline is None:
        raise RuntimeError("Neighborhood diagnostic baseline is invalid.")

    result = {
        "test_steps": int(test_steps),
        "baseline": baseline,
        "q": {},
        "qdot": {},
    }
    n = teacher.n_dof

    for kind in ("q", "qdot"):
        physical_scale = scaler.q_scale if kind == "q" else scaler.qdot_scale
        for eps in epsilons:
            trials = []
            for joint in range(n):
                for sign in (-1.0, 1.0):
                    q = q0.copy()
                    qdot = qdot0.copy()
                    if kind == "q":
                        q[joint] += sign * eps * physical_scale[joint]
                        if q[joint] <= teacher.lower[joint] or q[joint] >= teacher.upper[joint]:
                            continue
                    else:
                        qdot[joint] += sign * eps * physical_scale[joint]

                    trial = evaluate_short_rollout(
                        teacher,
                        scaler,
                        driver,
                        network,
                        q,
                        qdot,
                        driver_seed=driver_seed,
                        sync_steps=sync_steps,
                        test_steps=test_steps,
                        sync_feedback_gain=sync_feedback_gain,
                    )
                    if trial is not None:
                        trials.append(trial)

            if not trials:
                result[kind][str(eps)] = {"valid_trials": 0}
                continue

            mean_rmse = float(np.mean([trial["rmse"] for trial in trials]))
            mean_q = float(np.mean([trial["q_rmse"] for trial in trials]))
            mean_qdot = float(np.mean([trial["qdot_rmse"] for trial in trials]))
            mean_tau = float(np.mean([trial["tau_rmse"] for trial in trials]))
            result[kind][str(eps)] = {
                "valid_trials": int(len(trials)),
                "rmse": mean_rmse,
                "q_rmse": mean_q,
                "qdot_rmse": mean_qdot,
                "tau_rmse": mean_tau,
                "rmse_ratio_to_baseline": mean_rmse / baseline["rmse"],
            }

    return result


def run_experiment(
    *,
    dt: float = 1e-3,
    n_neurons: int = 256,
    train_episodes: int = 30,
    train_steps: int = 250,
    sync_steps: int = 75,
    test_steps: int = 250,
    feedback_gain: float = 40.0,
    final_feedback_gain: float = 10.0,
    eta: float = 0.05,
    torque_fraction: float = 0.02,
    seed: int = 0,
    perturbation_diagnostic: bool = False,
    diagnostic_steps: int = 25,
) -> dict:
    """Train on iCub dynamics, switch feedback off, and test unseen motion."""
    teacher = ICubTeacher(dt=dt, integration_substeps=4)
    torque_scale = characteristic_torque_scale(teacher, seed=seed)
    scaler = StateScaler(teacher, torque_scale)
    driver = TorqueDriver(
        torque_scale,
        dt,
        fraction=torque_fraction,
        seed=seed,
    )
    network = PaperEBN(
        state_dim=3 * teacher.n_dof,
        n_neurons=n_neurons,
        dt=dt,
        eta=eta,
        feedback_gain=feedback_gain,
        seed=seed,
    )

    rng = np.random.default_rng(seed)
    training_rmse = []

    try:
        # TRAINING: start with strong teacher feedback and reduce it as
        # learning proceeds, following the procedure described in the paper.
        for episode in range(train_episodes):
            if train_episodes <= 1:
                network.feedback_gain = final_feedback_gain
            else:
                fraction = episode / (train_episodes - 1)
                network.feedback_gain = (
                    feedback_gain
                    + fraction * (final_feedback_gain - feedback_gain)
                )
            q0, qdot0 = sample_initial_state(teacher, scaler, rng)
            teacher.reset(q0, qdot0)
            gravity = teacher.inverse_dynamics(
                q0,
                np.zeros(teacher.n_dof),
                np.zeros(teacher.n_dof),
            )
            driver.reset(gravity, seed=seed + episode)
            network.reset_state()

            # Synchronize neural state to a new teacher initial condition.
            for _ in range(sync_steps):
                x_t, command, x_next = one_teacher_step(teacher, driver)
                if not inside_limits(x_next[:teacher.n_dof], teacher):
                    break
                network.step(
                    scaler.encode_command(command),
                    scaler.encode_state(x_t),
                    learn=False,
                )

            errors = []
            for _ in range(train_steps):
                x_t, command, x_next = one_teacher_step(teacher, driver)
                if not inside_limits(x_next[:teacher.n_dof], teacher):
                    break
                estimate = network.step(
                    scaler.encode_command(command),
                    scaler.encode_state(x_t),
                    learn=True,
                )
                errors.append(
                    np.sqrt(
                        np.mean(
                            (
                                scaler.encode_state(x_next)
                                - estimate
                            ) ** 2
                        )
                    )
                )
            training_rmse.append(
                float(np.sqrt(np.mean(np.square(errors))))
                if errors else float("nan")
            )

        # TEST: unseen initial state/input, learning OFF, k=0 after sync.
        # Use an independent fixed RNG so comparisons with different training
        # lengths are evaluated on exactly the same test initial condition.
        eval_rng = np.random.default_rng(seed + 100000)
        q0, qdot0 = sample_initial_state(teacher, scaler, eval_rng)
        teacher.reset(q0, qdot0)
        gravity = teacher.inverse_dynamics(
            q0,
            np.zeros(teacher.n_dof),
            np.zeros(teacher.n_dof),
        )
        driver.reset(gravity, seed=seed + 100000)
        network.reset_state()
        network.feedback_gain = final_feedback_gain

        for _ in range(sync_steps):
            x_t, command, x_next = one_teacher_step(teacher, driver)
            if not inside_limits(x_next[:teacher.n_dof], teacher):
                break
            network.step(
                scaler.encode_command(command),
                scaler.encode_state(x_t),
                learn=False,
            )

        old_gain = network.feedback_gain
        network.feedback_gain = 0.0

        teacher_norm = []
        estimate_norm = []
        teacher_physical = []
        estimate_physical = []
        spike_rows = []
        spike_times = []

        try:
            for t in range(test_steps):
                _, command, x_next = one_teacher_step(teacher, driver)
                if not inside_limits(x_next[:teacher.n_dof], teacher):
                    break

                estimate = network.step(
                    scaler.encode_command(command),
                    teacher_state=None,
                    learn=False,
                )

                target = scaler.encode_state(x_next)
                teacher_norm.append(target)
                estimate_norm.append(estimate)
                teacher_physical.append(x_next.copy())
                estimate_physical.append(scaler.decode_state(estimate))

                neurons = np.flatnonzero(network.spikes)
                for neuron in neurons:
                    for _ in range(int(network.spikes[neuron])):
                        spike_times.append(t * dt)
                        spike_rows.append(int(neuron))
        finally:
            network.feedback_gain = old_gain

        teacher_norm = np.asarray(teacher_norm)
        estimate_norm = np.asarray(estimate_norm)
        teacher_physical = np.asarray(teacher_physical)
        estimate_physical = np.asarray(estimate_physical)

        if len(teacher_norm) == 0:
            raise RuntimeError("Test trajectory left joint limits immediately.")

        error = teacher_norm - estimate_norm
        n = teacher.n_dof

        perturbation_summary = None
        if perturbation_diagnostic:
            perturbation_summary = run_neighborhood_diagnostic(
                teacher,
                scaler,
                driver,
                network,
                q0,
                qdot0,
                driver_seed=seed + 100000,
                sync_steps=sync_steps,
                test_steps=diagnostic_steps,
                sync_feedback_gain=final_feedback_gain,
            )

        horizon_rmse = {}
        for horizon in (1, 5, 10, 25, 50, 100, 200, 250):
            if horizon <= len(error):
                prefix = error[:horizon]
                horizon_rmse[horizon] = {
                    "rmse": float(np.sqrt(np.mean(prefix**2))),
                    "q_rmse": float(np.sqrt(np.mean(prefix[:, :n]**2))),
                    "qdot_rmse": float(
                        np.sqrt(np.mean(prefix[:, n:2*n]**2))
                    ),
                    "tau_rmse": float(
                        np.sqrt(np.mean(prefix[:, 2*n:]**2))
                    ),
                }

        return {
            "dt": dt,
            "joint_names": list(teacher.active_joint_names),
            "training_rmse": np.asarray(training_rmse),
            "teacher": teacher_physical,
            "estimate": estimate_physical,
            "normalized_error": error,
            "rmse": float(np.sqrt(np.mean(error**2))),
            "q_rmse": float(np.sqrt(np.mean(error[:, :n]**2))),
            "qdot_rmse": float(np.sqrt(np.mean(error[:, n:2*n]**2))),
            "tau_rmse": float(np.sqrt(np.mean(error[:, 2*n:]**2))),
            "executed_test_steps": int(len(error)),
            "horizon_rmse": horizon_rmse,
            "feedback_start": float(feedback_gain),
            "feedback_final": float(final_feedback_gain),
            "spike_times": np.asarray(spike_times),
            "spike_neurons": np.asarray(spike_rows, dtype=int),
            "slow_weight_norm": float(np.linalg.norm(network.W_slow)),
            "perturbation_diagnostic": perturbation_summary,
        }
    finally:
        teacher.close()


def save_results(result: dict, output: Path) -> None:
    import matplotlib.pyplot as plt

    output.parent.mkdir(parents=True, exist_ok=True)
    n = len(result["joint_names"])
    time = np.arange(len(result["teacher"])) * result["dt"]

    fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=False)

    for j, name in enumerate(result["joint_names"]):
        axes[0].plot(time, result["teacher"][:, j], label=f"{name} teacher")
        axes[0].plot(
            time,
            result["estimate"][:, j],
            linestyle="--",
            label=f"{name} EBN",
        )
    axes[0].set_ylabel("q [rad]")
    axes[0].set_title("iCub joint position: simulator vs paper EBN")

    for j, name in enumerate(result["joint_names"]):
        axes[1].plot(time, result["teacher"][:, n+j], label=name)
        axes[1].plot(
            time,
            result["estimate"][:, n+j],
            linestyle="--",
        )
    axes[1].set_ylabel("qdot [rad/s]")
    axes[1].set_title("Joint velocity")

    step_rmse = np.sqrt(np.mean(result["normalized_error"] ** 2, axis=1))
    axes[2].plot(time, step_rmse)
    axes[2].set_ylabel("normalized RMSE")
    axes[2].set_title("Autonomous error after feedback is removed")

    axes[3].scatter(
        result["spike_times"],
        result["spike_neurons"],
        s=3,
    )
    axes[3].set_xlabel("time [s]")
    axes[3].set_ylabel("neuron")
    axes[3].set_title("Spike raster")

    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)

    np.savez(
        output.with_suffix(".npz"),
        teacher=result["teacher"],
        estimate=result["estimate"],
        normalized_error=result["normalized_error"],
        training_rmse=result["training_rmse"],
        spike_times=result["spike_times"],
        spike_neurons=result["spike_neurons"],
    )

    metrics = {
        key: value
        for key, value in result.items()
        if key in (
            "rmse",
            "q_rmse",
            "qdot_rmse",
            "tau_rmse",
            "executed_test_steps",
            "slow_weight_norm",
        )
    }
    output.with_suffix(".json").write_text(json.dumps(metrics, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the paper EBN directly on iCubGenova11 arm dynamics."
    )
    parser.add_argument("--output", type=Path, default=Path("results/paper_on_icub.png"))
    parser.add_argument("--neurons", type=int, default=256)
    parser.add_argument("--train-episodes", type=int, default=30)
    parser.add_argument("--train-steps", type=int, default=250)
    parser.add_argument("--test-steps", type=int, default=250)
    parser.add_argument("--sync-steps", type=int, default=75)
    parser.add_argument("--feedback-gain", type=float, default=40.0)
    parser.add_argument("--final-feedback-gain", type=float, default=10.0)
    parser.add_argument("--eta", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--perturbation-diagnostic", action="store_true")
    parser.add_argument("--diagnostic-steps", type=int, default=25)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    result = run_experiment(
        n_neurons=args.neurons,
        train_episodes=args.train_episodes,
        train_steps=args.train_steps,
        sync_steps=args.sync_steps,
        test_steps=args.test_steps,
        feedback_gain=args.feedback_gain,
        final_feedback_gain=args.final_feedback_gain,
        eta=args.eta,
        seed=args.seed,
        perturbation_diagnostic=args.perturbation_diagnostic,
        diagnostic_steps=args.diagnostic_steps,
    )

    summary = {
        "rmse": result["rmse"],
        "q_rmse": result["q_rmse"],
        "qdot_rmse": result["qdot_rmse"],
        "tau_rmse": result["tau_rmse"],
        "executed_test_steps": result["executed_test_steps"],
        "slow_weight_norm": result["slow_weight_norm"],
        "horizon_rmse": result["horizon_rmse"],
        "feedback_start": result["feedback_start"],
        "feedback_final": result["feedback_final"],
        "perturbation_diagnostic": result["perturbation_diagnostic"],
    }
    print(json.dumps(summary, indent=2))

    if not args.no_plot:
        save_results(result, args.output)


if __name__ == "__main__":
    main()
