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


def teacher_vector_field_normalized(
    teacher: ICubTeacher,
    scaler: StateScaler,
    x_normalized: np.ndarray,
    command: np.ndarray,
    *,
    torque_alpha: float,
) -> np.ndarray:
    """Exact normalized continuous-time vector field of the rigid-body teacher."""
    x = scaler.decode_state(np.asarray(x_normalized, dtype=float))
    n = teacher.n_dof
    q = x[:n]
    qdot = x[n:2*n]
    tau = x[2*n:]
    command_tau = np.asarray(command[2*n:], dtype=float)

    qddot = teacher.acceleration(q, qdot, tau)
    tau_dot = -torque_alpha * tau + command_tau
    return np.concatenate([
        qdot / scaler.q_scale,
        qddot / scaler.qdot_scale,
        tau_dot / scaler.torque_scale,
    ])


def network_vector_field_effective(
    network: PaperEBN,
    command_normalized: np.ndarray,
    decoded_delta: np.ndarray | None = None,
) -> np.ndarray:
    """One-step effective decoded vector field around the current EBN state.

    A decoded perturbation is inserted with the minimum-norm rate change that
    satisfies D @ delta_r = decoded_delta. The original spiking state is
    restored after the probe, so diagnostics do not alter the experiment.
    """
    u0 = network.u.copy()
    r0 = network.r.copy()
    spikes0 = network.spikes.copy()
    gain0 = network.feedback_gain

    try:
        if decoded_delta is not None:
            delta = np.asarray(decoded_delta, dtype=float)
            gram = network.D @ network.D.T
            delta_r = network.D.T @ np.linalg.solve(gram, delta)
            network.r += delta_r

        x0 = network.decoded_state.copy()
        network.feedback_gain = 0.0
        x1 = network.step(
            np.asarray(command_normalized, dtype=float),
            teacher_state=None,
            learn=False,
        )
        return (x1 - x0) / network.dt
    finally:
        network.u[:] = u0
        network.r[:] = r0
        network.spikes[:] = spikes0
        network.feedback_gain = gain0


def _new_perturbation_accumulator(epsilons):
    return {
        kind: {
            float(eps): {
                "response_rmse": [],
                "response_cosine": [],
                "perturbed_field_rmse": [],
                "teacher_growth": [],
                "network_growth": [],
            }
            for eps in epsilons
        }
        for kind in ("q", "qdot")
    }


def perturbation_probe(
    teacher: ICubTeacher,
    scaler: StateScaler,
    driver: TorqueDriver,
    network: PaperEBN,
    x_t: np.ndarray,
    command: np.ndarray,
    accumulator: dict,
    *,
    epsilons=(0.005, 0.01, 0.02),
) -> float:
    """Compare teacher and learned local responses around one synchronized state."""
    x_n = scaler.encode_state(x_t)
    c_n = scaler.encode_command(command)
    f_teacher_0 = teacher_vector_field_normalized(
        teacher,
        scaler,
        x_n,
        command,
        torque_alpha=driver.alpha,
    )
    f_network_0 = network_vector_field_effective(network, c_n)
    baseline_rmse = float(np.sqrt(np.mean((f_network_0 - f_teacher_0) ** 2)))

    n = teacher.n_dof
    for kind, offset in (("q", 0), ("qdot", n)):
        for eps in epsilons:
            bucket = accumulator[kind][float(eps)]
            for joint in range(n):
                delta = np.zeros(3 * n)
                delta[offset + joint] = eps

                f_t_plus = teacher_vector_field_normalized(
                    teacher,
                    scaler,
                    x_n + delta,
                    command,
                    torque_alpha=driver.alpha,
                )
                f_t_minus = teacher_vector_field_normalized(
                    teacher,
                    scaler,
                    x_n - delta,
                    command,
                    torque_alpha=driver.alpha,
                )
                f_n_plus = network_vector_field_effective(network, c_n, delta)
                f_n_minus = network_vector_field_effective(network, c_n, -delta)

                teacher_response = (f_t_plus - f_t_minus) / (2.0 * eps)
                network_response = (f_n_plus - f_n_minus) / (2.0 * eps)
                response_error = network_response - teacher_response
                bucket["response_rmse"].append(
                    float(np.sqrt(np.mean(response_error**2)))
                )

                denom = (
                    np.linalg.norm(teacher_response)
                    * np.linalg.norm(network_response)
                )
                if denom > 1e-12:
                    bucket["response_cosine"].append(
                        float(
                            np.dot(teacher_response, network_response) / denom
                        )
                    )

                bucket["perturbed_field_rmse"].extend([
                    float(np.sqrt(np.mean((f_n_plus - f_t_plus) ** 2))),
                    float(np.sqrt(np.mean((f_n_minus - f_t_minus) ** 2))),
                ])

                teacher_growth_plus = np.linalg.norm(
                    delta + network.dt * (f_t_plus - f_teacher_0)
                ) / eps
                teacher_growth_minus = np.linalg.norm(
                    -delta + network.dt * (f_t_minus - f_teacher_0)
                ) / eps
                network_growth_plus = np.linalg.norm(
                    delta + network.dt * (f_n_plus - f_network_0)
                ) / eps
                network_growth_minus = np.linalg.norm(
                    -delta + network.dt * (f_n_minus - f_network_0)
                ) / eps

                bucket["teacher_growth"].append(
                    float(0.5 * (teacher_growth_plus + teacher_growth_minus))
                )
                bucket["network_growth"].append(
                    float(0.5 * (network_growth_plus + network_growth_minus))
                )

    return baseline_rmse


def summarize_perturbation_diagnostic(
    accumulator: dict,
    baseline_errors: list[float],
) -> dict:
    summary = {
        "anchors": int(len(baseline_errors)),
        "baseline_vector_field_rmse": (
            float(np.mean(baseline_errors)) if baseline_errors else float("nan")
        ),
    }
    for kind, by_eps in accumulator.items():
        summary[kind] = {}
        for eps, values in by_eps.items():
            teacher_growth = float(np.mean(values["teacher_growth"]))
            network_growth = float(np.mean(values["network_growth"]))
            summary[kind][str(eps)] = {
                "response_rmse": float(np.mean(values["response_rmse"])),
                "response_cosine": (
                    float(np.mean(values["response_cosine"]))
                    if values["response_cosine"]
                    else float("nan")
                ),
                "perturbed_field_rmse": float(
                    np.mean(values["perturbed_field_rmse"])
                ),
                "teacher_growth": teacher_growth,
                "network_growth": network_growth,
                "excess_growth": network_growth - teacher_growth,
            }
    return summary


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
    diagnostic_anchors: int = 10,
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

        perturbation_accumulator = _new_perturbation_accumulator(
            (0.005, 0.01, 0.02)
        )
        perturbation_baseline_errors = []

        for sync_index in range(sync_steps):
            x_t, command, x_next = one_teacher_step(teacher, driver)
            if not inside_limits(x_next[:teacher.n_dof], teacher):
                break

            if (
                perturbation_diagnostic
                and sync_index >= max(0, sync_steps - diagnostic_anchors)
            ):
                perturbation_baseline_errors.append(
                    perturbation_probe(
                        teacher,
                        scaler,
                        driver,
                        network,
                        x_t,
                        command,
                        perturbation_accumulator,
                    )
                )

            network.step(
                scaler.encode_command(command),
                scaler.encode_state(x_t),
                learn=False,
            )

        perturbation_summary = (
            summarize_perturbation_diagnostic(
                perturbation_accumulator,
                perturbation_baseline_errors,
            )
            if perturbation_diagnostic
            else None
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
    parser.add_argument("--diagnostic-anchors", type=int, default=10)
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
        diagnostic_anchors=args.diagnostic_anchors,
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
