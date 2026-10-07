"""Strict paper-form EBN experiment on the iCub right arm.

The network equations are exactly the architecture in Alemi et al. (2018).
The only task-specific adapter is how the iCub teacher is written in the
paper's required form

    x_dot = f(x) + c(t).

For the 7-DOF arm we use the second-order state

    x = [q, qdot]  (K = 14),

as the paper does for a mechanical system by providing both position and
velocity.  The command is an additive generalized-acceleration signal

    c(t) = [0, a_cmd(t)].

The iCub rigid-body teacher therefore obeys

    qddot = qddot_passive(q, qdot) + a_cmd(t),

which is exactly of the paper's additive-input form.  No torque state,
custom recurrent objective, DAgger, rollout loss, or auxiliary dynamics are
introduced.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .icub_teacher import ICubTeacher
from .paper_network import PaperEBN


class ArmCoordinates:
    """Fixed affine/unit transform for the 14-D mechanical state."""

    def __init__(self, teacher: ICubTeacher):
        self.n = teacher.n_dof
        self.q0 = 0.5 * (teacher.lower + teacher.upper)
        self.q_scale = np.maximum(
            0.5 * (teacher.upper - teacher.lower),
            1e-6,
        )
        # A fixed numerical unit for velocity.  This is task preprocessing,
        # not a change to the EBN equations.
        self.v_scale = np.maximum(self.q_scale, 0.5)

    def encode(self, q: np.ndarray, qdot: np.ndarray) -> np.ndarray:
        return np.concatenate([
            (np.asarray(q) - self.q0) / self.q_scale,
            np.asarray(qdot) / self.v_scale,
        ])

    def decode(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x = np.asarray(x)
        return (
            self.q0 + x[:self.n] * self.q_scale,
            x[self.n:] * self.v_scale,
        )

    def command(self, normalized_acceleration: np.ndarray) -> np.ndarray:
        """Paper input c(t) in the same normalized state coordinates."""
        return np.concatenate([
            np.zeros(self.n),
            np.asarray(normalized_acceleration, dtype=float),
        ])

    def physical_acceleration(
        self,
        normalized_acceleration: np.ndarray,
    ) -> np.ndarray:
        return np.asarray(normalized_acceleration, dtype=float) * self.v_scale


class FilteredRandomInput:
    """Filtered random command c(t), as used in the paper."""

    def __init__(
        self,
        n: int,
        dt: float,
        *,
        amplitude: float = 0.35,
        time_constant: float = 0.05,
        resample_time: float = 0.05,
        seed: int = 0,
    ):
        self.n = int(n)
        self.dt = float(dt)
        self.amplitude = float(amplitude)
        self.beta = 1.0 / float(time_constant)
        self.resample_steps = max(1, int(round(resample_time / dt)))
        self.rng = np.random.default_rng(seed)
        self.value = np.zeros(self.n)
        self.target = np.zeros(self.n)
        self.index = 0

    def reset(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)
        self.value.fill(0.0)
        self.target.fill(0.0)
        self.index = 0

    def step(self) -> np.ndarray:
        if self.index % self.resample_steps == 0:
            self.target = self.rng.uniform(
                -self.amplitude,
                self.amplitude,
                size=self.n,
            )
        self.value += self.dt * self.beta * (self.target - self.value)
        self.index += 1
        return self.value.copy()


def feedback_for_iteration(
    iteration: int,
    n_iterations: int,
    start: float,
    end: float,
) -> float:
    """Large feedback initially, reduced as learning progresses (paper text)."""
    if n_iterations <= 1:
        return float(end)
    fraction = iteration / (n_iterations - 1)
    return float(start + fraction * (end - start))


def run_teacher_network_step(
    teacher: ICubTeacher,
    coords: ArmCoordinates,
    command_source: FilteredRandomInput,
    network: PaperEBN,
    *,
    learn: bool,
    input_enabled: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    q, qdot = teacher.state()
    x_t = coords.encode(q, qdot)

    if input_enabled:
        a_normalized = command_source.step()
    else:
        a_normalized = np.zeros(teacher.n_dof)

    c_t = coords.command(a_normalized)
    q_next, qdot_next = teacher.step_additive_acceleration(
        coords.physical_acceleration(a_normalized)
    )
    x_next = coords.encode(q_next, qdot_next)

    xhat_next = network.step(
        c_t,
        teacher_state=x_t if learn or network.feedback_gain != 0.0 else None,
        learn=learn,
    )
    return x_next, xhat_next, c_t


def run_experiment(
    *,
    dt: float = 1e-3,
    n_neurons: int = 200,
    train_iterations: int = 500,
    steps_per_iteration: int = 50,
    test_input_steps: int = 50,
    test_free_steps: int = 50,
    feedback_start: float = 40.0,
    feedback_end: float = 5.0,
    eta: float = 0.05,
    input_amplitude: float = 0.05,
    seed: int = 0,
) -> dict:
    """Train exactly with Eq. 11/12, then test with k_test=0."""
    teacher = ICubTeacher(dt=dt, integration_substeps=4)
    coords = ArmCoordinates(teacher)
    source = FilteredRandomInput(
        teacher.n_dof,
        dt,
        amplitude=input_amplitude,
        seed=seed,
    )
    network = PaperEBN(
        state_dim=2 * teacher.n_dof,
        n_neurons=n_neurons,
        dt=dt,
        eta=eta,
        feedback_gain=feedback_start,
        seed=seed,
    )

    training_rmse = np.empty(train_iterations)

    try:
        # Each learning iteration starts from the same well-defined mechanical
        # state.  W_slow persists; neural state is reset.
        for iteration in range(train_iterations):
            teacher.reset(coords.q0, np.zeros(teacher.n_dof))
            network.reset_state()
            source.reset(seed + iteration)

            network.feedback_gain = feedback_for_iteration(
                iteration,
                train_iterations,
                feedback_start,
                feedback_end,
            )

            squared_error = []
            for _ in range(steps_per_iteration):
                target, estimate, _ = run_teacher_network_step(
                    teacher,
                    coords,
                    source,
                    network,
                    learn=True,
                )
                if np.any(teacher.q < teacher.lower) or np.any(teacher.q > teacher.upper):
                    raise RuntimeError(
                        "Training trajectory left the iCub hardware joint range; "
                        "reduce input_amplitude or iteration duration."
                    )
                squared_error.append(np.mean((target - estimate) ** 2))

            training_rmse[iteration] = np.sqrt(np.mean(squared_error))

        # Paper test phase: unseen random input, k_test = 0, learning disabled.
        teacher.reset(coords.q0, np.zeros(teacher.n_dof))
        network.reset_state()
        source.reset(seed + 1_000_000)
        network.feedback_gain = 0.0

        total_steps = test_input_steps + test_free_steps
        targets = np.empty((total_steps, 2 * teacher.n_dof))
        estimates = np.empty_like(targets)
        spikes = []

        for t in range(total_steps):
            target, estimate, _ = run_teacher_network_step(
                teacher,
                coords,
                source,
                network,
                learn=False,
                input_enabled=t < test_input_steps,
            )
            if np.any(teacher.q < teacher.lower) or np.any(teacher.q > teacher.upper):
                raise RuntimeError(
                    "Test trajectory left the iCub hardware joint range; "
                    "reduce input amplitude or test duration."
                )
            targets[t] = target
            estimates[t] = estimate
            for neuron in np.flatnonzero(network.spikes):
                spikes.extend(
                    (t * dt, int(neuron))
                    for _ in range(int(network.spikes[neuron]))
                )

        error = targets - estimates
        n = teacher.n_dof
        horizons = {}
        for horizon in (1, 5, 10, 25, 50, 100, total_steps):
            if horizon <= total_steps:
                prefix = error[:horizon]
                horizons[horizon] = {
                    "rmse": float(np.sqrt(np.mean(prefix**2))),
                    "q_rmse": float(np.sqrt(np.mean(prefix[:, :n]**2))),
                    "qdot_rmse": float(np.sqrt(np.mean(prefix[:, n:]**2))),
                }

        teacher_physical = np.empty_like(targets)
        estimate_physical = np.empty_like(estimates)
        for i in range(total_steps):
            tq, tv = coords.decode(targets[i])
            eq, ev = coords.decode(estimates[i])
            teacher_physical[i] = np.concatenate([tq, tv])
            estimate_physical[i] = np.concatenate([eq, ev])

        return {
            "dt": dt,
            "joint_names": list(teacher.active_joint_names),
            "training_rmse": training_rmse,
            "teacher": teacher_physical,
            "estimate": estimate_physical,
            "normalized_error": error,
            "rmse": float(np.sqrt(np.mean(error**2))),
            "q_rmse": float(np.sqrt(np.mean(error[:, :n]**2))),
            "qdot_rmse": float(np.sqrt(np.mean(error[:, n:]**2))),
            "horizon_rmse": horizons,
            "test_input_steps": int(test_input_steps),
            "test_free_steps": int(test_free_steps),
            "slow_weight_norm": float(np.linalg.norm(network.W_slow)),
            "spike_times": np.asarray([item[0] for item in spikes]),
            "spike_neurons": np.asarray(
                [item[1] for item in spikes],
                dtype=int,
            ),
        }
    finally:
        teacher.close()


def save_results(result: dict, output: Path) -> None:
    import matplotlib.pyplot as plt

    output.parent.mkdir(parents=True, exist_ok=True)
    n = len(result["joint_names"])
    time = np.arange(len(result["teacher"])) * result["dt"]

    fig, axes = plt.subplots(4, 1, figsize=(12, 12))

    for j in range(n):
        axes[0].plot(time, result["teacher"][:, j])
        axes[0].plot(time, result["estimate"][:, j], linestyle="--")
    axes[0].set_ylabel("q [rad]")
    axes[0].set_title("iCub position: teacher / EBN")

    for j in range(n):
        axes[1].plot(time, result["teacher"][:, n + j])
        axes[1].plot(time, result["estimate"][:, n + j], linestyle="--")
    axes[1].set_ylabel("qdot [rad/s]")
    axes[1].set_title("iCub velocity: teacher / EBN")

    axes[2].plot(
        time,
        np.sqrt(np.mean(result["normalized_error"] ** 2, axis=1)),
    )
    axes[2].axvline(
        result["test_input_steps"] * result["dt"],
        linestyle="--",
    )
    axes[2].set_ylabel("normalized RMSE")
    axes[2].set_title("k_test = 0; dashed line = command switched off")

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

    output.with_suffix(".json").write_text(
        json.dumps(
            {
                "rmse": result["rmse"],
                "q_rmse": result["q_rmse"],
                "qdot_rmse": result["qdot_rmse"],
                "horizon_rmse": result["horizon_rmse"],
                "slow_weight_norm": result["slow_weight_norm"],
            },
            indent=2,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Strict Alemi et al. paper-form EBN on iCub arm dynamics."
    )
    parser.add_argument("--output", type=Path, default=Path("results/paper_on_icub.png"))
    parser.add_argument("--neurons", type=int, default=200)
    parser.add_argument("--train-iterations", type=int, default=500)
    parser.add_argument("--steps-per-iteration", type=int, default=50)
    parser.add_argument("--test-input-steps", type=int, default=50)
    parser.add_argument("--test-free-steps", type=int, default=50)
    parser.add_argument("--feedback-start", type=float, default=40.0)
    parser.add_argument("--feedback-end", type=float, default=5.0)
    parser.add_argument("--eta", type=float, default=0.05)
    parser.add_argument("--input-amplitude", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    result = run_experiment(
        n_neurons=args.neurons,
        train_iterations=args.train_iterations,
        steps_per_iteration=args.steps_per_iteration,
        test_input_steps=args.test_input_steps,
        test_free_steps=args.test_free_steps,
        feedback_start=args.feedback_start,
        feedback_end=args.feedback_end,
        eta=args.eta,
        input_amplitude=args.input_amplitude,
        seed=args.seed,
    )

    print(json.dumps({
        "rmse": result["rmse"],
        "q_rmse": result["q_rmse"],
        "qdot_rmse": result["qdot_rmse"],
        "horizon_rmse": result["horizon_rmse"],
        "slow_weight_norm": result["slow_weight_norm"],
    }, indent=2))

    if not args.no_plot:
        save_results(result, args.output)


if __name__ == "__main__":
    main()
