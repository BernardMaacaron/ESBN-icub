import numpy as np
import pytest

pytest.importorskip("pybullet")
pytest.importorskip("icub_models")

from esbn_icub.arm_learning import (
    autonomous_steps,
    build_arm_experiment,
    evaluate_unseen_episode,
    train_episodes,
    train_steps,
)


def test_arm_learning_harness_runs():
    teacher, experiment, normalizer, net = build_arm_experiment(
        dt=1e-3,
        n_neurons=64,
        seed=0,
        torque_fraction=0.01,
        noise_std=0.05,
    )
    try:
        experiment.reset()
        train_error = train_steps(experiment, normalizer, net, 5)
        result = autonomous_steps(experiment, normalizer, net, 3)

        assert train_error.shape == (5,)
        assert np.all(np.isfinite(train_error))
        assert np.isfinite(result["rmse"])
        assert result["targets"].shape == (3, 21)
        assert result["estimates"].shape == (3, 21)
    finally:
        teacher.close()



def test_episodic_training_and_unseen_evaluation_run():
    teacher, experiment, normalizer, net = build_arm_experiment(
        dt=1e-3,
        n_neurons=64,
        seed=2,
        torque_fraction=0.01,
        noise_std=0.05,
    )
    try:
        episode_error = train_episodes(
            experiment,
            normalizer,
            net,
            n_episodes=2,
            steps_per_episode=5,
            seed=2,
        )
        result = evaluate_unseen_episode(
            experiment,
            normalizer,
            net,
            n_steps=3,
            sync_steps=3,
            seed=3,
        )

        assert episode_error.shape == (2,)
        assert np.all(np.isfinite(episode_error))
        assert np.isfinite(result["rmse"])
        assert result["targets"].shape == (3, 21)
    finally:
        teacher.close()
