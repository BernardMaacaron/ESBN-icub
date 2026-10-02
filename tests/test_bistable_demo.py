import numpy as np

from esbn_icub.bistable_demo import evaluate_bistable_attractors, run_bistable_demo, run_bistable_trials


def test_bistable_demo_runs_finite():
    result = run_bistable_demo(
        dt=1e-3,
        train_steps=1000,
        test_steps=250,
        n_neurons=32,
        seed=0,
    )

    assert np.isfinite(result["train_rmse_tail"])
    assert np.isfinite(result["test_rmse"])
    assert np.isfinite(result["slow_weight_norm"])
    assert result["slow_weight_norm"] > 0.0
    assert result["target_trace"].shape == (250,)
    assert result["estimate_trace"].shape == (250,)



def test_bistable_trials_runs_finite():
    result = run_bistable_trials(
        train_trials=2,
        train_trial_steps=100,
        command_steps=50,
        test_steps=100,
        n_neurons=20,
        seed=1,
    )

    for key in (
        "train_rmse_tail",
        "test_rmse",
        "final_attractor_error",
        "train_rate_hz",
        "test_rate_hz",
        "slow_weight_norm",
    ):
        assert np.isfinite(result[key])


def test_bistable_multi_trial_acceptance_metric_runs():
    from esbn_icub.alemi_ebn import AlemiEBN

    net = AlemiEBN(
        state_dim=1,
        n_neurons=20,
        dt=1e-3,
        feedback_gain=0.0,
        seed=0,
    )
    result = evaluate_bistable_attractors(
        net,
        n_trials=4,
        command_steps=5,
        settle_steps=5,
        seed=0,
    )
    assert 0.0 <= result["success_rate"] <= 1.0
    assert np.isfinite(result["mean_attractor_error"])
