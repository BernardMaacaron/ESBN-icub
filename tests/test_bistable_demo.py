import numpy as np

from esbn_icub.bistable_demo import run_bistable_demo, run_bistable_trials


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
