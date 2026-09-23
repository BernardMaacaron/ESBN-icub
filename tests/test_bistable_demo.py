import numpy as np

from esbn_icub.bistable_demo import run_bistable_demo


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
