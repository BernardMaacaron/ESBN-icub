import numpy as np

from esbn_icub.paper_on_icub import run_experiment


def test_end_to_end_paper_on_icub_smoke():
    result = run_experiment(
        n_neurons=64,
        train_episodes=2,
        train_steps=25,
        sync_steps=10,
        test_steps=25,
        seed=2,
    )

    assert result["teacher"].shape[1] == 21
    assert result["estimate"].shape == result["teacher"].shape
    assert result["executed_test_steps"] > 0
    assert np.isfinite(result["rmse"])
    assert np.isfinite(result["slow_weight_norm"])
