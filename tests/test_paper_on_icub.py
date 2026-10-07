import numpy as np

from esbn_icub.paper_on_icub import run_experiment


def test_end_to_end_paper_on_icub_smoke():
    result = run_experiment(
        n_neurons=64,
        train_iterations=2,
        steps_per_iteration=25,
        test_input_steps=10,
        test_free_steps=10,
        input_amplitude=0.05,
        seed=2,
    )

    assert result["teacher"].shape[1] == 14
    assert result["estimate"].shape == result["teacher"].shape
    assert result["teacher"].shape[0] == 20
    assert np.isfinite(result["rmse"])
    assert np.isfinite(result["slow_weight_norm"])
