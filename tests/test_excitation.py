import numpy as np

from esbn_icub.excitation import FilteredTorque


def test_filtered_torque_respects_limits():
    process = FilteredTorque(
        limits=np.array([0.1, 0.2]),
        dt=0.05,
        noise_std=100.0,
        seed=0,
    )
    for _ in range(100):
        tau, xi = process.step()
        assert tau.shape == (2,)
        assert xi.shape == (2,)
        assert np.all(np.abs(tau) <= process.limits + 1e-12)


def test_reset_with_seed_reproduces_excitation():
    process = FilteredTorque(
        limits=np.array([0.2, 0.3]),
        dt=0.01,
        noise_std=0.5,
        seed=123,
    )
    seq_a = [process.step()[1].copy() for _ in range(20)]
    process.reset(seed=123)
    seq_b = [process.step()[1].copy() for _ in range(20)]
    np.testing.assert_allclose(seq_a, seq_b)
