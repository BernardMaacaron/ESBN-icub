import numpy as np

from esbn_icub.alemi_ebn import AlemiEBN


def test_shapes_and_decode():
    net = AlemiEBN(state_dim=3, n_neurons=20, dt=1e-3, seed=1)
    assert net.D.shape == (3, 20)
    assert net.W_fast.shape == (20, 20)
    assert net.W_slow.shape == (20, 20)
    assert net.decoded_state.shape == (3,)


def test_local_slow_weight_update_matches_outer_product_without_spikes():
    net = AlemiEBN(
        state_dim=2,
        n_neurons=6,
        dt=1e-3,
        eta=0.2,
        feedback_gain=0.0,
        seed=2,
    )
    net.u[:] = -100.0
    target = np.array([0.3, -0.2])
    command = np.zeros(2)

    error = target - net.decoded_state
    psi = net.basis()
    expected = net.eta * net.dt * np.outer(net.D.T @ error, psi)

    net.step(command, target, learn=True)
    np.testing.assert_allclose(net.W_slow, expected)


def test_no_learning_when_disabled():
    net = AlemiEBN(state_dim=2, n_neurons=10, dt=1e-3, seed=3)
    before = net.W_slow.copy()
    net.step(np.zeros(2), np.ones(2), learn=False)
    np.testing.assert_array_equal(net.W_slow, before)
