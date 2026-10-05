import numpy as np

from esbn_icub.paper_network import PaperEBN


def test_paper_network_shapes_and_local_update():
    net = PaperEBN(
        state_dim=3,
        n_neurons=24,
        dt=1e-3,
        eta=0.2,
        feedback_gain=0.0,
        seed=1,
    )

    assert net.D.shape == (3, 24)
    assert net.W_fast.shape == (24, 24)
    assert net.W_slow.shape == (24, 24)

    target = np.array([0.2, -0.1, 0.05])
    command = np.zeros(3)
    error = target - net.decoded_state
    psi = np.tanh(net.M @ net.r + net.theta)
    expected = net.eta * net.dt * np.outer(net.D.T @ error, psi)

    net.step(command, target, learn=True)

    np.testing.assert_allclose(net.W_slow, expected)
