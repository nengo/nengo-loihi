import nengo
import numpy as np
import pytest


@pytest.mark.xfail(pytest.config.getoption("--target") == "loihi",
                   reason="Hangs indefinitely on Loihi")
@pytest.mark.parametrize('N', [100, 300])
def test_pes_comm_channel(allclose, Simulator, seed, plt, N):
    input_fn = lambda t: np.sin(t*2*np.pi)

    with nengo.Network(seed=seed) as model:
        stim = nengo.Node(input_fn)

        a = nengo.Ensemble(N, 1)

        b = nengo.Node(None, size_in=1, size_out=1)

        nengo.Connection(stim, a, synapse=None)
        conn = nengo.Connection(
            a, b, function=lambda x: 0, synapse=0.01,
            learning_rule_type=nengo.PES(learning_rate=1e-3))

        error = nengo.Node(None, size_in=1)
        nengo.Connection(b, error)
        nengo.Connection(stim, error, transform=-1)
        nengo.Connection(error, conn.learning_rule)

        p_stim = nengo.Probe(stim)
        p_a = nengo.Probe(a, synapse=0.02)
        p_b = nengo.Probe(b, synapse=0.02)

    with Simulator(model, precompute=False) as sim:
        sim.run(5.0)

    t = sim.trange()
    plt.subplot(211)
    plt.plot(t, sim.data[p_stim])
    plt.plot(t, sim.data[p_a])
    plt.plot(t, sim.data[p_b])

    # --- fit input_fn to output, determine magnitude
    #   The larger the magnitude is, the closer the output is to the input
    x = input_fn(t)[t > 4]
    y = sim.data[p_b][t > 4][:, 0]
    m = np.linspace(0, 1, 21)
    errors = np.abs(y - m[:, None]*x).mean(axis=1)
    m_best = m[np.argmin(errors)]

    plt.subplot(212)
    plt.plot(t[t > 4], x)
    plt.plot(t[t > 4], y)
    plt.plot(t[t > 4], m_best * x, ':')

    assert allclose(
        sim.data[p_a][t > 0.1], sim.data[p_stim][t > 0.1], atol=0.2, rtol=0.2)
    assert errors.min() < 0.3, "Not able to fit correctly"
    assert m_best > (0.3 if N < 150 else 0.6)
