import nengo
import numpy as np
import pytest


@pytest.mark.parametrize('n_per_dim', [120, 200])
@pytest.mark.parametrize('dims', [1, 3])
def test_pes_comm_channel(allclose, plt, seed, Simulator, n_per_dim, dims):
    scale = np.linspace(1, 0, dims + 1)[:-1]
    input_fn = lambda t: np.sin(t * 2 * np.pi) * scale

    with nengo.Network(seed=seed) as model:
        stim = nengo.Node(input_fn)

        pre = nengo.Ensemble(n_per_dim * dims, dims)
        post = nengo.Node(size_in=dims)

        nengo.Connection(stim, pre, synapse=None)
        conn = nengo.Connection(
            pre, post,
            function=lambda x: np.zeros(dims),
            synapse=0.01,
            learning_rule_type=nengo.PES(learning_rate=1e-3))

        nengo.Connection(post, conn.learning_rule)
        nengo.Connection(stim, conn.learning_rule, transform=-1)

        p_stim = nengo.Probe(stim, synapse=0.02)
        p_pre = nengo.Probe(pre, synapse=0.02)
        p_post = nengo.Probe(post, synapse=0.02)

    with Simulator(model) as sim:
        sim.run(5.0)

    t = sim.trange()
    plt.subplot(211)
    plt.plot(t, sim.data[p_stim])
    plt.plot(t, sim.data[p_pre])
    plt.plot(t, sim.data[p_post])

    # --- fit input_fn to output, determine magnitude
    #     The larger the magnitude, the closer the output is to the input
    x = np.array([input_fn(tt)[0] for tt in t[t > 4]])
    y = sim.data[p_post][t > 4][:, 0]
    m = np.linspace(0, 1, 21)
    errors = np.abs(y - m[:, None]*x).mean(axis=1)
    m_best = m[np.argmin(errors)]

    plt.subplot(212)
    plt.plot(t[t > 4], x)
    plt.plot(t[t > 4], y)
    plt.plot(t[t > 4], m_best * x, ':')

    assert allclose(sim.data[p_pre][t > 0.1],
                    sim.data[p_stim][t > 0.1],
                    atol=0.15,
                    rtol=0.15)
    assert np.min(errors) < 0.3, "Not able to fit correctly"
    assert m_best > (0.3 if n_per_dim < 150 else 0.6)


def test_multiple_pes(allclose, plt, seed, Simulator):
    n_errors = 5
    targets = np.linspace(-0.9, 0.9, n_errors)
    with nengo.Network(seed=seed) as model:
        pre_ea = nengo.networks.EnsembleArray(200, n_ensembles=n_errors)
        output = nengo.Node(size_in=n_errors)

        target = nengo.Node(targets)

        for i in range(n_errors):
            conn = nengo.Connection(
                pre_ea.ea_ensembles[i],
                output[i],
                learning_rule_type=nengo.PES(learning_rate=5e-4),
            )
            nengo.Connection(target[i], conn.learning_rule, transform=-1)
            nengo.Connection(output[i], conn.learning_rule)

        probe = nengo.Probe(output, synapse=0.1)
    with Simulator(model) as sim:
        sim.run(1.0)
    t = sim.trange()

    plt.plot(t, sim.data[probe])
    for target, style in zip(targets, plt.rcParams["axes.prop_cycle"]):
        plt.axhline(target, **style)

    for i, target in enumerate(targets):
        assert allclose(sim.data[probe][t > 0.8, i], target, atol=0.1)
