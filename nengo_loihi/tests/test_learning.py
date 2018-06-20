import nengo
import numpy as np


def test_learning(Simulator, seed, plt):
    with nengo.Network(seed=seed) as model:
        stim = nengo.Node(lambda t: [np.sin(t*2*np.pi)])

        a = nengo.Ensemble(23, 1,
                           max_rates=nengo.dists.Uniform(100, 120),
                           intercepts=nengo.dists.Uniform(-0.5, 0.5))

        b = nengo.Node(None, size_in=1)

        nengo.Connection(stim, a)
        conn = nengo.Connection(a, b,
                                learning_rule_type=nengo.PES(),
                                function=lambda x: 0)

        error = nengo.Node(None, size_in=1)
        nengo.Connection(b, error)
        nengo.Connection(stim, error, transform=-1)
        nengo.Connection(error, conn.learning_rule)

        p_stim = nengo.Probe(stim)
        p_a = nengo.Probe(a, synapse=0.03)
        p_b = nengo.Probe(b, synapse=0.03)

    with Simulator(model, precompute=False) as sim:
        sim.run(5.0)

    plt.plot(sim.trange(), sim.data[p_stim])
    plt.plot(sim.trange(), sim.data[p_a])
    plt.plot(sim.trange(), sim.data[p_b])
