import numpy as np
import pytest

import nengo
import nengo_loihi


def test_ens_decoded_on_host(Simulator, seed, plt):
    view_synapse = nengo.synapses.Alpha(0.03)

    with nengo.Network(seed=seed) as model:
        nengo_loihi.add_params(model)

        stim = nengo.Node(lambda t: [np.sin(t*2*np.pi)])

        a = nengo.Ensemble(100, 1)
        model.config[a].on_chip = False

        b = nengo.Ensemble(100, 1,
                           max_rates=nengo.dists.Uniform(100, 120),
                           intercepts=nengo.dists.Uniform(-0.5, 0.5))

        nengo.Connection(stim, a)

        nengo.Connection(a, b, function=lambda x: -x)

        p_stim = nengo.Probe(stim, synapse=view_synapse)
        p_a = nengo.Probe(a, synapse=view_synapse)
        p_b = nengo.Probe(b, synapse=view_synapse)

    with Simulator(model) as sim:
        sim.run(1.0)

    plt.plot(sim.trange(), sim.data[p_stim])
    plt.plot(sim.trange(), sim.data[p_a])
    plt.plot(sim.trange(), sim.data[p_b])

    assert np.allclose(sim.data[p_a], sim.data[p_stim], atol=0.05, rtol=0.01)
    assert np.allclose(sim.data[p_b], -sim.data[p_a], atol=0.15, rtol=0.1)


@pytest.mark.xfail
def test_ens_neurons_on_host(Simulator, seed, plt):
    with nengo.Network(seed=seed) as model:
        nengo_loihi.add_params(model)

        stim = nengo.Node(lambda t: [np.sin(t*2*np.pi)])

        a = nengo.Ensemble(100, 1)
        model.config[a].on_chip = False

        b = nengo.Ensemble(100, 1,
                           max_rates=nengo.dists.Uniform(100, 120),
                           intercepts=nengo.dists.Uniform(-0.5, 0.5))

        nengo.Connection(stim, a)

        nengo.Connection(a.neurons, b, transform=np.zeros((1, 100)))

        p_stim = nengo.Probe(stim)
        p_a = nengo.Probe(a)
        p_b = nengo.Probe(b)

    with Simulator(model) as sim:
        sim.run(1.0)

    plt.plot(sim.trange(), sim.data[p_stim])
    plt.plot(sim.trange(), sim.data[p_a])
    plt.plot(sim.trange(), sim.data[p_b])
