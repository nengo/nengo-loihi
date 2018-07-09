import numpy as np
import pytest

import nengo
import nengo_loihi


@pytest.mark.parametrize('precompute', [True, False])
def test_ens_decoded_on_host(precompute, allclose, Simulator, seed, plt):
    out_synapse = nengo.synapses.Alpha(0.03)

    with nengo.Network(seed=seed) as model:
        nengo_loihi.add_params(model)

        stim = nengo.Node(lambda t: [np.sin(t * 2 * np.pi)])

        a = nengo.Ensemble(100, 1)
        model.config[a].on_chip = False

        b = nengo.Ensemble(100, 1)

        nengo.Connection(stim, a)

        nengo.Connection(a, b, function=lambda x: -x)

        p_stim = nengo.Probe(stim, synapse=out_synapse)
        p_a = nengo.Probe(a, synapse=out_synapse)
        p_b = nengo.Probe(b, synapse=out_synapse)

    with Simulator(model, precompute=precompute) as sim:
        sim.run(1.0)

    plt.plot(sim.trange(), sim.data[p_stim])
    plt.plot(sim.trange(), sim.data[p_a])
    plt.plot(sim.trange(), sim.data[p_b])

    assert allclose(sim.data[p_a], sim.data[p_stim], atol=0.05, rtol=0.01)
    assert allclose(sim.data[p_b], -sim.data[p_a], atol=0.15, rtol=0.1)


@pytest.mark.parametrize('precompute', [True, False])
def test_ens_neurons_on_host(precompute, allclose, Simulator, seed, plt):
    out_synapse = nengo.synapses.Alpha(0.03)

    n = 50

    with nengo.Network(seed=seed) as model:
        nengo_loihi.add_params(model)

        stim = nengo.Node(lambda t: [np.sin(t * 2 * np.pi)])

        a = nengo.Ensemble(n, 1)
        model.config[a].on_chip = False
        nengo.Connection(stim, a)

        b = nengo.Ensemble(n, 1)
        nengo.Connection(a.neurons, b.neurons, transform=np.eye(n))

        c = nengo.Node(size_in=1)
        nengo.Connection(b, c)

        p_stim = nengo.Probe(stim, synapse=out_synapse)
        p_a = nengo.Probe(a, synapse=out_synapse)
        p_b = nengo.Probe(b, synapse=out_synapse)
        p_c = nengo.Probe(c, synapse=out_synapse)

    with Simulator(model, precompute=precompute) as sim:
        sim.run(1.0)

    with model:
        model.config[a].on_chip = True

    with Simulator(model, precompute=precompute) as sim2:
        sim2.run(1.0)

    plt.plot(sim.trange(), sim.data[p_stim])
    plt.plot(sim.trange(), sim.data[p_a])
    plt.plot(sim.trange(), sim.data[p_b])
    plt.plot(sim.trange(), sim.data[p_c])
    plt.plot(sim2.trange(), sim2.data[p_a])
    plt.plot(sim2.trange(), sim2.data[p_b])
    plt.plot(sim2.trange(), sim.data[p_c])

    assert allclose(sim.data[p_a], sim2.data[p_a], atol=0.15)
    assert allclose(sim.data[p_b], sim2.data[p_b], atol=0.15)
    assert allclose(sim.data[p_c], sim2.data[p_c], atol=0.15)
