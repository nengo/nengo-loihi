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


@pytest.mark.parametrize('seed_ens', [True, False])
@pytest.mark.parametrize('precompute', [True, False])
def test_n2n_on_host(precompute, allclose, Simulator, seed_ens, seed, plt):
    """Ensure that neuron to neuron connections work on and off chip."""

    n_neurons = 50
    # When the ensemble is seeded, the output plots will make more sense,
    # but the test should work whether they're seeded or not.
    ens_seed = (seed + 1) if seed_ens else None
    if not seed_ens:
        pytest.xfail("Seeds change when moving ensembles off/on chip")

    with nengo.Network(seed=seed) as model:
        nengo_loihi.add_params(model)

        stim = nengo.Node(lambda t: [np.sin(t * 2 * np.pi)])

        # pre receives stimulation and represents the sine wave
        pre = nengo.Ensemble(n_neurons, dimensions=1, seed=ens_seed)
        model.config[pre].on_chip = False
        nengo.Connection(stim, pre)

        # post has pre's neural activity forwarded to it.
        # Since the neuron parameters are the same, it should also represent
        # the same sine wave.
        # The 0.015 scaling is chosen so the values match visually,
        # though a more principled reason would be better.
        post = nengo.Ensemble(n_neurons, dimensions=1, seed=ens_seed)
        nengo.Connection(pre.neurons, post.neurons,
                         transform=np.eye(n_neurons) * 0.015)

        p_synapse = nengo.synapses.Alpha(0.03)
        p_stim = nengo.Probe(stim, synapse=p_synapse)
        p_pre = nengo.Probe(pre, synapse=p_synapse)
        p_post = nengo.Probe(post, synapse=p_synapse)

    with Simulator(model, precompute=precompute) as sim:
        sim.run(1.0)
    t = sim.trange()

    model.config[pre].on_chip = True

    with Simulator(model, precompute=precompute) as sim2:
        sim2.run(1.0)
    t2 = sim2.trange()

    plt.plot(t, sim.data[p_stim], c="k", label="input")
    plt.plot(t, sim.data[p_pre], label="pre off-chip")
    plt.plot(t, sim.data[p_post], label="post (pre off-chip)")
    plt.plot(t2, sim2.data[p_pre], label="pre on-chip")
    plt.plot(t2, sim2.data[p_post], label="post (pre on-chip)")
    plt.legend()

    assert allclose(sim.data[p_pre], sim2.data[p_pre], atol=0.1)
    assert allclose(sim.data[p_post], sim2.data[p_post], atol=0.1)
