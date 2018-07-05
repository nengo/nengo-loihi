import numpy as np
import pytest

import nengo
from nengo.dists import Uniform, UniformHypersphere
import nengo_loihi


@pytest.mark.parametrize('precompute', [True, False])
def test_ens_decoded_on_host(precompute, allclose, Simulator, seed, plt):
    out_synapse = nengo.synapses.Alpha(0.03)

    with nengo.Network(seed=seed) as model:
        nengo_loihi.add_params(model)

        stim = nengo.Node(lambda t: [np.sin(t*2*np.pi)])

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
def test_ens_neurons_on_host(precompute, allclose, Simulator, seed, plt, rng):
    out_synapse = nengo.synapses.Alpha(0.03)

    n = 50
    transform = 0.0125 * np.eye(n)

    # Sample the distributions early so `a` and `b` have the same params
    ens_params = dict(
        encoders=UniformHypersphere(surface=True).sample(n, d=1, rng=rng),
        max_rates=Uniform(100, 120).sample(n, rng=rng),
        intercepts=Uniform(-0.5, 0.5).sample(n, rng=rng),
    )

    with nengo.Network(seed=seed) as model:
        nengo_loihi.add_params(model)

        stim = nengo.Node(lambda t: [np.sin(t*2*np.pi)])

        a = nengo.Ensemble(n, 1, **ens_params)
        model.config[a].on_chip = False
        nengo.Connection(stim, a)

        b = nengo.Ensemble(n, 1, **ens_params)
        nengo.Connection(a.neurons, b.neurons, transform=transform)

        p_stim = nengo.Probe(stim, synapse=out_synapse)
        p_a = nengo.Probe(a, synapse=out_synapse)
        p_b = nengo.Probe(b, synapse=out_synapse)

    with Simulator(model, precompute=precompute) as sim:
        sim.run(1.0)

    plt.plot(sim.trange(), sim.data[p_stim])
    plt.plot(sim.trange(), sim.data[p_a])
    plt.plot(sim.trange(), sim.data[p_b])

    assert allclose(sim.data[p_b], sim.data[p_a], atol=0.15, rtol=0.15)


@pytest.mark.parametrize('precompute', [True, False])
def test_post_node_on_host(precompute, allclose, Simulator, seed, plt, rng):
    out_synapse = nengo.synapses.Alpha(0.03)

    n = 50
    transform = 0.0125 * np.eye(n)

    neuron_type = nengo.LIF()
    encoders = UniformHypersphere(surface=True).sample(n, d=1, rng=rng)
    max_rates = Uniform(100, 120).sample(n, rng=rng)
    intercepts = Uniform(-0.5, 0.5).sample(n, rng=rng)
    ens_params = dict(
        encoders=encoders, max_rates=max_rates, intercepts=intercepts,
        neuron_type=neuron_type)

    solver = nengo.solvers.LstsqL2()
    eval_points = UniformHypersphere(surface=False).sample(1000, d=1, rng=rng)
    gain, bias = neuron_type.gain_bias(max_rates, intercepts)
    A = neuron_type.rates(np.dot(eval_points, encoders.T), gain, bias)
    D, _ = solver(A, eval_points)

    with nengo.Network(seed=seed) as model:
        nengo_loihi.add_params(model)

        stim = nengo.Node(lambda t: [np.sin(t*2*np.pi)])

        a = nengo.Ensemble(n, 1, **ens_params)
        model.config[a].on_chip = False
        nengo.Connection(stim, a)

        b = nengo.Ensemble(n, 1, **ens_params)
        nengo.Connection(a.neurons, b.neurons, transform=transform)

        v = nengo.Node(size_in=1)
        nengo.Connection(b.neurons, v, transform=D.T)

        p_stim = nengo.Probe(stim, synapse=out_synapse)
        p_a = nengo.Probe(a, synapse=out_synapse)
        p_b = nengo.Probe(b, synapse=out_synapse)

    with Simulator(model, precompute=precompute) as sim:
        sim.run(1.0)

    plt.plot(sim.trange(), sim.data[p_stim])
    plt.plot(sim.trange(), sim.data[p_a])
    plt.plot(sim.trange(), sim.data[p_b])

    assert allclose(sim.data[p_b], sim.data[p_a], atol=0.15, rtol=0.15)
