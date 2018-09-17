import nengo
from nengo.utils.matplotlib import rasterplot
import numpy as np
import pytest

import nengo_loihi


@pytest.mark.parametrize('weight_solver', [False, True])
@pytest.mark.parametrize('target_value', [-0.75, 0.4, 1.0])
def test_ens_ens_constant(
        allclose, weight_solver, target_value, Simulator, seed, plt):
    a_fn = lambda x: x + target_value
    solver = nengo.solvers.LstsqL2(weights=weight_solver)

    bnp = None
    with nengo.Network(seed=seed) as model:
        a = nengo.Ensemble(100, 1, label='a')

        b = nengo.Ensemble(101, 1, label='b')
        nengo.Connection(a, b, function=a_fn, solver=solver)
        bp = nengo.Probe(b)
        bnp = nengo.Probe(b.neurons)

        c = nengo.Ensemble(1, 1, label='c')
        bc_conn = nengo.Connection(b, c)

    with Simulator(model) as sim:
        sim.run(1.0)

    t = sim.trange()
    bcount = (sim.data[bnp] > 0).mean(axis=0)

    b_decoders = sim.data[bc_conn].weights
    dec_value = np.dot(b_decoders, bcount)

    output_filter = nengo.synapses.Alpha(0.03)
    target_output = target_value * np.ones_like(t)
    sim_output = output_filter.filt(sim.data[bp])
    plt.plot(t, target_output, 'k')
    plt.plot(t, sim_output)

    assert allclose(dec_value, target_value, rtol=0.1, atol=0.1)
    t_check = t > 0.5
    assert allclose(
        sim_output[t_check], target_output[t_check], rtol=0.15, atol=0.15)


@pytest.mark.parametrize('precompute', [True, False])
def test_node_to_neurons(precompute, allclose, Simulator, plt):
    tfinal = 1.0

    x = np.array([0.7, 0.3])
    A = np.array([[1, 1],
                  [1, -1],
                  [1, -0.5]])
    y = np.dot(A, x)

    gain = [3] * len(y)
    bias = [0] * len(y)

    neuron_type = nengo.LIF()
    z = neuron_type.rates(y, gain, bias)

    with nengo.Network() as model:
        u = nengo.Node(x, label='u')
        a = nengo.Ensemble(len(y), 1, label='a',
                           neuron_type=neuron_type, gain=gain, bias=bias)
        ap = nengo.Probe(a.neurons)
        nengo.Connection(u, a.neurons, synapse=None, transform=A)

    with Simulator(model, precompute=precompute) as sim:
        sim.run(tfinal)

    tsum = 0.5
    t = sim.trange()
    rates = (sim.data[ap][t > t[-1] - tsum] > 0).sum(axis=0) / tsum

    bar_width = 0.35
    plt.bar(np.arange(len(z)), z, bar_width, color='k', label='z')
    plt.bar(np.arange(len(z)) + bar_width, rates, bar_width, label='rates')
    plt.legend(loc='best')

    assert allclose(rates, z, atol=3, rtol=0.1)


@pytest.mark.parametrize("factor", [0.11, 0.26, 0.51, 1.01])
def test_neuron_to_neuron(Simulator, factor, seed, allclose):
    # note: we use these weird factor values so that voltages don't line up
    # exactly with the firing threshold.  since loihi neurons fire when
    # voltage > threshold (rather than >=), if the voltages line up
    # exactly then we need an extra spike each time to push `b` over threshold

    with nengo.Network(seed=seed) as net:
        n = 10
        stim = nengo.Node(lambda t: [np.sin(t * 2 * np.pi)])
        a = nengo.Ensemble(n, 1)
        nengo.Connection(stim, a, synapse=None)

        b = nengo.Ensemble(n, 1, neuron_type=nengo.SpikingRectifiedLinear(),
                           gain=np.ones(n), bias=np.zeros(n))
        nengo.Connection(a.neurons, b.neurons, synapse=None,
                         transform=np.eye(n) * factor)

        p_a = nengo.Probe(a.neurons)
        p_b = nengo.Probe(b.neurons)

    with Simulator(net) as sim:
        sim.run(1.0)

    assert allclose(np.sum(sim.data[p_b] > 0, axis=0),
                    np.floor(np.sum(sim.data[p_a] > 0, axis=0) * factor),
                    atol=1)


def test_ensemble_to_neurons(Simulator, seed, allclose, plt):
    with nengo.Network(seed=seed) as net:
        stim = nengo.Node(lambda t: [np.sin(t * 2 * np.pi)])
        pre = nengo.Ensemble(20, 1)
        nengo.Connection(stim, pre, synapse=None)

        post = nengo.Ensemble(2, 1,
                              gain=[1., 1.], bias=[0., 0.])

        # On and off neurons
        nengo.Connection(pre, post.neurons,
                         synapse=None, transform=[[5], [-5]])

        p_pre = nengo.Probe(pre, synapse=nengo.synapses.Alpha(0.03))
        p_post = nengo.Probe(post.neurons)

    # Compare to Nengo
    with nengo.Simulator(net) as nengosim:
        nengosim.run(1.0)

    with Simulator(net) as sim:
        sim.run(1.0)

    t = sim.trange()
    plt.subplot(2, 1, 1)
    plt.title("Reference Nengo")
    plt.plot(t, nengosim.data[p_pre], c='k')
    plt.ylabel("Decoded pre value")
    plt.xlabel("Time (s)")
    plt.twinx()
    rasterplot(t, nengosim.data[p_post])
    plt.ylabel("post neuron number")
    plt.subplot(2, 1, 2)
    plt.title("Nengo Loihi")
    plt.plot(t, sim.data[p_pre], c='k')
    plt.ylabel("Decoded pre value")
    plt.xlabel("Time (s)")
    plt.twinx()
    rasterplot(t, sim.data[p_post])
    plt.ylabel("post neuron number")

    plt.tight_layout()

    # Compare the number of spikes for each neuron.
    # We'll let them be off by 5 for now.
    assert allclose(np.sum(sim.data[p_post], axis=0) * sim.dt,
                    np.sum(nengosim.data[p_post], axis=0) * nengosim.dt,
                    atol=5)


@pytest.mark.parametrize('pre_on_chip', [True, False])
def test_neurons_to_ensemble(pre_on_chip, Simulator, seed, rng, allclose, plt):
    with nengo.Network(seed=seed) as net:
        nengo_loihi.add_params(net)

        stim = nengo.Node(lambda t: [np.sin(t * 2 * np.pi)])

        pre_max_rate = 100
        n_pre = 50
        pre = nengo.Ensemble(n_pre, 1,
                             max_rates=pre_max_rate * np.ones(n_pre),
                             intercepts=np.linspace(-1, 0.9, n_pre))
        net.config[pre].on_chip = pre_on_chip
        nengo.Connection(stim, pre, synapse=None)

        pre_max = pre_max_rate * n_pre
        n_post = 51
        post_encoders = np.ones((n_post, n_pre))
        post = nengo.Ensemble(n_post, n_pre,
                              radius=pre_max,
                              encoders=post_encoders,
                              normalize_encoders=False,
                              max_rates=nengo.dists.Uniform(100, 150),
                              intercepts=nengo.dists.Uniform(0, 0.5),
                              )

        nengo.Connection(pre.neurons, post, synapse=0.005)

        p_pre = nengo.Probe(pre, synapse=nengo.synapses.Alpha(0.03))
        p_post = nengo.Probe(post, synapse=nengo.synapses.Alpha(0.03))

    with nengo.Simulator(net) as nengosim:
        nengosim.run(1.0)

    with Simulator(net, precompute=False) as sim:
        sim.run(1.0)

    y0 = nengosim.data[p_post].sum(axis=1)
    y1 = sim.data[p_post].sum(axis=1)

    t = sim.trange()
    plt.subplot(2, 1, 1)
    plt.plot(t, nengosim.data[p_pre], c='k')
    plt.plot(t, sim.data[p_pre], c='g')
    plt.ylabel("Decoded pre value")
    plt.xlabel("Time (s)")

    plt.subplot(2, 1, 2)
    plt.plot(t, y0, c='k')
    plt.plot(t, y1, c='g')
    plt.ylabel("Decoded post value")
    plt.xlabel("Time (s)")

    assert allclose(y1, y0, rtol=1e-1, atol=0.05 * y0.max())
