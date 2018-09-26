import numpy as np
import nengo
import pytest


@pytest.mark.parametrize("tau_ref", [0.001, 0.003, 0.005])
def test_lif_response_curves(tau_ref, Simulator, plt):
    n = 256
    encoders = np.ones((n, 1))
    gain = np.zeros(n)
    bias = np.linspace(1, 30, n)

    with nengo.Network() as model:
        a = nengo.Ensemble(n, 1,
                           neuron_type=nengo.LIF(tau_ref=tau_ref),
                           encoders=encoders,
                           gain=gain,
                           bias=bias)
        ap = nengo.Probe(a.neurons)

    dt = 0.001
    with Simulator(model, dt=dt) as sim:
        sim.run(1.0)

    scount = np.sum(sim.data[ap] > 0, axis=0)

    upper_bound = nengo.LIF(tau_ref=tau_ref).rates(0., gain, bias)
    lower_bound = nengo.LIF(tau_ref=tau_ref + dt).rates(0., gain, bias)
    mid = nengo.LIF(tau_ref=tau_ref + 0.5 * dt).rates(0., gain, bias)
    plt.title("tau_ref=%.3f" % tau_ref)
    plt.plot(bias, upper_bound, "k")
    plt.plot(bias, lower_bound, "k")
    plt.plot(bias, mid, "b")
    plt.plot(bias, scount, "g", label="Spike count on Loihi")
    plt.xlabel("Bias current")
    plt.ylabel("Firing rate (Hz)")
    plt.legend(loc="best")

    assert np.all(scount <= upper_bound + 1)
    assert np.all(scount >= lower_bound - 1)


def test_relu_response_curves(Simulator, plt, allclose):
    n = 256
    encoders = np.ones((n, 1))
    gain = np.zeros(n)
    bias = np.linspace(0, 50, n)

    with nengo.Network() as model:
        a = nengo.Ensemble(n, 1,
                           neuron_type=nengo.SpikingRectifiedLinear(),
                           encoders=encoders,
                           gain=gain,
                           bias=bias)
        ap = nengo.Probe(a.neurons)

    dt = 0.001
    t_final = 1.0
    with Simulator(model, dt=dt) as sim:
        sim.run(t_final)

    scount = np.sum(sim.data[ap] > 0, axis=0)
    actual = nengo.SpikingRectifiedLinear().rates(0., gain, bias)
    plt.plot(bias, actual, "b", label="Ideal")
    plt.plot(bias, scount, "g", label="Loihi")
    plt.xlabel("Bias current")
    plt.ylabel("Firing rate (Hz)")
    plt.legend(loc="best")

    assert allclose(actual, scount, atol=5)


@pytest.mark.parametrize("amplitude", (0.1, 0.5, 1))
@pytest.mark.parametrize(
    "neuron_type", (nengo.SpikingRectifiedLinear, nengo.LIF))
def test_amplitude(Simulator, amplitude, neuron_type, seed, plt, allclose):
    with nengo.Network(seed=seed) as net:
        a = nengo.Node([0.5])
        n = 100
        ens = nengo.Ensemble(
            n, 1, neuron_type=neuron_type(amplitude=amplitude))
        ens2 = nengo.Ensemble(n, 1, gain=np.ones(n), bias=np.zeros(n),
                              neuron_type=nengo.SpikingRectifiedLinear())
        nengo.Connection(a, ens)

        # note: slight boost on transform so that the post neurons are pushed
        # over threshold, rather than ==threshold
        nengo.Connection(ens.neurons, ens2.neurons, synapse=None,
                         transform=np.eye(n) * 1.02)

        node = nengo.Node(size_in=n)
        nengo.Connection(ens.neurons, node, synapse=None)

        ens_p = nengo.Probe(ens, synapse=0.1)
        neuron_p = nengo.Probe(ens.neurons)
        indirect_p = nengo.Probe(node)
        neuron2_p = nengo.Probe(ens2.neurons)

    with Simulator(net, precompute=True) as sim:
        sim.run(1)

    spikemean1 = np.mean(sim.data[neuron_p], axis=0)
    spikemean2 = np.mean(sim.data[neuron2_p], axis=0)

    plt.subplot(211)
    plt.plot(sim.trange(), sim.data[ens_p])
    plt.subplot(212)
    i = np.argsort(spikemean1)
    plt.plot(spikemean1[i])
    plt.plot(spikemean2[i], linestyle='--')

    assert allclose(sim.data[ens_p][sim.trange() > 0.9], 0.5, atol=0.05)
    assert np.max(sim.data[neuron_p]) == amplitude / sim.dt

    # the identity neuron-to-neuron connection causes `ens2` to fire at
    # `amplitude` * the firing rate of `ens` (i.e., the same overall firing
    # rate as `ens`)
    assert allclose(spikemean1, spikemean2, atol=1)

    # note: one-timestep delay, despite synapse=None
    assert allclose(sim.data[neuron_p][:-1], sim.data[indirect_p][1:])
