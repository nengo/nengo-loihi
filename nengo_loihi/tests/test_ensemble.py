import nengo
import numpy as np
import pytest
from nengo.exceptions import BuildError

from nengo_loihi.builder import Model
from nengo_loihi.neurons import nengo_rates


@pytest.mark.parametrize("tau_ref", [0.001, 0.003, 0.005])
def test_lif_response_curves(tau_ref, Simulator, plt):
    n = 256
    encoders = np.ones((n, 1))
    gain = np.zeros(n)
    bias = np.linspace(1, 30, n)

    with nengo.Network() as model:
        a = nengo.Ensemble(
            n,
            1,
            neuron_type=nengo.LIF(tau_ref=tau_ref),
            encoders=encoders,
            gain=gain,
            bias=bias,
        )
        ap = nengo.Probe(a.neurons)

    dt = 0.001
    with Simulator(model, dt=dt) as sim:
        sim.run(1.0)

    scount = np.sum(sim.data[ap] > 0, axis=0)

    def rates(tau_ref, gain=gain, bias=bias):
        lif = nengo.LIF(tau_ref=tau_ref)
        return nengo_rates(lif, 0.0, gain, bias).squeeze(axis=0)

    upper_bound = rates(tau_ref=tau_ref)
    lower_bound = rates(tau_ref=tau_ref + dt)
    mid = rates(tau_ref=tau_ref + 0.5 * dt)
    plt.title("tau_ref=%.3f" % tau_ref)
    plt.plot(bias, upper_bound, "k")
    plt.plot(bias, lower_bound, "k")
    plt.plot(bias, mid, "b")
    plt.plot(bias, scount, "g", label="Spike count on Loihi")
    plt.xlabel("Bias current")
    plt.ylabel("Firing rate (Hz)")
    plt.legend(loc="best")

    assert scount.shape == upper_bound.shape == lower_bound.shape
    assert np.all(scount <= upper_bound + 1)
    assert np.all(scount >= lower_bound - 1)


def test_relu_response_curves(Simulator, plt, allclose):
    n = 256
    encoders = np.ones((n, 1))
    gain = np.zeros(n)
    bias = np.linspace(0, 50, n)

    with nengo.Network() as model:
        a = nengo.Ensemble(
            n,
            1,
            neuron_type=nengo.SpikingRectifiedLinear(),
            encoders=encoders,
            gain=gain,
            bias=bias,
        )
        ap = nengo.Probe(a.neurons)

    dt = 0.001
    t_final = 1.0
    with Simulator(model, dt=dt) as sim:
        sim.run(t_final)

    scount = np.sum(sim.data[ap] > 0, axis=0)
    actual = nengo.SpikingRectifiedLinear().rates(0.0, gain, bias)
    plt.plot(bias, actual, "b", label="Ideal")
    plt.plot(bias, scount, "g", label="Loihi")
    plt.xlabel("Bias current")
    plt.ylabel("Firing rate (Hz)")
    plt.legend(loc="best")

    assert allclose(actual, scount, atol=5)


@pytest.mark.parametrize("amplitude", (0.1, 0.5, 1))
@pytest.mark.parametrize("neuron_type", (nengo.SpikingRectifiedLinear, nengo.LIF))
def test_amplitude(Simulator, amplitude, neuron_type, seed, plt, allclose):
    with nengo.Network(seed=seed) as net:
        a = nengo.Node([0.5])
        n = 100
        ens = nengo.Ensemble(n, 1, neuron_type=neuron_type(amplitude=amplitude))
        ens2 = nengo.Ensemble(
            n,
            1,
            gain=np.ones(n),
            bias=np.zeros(n),
            neuron_type=nengo.SpikingRectifiedLinear(),
        )
        nengo.Connection(a, ens)

        # note: slight boost on transform so that the post neurons are pushed
        # over threshold, rather than ==threshold
        nengo.Connection(
            ens.neurons, ens2.neurons, synapse=None, transform=np.eye(n) * 1.02
        )

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
    plt.plot(spikemean2[i], linestyle="--")

    assert allclose(sim.data[ens_p][sim.trange() > 0.9], 0.5, atol=0.05)
    assert np.max(sim.data[neuron_p]) == amplitude / sim.dt

    # the identity neuron-to-neuron connection causes `ens2` to fire at
    # `amplitude` * the firing rate of `ens` (i.e., the same overall firing
    # rate as `ens`)
    assert allclose(spikemean1, spikemean2, atol=1)

    # note: one-timestep delay, despite synapse=None
    assert allclose(sim.data[neuron_p][:-1], sim.data[indirect_p][1:])


def test_bad_gain_error(Simulator):
    with nengo.Network() as net:
        nengo.Ensemble(5, 1, intercepts=nengo.dists.Choice([2.0]))

    model = Model()
    model.intercept_limit = 10.0
    with pytest.raises(BuildError, match="negative.*gain"):
        with Simulator(net, model=model):
            pass


def test_neuron_build_errors(Simulator):
    # unsupported neuron type
    with nengo.Network() as net:
        nengo.Ensemble(5, 1, neuron_type=nengo.neurons.Sigmoid(tau_ref=0.005))

    with pytest.raises(BuildError, match="type 'Sigmoid' cannot be simulated"):
        with Simulator(net):
            pass

    # unsupported RegularSpiking type
    with nengo.Network() as net:
        nengo.Ensemble(
            5, 1, neuron_type=nengo.RegularSpiking(nengo.Sigmoid(tau_ref=0.005))
        )

    with pytest.raises(BuildError, match="RegularSpiking.*'Sigmoid'.*cannot be simu"):
        with Simulator(net):
            pass

    # amplitude with RegularSpiking base type
    with nengo.Network() as net:
        nengo.Ensemble(
            5, 1, neuron_type=nengo.RegularSpiking(nengo.LIFRate(amplitude=0.5))
        )

    with pytest.raises(BuildError, match="Amplitude is not supported on RegularSpikin"):
        with Simulator(net):
            pass

    # non-zero initial voltage warning
    with nengo.Network() as net:
        nengo.Ensemble(
            5,
            1,
            neuron_type=nengo.LIF(initial_state={"voltage": nengo.dists.Uniform(0, 1)}),
        )

    with pytest.warns(Warning, match="initial values for 'voltage' being non-zero"):
        with Simulator(net):
            pass


@pytest.mark.parametrize("radius", [0.01, 1, 100])
def test_radius_probe(Simulator, seed, radius):
    with nengo.Network(seed=seed) as model:
        stim = nengo.Node(radius / 2.0)
        ens = nengo.Ensemble(
            n_neurons=100,
            dimensions=1,
            radius=radius,
            intercepts=nengo.dists.Uniform(-0.95, 0.95),
        )
        nengo.Connection(stim, ens)
        p = nengo.Probe(ens, synapse=0.1)
    with Simulator(model, precompute=True) as sim:
        sim.run(0.5)

    assert np.allclose(sim.data[p][-1:], radius / 2.0, rtol=0.1)


@pytest.mark.parametrize("radius1", [0.01, 100])
@pytest.mark.parametrize("radius2", [0.01, 100])
@pytest.mark.parametrize("weights", [True, False])
def test_radius_ens_ens(Simulator, seed, radius1, radius2, weights):
    with nengo.Network(seed=seed) as model:
        stim = nengo.Node(radius1 / 2.0)
        a = nengo.Ensemble(
            n_neurons=100,
            dimensions=1,
            radius=radius1,
            intercepts=nengo.dists.Uniform(-0.95, 0.95),
        )
        b = nengo.Ensemble(
            n_neurons=100,
            dimensions=1,
            radius=radius2,
            intercepts=nengo.dists.Uniform(-0.95, 0.95),
        )
        nengo.Connection(stim, a)
        nengo.Connection(
            a,
            b,
            synapse=0.01,
            transform=radius2 / radius1,
            solver=nengo.solvers.LstsqL2(weights=weights),
        )
        p = nengo.Probe(b, synapse=0.1)
    with Simulator(model, precompute=True) as sim:
        sim.run(0.4)

    assert np.allclose(sim.data[p][-1:], radius2 / 2.0, rtol=0.2)
