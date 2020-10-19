import nengo
import numpy as np
import pytest

from nengo_loihi import BlockShape, add_params


def test_spike_units(Simulator, seed):
    with nengo.Network(seed=seed) as model:
        a = nengo.Ensemble(100, 1)
        p = nengo.Probe(a.neurons)
    with Simulator(model) as sim:
        sim.run(0.1)

    values = np.unique(sim.data[p])
    assert values[0] == 0
    assert values[1] == int(1.0 / sim.dt)
    assert len(values) == 2


@pytest.mark.parametrize("dim", [1, 3])
def test_voltage_decode(allclose, Simulator, seed, plt, dim):
    with nengo.Network(seed=seed) as model:
        stim = nengo.Node(lambda t: [np.sin(2 * np.pi * t) / np.sqrt(dim)] * dim)
        p_stim = nengo.Probe(stim, synapse=0.01)

        a = nengo.Ensemble(100 * 3, dim, intercepts=nengo.dists.Uniform(-0.95, 0.95))
        nengo.Connection(stim, a)

        p_a = nengo.Probe(a, synapse=0.01)

    with Simulator(model, precompute=True) as sim:
        sim.run(1.0)

    plt.plot(sim.trange(), sim.data[p_a])
    plt.plot(sim.trange(), sim.data[p_stim])

    assert allclose(sim.data[p_stim], sim.data[p_a], atol=0.3)


def test_repeated_probes(Simulator):
    with nengo.Network() as net:
        ens = nengo.Ensemble(1024, 1)
        nengo.Probe(ens.neurons)

    for _ in range(5):
        with Simulator(net) as sim:
            sim.run(0.1)


@pytest.mark.filterwarnings("ignore:Model is precomputable.")
@pytest.mark.parametrize("precompute", [True, False])
@pytest.mark.parametrize("probe_target", ["input", "voltage"])
def test_neuron_probes(precompute, probe_target, Simulator, seed, plt, allclose):
    simtime = 0.3

    with nengo.Network(seed=seed) as model:
        stim = nengo.Node(lambda t: [np.sin(t * 2 * np.pi / simtime)])

        a = nengo.Ensemble(
            1,
            1,
            neuron_type=nengo.LIF(min_voltage=-1),
            encoders=nengo.dists.Choice([[1]]),
            max_rates=nengo.dists.Choice([100]),
            intercepts=nengo.dists.Choice([0.0]),
        )
        nengo.Connection(stim, a, synapse=None)

        p_stim = nengo.Probe(stim, synapse=0.005)
        p_neurons = nengo.Probe(a.neurons, probe_target)

        probe_synapse = nengo.Alpha(0.01)
        p_stim_f = nengo.Probe(
            stim, synapse=probe_synapse.combine(nengo.Lowpass(0.005))
        )
        p_neurons_f = nengo.Probe(a.neurons, probe_target, synapse=probe_synapse)

    with Simulator(model, precompute=precompute) as sim:
        sim.run(simtime)

    scale = float(sim.data[p_neurons].max())
    t = sim.trange()
    x = sim.data[p_stim]
    xf = sim.data[p_stim_f]
    y = sim.data[p_neurons] / scale
    yf = sim.data[p_neurons_f] / scale
    plt.plot(t, x, label="stim")
    plt.plot(t, xf, label="stim filt")
    plt.plot(t, y, label="loihi")
    plt.plot(t, yf, label="loihi filt")
    plt.legend()

    if probe_target == "input":
        # shape of current input should roughly match stimulus
        assert allclose(y, x, atol=0.4, rtol=0)  # noisy, so rough match
        assert allclose(yf, xf, atol=0.05, rtol=0)  # tight match
    elif probe_target == "voltage":
        # check for voltage fluctuations (spiking) when stimulus is positive,
        # and negative voltage when stimulus is most negative
        spos = (t > 0.1 * simtime) & (t < 0.4 * simtime)
        assert allclose(yf[spos], 0.5, atol=0.1, rtol=0.1)
        assert y[spos].std() > 0.25

        sneg = (t > 0.7 * simtime) & (t < 0.9 * simtime)
        assert np.all(y[sneg] < 0)


def test_neuron_probe_with_synapse(Simulator, seed, allclose):
    synapse = nengo.Lowpass(0.01)
    with nengo.Network(seed=seed) as net:
        ens = nengo.Ensemble(10, 1)
        p_nosynapse = nengo.Probe(ens.neurons, synapse=None)
        p_synapse = nengo.Probe(ens.neurons, synapse=synapse)

    with Simulator(net) as sim:
        sim.run(0.1)

    assert allclose(sim.data[p_synapse], synapse.filt(sim.data[p_nosynapse]))


@pytest.mark.parametrize("precompute", [True, False])
def test_probe_filter_twice(precompute, plt, seed, Simulator):
    with nengo.Network(seed=seed) as net:
        stim = nengo.Node([1])
        ens = nengo.Ensemble(100, 1)
        probe = nengo.Probe(ens, synapse=0.01)
        nengo.Connection(stim, ens)

    with Simulator(net, precompute=precompute) as sim0:
        sim0.run(0.04)

    with Simulator(net, precompute=precompute) as sim1:
        sim1.run(0.02)
        sim1.run(0.02)

    plt.plot(sim0.trange(), sim0.data[probe])
    plt.plot(sim1.trange(), sim1.data[probe])

    assert np.all(sim0.data[probe] == sim1.data[probe])


def test_probe_split_blocks(Simulator, seed, plt):
    n_neurons = 80
    gain = np.ones(n_neurons)
    bias = np.linspace(0, 20, n_neurons)
    simtime = 0.2

    with nengo.Network(seed=seed) as net:
        ens = nengo.Ensemble(n_neurons, 1, gain=gain, bias=bias)

        probe = nengo.Probe(ens.neurons)

        probe1_slice = slice(3, 33)
        probe1 = nengo.Probe(ens.neurons[probe1_slice])

        probe2_slice = slice(7, 52, 3)
        probe2 = nengo.Probe(ens.neurons[probe2_slice])

        probe3_slice = [2, 5, 17, 21, 36, 49, 52, 69, 73]  # randomly chosen inds
        probe3 = nengo.Probe(ens.neurons[probe3_slice])

    # run without splitting ensemble
    with Simulator(net) as sim1:
        assert len(sim1.model.blocks) == 1
        sim1.run(simtime)

    # run with splitting ensemble
    with net:
        add_params(net)
        net.config[ens].block_shape = BlockShape((5, 4), (10, 8))

    with Simulator(net) as sim2:
        assert len(sim2.model.blocks) == 4
        sim2.run(simtime)

    for k, sim in enumerate((sim1, sim2)):
        plt.subplot(2, 1, k + 1)
        plt.plot(bias, sim.data[probe].mean(axis=0))
        plt.plot(bias[probe1_slice], sim.data[probe1].mean(axis=0))
        plt.plot(bias[probe2_slice], sim.data[probe2].mean(axis=0), ".")
        plt.plot(bias[probe3_slice], sim.data[probe3].mean(axis=0), "x")

    # ensure rates increase and not everything is zero
    for sim in (sim1, sim2):
        diffs = np.diff(sim.data[probe].mean(axis=0))
        assert (diffs >= 0).all() and (diffs > 1).sum() > 10

    # ensure slices match unsliced probe
    for sim in (sim1, sim2):
        assert np.array_equal(sim.data[probe1], sim.data[probe][:, probe1_slice])
        assert np.array_equal(sim.data[probe2], sim.data[probe][:, probe2_slice])
        assert np.array_equal(sim.data[probe3], sim.data[probe][:, probe3_slice])

    # ensure split and unsplit simulators match
    for p in (probe, probe1, probe2, probe3):
        assert np.array_equal(sim1.data[p], sim2.data[p])


def piecewise_net(n_pres, pres_time, seed):
    values = np.linspace(-1, 1, n_pres)
    with nengo.Network(seed=seed) as net:
        add_params(net)
        inp = nengo.Node(nengo.processes.PresentInput(values, pres_time), size_out=1)
        ens = nengo.Ensemble(100, 1)
        nengo.Connection(inp, ens)

        net.probe = nengo.Probe(ens, synapse=nengo.Alpha(0.01))
        node = nengo.Node(size_in=1)
        nengo.Connection(ens, node, synapse=nengo.Alpha(0.01))
        net.node_probe = nengo.Probe(node)

    return net, values


@pytest.mark.parametrize("precompute", [False, True])
def test_clear_probes(Simulator, seed, plt, allclose, precompute):
    n_pres = 5
    pres_time = 0.1
    net, values = piecewise_net(n_pres, pres_time, seed)

    outputs = {"probe": [], "node": []}
    with Simulator(net, precompute=precompute) as sim:
        for _ in range(n_pres):
            sim.clear_probes()
            sim.run(pres_time)
            outputs["probe"].append(np.copy(np.squeeze(sim.data[net.probe], axis=-1)))
            outputs["node"].append(
                np.copy(np.squeeze(sim.data[net.node_probe], axis=-1))
            )

    for key, output in outputs.items():
        for i, line in enumerate(output):
            plt.plot(
                line,
                color="b" if key == "probe" else "r",
                label=key if i == 0 else None,
            )
    plt.legend(loc="best")

    expected_shape = (int(pres_time / 0.001),)
    for key, output in outputs.items():
        # check that results from each presentation have the same shape
        assert all(
            x.shape == expected_shape for x in output
        ), f"shapes: {[x.shape for x in output]}"

        # check that the last-timestep values match the presented values
        # within neural tolerances
        assert allclose([x[-1] for x in output], values, atol=0.05, rtol=0.03)

    # node output is delayed, since it has DecodeNeurons, hence higher tolerance
    assert allclose(outputs["node"], outputs["probe"], atol=0.15)


def test_probe_precompute(Simulator, seed, allclose):
    n_pres = 5
    pres_time = 0.1
    net, _ = piecewise_net(n_pres, pres_time, seed)
    with Simulator(net, precompute=False) as sim:
        sim.run(pres_time * n_pres)
    with Simulator(net, precompute=True) as sim_precompute:
        sim_precompute.run(pres_time * n_pres)

    assert allclose(sim.data[net.probe], sim_precompute.data[net.probe])
    assert allclose(sim.data[net.node_probe], sim_precompute.data[net.node_probe])
