"""Test block splitting (``split_blocks.py``).

See ``test_conv.py:test_conv_deepnet`` for a test with block splitting
in a deep convnet.
"""

import nengo
import numpy as np
import pytest

import nengo_loihi


def test_split_ensembles(Simulator, seed, rng, plt, allclose):
    b_fn = lambda x: x ** 2

    with nengo.Network(seed=seed) as net:
        nengo_loihi.add_params(net)

        u = nengo.Node(lambda t: np.sin(4 * np.pi * t))
        up = nengo.Probe(u, synapse=0.03)

        # test auto-splitting large block
        a = nengo.Ensemble(1100, 1, label="a")
        nengo.Connection(u, a, synapse=None)

        ap = nengo.Probe(a, synapse=0.03)

        # test connection between two split blocks
        b = nengo.Ensemble(400, 1, label="b", seed=seed + 1)
        nengo.Connection(a, b, function=b_fn, seed=seed + 2)
        net.config[b].block_shape = nengo_loihi.BlockShape((134,), (400,))

        bp = nengo.Probe(b, synapse=0.03)
        bp10 = nengo.Probe(b[:10], synapse=0.03)

        # have one block not be split
        c = nengo.Ensemble(400, 1, label="c", seed=seed + 1)
        nengo.Connection(a, c, function=b_fn, seed=seed + 2)

        cp = nengo.Probe(c, synapse=0.03)

        # TODO: uncomment when we allow connections to neuron slices

        # ensemble with input to not all blocks, to check synapse splitting,
        # specifically the `if len(axon_ids) == 0: continue` in `split_syanpse`.
        # However we currently don't allow connections to neuron slices, so ignore.
        # d_enc = nengo.dists.UniformHypersphere(surface=True).sample(400, d=1, rng=rng)
        # d = nengo.Ensemble(400, 1, label="d", encoders=d_enc)
        # net.config[d].block_shape = nengo_loihi.BlockShape((134,), (400,))
        # nengo.Connection(a, d.neurons[:200], transform=d_enc[:200])
        # nengo.Connection(a, d.neurons[200:], transform=d_enc[200:])

    with Simulator(net) as sim:
        sim.run(0.5)

        assert len(sim.model.objs[a]["out"]) == 2
        assert len(sim.model.objs[b]["out"]) == 3
        assert len(sim.model.objs[c]["out"]) == 1

    y = b_fn(nengo.synapses.Lowpass(0.01).filt(sim.data[up], dt=sim.dt))
    plt.plot(sim.trange(), sim.data[up], "k", label="Ideal x")
    plt.plot(sim.trange(), sim.data[ap], label="x")
    plt.plot(sim.trange(), y, "k", label="Ideal x ** 2")
    plt.plot(sim.trange(), sim.data[bp], label="x ** 2, two blocks")
    plt.plot(sim.trange(), sim.data[cp], label="x ** 2, one block")
    plt.legend()

    assert allclose(sim.data[ap], sim.data[up], atol=0.15)

    assert allclose(sim.data[bp10], sim.data[bp][:, :10])

    # b and c have same seeds, so should be very similar. However, since one
    # is split and one is not, discretizing the blocks after splitting means
    # that there will be slight numerical differences.
    assert allclose(sim.data[cp], sim.data[bp], atol=0.02)

    assert allclose(sim.data[bp], y, atol=0.2)


@pytest.mark.parametrize("n_neurons", [8, 40, 200])
def test_split_probe(allclose, n_neurons, plt, seed, Simulator):
    """Tests combining "transformed" probes that are split across multiple blocks.

    "Transformed" probes occur when probing decoded values from an ensemble.
    The values that are actually probed are the voltages from two compartments
    per dimensions on a block, one tuned positively and one negatively for each
    dimension. Therefore, in order to split the decoded values across two blocks,
    we need to probe an ensemble with more than 512 dimensions, since a block
    can contain 1024 compartments. A real ensemble (i.e., one that would represent
    a 513-dimensional signal reliably) would take a long time to build and simulate,
    so we instead use an undersized ensemble and expect it to do a poor job
    representing its input. To ensure the behavior is approximately correct,
    we compare it to the same network simulated with Nengo core.
    """

    with nengo.Network(seed=seed) as net:
        stim = nengo.Node(np.ones(513) * 0.5)
        ens = nengo.Ensemble(n_neurons, 513)
        nengo.Connection(stim, ens, synapse=None)
        probe = nengo.Probe(ens, synapse=0.02)

    with nengo.Simulator(net) as nengo_sim:
        nengo_sim.run(0.05)

    with Simulator(net) as loihi_sim:
        loihi_sim.run(nengo_sim.time)

    t = nengo_sim.trange()
    nengo_mean = np.mean(nengo_sim.data[probe], axis=1)
    loihi_mean = np.mean(loihi_sim.data[probe], axis=1)
    plt.plot(t, nengo_mean, label="Nengo core")
    plt.plot(t, loihi_mean, label="NengoLoihi")
    plt.legend()

    assert len(loihi_sim.model.objs[ens]["out"]) == (2 if n_neurons >= 200 else 1)
    assert allclose(nengo_mean, loihi_mean, atol=0.005)


@pytest.mark.parametrize(
    "probe_slice", [slice(None, None, 3), slice(0, 3), slice(None, None, 5)]
)
def test_sliced_probe(allclose, probe_slice, Simulator):
    n_neurons = 16
    # The bias should be the same for each block, but different within the block
    # to see some variety. This gives bias 0, 1, 2, 3 to the four neurons in the
    # 2 by 2 block.
    bias = np.tile(np.arange(4).reshape((2, 2)), (2, 2)).flatten()

    with nengo.Network() as net:
        e = nengo.Ensemble(n_neurons, 1, gain=np.zeros(n_neurons), bias=bias)
        p = nengo.Probe(e.neurons[probe_slice], "voltage", synapse=0.002)

    with Simulator(net) as ref_sim:
        ref_sim.run(0.01)

    ref_voltages = ref_sim.data[p]

    with net:
        nengo_loihi.add_params(net)
        net.config[e].block_shape = nengo_loihi.BlockShape((2, 2), (n_neurons // 4, 4))

    with Simulator(net) as split_sim:
        split_sim.run(ref_sim.time)

    split_voltages = split_sim.data[p]

    assert np.all(ref_sim.data[e].gain == split_sim.data[e].gain)
    assert np.all(ref_sim.data[e].bias == split_sim.data[e].bias)

    assert allclose(split_voltages, ref_voltages)
