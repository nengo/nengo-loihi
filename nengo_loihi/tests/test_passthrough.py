import nengo
import numpy as np
import pytest
from nengo.exceptions import BuildError

from nengo_loihi.decode_neurons import OnOffDecodeNeurons
from nengo_loihi.passthrough import PassthroughSplit
from nengo_loihi.splitter import HostChipSplit

default_node_neurons = OnOffDecodeNeurons()


def test_passthrough_placement():
    with nengo.Network() as net:
        stim = nengo.Node(0)
        a = nengo.Node(None, size_in=1)  # should be off-chip
        b = nengo.Ensemble(10, 1)
        c = nengo.Node(None, size_in=1)  # should be removed
        d = nengo.Node(None, size_in=1)  # should be removed
        e = nengo.Node(None, size_in=1)  # should be removed
        f = nengo.Ensemble(10, 1)
        g = nengo.Node(None, size_in=1)  # should be off-chip
        nengo.Connection(stim, a)
        nengo.Connection(a, b)
        conn_bc = nengo.Connection(b, c)
        conn_cd = nengo.Connection(c, d)
        conn_de = nengo.Connection(d, e)
        conn_ef = nengo.Connection(e, f)
        nengo.Connection(f, g)
        nengo.Probe(g)

    split = PassthroughSplit(net, HostChipSplit(net))

    assert split.to_remove == {c, d, e, conn_bc, conn_cd, conn_de, conn_ef}
    assert len(split.to_add) == 1
    conn = next(iter(split.to_add))
    assert conn.pre is b
    assert conn.post is f


@pytest.mark.parametrize("d1", [1, 3])
@pytest.mark.parametrize("d2", [1, 3])
@pytest.mark.parametrize("d3", [1, 3])
def test_transform_merging(d1, d2, d3):
    with nengo.Network() as net:
        a = nengo.Ensemble(10, d1)
        b = nengo.Node(None, size_in=d2)
        c = nengo.Ensemble(10, d3)

        t1 = np.random.uniform(-1, 1, (d2, d1))
        t2 = np.random.uniform(-1, 1, (d3, d2))

        conn_ab = nengo.Connection(a, b, transform=t1)
        conn_bc = nengo.Connection(b, c, transform=t2)

    split = PassthroughSplit(net, HostChipSplit(net))

    assert split.to_remove == {b, conn_ab, conn_bc}

    assert len(split.to_add) == 1
    conn = next(iter(split.to_add))
    assert np.allclose(conn.transform.init, np.dot(t2, t1))


@pytest.mark.parametrize("n_ensembles", [1, 3])
@pytest.mark.parametrize("ens_dimensions", [1, 3])
def test_identity_array(n_ensembles, ens_dimensions):
    with nengo.Network() as net:
        a = nengo.networks.EnsembleArray(10, n_ensembles, ens_dimensions)
        b = nengo.networks.EnsembleArray(10, n_ensembles, ens_dimensions)
        nengo.Connection(a.output, b.input)

    split = PassthroughSplit(net, HostChipSplit(net))

    assert len(split.to_add) == n_ensembles

    pre = set()
    post = set()
    for conn in split.to_add:
        assert conn.pre in a.all_ensembles or conn.pre_obj is a.input
        assert conn.post in b.all_ensembles
        assert np.allclose(conn.transform.init, np.eye(ens_dimensions))
        pre.add(conn.pre)
        post.add(conn.post)
    assert len(pre) == n_ensembles
    assert len(post) == n_ensembles


@pytest.mark.parametrize("n_ensembles", [1, 3])
@pytest.mark.parametrize("ens_dimensions", [1, 3])
def test_full_array(n_ensembles, ens_dimensions):
    with nengo.Network() as net:
        a = nengo.networks.EnsembleArray(10, n_ensembles, ens_dimensions)
        b = nengo.networks.EnsembleArray(10, n_ensembles, ens_dimensions)
        D = n_ensembles * ens_dimensions
        nengo.Connection(a.output, b.input, transform=np.ones((D, D)))

    split = PassthroughSplit(net, HostChipSplit(net))

    assert len(split.to_add) == n_ensembles ** 2

    pairs = set()
    for conn in split.to_add:
        assert conn.pre in a.all_ensembles
        assert conn.post in b.all_ensembles
        assert np.allclose(
            conn.transform.init, np.ones((ens_dimensions, ens_dimensions))
        )
        pairs.add((conn.pre, conn.post))
    assert len(pairs) == n_ensembles ** 2


def test_synapse_merging(Simulator, seed):
    with nengo.Network(seed=seed) as net:
        a = nengo.networks.EnsembleArray(10, n_ensembles=2)
        b = nengo.Node(None, size_in=2)
        c = nengo.networks.EnsembleArray(10, n_ensembles=2)
        nengo.Connection(a.output[0], b[0], synapse=None)
        nengo.Connection(a.output[1], b[1], synapse=0.1)
        nengo.Connection(b[0], c.input[0], synapse=None)
        nengo.Connection(b[0], c.input[1], synapse=0.2)
        nengo.Connection(b[1], c.input[0], synapse=None)
        nengo.Connection(b[1], c.input[1], synapse=0.2)

    split = PassthroughSplit(net, HostChipSplit(net))

    assert len(split.to_add) == 4

    desired_filters = {
        ("0", "0"): None,
        ("0", "1"): 0.2,
        ("1", "0"): 0.1,
        ("1", "1"): 0.3,
    }
    for conn in split.to_add:
        if desired_filters[(conn.pre.label, conn.post.label)] is None:
            assert conn.synapse is None
        else:
            assert isinstance(conn.synapse, nengo.Lowpass)
            assert np.allclose(
                conn.synapse.tau, desired_filters[(conn.pre.label, conn.post.label)]
            )

    # check that model builds/runs, and issues the warning
    with pytest.warns(UserWarning) as record:
        with Simulator(net, remove_passthrough=True) as sim:
            sim.step()

    assert any("Combining two Lowpass synapses" in r.message.args[0] for r in record)


def test_no_input(Simulator, seed, allclose):
    # check that remove_passthrough does not change the behaviour of the
    # network

    with nengo.Network(seed=seed) as net:
        a = nengo.Node(size_in=1)
        b = nengo.Ensemble(200, 1)
        c = nengo.Node(size_in=1)
        nengo.Connection(a, b, synapse=None)
        nengo.Connection(b, c, synapse=None)
        p = nengo.Probe(c)

    with Simulator(net, remove_passthrough=False) as sim_base:
        sim_base.run_steps(100)

    with Simulator(net, remove_passthrough=True) as sim_remove:
        sim_remove.run_steps(100)

    assert allclose(sim_base.data[p], sim_remove.data[p])


def test_transform_errors(Simulator):
    def make_net(transform1=1.0, transform2=1.0):
        with nengo.Network() as net:
            a = nengo.Ensemble(3, 2)
            q = nengo.Node(size_in=2)
            b = nengo.Ensemble(4, 2)
            nengo.Connection(a, q, transform=transform1)
            nengo.Connection(q, b, transform=transform2)

        return net

    net = make_net(transform1=[1, 1])
    with pytest.raises(BuildError, match="transform"):
        with Simulator(net, remove_passthrough=True):
            pass

    net = make_net(transform2=[1, 1])
    with pytest.raises(BuildError, match="transform"):
        with Simulator(net, remove_passthrough=True):
            pass

    net = make_net()
    with pytest.warns(UserWarning, match="synapse"):
        with Simulator(net, remove_passthrough=True):
            pass


def test_cluster_errors(Simulator, seed, plt):
    """Test that situations with ClusterErrors keep passthrough nodes"""
    simtime = 0.2

    def make_net(learn_error=False, loop=False):
        probes = {}
        with nengo.Network(seed=seed) as net:
            u = nengo.Node(lambda t: -(np.sin((2 * np.pi / simtime) * t)))
            a = nengo.Ensemble(50, 1)
            q = nengo.Node(size_in=1, label="q")
            b = nengo.Ensemble(50, 1)

            nengo.Connection(u, a, synapse=None)
            nengo.Connection(a, q)
            nengo.Connection(q, b)

            if learn_error:
                ab = nengo.Connection(a, b, learning_rule_type=nengo.PES())
                nengo.Connection(q, ab.learning_rule)

            if loop:
                p = nengo.Node(size_in=1, label="p")
                nengo.Connection(q, p)
                nengo.Connection(p, q, transform=0.5)

            probes["b"] = nengo.Probe(b, synapse=0.02)

        return net, probes

    # Since `PassthroughSplit` catches its own cluster errors, we won't see
    # the error here. We ensure identical behaviour (so nodes are not removed).

    # Test learning rule node input
    net, probes = make_net(learn_error=True)
    with Simulator(net, remove_passthrough=False) as sim0:
        sim0.run(simtime)

    with Simulator(net, remove_passthrough=True) as sim1:
        sim1.run(simtime)

    y0_learn_error = sim0.data[probes["b"]]
    y1_learn_error = sim1.data[probes["b"]]
    plt.subplot(211)
    plt.plot(sim0.trange(), y0_learn_error)
    plt.plot(sim1.trange(), y1_learn_error)

    # Test loop
    net, probes = make_net(loop=True)
    with Simulator(net, remove_passthrough=False) as sim0:
        sim0.run(simtime)

    with Simulator(net, remove_passthrough=True) as sim1:
        sim1.run(simtime)

    y0_loop = sim0.data[probes["b"]]
    y1_loop = sim1.data[probes["b"]]
    plt.subplot(212)
    plt.plot(sim0.trange(), y0_loop)
    plt.plot(sim1.trange(), y1_loop)

    assert np.allclose(y1_learn_error, y0_learn_error)
    assert np.allclose(y1_loop, y0_loop)
