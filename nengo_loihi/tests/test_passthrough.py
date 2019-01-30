import pytest
import nengo
import numpy as np

import nengo_loihi
from nengo_loihi.decode_neurons import OnOffDecodeNeurons
from nengo_loihi.inputs import ChipReceiveNode
from nengo_loihi.splitter import split

default_node_neurons = OnOffDecodeNeurons()


def test_passthrough_placement():
    with nengo.Network() as model:
        stim = nengo.Node(0)
        a = nengo.Node(None, size_in=1)   # should be off-chip
        b = nengo.Ensemble(10, 1)
        c = nengo.Node(None, size_in=1)   # should be removed
        d = nengo.Node(None, size_in=1)   # should be removed
        e = nengo.Node(None, size_in=1)   # should be removed
        f = nengo.Ensemble(10, 1)
        g = nengo.Node(None, size_in=1)   # should be off-chip
        nengo.Connection(stim, a)
        nengo.Connection(a, b)
        nengo.Connection(b, c)
        nengo.Connection(c, d)
        nengo.Connection(d, e)
        nengo.Connection(e, f)
        nengo.Connection(f, g)
        nengo.Probe(g)

    nengo_loihi.add_params(model)
    networks = split(model,
                     precompute=False,
                     node_neurons=default_node_neurons,
                     node_tau=0.005,
                     remove_passthrough=True)
    chip = networks.chip
    host = networks.host

    assert a in host.nodes
    assert a not in chip.nodes
    assert c not in host.nodes
    assert c not in chip.nodes
    assert d not in host.nodes
    assert d not in chip.nodes
    assert e not in host.nodes
    assert e not in chip.nodes
    assert g in host.nodes
    assert g not in chip.nodes


@pytest.mark.parametrize("d1", [1, 3])
@pytest.mark.parametrize("d2", [1, 3])
@pytest.mark.parametrize("d3", [1, 3])
def test_transform_merging(d1, d2, d3):
    with nengo.Network() as model:
        a = nengo.Ensemble(10, d1)
        b = nengo.Node(None, size_in=d2)
        c = nengo.Ensemble(10, d3)

        t1 = np.random.uniform(-1, 1, (d2, d1))
        t2 = np.random.uniform(-1, 1, (d3, d2))

        nengo.Connection(a, b, transform=t1)
        nengo.Connection(b, c, transform=t2)

    nengo_loihi.add_params(model)
    networks = split(model,
                     precompute=False,
                     node_neurons=default_node_neurons,
                     node_tau=0.005,
                     remove_passthrough=True)
    chip = networks.chip

    assert len(chip.connections) == 1
    conn = chip.connections[0]
    assert np.allclose(conn.transform, np.dot(t2, t1))


@pytest.mark.parametrize("n_ensembles", [1, 3])
@pytest.mark.parametrize("ens_dimensions", [1, 3])
def test_identity_array(n_ensembles, ens_dimensions):
    with nengo.Network() as model:
        a = nengo.networks.EnsembleArray(10, n_ensembles, ens_dimensions)
        b = nengo.networks.EnsembleArray(10, n_ensembles, ens_dimensions)
        nengo.Connection(a.output, b.input)

    nengo_loihi.add_params(model)
    networks = split(model,
                     precompute=False,
                     node_neurons=default_node_neurons,
                     node_tau=0.005,
                     remove_passthrough=True)

    # ignore the a.input -> a.ensemble connections
    connections = [conn for conn in networks.chip.connections
                   if not (isinstance(conn.pre_obj, ChipReceiveNode)
                           and conn.post_obj in a.ensembles)]

    assert len(connections) == n_ensembles
    pre = set()
    post = set()
    for conn in connections:
        assert conn.pre in a.all_ensembles or conn.pre_obj is a.input
        assert conn.post in b.all_ensembles
        assert np.allclose(conn.transform, np.eye(ens_dimensions))
        pre.add(conn.pre)
        post.add(conn.post)
    assert len(pre) == n_ensembles
    assert len(post) == n_ensembles


@pytest.mark.parametrize("n_ensembles", [1, 3])
@pytest.mark.parametrize("ens_dimensions", [1, 3])
def test_full_array(n_ensembles, ens_dimensions):
    with nengo.Network() as model:
        a = nengo.networks.EnsembleArray(10, n_ensembles, ens_dimensions)
        b = nengo.networks.EnsembleArray(10, n_ensembles, ens_dimensions)
        D = n_ensembles * ens_dimensions
        nengo.Connection(a.output, b.input, transform=np.ones((D, D)))

    nengo_loihi.add_params(model)
    networks = split(model,
                     precompute=False,
                     node_neurons=default_node_neurons,
                     node_tau=0.005,
                     remove_passthrough=True)

    # ignore the a.input -> a.ensemble connections
    connections = [conn for conn in networks.chip.connections
                   if not (isinstance(conn.pre_obj, ChipReceiveNode)
                           and conn.post_obj in a.ensembles)]

    assert len(connections) == n_ensembles ** 2
    pairs = set()
    for conn in connections:
        assert conn.pre in a.all_ensembles
        assert conn.post in b.all_ensembles
        assert np.allclose(conn.transform,
                           np.ones((ens_dimensions, ens_dimensions)))
        pairs.add((conn.pre, conn.post))
    assert len(pairs) == n_ensembles ** 2


def test_synapse_merging(Simulator, seed):
    with nengo.Network(seed=seed) as model:
        a = nengo.networks.EnsembleArray(10, n_ensembles=2)
        b = nengo.Node(None, size_in=2)
        c = nengo.networks.EnsembleArray(10, n_ensembles=2)
        nengo.Connection(a.output[0], b[0], synapse=None)
        nengo.Connection(a.output[1], b[1], synapse=0.1)
        nengo.Connection(b[0], c.input[0], synapse=None)
        nengo.Connection(b[0], c.input[1], synapse=0.2)
        nengo.Connection(b[1], c.input[0], synapse=None)
        nengo.Connection(b[1], c.input[1], synapse=0.2)

    nengo_loihi.add_params(model)
    networks = split(model,
                     precompute=False,
                     node_neurons=default_node_neurons,
                     node_tau=0.005,
                     remove_passthrough=True)

    # ignore the a.input -> a.ensemble connections
    connections = [conn for conn in networks.chip.connections
                   if not (isinstance(conn.pre_obj, ChipReceiveNode)
                           and conn.post_obj in a.ensembles)]

    assert len(connections) == 4
    desired_filters = {
        ('0', '0'): None,
        ('0', '1'): 0.2,
        ('1', '0'): 0.1,
        ('1', '1'): 0.3,
    }
    for conn in connections:
        if desired_filters[(conn.pre.label, conn.post.label)] is None:
            assert conn.synapse is None
        else:
            assert isinstance(conn.synapse, nengo.Lowpass)
            assert np.allclose(
                conn.synapse.tau,
                desired_filters[(conn.pre.label, conn.post.label)])

    # check that model builds/runs correctly
    with Simulator(model, remove_passthrough=True) as sim:
        sim.step()


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
