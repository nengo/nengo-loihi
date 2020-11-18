import nengo
import numpy as np
import pytest
from nengo.exceptions import BuildError

from nengo_loihi.config import add_params
from nengo_loihi.splitter import Split


def test_place_nodes():
    # all nodes go on the host
    # ChipReceiveNodes and HostSendNodes are created later by the builder

    with nengo.Network() as net:
        offchip1 = nengo.Node(0)
        with nengo.Network():
            offchip2 = nengo.Node(np.sin)
            ensemble = nengo.Ensemble(100, 1)
            offchip3 = nengo.Node(size_in=1)
            nengo.Connection(ensemble, offchip3)

    with nengo.Network():
        nowhere = nengo.Node(0)

    split = Split(net)
    assert not split.on_chip(offchip1)
    assert not split.on_chip(offchip2)
    assert not split.on_chip(offchip3)

    with pytest.raises(BuildError, match="not a part of the network"):
        split.on_chip(nowhere)


def test_place_ensembles():
    # builder will move the learning stuff onto the host

    with nengo.Network() as net:
        add_params(net)
        offchip = nengo.Ensemble(10, 1, label="offchip")
        net.config[offchip].on_chip = False
        direct = nengo.Ensemble(1, 1, neuron_type=nengo.Direct(), label="direct")
        with nengo.Network():
            onchip = nengo.Ensemble(20, 1, label="onchip")
        pre = nengo.Ensemble(10, 1, label="pre")
        post = nengo.Ensemble(10, 1, label="post")
        error = nengo.Ensemble(10, 1, label="error")
        conn = nengo.Connection(pre, post, learning_rule_type=nengo.PES())
        nengo.Connection(error, conn.learning_rule)

    split = Split(net)
    assert not split.on_chip(offchip)
    assert not split.on_chip(direct)
    assert split.on_chip(onchip)
    assert split.on_chip(pre)
    assert not split.on_chip(post)
    assert not split.on_chip(error)

    for obj in net.all_ensembles + net.all_nodes:
        assert not split.precomputable(obj)

    with pytest.raises(BuildError, match="Locations are only established"):
        split.on_chip(conn)


def test_place_internetwork_connections():
    with nengo.Network() as net:
        add_params(net)
        offchip = nengo.Ensemble(10, 1)
        net.config[offchip].on_chip = False
        onchip = nengo.Ensemble(10, 1)

        onon = nengo.Connection(onchip, onchip)
        onoff = nengo.Connection(onchip, offchip)
        offon = nengo.Connection(offchip, onchip)
        offoff = nengo.Connection(offchip, offchip)

    split = Split(net)

    assert split.on_chip(onon.pre)
    assert split.on_chip(onon.post)

    assert split.on_chip(onoff.pre)
    assert not split.on_chip(onoff.post)

    assert not split.on_chip(offon.pre)
    assert split.on_chip(offon.post)

    assert not split.on_chip(offoff.pre)
    assert not split.on_chip(offoff.post)


def test_split_host_to_learning_rule():
    with nengo.Network() as net:
        add_params(net)
        pre = nengo.Ensemble(10, 1, label="pre")
        post = nengo.Ensemble(10, 1, label="post")
        err_onchip = nengo.Ensemble(10, 1, label="err_onchip")
        err_offchip = nengo.Ensemble(10, 1, label="err_offchip")
        net.config[err_offchip].on_chip = False
        ens_conn = nengo.Connection(pre, post, learning_rule_type=nengo.PES())
        neurons_conn = nengo.Connection(
            pre.neurons, post.neurons, transform=1.0, learning_rule_type=nengo.PES()
        )
        nengo.Connection(err_onchip, ens_conn.learning_rule)
        nengo.Connection(err_onchip, neurons_conn.learning_rule)
        nengo.Connection(err_offchip, ens_conn.learning_rule)
        nengo.Connection(err_offchip, neurons_conn.learning_rule)

    split = Split(net)

    assert split.on_chip(pre)
    assert not split.on_chip(post)

    assert not split.on_chip(err_onchip)
    assert not split.on_chip(err_offchip)


def test_precompute_host_to_learning_rule_unsupported():
    with nengo.Network() as net:
        pre = nengo.Ensemble(10, 1, label="pre")
        post = nengo.Ensemble(10, 1, label="post")
        nengo.Connection(pre, post, learning_rule_type=nengo.PES())

    with pytest.raises(BuildError, match="learning rules"):
        Split(net, precompute=True)


def test_place_probes():
    with nengo.Network() as net:
        add_params(net)
        offchip1 = nengo.Node(0)
        with nengo.Network():
            onchip1 = nengo.Ensemble(10, 1)
            offchip2 = nengo.Ensemble(10, 1)
            net.config[offchip2].on_chip = False
        onchip2 = nengo.Ensemble(10, 1)
        nengo.Connection(onchip1, onchip2)
        nengo.Connection(offchip1, offchip2)
        offchip_probes = [nengo.Probe(offchip1), nengo.Probe(offchip2)]
        onchip_probes = [nengo.Probe(onchip1), nengo.Probe(onchip2)]

    split = Split(net)
    assert split.on_chip(onchip1)
    assert split.on_chip(onchip2)
    assert not split.on_chip(offchip1)
    assert not split.on_chip(offchip2)
    assert not any(split.on_chip(p) for p in offchip_probes)
    assert all(split.on_chip(p) for p in onchip_probes)


def test_split_pre_from_host():
    with nengo.Network() as net:
        add_params(net)
        pre_1 = nengo.Node(0, label="pre_1")
        pre_2 = nengo.Ensemble(10, 1, label="pre_2")
        pre_3 = nengo.Node(size_in=1, label="pre_3")
        pre_4 = nengo.Ensemble(1, 1, label="pre_4")
        pre_5 = nengo.Probe(pre_4)

        onchip = nengo.Ensemble(1, 1, label="onchip")
        post1 = nengo.Ensemble(10, 1, label="post1")
        post2 = nengo.Node(size_in=1, label="post2")
        post3 = nengo.Probe(post2, label="post3")

        nengo.Connection(pre_1, pre_2)
        nengo.Connection(pre_2, pre_3)
        nengo.Connection(pre_3, pre_4)
        nengo.Connection(pre_4.neurons, onchip)
        nengo.Connection(onchip, post1)
        nengo.Connection(post1, post2)

        net.config[pre_2].on_chip = False
        net.config[pre_4].on_chip = False
        net.config[post1].on_chip = False

    split = Split(net, precompute=True)

    host_precomputable = {pre_1, pre_2, pre_3, pre_4, pre_5}
    for obj in host_precomputable:
        assert not split.on_chip(obj)
        assert split.precomputable(obj)

    host_nonprecomputable = {post1, post2, post3}
    for obj in host_nonprecomputable:
        assert not split.on_chip(obj)
        assert not split.precomputable(obj)

    assert split.on_chip(onchip)
    assert not split.precomputable(onchip)

    assert not split.precomputable(nengo.Node(0, add_to_container=False))


def test_split_precompute_loop_error():
    with nengo.Network() as net:
        node_offchip = nengo.Node(lambda t, x: x + 1, size_in=1, size_out=1)
        ens_onchip = nengo.Ensemble(10, 1)
        nengo.Connection(node_offchip, ens_onchip)
        nengo.Connection(ens_onchip, node_offchip)

    with pytest.raises(BuildError, match="Cannot precompute"):
        Split(net, precompute=True)


def test_chip_learning_errors():
    with nengo.Network() as net:
        add_params(net)

        a = nengo.Ensemble(100, 1)
        b = nengo.Ensemble(100, 1)
        net.config[b].on_chip = True

        nengo.Connection(a, b, learning_rule_type=nengo.PES())

    with pytest.raises(BuildError, match="Post ensemble"):
        Split(net)

    with nengo.Network() as net:
        add_params(net)

        a = nengo.Ensemble(100, 1)
        b = nengo.Ensemble(100, 1)
        error = nengo.Ensemble(100, 1)
        net.config[error].on_chip = True

        conn = nengo.Connection(a, b, learning_rule_type=nengo.PES())
        nengo.Connection(error, conn.learning_rule)

    with pytest.raises(BuildError, match="Pre ensemble"):
        Split(net)


@pytest.mark.parametrize("remove_passthrough", [True, False])
def test_split_remove_passthrough(remove_passthrough):
    with nengo.Network() as net:
        keep1 = nengo.Node(0, label="keep1")
        keep2 = nengo.Node(lambda t, x: x, size_in=1, label="keep2")
        keep3 = nengo.Node(size_in=1, label="keep3")

        chip1 = nengo.Ensemble(10, 1, label="chip1")
        discard1 = nengo.Node(size_in=1, label="discard1")
        chip2 = nengo.Ensemble(10, 1, label="chip2")
        discard2 = nengo.Node(size_in=1, label="discard2")
        chip3 = nengo.Ensemble(10, 1, label="chip3")

        keep4 = nengo.Node(size_in=1, label="keep4")
        probe = nengo.Probe(keep4)

        nengo.Connection(keep1, keep2)
        nengo.Connection(keep2, keep3)
        nengo.Connection(keep3, chip1)
        conn1 = nengo.Connection(chip1, discard1)
        conn2 = nengo.Connection(discard1, chip2)
        conn3 = nengo.Connection(chip2, discard2)
        conn4 = nengo.Connection(discard2, chip3)
        nengo.Connection(chip3, keep4)

    split = Split(net, remove_passthrough=remove_passthrough)
    assert not split.on_chip(probe)

    if remove_passthrough:
        assert split.passthrough.to_remove == {
            conn1,
            conn2,
            conn3,
            conn4,
            discard1,
            discard2,
        }

        conns = list(split.passthrough.to_add)
        assert len(conns) == 2

        prepost = {(conn.pre, conn.post) for conn in conns}
        assert prepost == {(chip1, chip2), (chip2, chip3)}

    else:
        assert split.passthrough.to_remove == set()
        assert split.passthrough.to_add == set()


def test_sliced_passthrough_bug():
    with nengo.Network() as model:
        a = nengo.Ensemble(1, 1, label="a")
        passthrough = nengo.Node(size_in=1, label="passthrough")

        nengo.Connection(a, passthrough)
        p = nengo.Probe(passthrough[0])

    split = Split(model, remove_passthrough=True)

    assert len(split.passthrough.to_add) == 0
    assert len(split.passthrough.to_remove) == 0

    assert split.on_chip(a)
    assert not split.on_chip(passthrough)
    assert not split.on_chip(p)


def test_precompute_remove_passthrough():
    with nengo.Network() as net:
        host = nengo.Node(0, label="host")
        onchip1 = nengo.Ensemble(1, 1, label="onchip1")
        passthrough1 = nengo.Node(size_in=1, label="passthrough1")
        onchip2 = nengo.Ensemble(1, 1, label="onchip2")
        passthrough2 = nengo.Node(size_in=1, label="passthrough2")
        onchip3 = nengo.Ensemble(1, 1, label="onchip3")

        nengo.Connection(host, onchip1)
        nengo.Connection(onchip1, passthrough1)
        nengo.Connection(passthrough1, onchip2)
        nengo.Connection(onchip2, passthrough2)
        nengo.Connection(passthrough2, onchip3)

    split = Split(net, precompute=True, remove_passthrough=True)

    assert split.precomputable(host)
    assert not split.on_chip(host)

    for obj in (onchip1, passthrough1, onchip2, passthrough2, onchip3):
        assert not split.precomputable(obj)

    for obj in (onchip1, onchip2, onchip3):
        assert split.on_chip(obj)
