import inspect

import nengo
import numpy as np
import pytest

import nengo_loihi


def test_cx_model_validate_notempty(Simulator):
    with nengo.Network() as model:
        nengo_loihi.add_params(model)

        a = nengo.Ensemble(10, 1)
        model.config[a].on_chip = False

    with pytest.raises(nengo.exceptions.BuildError):
        with Simulator(model):
            pass


@pytest.mark.parametrize("precompute", [True, False])
def test_probedict_fallbacks(precompute, Simulator):
    with nengo.Network() as net:
        nengo_loihi.add_params(net)
        node_a = nengo.Node(0)
        with nengo.Network():
            ens_b = nengo.Ensemble(10, 1)
            conn_ab = nengo.Connection(node_a, ens_b)
        ens_c = nengo.Ensemble(5, 1)
        net.config[ens_c].on_chip = False
        conn_bc = nengo.Connection(ens_b, ens_c)
        probe_a = nengo.Probe(node_a)
        probe_c = nengo.Probe(ens_c)

    with Simulator(net, precompute=precompute) as sim:
        sim.run(0.002)

    assert node_a in sim.data
    assert ens_b in sim.data
    assert ens_c in sim.data
    assert probe_a in sim.data
    assert probe_c in sim.data

    # TODO: connections are currently not probeable as they are
    #       replaced in the splitting process
    assert conn_ab  # in sim.data
    assert conn_bc  # in sim.data


@pytest.mark.parametrize(
    "dt, pre_on_chip",
    [(2e-4, True), (3e-4, False), (4e-4, True), (2e-3, True)]
)
def test_dt(dt, pre_on_chip, Simulator, seed, plt, allclose):
    function = lambda x: x**2
    probe_synapse = nengo.Alpha(0.01)

    ens_params = dict(
        intercepts=nengo.dists.Uniform(-0.9, 0.9),
        max_rates=nengo.dists.Uniform(100, 120))

    with nengo.Network(seed=seed) as model:
        nengo_loihi.add_params(model)

        stim = nengo.Node(lambda t: -np.sin(2 * np.pi * t))
        stim_p = nengo.Probe(stim, synapse=probe_synapse)

        pre = nengo.Ensemble(100, 1, **ens_params)
        model.config[pre].on_chip = pre_on_chip
        pre_p = nengo.Probe(pre, synapse=probe_synapse)

        post = nengo.Ensemble(101, 1, **ens_params)
        post_p = nengo.Probe(post, synapse=probe_synapse)

        nengo.Connection(stim, pre)
        nengo.Connection(pre, post, function=function,
                         solver=nengo.solvers.LstsqL2(weights=True))

    with Simulator(model) as sim:
        sim.run(1.0)

    x = sim.data[stim_p]
    y = function(x)
    plt.plot(sim.trange(), x, 'k--')
    plt.plot(sim.trange(), y, 'k--')
    plt.plot(sim.trange(), sim.data[pre_p])
    plt.plot(sim.trange(), sim.data[post_p])

    assert allclose(sim.data[pre_p], x, rtol=0.1, atol=0.1)
    assert allclose(sim.data[post_p], y, rtol=0.1, atol=0.1)


@pytest.mark.parametrize('simtype', ['simreal', None])
def test_nengo_comm_channel_compare(simtype, Simulator, seed, plt, allclose):
    if simtype == 'simreal':
        Simulator = lambda *args: nengo_loihi.Simulator(
            *args, target='simreal')

    with nengo.Network(seed=seed) as model:
        u = nengo.Node(lambda t: np.sin(6*t))
        a = nengo.Ensemble(50, 1)
        b = nengo.Ensemble(50, 1)
        nengo.Connection(u, a)
        nengo.Connection(a, b, function=lambda x: x**2,
                         solver=nengo.solvers.LstsqL2(weights=True))

        ap = nengo.Probe(a, synapse=0.03)
        bp = nengo.Probe(b, synapse=0.03)

    with nengo.Simulator(model) as nengo_sim:
        nengo_sim.run(1.0)

    with Simulator(model) as loihi_sim:
        loihi_sim.run(1.0)

    plt.subplot(2, 1, 1)
    plt.plot(nengo_sim.trange(), nengo_sim.data[ap])
    plt.plot(loihi_sim.trange(), loihi_sim.data[ap])

    plt.subplot(2, 1, 2)
    plt.plot(nengo_sim.trange(), nengo_sim.data[bp])
    plt.plot(loihi_sim.trange(), loihi_sim.data[bp])

    assert allclose(loihi_sim.data[ap], nengo_sim.data[ap], atol=0.1, rtol=0.2)
    assert allclose(loihi_sim.data[bp], nengo_sim.data[bp], atol=0.1, rtol=0.2)


@pytest.mark.parametrize("precompute", (True, False))
def test_close(Simulator, precompute):
    with nengo.Network() as net:
        a = nengo.Node([0])
        b = nengo.Ensemble(10, 1)
        c = nengo.Node(size_in=1)
        nengo.Connection(a, b)
        nengo.Connection(b, c)

    with Simulator(net, precompute=precompute) as sim:
        pass

    assert sim.closed
    assert all(s.closed for s in sim.sims.values())


def test_all_run_steps(Simulator):
    # Case 1. No objects on host, so no host and no host_pre
    with nengo.Network() as net:
        pre = nengo.Ensemble(10, 1)
        post = nengo.Ensemble(10, 1)
        nengo.Connection(pre, post)

    # 1a. precompute=False, no host
    with Simulator(net) as sim:
        sim.run(0.001)
    # Since no objects on host, we should be precomputing even if we did not
    # explicitly request precomputing
    assert sim.precompute
    assert inspect.ismethod(sim._run_steps)
    assert sim._run_steps.__name__ == "run_steps"

    # 1b. precompute=True, no host, no host_pre
    with pytest.warns(UserWarning) as record:
        with Simulator(net, precompute=True) as sim:
            sim.run(0.001)
    assert any("No precomputable objects" in r.message.args[0] for r in record)
    assert inspect.ismethod(sim._run_steps)
    assert sim._run_steps.__name__ == "run_steps"

    # Case 2: Add a precomputable off-chip object, so we have either host or
    # host_pre but not both host and host_pre
    with net:
        stim = nengo.Node(1)
        stim_conn = nengo.Connection(stim, pre)

    # 2a. precompute=False, host
    with Simulator(net) as sim:
        sim.run(0.001)
    assert sim._run_steps.__name__.endswith("_bidirectional_with_host")

    # 2b. precompute=True, no host, host_pre
    with Simulator(net, precompute=True) as sim:
        sim.run(0.001)
    assert sim._run_steps.__name__.endswith("_precomputed_host_pre_only")

    # Case 3: Add a non-precomputable off-chip object so we have host
    # and host_pre
    with net:
        out = nengo.Node(size_in=1)
        nengo.Connection(post, out)
        nengo.Probe(out)  # probe to prevent `out` from being optimized away

    # 3a. precompute=False, host (same as 2a)
    with Simulator(net) as sim:
        sim.run(0.001)
    assert sim._run_steps.__name__.endswith("_bidirectional_with_host")

    # 3b. precompute=True, host, host_pre
    with Simulator(net, precompute=True) as sim:
        sim.run(0.001)
    assert sim._run_steps.__name__.endswith("_precomputed_host_pre_and_host")

    # Case 4: Delete the precomputable off-chip object, so we have host only
    net.nodes.remove(stim)
    net.connections.remove(stim_conn)

    # 4a. precompute=False, host (same as 2a and 3a)
    with Simulator(net) as sim:
        sim.run(0.001)
    assert sim._run_steps.__name__.endswith("_bidirectional_with_host")

    # 4b. precompute=True, host, no host_pre
    with pytest.warns(UserWarning) as record:
        with Simulator(net, precompute=True) as sim:
            sim.run(0.001)
    assert any("No precomputable objects" in r.message.args[0] for r in record)
    assert sim._run_steps.__name__.endswith("_precomputed_host_only")


def test_no_precomputable(Simulator):
    with nengo.Network() as net:
        active_ens = nengo.Ensemble(10, 1,
                                    gain=np.ones(10) * 10,
                                    bias=np.ones(10) * 10)
        out = nengo.Node(size_in=10)
        nengo.Connection(active_ens.neurons, out)
        out_p = nengo.Probe(out)

    with pytest.warns(UserWarning) as record:
        with Simulator(net, precompute=True) as sim:
            sim.run(0.01)

    assert sim._run_steps.__name__.endswith("precomputed_host_only")
    # Should warn that no objects are precomputable
    assert any("No precomputable objects" in r.message.args[0] for r in record)
    # But still mark the sim as precomputable for speed reasons, because
    # there are no inputs that depend on outputs in this case
    assert sim.precompute
    assert sim.data[out_p].shape[0] == sim.trange().shape[0]
    assert np.all(sim.data[out_p][-1] > 100)


def test_all_onchip(Simulator):
    with nengo.Network() as net:
        active_ens = nengo.Ensemble(10, 1,
                                    gain=np.ones(10) * 10,
                                    bias=np.ones(10) * 10)
        out = nengo.Ensemble(10, 1, gain=np.ones(10), bias=np.ones(10))
        nengo.Connection(active_ens.neurons, out.neurons,
                         transform=np.eye(10) * 10)
        out_p = nengo.Probe(out.neurons)

    with Simulator(net) as sim:
        sim.run(0.01)

    # Though we did not specify precompute, the model should be marked as
    # precomputable because there are no off-chip objects
    assert sim.precompute
    assert inspect.ismethod(sim._run_steps)
    assert sim._run_steps.__name__ == "run_steps"
    assert sim.data[out_p].shape[0] == sim.trange().shape[0]
    assert np.all(sim.data[out_p][-1] > 100)
