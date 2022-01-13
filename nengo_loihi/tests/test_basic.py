import nengo
import numpy as np
import pytest


@pytest.mark.parametrize("weights", [True, False])
def test_ens_ens(allclose, plt, seed, Simulator, weights):
    with nengo.Network(seed=seed) as model:
        a = nengo.Ensemble(100, 1, label="a")
        ap = nengo.Probe(a)
        anp = nengo.Probe(a.neurons)
        avp = nengo.Probe(a.neurons[:5], "voltage")

        b = nengo.Ensemble(101, 1, label="b")
        solver = nengo.solvers.LstsqL2(weights=weights)
        nengo.Connection(a, b, function=lambda x: x + 0.5, solver=solver)
        bp = nengo.Probe(b)
        bnp = nengo.Probe(b.neurons)
        bup = nengo.Probe(b.neurons[:5], "input")
        bvp = nengo.Probe(b.neurons[:5], "voltage")

        c = nengo.Ensemble(1, 1, label="c")
        bc_conn = nengo.Connection(b, c)

    with Simulator(model) as sim:
        sim.run(1.0)

    plt.figure()
    output_filter = nengo.synapses.Alpha(0.02)
    a = output_filter.filtfilt(sim.data[ap])
    b = output_filter.filtfilt(sim.data[bp])
    t = sim.trange()
    plt.plot(t, a)
    plt.plot(t, b)

    assert anp in sim.data
    assert avp in sim.data
    assert allclose(a, 0.0, atol=0.03)

    assert bup in sim.data
    assert bvp in sim.data
    firing_rate = sim.data[bnp].mean(axis=0)
    b_decoders = sim.data[bc_conn].weights
    decoded = sim.dt * b_decoders.dot(firing_rate)
    assert 30 < firing_rate.mean() < 50
    assert allclose(decoded, 0.5, atol=0.075)
    assert allclose(b[t > 0.1], 0.5, atol=0.075)


@pytest.mark.parametrize(
    "weights", [True, pytest.param(False, marks=pytest.mark.xfail)]
)
def test_ens_ens_slice(allclose, plt, seed, Simulator, weights):
    b_vals = np.array([-0.5, 0.75])

    with nengo.Network(seed=seed) as model:
        a = nengo.Ensemble(100, 2, label="a")
        b = nengo.Ensemble(101, 2, label="b")
        bp = nengo.Probe(b)
        nengo.Connection(a, b, function=lambda x: x + b_vals)

        c = nengo.Ensemble(102, 2, label="c")
        cp = nengo.Probe(c)
        solver = nengo.solvers.LstsqL2(weights=weights)
        nengo.Connection(b[1], c[0], solver=solver)
        nengo.Connection(b[0], c[1], solver=solver)

    with Simulator(model) as sim:
        sim.run(1.0)

    output_filter = nengo.synapses.Alpha(0.02)
    t = sim.trange()
    b = output_filter.filtfilt(sim.data[bp])
    c = output_filter.filtfilt(sim.data[cp])
    plt.plot(t, b)
    plt.plot(t, c)
    plt.legend(
        [f"b{d}" for d in range(b.shape[1])] + [f"c{d}" for d in range(c.shape[1])]
    )

    assert allclose(b[t > 0.15, 0], b_vals[0], atol=0.15)
    assert allclose(b[t > 0.15, 1], b_vals[1], atol=0.2)
    assert allclose(c[t > 0.15, 0], b_vals[1], atol=0.2)
    assert allclose(c[t > 0.15, 1], b_vals[0], atol=0.2)


@pytest.mark.parametrize("dims", [2])
@pytest.mark.parametrize("weights", [True, False])
def test_node_ens_ens(allclose, plt, seed, Simulator, dims, weights):
    runtime = 1.0
    function = lambda x: x ** 2

    with nengo.Network(seed=seed) as model:
        # fix the seed of the input process to compare fairly between parametrizations
        u_process = nengo.processes.WhiteSignal(runtime, high=1, rms=0.5, seed=1)
        u = nengo.Node(output=u_process, size_out=dims)
        up = nengo.Probe(u, synapse=None)

        a = nengo.Ensemble(100, dims, label="a")
        nengo.Connection(u, a)
        ap = nengo.Probe(a)
        anp = nengo.Probe(a.neurons)
        avp = nengo.Probe(a.neurons[:5], "voltage")

        b = nengo.Ensemble(101, dims, label="b")
        solver = nengo.solvers.LstsqL2(weights=weights)
        nengo.Connection(a, b, function=function, solver=solver, synapse=None)
        bp = nengo.Probe(b)
        bup = nengo.Probe(b.neurons[:5], "input")
        bvp = nengo.Probe(b.neurons[:5], "voltage")

    with Simulator(model, precompute=True) as sim:
        sim.run(runtime)

    output_filter = nengo.synapses.Alpha(0.02)
    u = output_filter.filtfilt(sim.data[up])
    a = output_filter.filtfilt(sim.data[ap])
    b = output_filter.filtfilt(sim.data[bp])

    plt.figure(figsize=(8, 6))
    t = sim.trange()
    plt.subplot(411)
    plt.plot(t, u[:, 0], "b", label="u[0]")
    plt.plot(t, a[:, 0], "g", label="a[0]")
    plt.ylim([-1, 1])
    plt.legend(loc=0)

    plt.subplot(412)
    plt.plot(t, u[:, 1], "b", label="u[1]")
    plt.plot(t, a[:, 1], "g", label="a[1]")
    plt.ylim([-1, 1])
    plt.legend(loc=0)

    plt.subplot(413)
    plt.plot(t, function(a[:, 0]), c="b", label="f(a[0])")
    plt.plot(t, b[:, 0], c="g", label="b[0]")
    plt.ylim([-0.05, 1])
    plt.legend(loc=0)

    plt.subplot(414)
    plt.plot(t, function(a[:, 1]), c="b", label="f(a[1])")
    plt.plot(t, b[:, 1], c="g", label="b[1]")
    plt.ylim([-0.05, 1])
    plt.legend(loc=0)

    tmask = t > 0.1  # ignore transients at the beginning
    assert anp in sim.data
    assert avp in sim.data
    assert allclose(a[tmask], np.clip(u[tmask], -1, 1), atol=0.1, rtol=0.1)

    assert bup in sim.data
    assert bvp in sim.data
    assert allclose(b[tmask], function(a[tmask]), atol=0.1, rtol=0.15)
