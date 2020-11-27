import nengo
import numpy as np
import pytest
from nengo.exceptions import SimulationError, ValidationError
from nengo.utils.numpy import rms

import nengo_loihi
from nengo_loihi.builder import Model
from nengo_loihi.hardware.allocators import Greedy, RoundRobin
from nengo_loihi.tests import require_partition


def pes_network(
    n_per_dim,
    dims,
    seed,
    learning_rule_type=nengo.PES(learning_rate=1e-3),
    input_scale=None,
    error_scale=1.0,
    learn_synapse=0.005,
    probe_synapse=0.02,
    period=1.0,
):
    if input_scale is None:
        input_scale = np.linspace(1, 0, dims + 1)[:-1]
    assert input_scale.size == dims

    input_fn = lambda t: np.sin(t * 2 * np.pi / period) * input_scale

    probes = {}
    with nengo.Network(seed=seed) as model:
        stim = nengo.Node(input_fn)

        pre = nengo.Ensemble(n_per_dim * dims, dims)
        post = nengo.Node(size_in=dims)

        nengo.Connection(stim, pre, synapse=None)
        conn = nengo.Connection(
            pre,
            post,
            function=lambda x: np.zeros(dims),
            synapse=learn_synapse,
            learning_rule_type=learning_rule_type,
        )

        nengo.Connection(post, conn.learning_rule, transform=error_scale)
        nengo.Connection(stim, conn.learning_rule, transform=-error_scale)

        probes["stim"] = nengo.Probe(stim, synapse=probe_synapse)
        probes["pre"] = nengo.Probe(pre, synapse=probe_synapse)
        probes["post"] = nengo.Probe(post, synapse=probe_synapse)

    return model, probes


@pytest.mark.xfail(reason="Multi-chip learning fails intermittently due to timeout")
@pytest.mark.parametrize("dims", (1, 3))
def test_pes_comm_channel(dims, request, allclose, plt, seed, Simulator):
    n_per_dim = 300
    tau = 0.01
    simtime = 1.5
    model, probes = pes_network(
        n_per_dim,
        dims,
        seed,
        learn_synapse=tau,
        learning_rule_type=nengo.PES(learning_rate=1e-2),
        period=simtime / 2,
    )

    # with NxSDK 0.9.8, only Nahuku32 is working with multi-chip SNIPs
    success = require_partition(
        "nahuku32",
        request=request,
        lmt_options="--skip-power=1",
    )
    allocator = RoundRobin() if success else Greedy()

    with nengo.Simulator(model) as nengo_sim:
        nengo_sim.run(simtime)

    with Simulator(model, hardware_options={"allocator": allocator}) as loihi_sim:
        loihi_sim.run(simtime)

    with Simulator(model, target="simreal") as real_sim:
        real_sim.run(simtime)

    t = nengo_sim.trange()
    pre_tmask = t > 0.1
    post_tmask = t > simtime / 2

    dec_tau = loihi_sim.model.decode_tau
    y = nengo_sim.data[probes["stim"]]
    y_dpre = nengo.Lowpass(dec_tau).filt(y)
    y_dpost = nengo.Lowpass(tau).combine(nengo.Lowpass(dec_tau)).filt(y_dpre)
    y_nengo = nengo_sim.data[probes["post"]]
    y_loihi = loihi_sim.data[probes["post"]]
    y_real = real_sim.data[probes["post"]]

    plt.subplot(211)
    plt.plot(t, y_dpost, "k", label="target")
    plt.plot(t, y_nengo, "b", label="nengo")
    plt.plot(t, y_loihi, "g", label="loihi")
    plt.plot(t, y_real, "r:", label="real")
    plt.legend()

    plt.subplot(212)
    plt.plot(t[post_tmask], y_loihi[post_tmask] - y_dpost[post_tmask], "k")
    plt.plot(t[post_tmask], y_loihi[post_tmask] - y_nengo[post_tmask], "b")

    x_loihi = loihi_sim.data[probes["pre"]]
    assert allclose(x_loihi[pre_tmask], y_dpre[pre_tmask], atol=0.18, rtol=0.05)

    assert allclose(y_loihi[post_tmask], y_dpost[post_tmask], atol=0.18, rtol=0.05)
    assert allclose(y_loihi, y_nengo, atol=0.2, rtol=0.2)

    assert allclose(y_real[post_tmask], y_dpost[post_tmask], atol=0.18, rtol=0.05)
    assert allclose(y_real, y_nengo, atol=0.2, rtol=0.2)


@pytest.mark.xfail(reason="Multi-chip learning fails intermittently due to timeout")
def test_pes_overflow(request, plt, seed, Simulator):
    dims = 3
    n_per_dim = 300
    tau = 0.01
    simtime = 0.6
    model, probes = pes_network(
        n_per_dim,
        dims,
        seed,
        learn_synapse=tau,
        input_scale=np.linspace(1, 0.7, dims),
        learning_rule_type=nengo.PES(learning_rate=1e-2),
        period=simtime,
    )

    loihi_model = Model()
    # set learning_wgt_exp low to create overflow in weight values
    loihi_model.pes_wgt_exp = -2

    # with NxSDK 0.9.8, only Nahuku32 is working with multi-chip SNIPs
    success = require_partition(
        "nahuku32",
        request=request,
        lmt_options="--skip-power=1",
    )
    allocator = RoundRobin() if success else Greedy()

    with Simulator(
        model, model=loihi_model, hardware_options={"allocator": allocator}
    ) as loihi_sim:
        loihi_sim.run(simtime)

    t = loihi_sim.trange()
    post_tmask = t > simtime - 0.1

    dec_tau = loihi_sim.model.decode_tau
    y = loihi_sim.data[probes["stim"]]
    y_dpre = nengo.Lowpass(dec_tau).filt(y)
    y_dpost = nengo.Lowpass(tau).combine(nengo.Lowpass(dec_tau)).filt(y_dpre)
    y_loihi = loihi_sim.data[probes["post"]]

    plt.plot(t, y_dpost, "k", label="target")
    plt.plot(t, y_loihi, "g", label="loihi")

    # --- fit output to scaled version of target output
    z_ref0 = y_dpost[post_tmask][:, 0]
    z_loihi = y_loihi[post_tmask]
    scale = np.linspace(0, 1, 50)
    E = np.abs(z_loihi - scale[:, None, None] * z_ref0[:, None])
    errors = E.mean(axis=1)  # average over time (errors is: scales x dims)
    for j in range(dims):
        errors_j = errors[:, j]
        i = np.argmin(errors_j)
        assert errors_j[i] < 0.1, (
            "Learning output for dim %d did not match "
            "any scaled version of the target output" % j
        )
        assert scale[i] > 0.25, "Learning output for dim %d is too small" % j
        assert scale[i] < 0.9, (
            "Learning output for dim %d is too large "
            "(weights or traces not clipping as expected)" % j
        )


@pytest.mark.xfail(reason="Multi-chip learning fails intermittently due to timeout")
def test_pes_error_clip(request, plt, seed, Simulator):
    dims = 2
    n_per_dim = 120
    tau = 0.01
    error_scale = 5.0  # scale up error signal so it clips
    simtime = 0.3
    model, probes = pes_network(
        n_per_dim,
        dims,
        seed,
        learn_synapse=tau,
        learning_rule_type=nengo.PES(learning_rate=1e-2 / error_scale),
        input_scale=np.array([1.0, -1.0]),
        error_scale=error_scale,
        period=simtime,
    )

    # with NxSDK 0.9.8, only Nahuku32 is working with multi-chip SNIPs
    success = require_partition(
        "nahuku32",
        request=request,
        lmt_options="--skip-power=1",
    )
    allocator = RoundRobin() if success else Greedy()

    with pytest.warns(UserWarning, match=r".*PES error.*pes_error_scale.*"):
        with Simulator(model, hardware_options={"allocator": allocator}) as loihi_sim:
            loihi_sim.run(simtime)

    t = loihi_sim.trange()
    post_tmask = t > simtime - 1.0

    dec_tau = loihi_sim.model.decode_tau
    y = loihi_sim.data[probes["stim"]]
    y_dpre = nengo.Lowpass(dec_tau).filt(y)
    y_dpost = nengo.Lowpass(tau).combine(nengo.Lowpass(dec_tau)).filt(y_dpre)
    y_loihi = loihi_sim.data[probes["post"]]

    plt.plot(t, y_dpost, "k", label="target")
    plt.plot(t, y_loihi, "g", label="loihi")

    # --- assert that we've learned something, but not everything
    error = rms(y_loihi[post_tmask] - y_dpost[post_tmask]) / rms(y_dpost[post_tmask])
    assert error < 0.5
    assert error > 0.05
    # ^ error on emulator vs chip is quite different, hence large tolerances


@pytest.mark.xfail(reason="Multi-chip learning fails intermittently due to timeout")
@pytest.mark.parametrize("init_function", [None, lambda x: 0])
def test_multiple_pes(init_function, request, allclose, plt, seed, Simulator):
    n_errors = 5
    targets = np.linspace(-0.9, 0.9, n_errors)
    with nengo.Network(seed=seed) as model:
        pre_ea = nengo.networks.EnsembleArray(200, n_ensembles=n_errors)
        output = nengo.Node(size_in=n_errors)

        target = nengo.Node(targets)

        for i in range(n_errors):
            conn = nengo.Connection(
                pre_ea.ea_ensembles[i],
                output[i],
                function=init_function,
                learning_rule_type=nengo.PES(learning_rate=1e-2),
            )
            nengo.Connection(target[i], conn.learning_rule, transform=-1)
            nengo.Connection(output[i], conn.learning_rule)

        probe = nengo.Probe(output, synapse=0.1)

    # with NxSDK 0.9.8, only Nahuku32 is working with multi-chip SNIPs
    success = require_partition(
        "nahuku32",
        request=request,
        lmt_options="--skip-power=1",
    )
    allocator = RoundRobin() if success else Greedy()

    simtime = 0.6
    with Simulator(model, hardware_options={"allocator": allocator}) as sim:
        sim.run(simtime)

    t = sim.trange()
    tmask = t > simtime * 0.85

    plt.plot(t, sim.data[probe])
    for target, style in zip(targets, plt.rcParams["axes.prop_cycle"]):
        plt.axhline(target, **style)

    for i, target in enumerate(targets):
        assert allclose(sim.data[probe][tmask, i], target, atol=0.1, rtol=0.1), (
            "Target %d not close" % i
        )


# TODO: Revisit when we can blacklist boards
@pytest.mark.xfail
def test_pes_deterministic(request, Simulator, seed, allclose):
    """Ensure that learning output is the same between runs"""
    # Make a network with lots of objects, so dictionary order has an effect
    n_errors = 3
    targets = np.linspace(-0.8, 0.95, n_errors)
    with nengo.Network(seed=seed) as model:
        pre_ea = nengo.networks.EnsembleArray(100, n_ensembles=n_errors)
        output = nengo.Node(size_in=n_errors)

        target = nengo.Node(targets)

        for i in range(n_errors):
            conn = nengo.Connection(
                pre_ea.ea_ensembles[i],
                output[i],
                learning_rule_type=nengo.PES(learning_rate=1e-2),
            )
            nengo.Connection(target[i], conn.learning_rule, transform=-1)
            nengo.Connection(output[i], conn.learning_rule)

        probe = nengo.Probe(output, synapse=0.005)

    # with NxSDK 0.9.8, only Nahuku32 is working with multi-chip SNIPs
    success = require_partition(
        "nahuku32",
        request=request,
        lmt_options="--skip-power=1",
    )
    allocator = RoundRobin() if success else Greedy()

    # some random aspects (e.g. dictionary order) only have a few combinations,
    # so more sims makes it less likely we'll get the same order by chance,
    # if things are truly non-deterministic
    n_sims = 3
    simtime = 0.1
    sims = []
    for _ in range(n_sims):
        with Simulator(model, hardware_options={"allocator": allocator}) as sim:
            sim.run(simtime)
        sims.append(sim)

    sim0 = sims[0]
    for sim in sims[1:]:
        assert allclose(sim.data[probe], sim0.data[probe])


# TODO: Revisit when we can blacklist boards
@pytest.mark.xfail
def test_learning_seed(Simulator, request, seed):
    # with NxSDK 0.9.8, only Nahuku32 is working with multi-chip SNIPs
    require_partition(
        "nahuku32",
        request=request,
        lmt_options="--skip-power=1",
        action="fail" if nengo_loihi.version.dev is None else "skip",
    )

    n_per_dim = 120
    dims = 1
    tau = 0.005
    simtime = 0.2
    model, probes = pes_network(
        n_per_dim,
        dims,
        seed,
        learn_synapse=tau,
        learning_rule_type=nengo.PES(learning_rate=1e-2),
        period=simtime / 2,
    )

    sim_args = dict(hardware_options={"allocator": RoundRobin()})

    with Simulator(model, seed=seed, **sim_args) as sim:
        sim.run(simtime)

    with Simulator(model, seed=seed, **sim_args) as sim1:
        sim1.run(simtime)

    with Simulator(model, seed=seed + 1, **sim_args) as sim2:
        sim2.run(simtime)

    assert np.allclose(sim1.data[probes["post"]], sim.data[probes["post"]])
    assert not np.allclose(sim2.data[probes["post"]], sim.data[probes["post"]])


def test_pes_pre_synapse_type_error(Simulator):
    with nengo.Network() as model:
        pre = nengo.Ensemble(10, 1)
        post = nengo.Node(size_in=1)
        rule_type = nengo.PES(pre_synapse=nengo.Alpha(0.005))
        conn = nengo.Connection(pre, post, learning_rule_type=rule_type)
        nengo.Connection(post, conn.learning_rule)

    with pytest.raises(ValidationError, match="pre-synapses for learning"):
        with Simulator(model):
            pass


def test_pes_trace_increment_clip_warning(seed, Simulator):
    dims = 2
    n_per_dim = 120
    model, _ = pes_network(
        n_per_dim, dims, seed, learning_rule_type=nengo.PES(learning_rate=1e-1)
    )

    with pytest.warns(UserWarning, match="Trace increment exceeds upper"):
        with Simulator(model):
            pass


def test_drop_trace_spikes(Simulator, seed):
    with nengo.Network(seed=seed) as net:
        a = nengo.Ensemble(
            10,
            1,
            gain=nengo.dists.Choice([1]),
            bias=nengo.dists.Choice([2000]),
            neuron_type=nengo.SpikingRectifiedLinear(),
        )
        b = nengo.Node(size_in=1)

        conn = nengo.Connection(
            a, b, transform=[[100]], learning_rule_type=nengo.PES(1)
        )

        nengo.Connection(b, conn.learning_rule)

    with Simulator(net, target="sim") as sim:
        with pytest.raises(SimulationError, match="Synaptic trace spikes lost"):
            sim.run(1.0)
