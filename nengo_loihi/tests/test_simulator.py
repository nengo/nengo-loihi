import inspect
import re

import nengo
import numpy as np
import pytest
from nengo.exceptions import BuildError, ReadonlyError, ValidationError

import nengo_loihi
from nengo_loihi.block import Axon, LoihiBlock, Synapse
from nengo_loihi.builder import Model
from nengo_loihi.builder.discretize import discretize_model
from nengo_loihi.emulator import EmulatorInterface
from nengo_loihi.hardware import HardwareInterface
from nengo_loihi.inputs import SpikeInput
from nengo_loihi.probe import LoihiProbe
from nengo_loihi.simulator import Timers


def test_none_network(Simulator):
    with pytest.raises(ValidationError, match="network parameter"):
        Simulator(None)


def test_model_validate_notempty(Simulator):
    with nengo.Network() as model:
        nengo_loihi.add_params(model)

        a = nengo.Ensemble(10, 1)
        model.config[a].on_chip = False

    assert nengo.rc.get("decoder_cache", "enabled")

    with pytest.raises(BuildError, match="No neurons marked"):
        with Simulator(model):
            pass

    # Ensure cache config not changed
    assert nengo.rc.get("decoder_cache", "enabled")


@pytest.mark.filterwarnings("ignore:Model is precomputable.")
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


def test_probedict_interface(Simulator):
    with nengo.Network(label="net") as net:
        u = nengo.Node(1, label="u")
        a = nengo.Ensemble(9, 1, label="a")
        nengo.Connection(u, a)

    with Simulator(net) as sim:
        pass

    objs = [u, a]
    count = 0
    for o in sim.data:
        count += 1
        if o in objs:
            objs.remove(o)
    assert len(sim.data) == count
    assert len(objs) == 0, "Objects did not appear in probedict: %s" % objs


@pytest.mark.xfail
@pytest.mark.parametrize(
    "dt, pre_on_chip", [(2e-4, True), (3e-4, False), (4e-4, True), (2e-3, True)]
)
def test_dt(dt, pre_on_chip, Simulator, seed, plt, allclose):
    function = lambda x: x ** 2
    probe_synapse = nengo.Alpha(0.01)
    simtime = 0.2

    ens_params = dict(
        intercepts=nengo.dists.Uniform(-0.9, 0.9),
        max_rates=nengo.dists.Uniform(100, 120),
    )

    with nengo.Network(seed=seed) as model:
        nengo_loihi.add_params(model)

        stim = nengo.Node(lambda t: -(np.sin(2 * np.pi * t / simtime)))
        stim_p = nengo.Probe(stim, synapse=probe_synapse)

        pre = nengo.Ensemble(100, 1, **ens_params)
        model.config[pre].on_chip = pre_on_chip
        pre_p = nengo.Probe(pre, synapse=probe_synapse)

        post = nengo.Ensemble(101, 1, **ens_params)
        post_p = nengo.Probe(post, synapse=probe_synapse)

        nengo.Connection(stim, pre)
        nengo.Connection(
            pre, post, function=function, solver=nengo.solvers.LstsqL2(weights=True)
        )

    with Simulator(model, dt=dt) as sim:
        sim.run(simtime)

    x = sim.data[stim_p]
    y = function(x)
    plt.plot(sim.trange(), x, "k--")
    plt.plot(sim.trange(), y, "k--")
    plt.plot(sim.trange(), sim.data[pre_p])
    plt.plot(sim.trange(), sim.data[post_p])

    assert allclose(sim.data[pre_p], x, rtol=0.1, atol=0.1)
    assert allclose(sim.data[post_p], y, rtol=0.1, atol=0.1)


@pytest.mark.parametrize("simtype", ["simreal", None])
def test_nengo_comm_channel_compare(simtype, Simulator, seed, plt, allclose):
    if simtype == "simreal":
        Simulator = lambda *args: nengo_loihi.Simulator(*args, target="simreal")

    simtime = 0.6

    with nengo.Network(seed=seed) as model:
        u = nengo.Node(lambda t: np.sin(6 * t / simtime))
        a = nengo.Ensemble(50, 1)
        b = nengo.Ensemble(50, 1)
        nengo.Connection(u, a)
        nengo.Connection(
            a, b, function=lambda x: x ** 2, solver=nengo.solvers.LstsqL2(weights=True)
        )

        ap = nengo.Probe(a, synapse=nengo.synapses.Alpha(0.02))
        bp = nengo.Probe(b, synapse=nengo.synapses.Alpha(0.02))

    with nengo.Simulator(model) as nengo_sim:
        nengo_sim.run(simtime)

    with Simulator(model) as loihi_sim:
        loihi_sim.run(simtime)

    plt.subplot(2, 1, 1)
    plt.plot(nengo_sim.trange(), nengo_sim.data[ap])
    plt.plot(loihi_sim.trange(), loihi_sim.data[ap])

    plt.subplot(2, 1, 2)
    plt.plot(nengo_sim.trange(), nengo_sim.data[bp])
    plt.plot(loihi_sim.trange(), loihi_sim.data[bp])

    assert allclose(loihi_sim.data[ap], nengo_sim.data[ap], atol=0.07, xtol=3)
    assert allclose(loihi_sim.data[bp], nengo_sim.data[bp], atol=0.07, xtol=6)


@pytest.mark.filterwarnings("ignore:Model is precomputable.")
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


class TestRunSteps:
    @staticmethod
    def simple_prepost():
        with nengo.Network() as net:
            net.pre = nengo.Ensemble(10, 1)
            net.post = nengo.Ensemble(10, 1)
            nengo.Connection(net.pre, net.post)
        return net

    def test_no_host_objs(self, Simulator):
        """No host objects, so no host and no host_pre."""
        net = self.simple_prepost()

        # precompute=None, no host, no host_pre
        with Simulator(net, precompute=None) as sim:
            sim.run(0.001)

            # Since no objects on host, we should be precomputing even if we did not
            # explicitly request precomputing
            assert sim.precompute
            assert inspect.ismethod(sim._runner.run_steps)
            assert sim._runner.run_steps.__name__.endswith("_only")

        # precompute=False, no host
        with pytest.warns(UserWarning, match="Model is precomputable"):
            with Simulator(net, precompute=False) as sim:
                sim.run(0.001)
                assert inspect.ismethod(sim._runner.run_steps)
                assert sim._runner.run_steps.__name__.endswith("_only")

        # precompute=True, no host, no host_pre
        with Simulator(net, precompute=True) as sim:
            sim.run(0.001)
            assert inspect.ismethod(sim._runner.run_steps)
            assert sim._runner.run_steps.__name__.endswith("_only")

    def test_all_precomputable(self, Simulator):
        """One precomputable host object.

        We should have either a host or host_pre, but not both.
        """
        net = self.simple_prepost()
        with net:
            stim = nengo.Node(1)
            nengo.Connection(stim, net.pre)

        # precompute=None, no host
        with Simulator(net, precompute=None) as sim:
            sim.run(0.001)
            assert sim.precompute
            assert sim._runner.run_steps.__name__.endswith("_precomputed_host_pre_only")

        # precompute=False, no host_pre
        with pytest.warns(UserWarning, match="Model is precomputable"):
            with Simulator(net, precompute=False) as sim:
                sim.run(0.001)
                assert sim._runner.run_steps.__name__.endswith(
                    "_bidirectional_with_host"
                )

        # precompute=True, no host
        with Simulator(net, precompute=True) as sim:
            sim.run(0.001)
            assert sim._runner.run_steps.__name__.endswith("_precomputed_host_pre_only")

    def test_precomputable_and_not(self, Simulator):
        """One precomputable host object and one non-precomputable host object.

        We will have host and host_pre, unless we request no host_pre.
        """

        net = self.simple_prepost()
        with net:
            stim = nengo.Node(1)
            nengo.Connection(stim, net.pre)
            out = nengo.Node(size_in=1)
            nengo.Connection(net.post, out)
            nengo.Probe(out)  # probe to prevent `out` from being optimized away

        # precompute=None
        with Simulator(net) as sim:
            sim.run(0.001)
            assert sim.precompute
            assert sim._runner.run_steps.__name__.endswith(
                "_precomputed_host_pre_and_host"
            )

        # precompute=False, no host_pre
        with pytest.warns(UserWarning, match="Model is precomputable"):
            with Simulator(net, precompute=False) as sim:
                sim.run(0.001)
                assert sim._runner.run_steps.__name__.endswith(
                    "_bidirectional_with_host"
                )

        # precompute=True
        with Simulator(net, precompute=True) as sim:
            sim.run(0.001)
            assert sim._runner.run_steps.__name__.endswith(
                "_precomputed_host_pre_and_host"
            )

    def test_all_non_precomputable(self, Simulator):
        """One non-precomputable host object.

        We will always have a host and never a host_pre.
        """

        net = self.simple_prepost()
        with net:
            out = nengo.Node(size_in=1)
            nengo.Connection(net.post, out)
            nengo.Probe(out)  # probe to prevent `out` from being optimized away

        # precompute=None, no host_pre
        with Simulator(net) as sim:
            sim.run(0.001)
            assert sim.precompute
            assert sim._runner.run_steps.__name__.endswith("_precomputed_host_only")

        # precompute=False, no host_pre
        with pytest.warns(UserWarning, match="Model is precomputable"):
            with Simulator(net, precompute=False) as sim:
                sim.run(0.001)
                assert sim._runner.run_steps.__name__.endswith(
                    "_bidirectional_with_host"
                )

        # precompute=True, no host_pre
        with Simulator(net, precompute=True) as sim:
            sim.run(0.001)
            assert sim._runner.run_steps.__name__.endswith("_precomputed_host_only")

    def test_feedback_loop(self, Simulator):
        """Chip input depends on output, nothing is precomputable.

        We will never have a host_pre.
        """
        net = self.simple_prepost()
        with net:
            feedback = nengo.Node(lambda t, x: x + t, size_in=1)
            nengo.Connection(net.post, feedback)
            nengo.Connection(feedback, net.pre)

        # precompute=None
        with Simulator(net) as sim:
            sim.run(0.001)
            assert not sim.precompute
            assert sim._runner.run_steps.__name__.endswith("_bidirectional_with_host")

        # precompute=False
        with Simulator(net, precompute=False) as sim:
            sim.run(0.001)
            assert sim._runner.run_steps.__name__.endswith("_bidirectional_with_host")

        # precompute=True, raises BuildError
        with pytest.raises(BuildError):
            with Simulator(net, precompute=True) as sim:
                sim.run(0.001)

    def test_all_onchip(self, Simulator):
        """All network elements simulated on-chip."""

        with nengo.Network() as net:
            active_ens = nengo.Ensemble(
                10, 1, gain=np.ones(10) * 10, bias=np.ones(10) * 10
            )
            out = nengo.Ensemble(10, 1, gain=np.ones(10), bias=np.ones(10))
            nengo.Connection(active_ens.neurons, out.neurons, transform=np.eye(10) * 10)
            out_p = nengo.Probe(out.neurons)

        with Simulator(net, precompute=None) as sim:
            sim.run(0.01)

            # Though we did not specify precompute, the model should be marked as
            # precomputable because there are no off-chip objects
            assert sim.precompute
            assert inspect.ismethod(sim._runner.run_steps)
            assert sim._runner.run_steps.__name__.endswith("_only")
            assert sim.data[out_p].shape[0] == sim.trange().shape[0]
            assert np.all(sim.data[out_p][-1] > 100)


def test_progressbar_values(Simulator):
    with nengo.Network() as model:
        nengo.Ensemble(1, 1)

    # both `None` and `False` are valid ways of specifying no progress bar
    with Simulator(model, progress_bar=None):
        pass

    with Simulator(model, progress_bar=False):
        pass

    # progress bar not yet implemented
    with pytest.warns(UserWarning, match="progress bar"):
        with Simulator(model, progress_bar=True):
            pass


def test_tau_s_warning(Simulator):
    with nengo.Network() as net:
        stim = nengo.Node(0)
        ens = nengo.Ensemble(10, 1)
        nengo.Connection(stim, ens, synapse=0.1)
        nengo.Connection(
            ens, ens, synapse=0.001, solver=nengo.solvers.LstsqL2(weights=True)
        )

    with pytest.warns(UserWarning) as record:
        with Simulator(net):
            pass

    assert any(
        rec.message.args[0]
        == ("tau_s is already set to 0.005, which is larger than 0.001. Using 0.005.")
        for rec in record
    )

    with net:
        nengo.Connection(
            ens, ens, synapse=0.1, solver=nengo.solvers.LstsqL2(weights=True)
        )
    with pytest.warns(UserWarning) as record:
        with Simulator(net):
            pass

    assert any(
        rec.message.args[0]
        == (
            "tau_s is currently 0.005, which is smaller than 0.1. "
            "Overwriting tau_s with 0.1."
        )
        for rec in record
    )


@pytest.mark.filterwarnings("ignore:Model is precomputable.")
@pytest.mark.xfail(
    nengo.version.version_info <= (2, 8, 0), reason="Nengo core controls seeds"
)
@pytest.mark.parametrize("precompute", [False, True])
def test_seeds(precompute, Simulator, seed):
    with nengo.Network(seed=seed) as net:
        nengo_loihi.add_params(net)

        e0 = nengo.Ensemble(1, 1, label="e0")
        e1 = nengo.Ensemble(1, 1, seed=2, label="e1")
        e2 = nengo.Ensemble(1, 1, label="e2")
        net.config[e2].on_chip = False
        nengo.Connection(e0, e1)
        nengo.Connection(e0, e2)

        with nengo.Network():
            n = nengo.Node(0)
            e = nengo.Ensemble(1, 1, label="e")
            nengo.Node(1)
            nengo.Connection(n, e)
            nengo.Probe(e)

        with nengo.Network(seed=8):
            nengo.Ensemble(8, 1, seed=3, label="unnamed")
            nengo.Node(1)

    def get_seed(sim, obj):
        return sim.model.seeds.get(
            obj, sim.model.host.seeds.get(obj, sim.model.host_pre.seeds.get(obj, None))
        )

    # --- test that seeds are the same as nengo ref simulator
    ref = nengo.Simulator(net)

    with Simulator(net, precompute=precompute) as sim:
        for obj in net.all_objects:
            assert get_seed(sim, obj) == ref.model.seeds.get(obj, None)

    # --- test that seeds that we set are preserved after splitting
    model = nengo_loihi.builder.Model()
    for i, obj in enumerate(net.all_objects):
        model.seeds[obj] = i

    with Simulator(net, model=model, precompute=precompute) as sim:
        for i, obj in enumerate(net.all_objects):
            assert get_seed(sim, obj) == i


def test_interface(Simulator, allclose):
    """Tests for the Simulator API for things that aren't covered elsewhere"""
    # test sim.time
    with nengo.Network() as model:
        nengo.Ensemble(2, 1)

    simtime = 0.003
    with Simulator(model) as sim:
        sim.run(simtime)

    assert allclose(sim.time, simtime)

    # test that sim.dt is read-only
    with pytest.raises(ReadonlyError, match="dt"):
        sim.dt = 0.002

    # test error for bad target
    with pytest.raises(ValidationError, match="target"):
        with Simulator(model, target="foo"):
            pass

    # test negative runtime
    with pytest.raises(ValidationError, match="[Mm]ust be positive"):
        with Simulator(model):
            sim.run(-0.1)

    # test zero step warning
    with pytest.warns(UserWarning, match="0 timesteps"):
        with Simulator(model):
            sim.run(1e-8)


@pytest.mark.target_loihi
def test_loihi_simulation_exception(Simulator):
    """Test that Loihi shuts down properly after exception during simulation"""

    def node_fn(t):
        if t < 0.002:
            return 0
        else:
            raise RuntimeError("exception to kill the simulation")

    with nengo.Network() as net:
        u = nengo.Node(node_fn)
        e = nengo.Ensemble(8, 1)
        nengo.Connection(u, e)

    with pytest.raises(RuntimeError, match="exception to kill"):
        with Simulator(net, precompute=False) as sim:
            sim.run(0.01)

    assert sim.sims["loihi"].closed


@pytest.mark.filterwarnings("ignore:Model is precomputable.")
@pytest.mark.parametrize("precompute", [True, False])
def test_double_run(precompute, Simulator, seed, allclose):
    simtime = 0.2
    with nengo.Network(seed=seed) as net:
        stim = nengo.Node(lambda t: np.sin((2 * np.pi / simtime) * t))
        ens = nengo.Ensemble(10, 1)
        probe = nengo.Probe(ens)
        nengo.Connection(stim, ens, synapse=None)

    with Simulator(net, precompute=True) as sim0:
        sim0.run(simtime)

    with Simulator(net, precompute=precompute) as sim1:
        sim1.run(simtime / 2)
        sim1.run(simtime / 2)

    assert allclose(sim1.time, sim0.time)
    assert len(sim1.trange()) == len(sim0.trange())
    assert allclose(sim1.data[probe], sim0.data[probe])


# These base-10 exp values translate to noiseExp of [5, 10, 13] on the chip.
@pytest.mark.parametrize("exp", [-4.5, -3, -2])
def test_simulator_noise(exp, request, plt, seed, allclose):
    # TODO: test that the mean falls within a number of standard errors
    # of the expected mean, and that non-zero offsets work correctly.
    # Currently, there is an unexpected negative bias for small noise
    # exponents, apparently because there is a probability of generating
    # the shifted equivalent of -128, whereas with e.g. exp = 7 all the
    # generated numbers fall in [-127, 127].
    offset = 0

    target = request.config.getoption("--target")
    n_compartments = 1000

    model = Model()
    block = LoihiBlock(n_compartments)
    model.add_block(block)

    block.compartment.configure_relu()
    block.compartment.vmin = -1
    block.compartment.enable_noise[:] = 1
    block.compartment.noise_exp = exp
    block.compartment.noise_offset = offset
    block.compartment.noise_at_membrane = 1

    probe = LoihiProbe(target=block, key="voltage")
    model.add_probe(probe)

    discretize_model(model)
    exp2 = block.compartment.noise_exp
    offset2 = block.compartment.noise_offset

    n_steps = 100
    if target == "loihi":
        with HardwareInterface(model, use_snips=False, seed=seed) as sim:
            sim.run_steps(n_steps)
            y = sim.get_probe_output(probe)
    else:
        with EmulatorInterface(model, seed=seed) as sim:
            sim.run_steps(n_steps)
            y = sim.get_probe_output(probe)

    t = np.arange(1, n_steps + 1)
    bias = offset2 * 2.0 ** (exp2 - 1)
    std = 2.0 ** exp2 / np.sqrt(3)  # divide by sqrt(3) for std of uniform -1..1
    rmean = t * bias
    rstd = np.sqrt(t) * std
    rerr = rstd / np.sqrt(n_compartments)
    ymean = y.mean(axis=1)
    ystd = y.std(axis=1)
    diffs = np.diff(np.vstack([np.zeros_like(y[0]), y]), axis=0)

    plt.subplot(311)
    plt.hist(diffs.ravel(), bins=256)

    plt.subplot(312)
    plt.plot(rmean, "k")
    plt.plot(rmean + 3 * rerr, "k--")
    plt.plot(rmean - 3 * rerr, "k--")
    plt.plot(ymean)
    plt.title("mean")

    plt.subplot(313)
    plt.plot(rstd, "k")
    plt.plot(ystd)
    plt.title("std")

    assert allclose(ystd, rstd, rtol=0.1, atol=1)


def test_population_input(request, allclose):
    target = request.config.getoption("--target")
    dt = 0.001

    n_inputs = 3
    n_axons = 1
    n_compartments = 2

    steps = 6
    spike_times_inds = [(1, [0]), (3, [1]), (5, [2])]

    model = Model()

    input = SpikeInput(n_inputs)
    model.add_input(input)
    spikes = [(input, ti, inds) for ti, inds in spike_times_inds]

    input_axon = Axon(n_axons)
    target_axons = np.zeros(n_inputs, dtype=int)
    atoms = np.arange(n_inputs)
    input_axon.set_compartment_axon_map(target_axons, atoms=atoms)
    input.add_axon(input_axon)

    block = LoihiBlock(n_compartments)
    block.compartment.configure_lif(tau_rc=0.0, tau_ref=0.0, dt=dt)
    block.compartment.configure_filter(0, dt=dt)
    model.add_block(block)

    synapse = Synapse(n_axons)
    weights = 0.1 * np.array([[[1, 2], [2, 3], [4, 5]]], dtype=float)
    indices = np.array([[[0, 1], [0, 1], [0, 1]]], dtype=int)
    axon_to_weight_map = np.zeros(n_axons, dtype=int)
    bases = np.zeros(n_axons, dtype=int)
    synapse.set_population_weights(
        weights, indices, axon_to_weight_map, bases, pop_type=32
    )
    block.add_synapse(synapse)
    input_axon.target = synapse

    probe = LoihiProbe(target=block, key="voltage")
    model.add_probe(probe)

    discretize_model(model)

    if target == "loihi":
        with HardwareInterface(model, use_snips=True) as sim:
            sim.run_steps(steps, blocking=False)
            for ti in range(1, steps + 1):
                spikes_i = [spike for spike in spikes if spike[1] == ti]
                sim.host2chip(spikes=spikes_i, errors=[])
                sim.chip2host(probes_receivers={})

            y = sim.get_probe_output(probe)
    else:
        for inp, ti, inds in spikes:
            inp.add_spikes(ti, inds)

        with EmulatorInterface(model) as sim:
            sim.run_steps(steps)
            y = sim.get_probe_output(probe)

    vth = block.compartment.vth[0]
    assert (block.compartment.vth == vth).all()
    z = y / vth
    assert allclose(z[[1, 3, 5]], weights[0], atol=4e-2, rtol=0)


@pytest.mark.filterwarnings("ignore:Model is precomputable.")
def test_precompute(allclose, Simulator, seed, plt):
    simtime = 0.2

    with nengo.Network(seed=seed) as model:
        D = 2
        stim = nengo.Node(lambda t: [np.sin(t * 2 * np.pi / simtime)] * D)

        a = nengo.Ensemble(100, D)

        nengo.Connection(stim, a)

        output = nengo.Node(size_in=D)

        nengo.Connection(a, output)

        p_stim = nengo.Probe(stim, synapse=0.03)
        p_a = nengo.Probe(a, synapse=0.03)
        p_out = nengo.Probe(output, synapse=0.03)

    with Simulator(model, precompute=False) as sim1:
        sim1.run(simtime)
    with Simulator(model, precompute=True) as sim2:
        sim2.run(simtime)

    plt.subplot(2, 1, 1)
    plt.plot(sim1.trange(), sim1.data[p_stim])
    plt.plot(sim1.trange(), sim1.data[p_a])
    plt.plot(sim1.trange(), sim1.data[p_out])
    plt.title("precompute=False")
    plt.subplot(2, 1, 2)
    plt.plot(sim2.trange(), sim2.data[p_stim])
    plt.plot(sim2.trange(), sim2.data[p_a])
    plt.plot(sim2.trange(), sim2.data[p_out])
    plt.title("precompute=True")

    # check that each is using the right placement
    assert stim in sim1.model.host.params
    assert stim not in sim1.model.host_pre.params
    assert stim not in sim2.model.host.params
    assert stim in sim2.model.host_pre.params

    assert p_stim not in sim1.model.params
    assert p_stim in sim1.model.host.params
    assert p_stim not in sim1.model.host_pre.params

    assert p_stim not in sim2.model.params
    assert p_stim not in sim2.model.host.params
    assert p_stim in sim2.model.host_pre.params

    for sim in (sim1, sim2):
        assert a in sim.model.params
        assert a not in sim.model.host.params
        assert a not in sim.model.host_pre.params

        assert output not in sim.model.params
        assert output in sim.model.host.params
        assert output not in sim.model.host_pre.params

        assert p_a in sim.model.params
        assert p_a not in sim.model.host.params
        assert p_a not in sim.model.host_pre.params

        assert p_out not in sim.model.params
        assert p_out in sim.model.host.params
        assert p_out not in sim.model.host_pre.params

    assert np.array_equal(sim1.data[p_stim], sim2.data[p_stim])
    assert sim1.target == sim2.target

    # precompute should not make a difference in outputs
    assert allclose(sim1.data[p_a], sim2.data[p_a])
    assert allclose(sim1.data[p_out], sim2.data[p_out])


@pytest.mark.target_loihi
def test_input_node_precompute(allclose, Simulator, plt):
    simtime = 1.0
    input_fn = lambda t: np.sin(6 * np.pi * t / simtime)
    targets = ["sim", "loihi"]
    x = {}
    u = {}
    v = {}
    for target in targets:
        n = 4
        with nengo.Network(seed=1) as model:
            inp = nengo.Node(input_fn)

            a = nengo.Ensemble(n, 1)
            ap = nengo.Probe(a, synapse=0.01)
            aup = nengo.Probe(a.neurons, "input")
            avp = nengo.Probe(a.neurons, "voltage")

            nengo.Connection(inp, a)

        with Simulator(model, precompute=True, target=target) as sim:
            print("Running in {}".format(target))
            sim.run(simtime)

        synapse = nengo.synapses.Lowpass(0.03)
        x[target] = synapse.filt(sim.data[ap])

        u[target] = sim.data[aup][:25]
        u[target] = (
            np.round(u[target] * 1000)
            if str(u[target].dtype).startswith("float")
            else u[target]
        )

        v[target] = sim.data[avp][:25]
        v[target] = (
            np.round(v[target] * 1000)
            if str(v[target].dtype).startswith("float")
            else v[target]
        )

        plt.plot(sim.trange(), x[target], label=target)

    t = sim.trange()
    u = input_fn(t)
    plt.plot(t, u, "k:", label="input")
    plt.legend(loc="best")

    assert allclose(x["sim"], x["loihi"], atol=0.1, rtol=0.01)


@pytest.mark.parametrize("remove_passthrough", [True, False])
def test_simulator_passthrough(remove_passthrough, Simulator):
    with nengo.Network() as model:
        host_input = nengo.Node(0)
        host_a = nengo.Node(size_in=1)
        host_b = nengo.Node(size_in=1)

        chip_x = nengo.Ensemble(10, 1)
        remove_c = nengo.Node(size_in=1)
        chip_y = nengo.Ensemble(10, 1)

        host_d = nengo.Node(size_in=1)

        conn_input_a = nengo.Connection(host_input, host_a)
        conn_a_b = nengo.Connection(host_a, host_b)
        conn_b_x = nengo.Connection(host_b, chip_x)
        conn_x_c = nengo.Connection(chip_x, remove_c)
        conn_c_y = nengo.Connection(remove_c, chip_y)
        conn_y_d = nengo.Connection(chip_y, host_d)

        probe_y = nengo.Probe(chip_y)
        probe_d = nengo.Probe(host_d)

    with Simulator(model, remove_passthrough=remove_passthrough) as sim:
        pass

    # model is only precomputable if the passthrough has been removed
    assert sim.precompute == remove_passthrough
    host_pre_params = (sim.model.host_pre if sim.precompute else sim.model.host).params

    assert host_input in host_pre_params
    assert probe_d in sim.model.host.params

    assert chip_x in sim.model.params
    assert chip_y in sim.model.params
    assert probe_y in sim.model.params

    # Passthrough nodes are not removed on the host
    assert host_a in host_pre_params
    assert host_b in host_pre_params
    assert host_d in sim.model.host.params
    assert conn_input_a in host_pre_params
    assert conn_a_b in host_pre_params

    if remove_passthrough:
        assert remove_c not in sim.model.host.params
    else:
        assert remove_c in sim.model.host.params

    # These connections currently aren't built in either case
    for model in (sim.model, sim.model.host):
        assert conn_b_x not in model.params
        assert conn_x_c not in model.params
        assert conn_c_y not in model.params
        assert conn_y_d not in model.params


def test_slicing_bugs(Simulator, seed):

    n = 50
    with nengo.Network() as model:
        a = nengo.Ensemble(n, 1, label="a")
        p0 = nengo.Probe(a[0])
        p = nengo.Probe(a)

    with Simulator(model) as sim:
        sim.run(0.1)

    assert np.allclose(sim.data[p0], sim.data[p])
    assert a in sim.model.params
    assert a not in sim.model.host.params

    with nengo.Network() as model:
        nengo_loihi.add_params(model)

        a = nengo.Ensemble(n, 1, label="a")

        b0 = nengo.Ensemble(n, 1, label="b0", seed=seed)
        model.config[b0].on_chip = False
        nengo.Connection(a[0], b0)

        b = nengo.Ensemble(n, 1, label="b", seed=seed)
        model.config[b].on_chip = False
        nengo.Connection(a, b)

        p0 = nengo.Probe(b0)
        p = nengo.Probe(b)

    with Simulator(model) as sim:
        sim.run(0.1)

    assert np.allclose(sim.data[p0], sim.data[p])
    assert a in sim.model.params
    assert a not in sim.model.host.params
    assert b not in sim.model.params
    assert b in sim.model.host.params


def test_network_unchanged(Simulator):
    with nengo.Network() as model:
        nengo.Ensemble(100, 1)
        with Simulator(model):
            pass
        assert model.all_networks == []


def test_timers():
    timers = Timers()
    timers.start("1")
    timers.start("2")
    timers.start("3")
    timers.stop("3")
    timers.reset("2")
    timers.stop("1")

    assert len(timers) == 3
    n_totals = 0
    for total in timers:
        n_totals += 1
    assert n_totals == 3
    assert timers["1"] > timers["3"]
    assert timers["2"] == 0.0
    assert "2" not in timers._last_start

    regex = re.compile(r"<Timers: {'1': [0-9\.]+, '2': [0-9\.]+, '3': [0-9\.]+}>")
    assert regex.fullmatch(repr(timers))
    assert regex.fullmatch(str(timers))
    assert repr(timers) == str(timers)
