import nengo
from nengo.exceptions import BuildError
from nengo.utils.matplotlib import rasterplot
import numpy as np
import pytest

from nengo_loihi.config import add_params


@pytest.mark.parametrize('weight_solver', [False, True])
@pytest.mark.parametrize('target_value', [-0.75, 0.4, 1.0])
def test_ens_ens_constant(
        allclose, weight_solver, target_value, Simulator, seed, plt):
    a_fn = lambda x: x + target_value
    solver = nengo.solvers.LstsqL2(weights=weight_solver)

    bnp = None
    with nengo.Network(seed=seed) as model:
        a = nengo.Ensemble(100, 1, label='a')

        b = nengo.Ensemble(101, 1, label='b')
        nengo.Connection(a, b, function=a_fn, solver=solver)
        bp = nengo.Probe(b)
        bnp = nengo.Probe(b.neurons)

        c = nengo.Ensemble(1, 1, label='c')
        bc_conn = nengo.Connection(b, c)

    with Simulator(model) as sim:
        sim.run(1.0)

    t = sim.trange()
    bcount = (sim.data[bnp] > 0).mean(axis=0)

    b_decoders = sim.data[bc_conn].weights
    dec_value = np.dot(b_decoders, bcount)

    output_filter = nengo.synapses.Alpha(0.03)
    target_output = target_value * np.ones_like(t)
    sim_output = output_filter.filt(sim.data[bp])
    plt.plot(t, target_output, 'k')
    plt.plot(t, sim_output)

    assert allclose(dec_value, target_value, rtol=0.1, atol=0.1)
    t_check = t > 0.5
    assert allclose(
        sim_output[t_check], target_output[t_check], rtol=0.15, atol=0.15)


@pytest.mark.parametrize('dt', [3e-4, 1e-3])
@pytest.mark.parametrize('precompute', [True, False])
def test_node_to_neurons(dt, precompute, allclose, Simulator, plt):
    tfinal = 0.4

    x = np.array([0.7, 0.3])
    A = np.array([[1, 1],
                  [1, -1],
                  [1, -0.5]])
    y = np.dot(A, x)

    gain = [3] * len(y)
    bias = [0] * len(y)

    neuron_type = nengo.LIF()
    z = neuron_type.rates(y, gain, bias)

    with nengo.Network() as model:
        u = nengo.Node(x, label='u')
        a = nengo.Ensemble(len(y), 1, label='a',
                           neuron_type=neuron_type, gain=gain, bias=bias)
        ap = nengo.Probe(a.neurons)
        nengo.Connection(u, a.neurons, transform=A)

    with Simulator(model, dt=dt, precompute=precompute) as sim:
        sim.run(tfinal)

    tsum = tfinal / 2
    t = sim.trange()
    rates = (sim.data[ap][t > t[-1] - tsum] > 0).sum(axis=0) / tsum

    bar_width = 0.35
    plt.bar(np.arange(len(z)), z, bar_width, color='k', label='z')
    plt.bar(np.arange(len(z)) + bar_width, rates, bar_width, label='rates')
    plt.legend(loc='best')

    assert allclose(rates, z, atol=3, rtol=0.1)


@pytest.mark.parametrize("factor", [0.11, 0.26, 1.01])
def test_neuron_to_neuron(Simulator, factor, seed, allclose, plt):
    # note: we use these weird factor values so that voltages don't line up
    # exactly with the firing threshold.  since loihi neurons fire when
    # voltage > threshold (rather than >=), if the voltages line up
    # exactly then we need an extra spike each time to push `b` over threshold
    dt = 5e-4
    simtime = 0.2

    with nengo.Network(seed=seed) as net:
        n = 10
        stim = nengo.Node(lambda t: [np.sin(t * 2 * np.pi / simtime)])
        a = nengo.Ensemble(n, 1)
        nengo.Connection(stim, a)

        b = nengo.Ensemble(n, 1, neuron_type=nengo.SpikingRectifiedLinear(),
                           gain=np.ones(n), bias=np.zeros(n))
        nengo.Connection(a.neurons, b.neurons, synapse=None,
                         transform=np.eye(n) * factor)

        p_a = nengo.Probe(a.neurons)
        p_b = nengo.Probe(b.neurons)

    with Simulator(net, dt=dt) as sim:
        sim.run(simtime)

    y_ref = np.floor(np.sum(sim.data[p_a] > 0, axis=0) * factor)
    y_sim = np.sum(sim.data[p_b] > 0, axis=0)
    plt.plot(y_ref, c='k')
    plt.plot(y_sim)

    assert allclose(y_sim, y_ref, atol=1)


def test_ensemble_to_neurons(Simulator, seed, allclose, plt):
    with nengo.Network(seed=seed) as net:
        stim = nengo.Node(lambda t: [np.sin(t * 2 * np.pi)])
        pre = nengo.Ensemble(20, 1)
        nengo.Connection(stim, pre)

        post = nengo.Ensemble(2, 1,
                              gain=[1., 1.], bias=[0., 0.])

        # On and off neurons
        nengo.Connection(pre, post.neurons,
                         synapse=None, transform=[[5], [-5]])

        p_pre = nengo.Probe(pre, synapse=nengo.synapses.Alpha(0.03))
        p_post = nengo.Probe(post.neurons)

    # Compare to Nengo
    with nengo.Simulator(net) as nengosim:
        nengosim.run(1.0)

    with Simulator(net) as sim:
        sim.run(1.0)

    t = sim.trange()
    plt.subplot(2, 1, 1)
    plt.title("Reference Nengo")
    plt.plot(t, nengosim.data[p_pre], c='k')
    plt.ylabel("Decoded pre value")
    plt.xlabel("Time (s)")
    plt.twinx()
    rasterplot(t, nengosim.data[p_post])
    plt.ylabel("post neuron number")
    plt.subplot(2, 1, 2)
    plt.title("Nengo Loihi")
    plt.plot(t, sim.data[p_pre], c='k')
    plt.ylabel("Decoded pre value")
    plt.xlabel("Time (s)")
    plt.twinx()
    rasterplot(t, sim.data[p_post])
    plt.ylabel("post neuron number")

    plt.tight_layout()

    # Compare the number of spikes for each neuron.
    # We'll let them be off by 5 for now.
    assert allclose(np.sum(sim.data[p_post], axis=0) * sim.dt,
                    np.sum(nengosim.data[p_post], axis=0) * nengosim.dt,
                    atol=5)


def test_dists(Simulator, seed):
    # check that distributions on connection transforms are handled correctly

    with nengo.Network(seed=seed) as net:
        a = nengo.Node([1])
        b = nengo.Ensemble(50, 1, radius=2)
        conn0 = nengo.Connection(a, b, transform=nengo.dists.Uniform(-1, 1))
        c = nengo.Ensemble(50, 1)
        nengo.Connection(b, c, transform=nengo.dists.Uniform(-1, 1),
                         seed=seed + 3)
        d = nengo.Ensemble(50, 1)
        conn1 = nengo.Connection(c.neurons, d.neurons,
                                 transform=nengo.dists.Uniform(-1, 1))

        add_params(net)
        net.config[d].on_chip = False

        p0 = nengo.Probe(c)
        p1 = nengo.Probe(d)
        p2 = nengo.Probe(b.neurons)

    simtime = 0.1

    with Simulator(net) as sim:
        sim.run(simtime)

    with Simulator(net) as sim2:
        sim2.run(simtime)

    assert np.allclose(sim.data[p0], sim2.data[p0])
    assert np.allclose(sim.data[p1], sim2.data[p1])
    assert np.allclose(sim.data[p2], sim2.data[p2])

    conn0.seed = seed + 1
    with Simulator(net) as sim2:
        sim2.run(simtime)

    assert not np.allclose(sim.data[p2], sim2.data[p2])

    conn0.seed = None
    conn1.seed = seed + 1
    with Simulator(net) as sim2:
        sim2.run(simtime)
    assert not np.allclose(sim.data[p1], sim2.data[p1])


def test_long_tau(Simulator):
    with nengo.Network() as model:
        u = nengo.Node(0)
        x = nengo.Ensemble(100, 1)
        nengo.Connection(u, x, synapse=0.1)
        nengo.Connection(x, x, synapse=0.1)

    with Simulator(model) as sim:
        sim.run(0.002)  # Ensure it at least runs


def test_zero_activity_error(Simulator):
    with nengo.Network() as net:
        a = nengo.Ensemble(5, 1,
                           encoders=nengo.dists.Choice([[1.]]),
                           intercepts=nengo.dists.Choice([0.]))
        b = nengo.Ensemble(5, 1)
        nengo.Connection(a, b, eval_points=[[-1]])

    with pytest.raises(BuildError, match="activit.*zero"):
        with Simulator(net):
            pass


def test_chip_to_host_function_points(Simulator, seed, plt, allclose):
    """Connection from chip to host that computes a function using points"""
    fn = lambda x: -x
    probe_syn = nengo.Lowpass(0.03)
    simtime = 0.3

    with nengo.Network(seed=seed) as net:
        u = nengo.Node(lambda t: np.sin((2*np.pi/simtime) * t))
        a = nengo.Ensemble(100, 1)
        # v has a function so it doesn't get removed as passthrough
        v = nengo.Node(lambda t, x: x + 1e-8, size_in=1)
        nengo.Connection(u, a, synapse=None)

        x = np.linspace(-1, 1, 1000).reshape(-1, 1)
        y = fn(x)
        nengo.Connection(a, v, synapse=None, eval_points=x, function=y)

        up = nengo.Probe(u, synapse=probe_syn.combine(nengo.Lowpass(0.005)))
        vp = nengo.Probe(v, synapse=probe_syn)

    with Simulator(net) as sim:
        sim.run(simtime)

    y_ref = fn(sim.data[up])
    plt.plot(sim.trange(), y_ref)
    plt.plot(sim.trange(), sim.data[vp])
    assert allclose(sim.data[vp], y_ref, atol=0.1)


@pytest.mark.parametrize("val", (-0.75, -0.5, 0, 0.5, 0.75))
@pytest.mark.parametrize("type", ("array", "func"))
def test_input_node(allclose, Simulator, val, type):
    with nengo.Network() as net:
        if type == "array":
            input = [val]
        else:
            input = lambda t: [val]
        a = nengo.Node(input)

        b = nengo.Ensemble(100, 1)
        nengo.Connection(a, b)

        # create a second path so that we test nodes with multiple outputs
        c = nengo.Ensemble(100, 1)
        nengo.Connection(a, c)

        p_b = nengo.Probe(b, synapse=0.1)
        p_c = nengo.Probe(c, synapse=0.1)

    with Simulator(net, precompute=True) as sim:
        sim.run(1.0)

    # TODO: seems like error margins should be smaller than this?
    assert allclose(sim.data[p_b][-100:], val, atol=0.15)
    assert allclose(sim.data[p_c][-100:], val, atol=0.15)


@pytest.mark.parametrize(
    "pre_d, post_d, func",
    [(1, 1, False),
     (1, 3, False),
     (3, 1, True),
     (3, 3, True)]
)
def test_ens2node(allclose, Simulator, seed, plt, pre_d, post_d, func):
    simtime = 0.5
    with nengo.Network(seed=seed) as model:
        stim = nengo.Node(lambda t: [np.sin(t * 2 * np.pi / simtime)] * pre_d)

        a = nengo.Ensemble(100, pre_d)

        nengo.Connection(stim, a)

        data = []
        output = nengo.Node(lambda t, x: data.append(x),
                            size_in=post_d, size_out=0)

        transform = np.identity(max(pre_d, post_d))
        transform = transform[:post_d, :pre_d]
        if func:
            def conn_func(x):
                return -x
        else:
            conn_func = None
        nengo.Connection(a, output, transform=transform, function=conn_func)

        p_stim = nengo.Probe(stim)

    with Simulator(model) as sim:
        sim.run(simtime)

    filt = nengo.synapses.Lowpass(0.03)
    filt_data = filt.filt(np.array(data))

    # TODO: improve the bounds on these tests
    if post_d >= pre_d:
        assert allclose(
            filt_data[:, :pre_d] * (-1 if func else 1),
            sim.data[p_stim][:, :pre_d],
            atol=0.6)
        assert allclose(filt_data[:, pre_d:], 0, atol=0.6)
    else:
        assert allclose(
            filt_data * (-1 if func else 1),
            sim.data[p_stim][:, :post_d],
            atol=0.6)

    plt.subplot(2, 1, 1)
    plt.plot(sim.trange(), sim.data[p_stim])
    plt.title('Input (should be %d sine waves)' % pre_d)
    plt.legend(['%d' % i for i in range(pre_d)], loc='best')
    plt.subplot(2, 1, 2)
    n_sine = min(pre_d, post_d)
    status = ' flipped' if func else ''
    n_flat = post_d - n_sine
    plt.title('Output (should be %d%s sine waves and %d flat lines)' %
              (n_sine, status, n_flat))
    plt.plot(sim.trange(), filt_data)
    plt.legend(['%d' % i for i in range(post_d)], loc='best')


def test_neurons2node(Simulator, seed, plt):
    with nengo.Network(seed=seed) as model:
        stim = nengo.Node(lambda t: [np.sin(t * 2 * np.pi)])
        p_stim = nengo.Probe(stim)

        a = nengo.Ensemble(100, 1,
                           intercepts=nengo.dists.Choice([0]))

        nengo.Connection(stim, a)

        data = []
        output = nengo.Node(lambda t, x: data.append(x),
                            size_in=a.n_neurons, size_out=0)
        nengo.Connection(a.neurons, output, synapse=None)

    with Simulator(model, precompute=True) as sim:
        sim.run(1.0)

    rasterplot(sim.trange(), np.array(data), ax=plt.gca())
    plt.twinx()
    plt.plot(sim.trange(), sim.data[p_stim])
    plt.title('Raster plot for sine input')

    pre = np.asarray(data[:len(data) // 2 - 100])
    post = np.asarray(data[len(data) // 2 + 100:])
    on_neurons = np.squeeze(sim.data[a].encoders == 1)
    assert np.sum(pre[:, on_neurons]) > 0
    assert np.sum(post[:, on_neurons]) == 0
    assert np.sum(pre[:, np.logical_not(on_neurons)]) == 0
    assert np.sum(post[:, np.logical_not(on_neurons)]) > 0


@pytest.mark.parametrize(
    "pre_d, post_d, func",
    [(1, 1, False),
     (1, 3, False),
     (3, 1, True),
     (3, 3, True)]
)
def test_node2ens(allclose, Simulator, seed, plt, pre_d, post_d, func):
    simtime = 0.5
    with nengo.Network(seed=seed) as model:
        stim = nengo.Node(lambda t: [np.sin(t * 2 * np.pi / simtime)] * pre_d)

        a = nengo.Ensemble(100, post_d)

        transform = np.identity(max(pre_d, post_d))
        transform = transform[:post_d, :pre_d]
        if func:
            def function(x):
                return -x
        else:
            function = None
        nengo.Connection(stim, a, transform=transform, function=function)

        p_stim = nengo.Probe(stim)
        p_a = nengo.Probe(a, synapse=0.03)

    with Simulator(model) as sim:
        sim.run(simtime)

    # TODO: improve the bounds on these tests
    if post_d >= pre_d:
        assert allclose(
            sim.data[p_stim][:, :pre_d],
            sim.data[p_a][:, :pre_d] * (-1 if func else 1),
            atol=0.6)
        assert allclose(
            sim.data[p_a][:, pre_d:] * (-1 if func else 1), 0, atol=0.6)
    else:
        assert allclose(
            sim.data[p_stim][:, :post_d],
            sim.data[p_a] * (-1 if func else 1),
            atol=0.6)

    plt.subplot(2, 1, 1)
    plt.plot(sim.trange(), sim.data[p_stim])
    plt.title('Input (should be %d sine waves)' % pre_d)
    plt.legend(['%d' % i for i in range(pre_d)], loc='best')
    plt.subplot(2, 1, 2)
    n_sine = min(pre_d, post_d)
    status = ' flipped' if func else ''
    n_flat = post_d - n_sine
    plt.title('Output (should be %d%s sine waves and %d flat lines)' %
              (n_sine, status, n_flat))
    plt.plot(sim.trange(), sim.data[p_a])
    plt.legend(['%d' % i for i in range(post_d)], loc='best')


@pytest.mark.parametrize('precompute', [True, False])
def test_ens_decoded_on_host(precompute, allclose, Simulator, seed, plt):
    out_synapse = nengo.synapses.Alpha(0.03)
    simtime = 0.6

    with nengo.Network(seed=seed) as model:
        add_params(model)

        stim = nengo.Node(lambda t: [np.sin(t * 2 * np.pi / simtime)])

        a = nengo.Ensemble(100, 1)
        model.config[a].on_chip = False

        b = nengo.Ensemble(100, 1)

        nengo.Connection(stim, a)

        nengo.Connection(a, b, function=lambda x: -x)

        p_stim = nengo.Probe(stim, synapse=out_synapse)
        p_a = nengo.Probe(a, synapse=out_synapse)
        p_b = nengo.Probe(b, synapse=out_synapse)

    with Simulator(model, precompute=precompute) as sim:
        sim.run(simtime)

    plt.plot(sim.trange(), sim.data[p_stim])
    plt.plot(sim.trange(), sim.data[p_a])
    plt.plot(sim.trange(), sim.data[p_b])

    assert allclose(sim.data[p_a], sim.data[p_stim], atol=0.05, rtol=0.01)
    assert allclose(sim.data[p_b], -sim.data[p_a], atol=0.15, rtol=0.1)


@pytest.mark.parametrize('seed_ens', [True, False])
@pytest.mark.parametrize('precompute', [True, False])
def test_n2n_on_host(precompute, allclose, Simulator, seed_ens, seed, plt):
    """Ensure that neuron to neuron connections work on and off chip."""

    if not seed_ens and nengo.version.version_info <= (2, 8, 0):
        plt.saveas = None
        pytest.xfail("Seeds change when moving ensembles off/on chip")

    n_neurons = 50
    # When the ensemble is seeded, the output plots will make more sense,
    # but the test should work whether they're seeded or not.
    ens_seed = (seed + 1) if seed_ens else None
    simtime = 1.0

    with nengo.Network(seed=seed) as model:
        add_params(model)

        stim = nengo.Node(lambda t: [np.sin(t * 2 * np.pi / simtime)])

        # pre receives stimulation and represents the sine wave
        pre = nengo.Ensemble(n_neurons, dimensions=1, seed=ens_seed)
        model.config[pre].on_chip = False
        nengo.Connection(stim, pre)

        # post has pre's neural activity forwarded to it.
        # Since the neuron parameters are the same, it should also represent
        # the same sine wave.
        # The 0.015 scaling is chosen so the values match visually,
        # though a more principled reason would be better.
        post = nengo.Ensemble(n_neurons, dimensions=1, seed=ens_seed)
        nengo.Connection(pre.neurons, post.neurons,
                         transform=np.eye(n_neurons) * 0.015)

        p_synapse = nengo.synapses.Alpha(0.03)
        p_stim = nengo.Probe(stim, synapse=p_synapse)
        p_pre = nengo.Probe(pre, synapse=p_synapse)
        p_post = nengo.Probe(post, synapse=p_synapse)

    with Simulator(model, precompute=precompute) as sim:
        sim.run(simtime)
    t = sim.trange()

    model.config[pre].on_chip = True

    with Simulator(model, precompute=precompute) as sim2:
        sim2.run(simtime)
    t2 = sim2.trange()

    plt.plot(t, sim.data[p_stim], c="k", label="input")
    plt.plot(t, sim.data[p_pre], label="pre off-chip")
    plt.plot(t, sim.data[p_post], label="post (pre off-chip)")
    plt.plot(t2, sim2.data[p_pre], label="pre on-chip")
    plt.plot(t2, sim2.data[p_post], label="post (pre on-chip)")
    plt.legend()

    assert allclose(sim.data[p_pre], sim2.data[p_pre], atol=0.1)
    assert allclose(sim.data[p_post], sim2.data[p_post], atol=0.1)
