import nengo
from nengo.exceptions import ValidationError, SimulationError
from nengo.utils.numpy import rms
import numpy as np
import pytest

import nengo_loihi.builder


def pes_network(
        n_per_dim,
        dims,
        seed,
        learning_rule_type=nengo.PES(learning_rate=1e-3),
        input_scale=None,
        error_scale=1.,
        learn_synapse=0.005,
        probe_synapse=0.02,
):
    if input_scale is None:
        input_scale = np.linspace(1, 0, dims + 1)[:-1]
    assert input_scale.size == dims

    input_fn = lambda t: np.sin(t * 2 * np.pi) * input_scale

    probes = {}
    with nengo.Network(seed=seed) as model:
        stim = nengo.Node(input_fn)

        pre = nengo.Ensemble(n_per_dim * dims, dims)
        post = nengo.Node(size_in=dims)

        nengo.Connection(stim, pre, synapse=None)
        conn = nengo.Connection(
            pre, post,
            function=lambda x: np.zeros(dims),
            synapse=learn_synapse,
            learning_rule_type=learning_rule_type)

        nengo.Connection(post, conn.learning_rule, transform=error_scale)
        nengo.Connection(stim, conn.learning_rule, transform=-error_scale)

        probes['stim'] = nengo.Probe(stim, synapse=probe_synapse)
        probes['pre'] = nengo.Probe(pre, synapse=probe_synapse)
        probes['post'] = nengo.Probe(post, synapse=probe_synapse)

    return model, probes


@pytest.mark.parametrize('n_per_dim', [120, 200])
@pytest.mark.parametrize('dims', [1, 3])
def test_pes_comm_channel(allclose, plt, seed, Simulator, n_per_dim, dims):
    tau = 0.01
    model, probes = pes_network(n_per_dim, dims, seed, learn_synapse=tau)

    simtime = 5.0
    with nengo.Simulator(model) as nengo_sim:
        nengo_sim.run(simtime)

    with Simulator(model) as loihi_sim:
        loihi_sim.run(simtime)

    with Simulator(model, target='simreal') as real_sim:
        real_sim.run(simtime)

    t = nengo_sim.trange()
    pre_tmask = t > 0.1
    post_tmask = t > simtime - 1.0

    dec_tau = loihi_sim.model.decode_tau
    y = nengo_sim.data[probes['stim']]
    y_dpre = nengo.Lowpass(dec_tau).filt(y)
    y_dpost = nengo.Lowpass(tau).combine(nengo.Lowpass(dec_tau)).filt(y_dpre)
    y_nengo = nengo_sim.data[probes['post']]
    y_loihi = loihi_sim.data[probes['post']]
    y_real = real_sim.data[probes['post']]

    plt.subplot(211)
    plt.plot(t, y_dpost, 'k', label='target')
    plt.plot(t, y_nengo, 'b', label='nengo')
    plt.plot(t, y_loihi, 'g', label='loihi')
    plt.plot(t, y_real, 'r:', label='real')
    plt.legend()

    plt.subplot(212)
    plt.plot(t[post_tmask], y_loihi[post_tmask] - y_dpost[post_tmask], 'k')
    plt.plot(t[post_tmask], y_loihi[post_tmask] - y_nengo[post_tmask], 'b')

    x_loihi = loihi_sim.data[probes['pre']]
    assert allclose(x_loihi[pre_tmask], y_dpre[pre_tmask],
                    atol=0.1, rtol=0.05)

    assert allclose(y_loihi[post_tmask], y_dpost[post_tmask],
                    atol=0.1, rtol=0.05)
    assert allclose(y_loihi, y_nengo, atol=0.2, rtol=0.2)

    assert allclose(y_real[post_tmask], y_dpost[post_tmask],
                    atol=0.1, rtol=0.05)
    assert allclose(y_real, y_nengo, atol=0.2, rtol=0.2)


def test_pes_overflow(allclose, plt, seed, Simulator):
    dims = 3
    n_per_dim = 120
    tau = 0.01
    model, probes = pes_network(n_per_dim, dims, seed, learn_synapse=tau,
                                input_scale=np.linspace(1, 0.7, dims))

    simtime = 3.0
    loihi_model = nengo_loihi.builder.Model()
    # set learning_wgt_exp low to create overflow in weight values
    loihi_model.pes_wgt_exp = -1

    with Simulator(model, model=loihi_model) as loihi_sim:
        loihi_sim.run(simtime)

    t = loihi_sim.trange()
    post_tmask = t > simtime - 1.0

    dec_tau = loihi_sim.model.decode_tau
    y = loihi_sim.data[probes['stim']]
    y_dpre = nengo.Lowpass(dec_tau).filt(y)
    y_dpost = nengo.Lowpass(tau).combine(nengo.Lowpass(dec_tau)).filt(y_dpre)
    y_loihi = loihi_sim.data[probes['post']]

    plt.plot(t, y_dpost, 'k', label='target')
    plt.plot(t, y_loihi, 'g', label='loihi')

    # --- fit output to scaled version of target output
    z_ref0 = y_dpost[post_tmask][:, 0]
    z_loihi = y_loihi[post_tmask]
    scale = np.linspace(0, 1, 50)
    E = np.abs(z_loihi - scale[:, None, None]*z_ref0[:, None])
    errors = E.mean(axis=1)  # average over time (errors is: scales x dims)
    for j in range(dims):
        errors_j = errors[:, j]
        i = np.argmin(errors_j)
        assert errors_j[i] < 0.1, ("Learning output for dim %d did not match "
                                   "any scaled version of the target output"
                                   % j)
        assert scale[i] > 0.4, "Learning output for dim %d is too small" % j
        assert scale[i] < 0.7, ("Learning output for dim %d is too large "
                                "(weights or traces not clipping as expected)"
                                % j)


def test_pes_error_clip(allclose, plt, seed, Simulator):
    dims = 2
    n_per_dim = 120
    tau = 0.01
    error_scale = 5.  # scale up error signal so it clips
    model, probes = pes_network(
        n_per_dim, dims, seed, learn_synapse=tau,
        learning_rule_type=nengo.PES(learning_rate=1e-3 / error_scale),
        input_scale=np.array([1., -1.]),
        error_scale=error_scale)

    simtime = 3.0
    with pytest.warns(UserWarning, match=r'.*PES error.*Clipping.'):
        with Simulator(model) as loihi_sim:
            loihi_sim.run(simtime)

    t = loihi_sim.trange()
    post_tmask = t > simtime - 1.0

    dec_tau = loihi_sim.model.decode_tau
    y = loihi_sim.data[probes['stim']]
    y_dpre = nengo.Lowpass(dec_tau).filt(y)
    y_dpost = nengo.Lowpass(tau).combine(nengo.Lowpass(dec_tau)).filt(y_dpre)
    y_loihi = loihi_sim.data[probes['post']]

    plt.plot(t, y_dpost, 'k', label='target')
    plt.plot(t, y_loihi, 'g', label='loihi')

    # --- assert that we've learned something, but not everything
    error = (rms(y_loihi[post_tmask] - y_dpost[post_tmask])
             / rms(y_dpost[post_tmask]))
    assert error < 0.5
    assert error > 0.05
    # ^ error on emulator vs chip is quite different, hence large tolerances


@pytest.mark.parametrize('init_function', [None, lambda x: 0])
def test_multiple_pes(init_function, allclose, plt, seed, Simulator):
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
                learning_rule_type=nengo.PES(learning_rate=3e-3),
            )
            nengo.Connection(target[i], conn.learning_rule, transform=-1)
            nengo.Connection(output[i], conn.learning_rule)

        probe = nengo.Probe(output, synapse=0.1)

    simtime = 2.5
    with Simulator(model) as sim:
        sim.run(simtime)

    t = sim.trange()
    tmask = t > simtime * 0.85

    plt.plot(t, sim.data[probe])
    for target, style in zip(targets, plt.rcParams["axes.prop_cycle"]):
        plt.axhline(target, **style)

    for i, target in enumerate(targets):
        assert allclose(sim.data[probe][tmask, i], target,
                        atol=0.05, rtol=0.05), "Target %d not close" % i


def test_pes_pre_synapse_type_error(Simulator):
    with nengo.Network() as model:
        pre = nengo.Ensemble(10, 1)
        post = nengo.Node(size_in=1)
        rule_type = nengo.PES(pre_synapse=nengo.Alpha(0.005))
        conn = nengo.Connection(pre, post, learning_rule_type=rule_type)
        nengo.Connection(post, conn.learning_rule)

    with pytest.raises(ValidationError):
        with Simulator(model):
            pass


def test_pes_trace_increment_clip_warning(seed, Simulator):
    dims = 2
    n_per_dim = 120
    model, _ = pes_network(
        n_per_dim, dims, seed,
        learning_rule_type=nengo.PES(learning_rate=1e-1))

    with pytest.warns(UserWarning, match="Trace increment exceeds upper"):
        with Simulator(model):
            pass


def test_drop_trace_spikes(Simulator, seed):
    with nengo.Network(seed=seed) as net:
        a = nengo.Ensemble(10, 1, gain=nengo.dists.Choice([1]),
                           bias=nengo.dists.Choice([2000]),
                           neuron_type=nengo.SpikingRectifiedLinear())
        b = nengo.Node(size_in=1)

        conn = nengo.Connection(a, b, learning_rule_type=nengo.PES(1))

        nengo.Connection(b, conn.learning_rule)

    with Simulator(net, target="sim") as sim:
        with pytest.raises(SimulationError):
            sim.run(1.0)
