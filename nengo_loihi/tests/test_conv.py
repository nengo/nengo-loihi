import os
import pickle

import numpy as np
import scipy.signal

import nengo
from nengo.dists import Uniform

import nengo_loihi
import nengo_loihi.loihi_cx as loihi_cx
from nengo_loihi.conv import Conv2dEnsemble
from nengo_loihi.neurons import loihi_rates

from nengo_extras.matplotlib import tile, imshow
from nengo_extras.vision import Gabor

home_dir = os.path.dirname(nengo_loihi.__file__)
test_dir = os.path.join(home_dir, 'tests')


def test_conv2d_weights(request, plt, seed, rng, allclose):
    target = request.config.getoption("--target")

    # load data
    with open(os.path.join(test_dir, 'mnist10.pkl'), 'rb') as f:
        test10 = pickle.load(f)

    test_x, test_y = test10[0][0].reshape(28, 28), test10[1][0]
    test_x = test_x[3:25, 3:25]
    test_x = 1.999 * test_x - 0.999

    filters = Gabor(freq=Uniform(0.5, 1)).generate(8, (7, 7), rng=rng)
    sti, stj = 2, 2
    tau_rc = 0.02
    tau_ref = 0.002
    tau_s = 0.005
    dt = 0.001

    encode_type = nengo.SpikingRectifiedLinear()
    encode_gain = 1./dt
    encode_bias = 0.
    neuron_type = nengo.LIF(tau_rc=tau_rc, tau_ref=tau_ref)
    neuron_gain = 1.
    neuron_bias = 1.

    pres_time = 1.0

    # --- compute ideal outputs
    def conv_pm(x, kernel):
        y0 = scipy.signal.correlate2d(x[0], kernel, mode='valid')[::sti, ::stj]
        y1 = scipy.signal.correlate2d(x[1], kernel, mode='valid')[::sti, ::stj]
        return [y0, -y1]

    ref_out = np.array([test_x, -test_x])
    ref_out = loihi_rates(encode_type, ref_out, encode_gain, encode_bias, dt)
    ref_out = ref_out / encode_gain
    ref_out = np.array([conv_pm(ref_out, kernel) for kernel in filters])
    ref_out = ref_out.sum(axis=1)  # sum positive and negative parts
    ref_out = loihi_rates(neuron_type, ref_out, neuron_gain, neuron_bias, dt)

    # --- compute nengo_loihi outputs
    inp_biases = np.array([test_x, -test_x])
    nk = inp_biases.shape[0]  # number of channels
    ni, nj = test_x.shape
    nij = ni * nj
    out_size = ref_out.size
    nf, nyi, nyj = ref_out.shape
    assert out_size <= 1024

    model = loihi_cx.CxModel()

    # input group
    inp = loihi_cx.CxGroup(ni * nj * nk, label='inp')
    assert inp.n <= 1024
    inp.configure_relu()
    inp.bias[:] = inp_biases.ravel()

    inp_ax = loihi_cx.CxAxons(nij, label='inp_ax')
    inp_ax.cx_to_axon_map = np.tile(np.arange(nij), nk)
    inp_ax.cx_atoms = np.concatenate([
        i * np.ones(nij, dtype=int) for i in range(nk)])
    inp.add_axons(inp_ax)

    inp_probe = loihi_cx.CxProbe(target=inp, key='s')
    inp.add_probe(inp_probe)

    model.add_group(inp)

    # conv group
    neurons = loihi_cx.CxGroup(out_size, label='neurons')
    assert neurons.n <= 1024
    neurons.configure_lif(tau_rc=tau_rc, tau_ref=tau_ref, dt=dt)
    neurons.configure_filter(tau_s, dt=dt)
    neurons.bias[:] = neuron_bias

    synapses = loihi_cx.CxSynapses(ni*nj, label='synapses')
    kernel = np.array([filters, -filters])  # two channels, pos and neg
    kernel = np.transpose(kernel, (0, 2, 3, 1))
    input_shape = (ni, nj, nk)
    synapses.set_conv2d_weights(kernel, input_shape, strides=(sti, stj))
    neurons.add_synapses(synapses)

    out_probe = loihi_cx.CxProbe(target=neurons, key='s')
    neurons.add_probe(out_probe)

    inp_ax.target = synapses
    model.add_group(neurons)

    # simulation
    model.discretize()

    n_steps = int(pres_time / dt)
    if target == 'loihi':
        with model.get_loihi(seed=seed) as sim:
            sim.run_steps(n_steps)

            sim_inp = np.column_stack([
                p.timeSeries.data for p in sim.board.probe_map[inp_probe]])
            sim_out = np.column_stack([
                p.timeSeries.data for p in sim.board.probe_map[out_probe]])
    else:
        sim = model.get_simulator(seed=seed)
        sim.run_steps(n_steps)

        sim_inp = sim.probe_outputs[inp_probe]
        sim_out = sim.probe_outputs[out_probe]

    sim_inp = np.sum(sim_inp, axis=0) / pres_time
    sim_inp.shape = (nk * ni, nj)

    sim_out = np.sum(sim_out, axis=0) / pres_time
    sim_out.shape = (nyi, nyj, nf)
    sim_out = np.transpose(sim_out, (2, 0, 1))

    out_max = max(ref_out.max(), sim_out.max())

    # --- plot results
    rows = 2
    cols = 2

    ax = plt.subplot(rows, cols, 1)
    tile(filters, cols=8, ax=ax)

    ax = plt.subplot(rows, cols, 2)
    tile(ref_out, vmin=0, vmax=out_max, cols=8, ax=ax)

    # ax = plt.subplot(rows, cols, 3)
    # imshow(sim_inp, vmin=0, vmax=1, ax=ax)

    ax = plt.subplot(rows, cols, 3)
    plt.hist(ref_out.ravel(), bins=31)
    plt.hist(sim_out.ravel(), bins=31)

    ax = plt.subplot(rows, cols, 4)
    # tile(sim_out, vmin=0, vmax=1, cols=8, ax=ax)
    tile(sim_out, vmin=0, vmax=out_max, cols=8, ax=ax)

    assert allclose(sim_out, ref_out, atol=10, rtol=1e-3)


def test_conv_ensemble(Simulator, seed, rng, plt):
    # load data
    with open(os.path.join(test_dir, 'mnist10.pkl'), 'rb') as f:
        test10 = pickle.load(f)

    test_x, test_y = test10[0][0].reshape(28, 28), test10[1][0]
    test_x = test_x[3:25, 3:25]
    test_x = 0.999 * test_x + 0.0005  # range (0, 1)
    # test_x = 1.999 * test_x - 0.999  # range (-1, 1)
    test_x = test_x[:, :, None]  # single channel

    filters = Gabor(freq=Uniform(0.5, 1)).generate(8, (7, 7), rng=rng)
    filters = filters[None, :, :, :]  # single channel
    filters = np.transpose(filters, (0, 2, 3, 1))  # filters last
    sti, stj = 2, 2
    tau_rc = 0.02
    tau_ref = 0.002
    tau_s = 0.005
    dt = 0.001

    # encode_type = nengo.SpikingRectifiedLinear()
    # encode_gain = 1./dt
    # encode_bias = 0.
    # neuron_type = nengo.LIF(tau_rc=tau_rc, tau_ref=tau_ref)
    # neuron_gain = 1.
    # neuron_bias = 1.

    neuron_type = nengo.LIF(tau_rc=tau_rc, tau_ref=tau_ref)

    pres_time = 1.0

    with nengo.Network(seed=seed) as model:
        nengo_loihi.add_params(model)

        u = nengo.Node(nengo.processes.PresentInput(
            [test_x.ravel()], pres_time), label='u')

        # encode image into spikes
        ni, nj, nk = test_x.shape
        assert nk == 1
        nk = 2
        a = nengo.Ensemble(ni * nj * nk, 1,
                           neuron_type=nengo.SpikingRectifiedLinear(),
                           max_rates=nengo.dists.Choice([100]),
                           intercepts=nengo.dists.Choice([0]),
                           label='a')
        model.config[a].on_chip = False

        nengo.Connection(u, a.neurons[0::2], transform=1, synapse=None)
        nengo.Connection(u, a.neurons[1::2], transform=-1, synapse=None)

        input_shape = (ni, nj, nk)
        filters = np.vstack([filters, -filters])
        gain, bias = neuron_type.gain_bias(max_rates=100, intercepts=0)
        gain = gain * 0.01  # account for `a` max_rates
        b = Conv2dEnsemble(input_shape, filters, strides=(sti, stj),
                           neuron_type=neuron_type,
                           # max_rates=nengo.dists.Choice([100]),
                           # intercepts=nengo.dists.Choice([0]),
                           gain=nengo.dists.Choice([gain[0]]),
                           bias=nengo.dists.Choice([bias[0]]),
                           label='b')
        ab = nengo.Connection(a.neurons, b, synapse=tau_s, label='Conn(a->b)')
        # model.config[ab].atoms =

        bp = nengo.Probe(b.neurons)

        output_shape = b.output_shape

    with nengo.Simulator(model, dt=dt, optimize=False) as sim:
        sim.run(pres_time)
    ref_out = sim.data[bp].mean(axis=0).reshape(output_shape)

    ndl_out = np.zeros_like(ref_out)
    # import nengo_dl
    # with nengo_dl.Simulator(model, dt=dt) as sim:
    #     sim.run(pres_time)
    # ndl_out = sim.data[bp].mean(axis=0).reshape(output_shape)

    sim_out = np.zeros_like(ref_out)
    # with Simulator(model, dt=dt) as sim:
    #     sim.run(pres_time)
    # sim_out = sim.data[bp].mean(axis=0).reshape(output_shape)

    out_max = max(ref_out.max(), sim_out.max())

    # --- plot results
    rows = 2
    cols = 3

    ax = plt.subplot(rows, cols, 1)
    imshow(test_x, vmin=0, vmax=1, ax=ax)

    ax = plt.subplot(rows, cols, 2)
    tile(np.transpose(filters[0], (2, 0, 1)), cols=8, ax=ax)

    ax = plt.subplot(rows, cols, 3)
    plt.hist(ref_out.ravel(), bins=31)
    plt.hist(sim_out.ravel(), bins=31)

    ax = plt.subplot(rows, cols, 4)
    tile(np.transpose(ref_out, (2, 0, 1)), vmin=0, vmax=out_max, cols=8, ax=ax)

    ax = plt.subplot(rows, cols, 5)
    tile(np.transpose(ndl_out, (2, 0, 1)), vmin=0, vmax=out_max, cols=8, ax=ax)

    ax = plt.subplot(rows, cols, 6)
    tile(np.transpose(sim_out, (2, 0, 1)), vmin=0, vmax=out_max, cols=8, ax=ax)

    # assert allclose(sim_out, ref_out, atol=10, rtol=1e-3)
