import os
import pickle

import nengo
from nengo.dists import Uniform
from nengo_extras.matplotlib import tile, imshow
from nengo_extras.vision import Gabor
import numpy as np
import pytest
import scipy.signal
try:
    import nengo_dl
except ImportError:
    nengo_dl = None

import nengo_loihi
import nengo_loihi.loihi_cx as loihi_cx
from nengo_loihi.conv import (
    Conv2D, conv2d_loihi_weights, ImageShape, ImageSlice, split_transform)
from nengo_loihi.loihi_cx import CxSimulator
from nengo_loihi.loihi_interface import LoihiSimulator
from nengo_loihi.neurons import (
    loihi_rates, LoihiLIF, LoihiSpikingRectifiedLinear)

home_dir = os.path.dirname(nengo_loihi.__file__)
test_dir = os.path.join(home_dir, 'tests')


@pytest.mark.parametrize('pop_type, out_channels_last', [
    (16, True), (32, True), (32, False)])
def test_pop_tiny(
        pop_type, out_channels_last, request, plt, seed, rng, allclose):
    nc = 2

    tau_rc = 0.02
    tau_ref = 0.001
    tau_s = 0.0
    dt = 0.001

    neuron_bias = 1.

    pres_time = 0.4

    sti, stj = 1, 1

    if nc == 1:
        filters = np.array([[-0.5, 2., -0.25],
                            [-0.75, 2., -1.0],
                            [-0.5, 3., -0.5],
                            [-1.0, 6., -0.25]]).reshape(1, 4, 1, 3)
        filters = np.transpose(filters, (0, 2, 3, 1))

        test_x = np.array([[1, 5, 1],
                           [2, 1, 2]])
        test_x = test_x[:, :, None]
    elif nc == 2:
        filters = np.array([[[-0.5, 2., -0.2],
                             [-0.7, 2., -1.0],
                             [-0.5, 3., -0.5],
                             [-1.0, 6., -0.2]],
                            [[-1.0, 2., -1.0],
                             [-0.5, 2., -0.5],
                             [-0.8, 3., -0.2],
                             [-1.0, 4., -0.2]]]).reshape(2, 4, 1, 3)
        filters = np.transpose(filters, (0, 2, 3, 1))

        test_x = np.array([[[1, 5, 1],
                            [2, 1, 2]],
                           [[0, 3, 1],
                            [4, 2, 1]]])
        test_x = np.transpose(test_x, (1, 2, 0))

    test_x = test_x / (test_x.max() + 0.001)

    # --- compute nengo_loihi outputs
    inp_biases = test_x
    inp_shape = ImageShape.from_shape(inp_biases.shape, channels_last=True)
    ni, nj, nk = inp_shape.shape(channels_last=True)
    nc, si, sj, nf = filters.shape
    nij = ni * nj
    nyi = 1 + (ni - si) // sti
    nyj = 1 + (nj - sj) // stj
    out_size = nyi * nyj * nf
    assert out_size <= 1024

    model = loihi_cx.CxModel()

    # input group
    inp = loihi_cx.CxGroup(ni * nj * nk, label='inp')
    assert inp.n <= 1024
    inp.configure_relu()
    inp.bias[:] = inp_biases.ravel()

    inp_ax = loihi_cx.CxAxons(nij, label='inp_ax')
    inp_ax.set_axon_map(inp_shape.pixel_idxs(), inp_shape.channel_idxs())
    inp.add_axons(inp_ax)

    model.add_group(inp)

    # conv group
    neurons = loihi_cx.CxGroup(out_size, label='neurons')
    assert neurons.n <= 1024
    neurons.configure_lif(tau_rc=tau_rc, tau_ref=tau_ref, dt=dt)
    neurons.configure_filter(tau_s, dt=dt)
    neurons.bias[:] = neuron_bias

    synapses = loihi_cx.CxSynapses(inp_shape.n_pixels, label='synapses')
    conv2d_transform = Conv2D.from_kernel(
        filters, inp_shape, strides=(sti, stj),
        output_channels_last=out_channels_last)
    weights, indices, axon_to_weight_map, cx_bases = conv2d_loihi_weights(
        conv2d_transform)
    synapses.set_population_weights(
        weights, indices, axon_to_weight_map, cx_bases, pop_type=pop_type)
    neurons.add_synapses(synapses)

    out_probe = loihi_cx.CxProbe(target=neurons, key='s')
    neurons.add_probe(out_probe)

    inp_ax.target = synapses
    model.add_group(neurons)

    # simulation
    model.discretize()

    n_steps = int(pres_time / dt)
    target = request.config.getoption("--target")
    if target == 'loihi':
        with LoihiSimulator(model, use_snips=False, seed=seed) as sim:
            sim.run_steps(n_steps)
            sim_out = sim.get_probe_output(out_probe)
    else:
        with CxSimulator(model, seed=seed) as sim:
            sim.run_steps(n_steps)
            sim_out = sim.get_probe_output(out_probe)

    sim_out = np.sum(sim_out, axis=0) * (dt / pres_time)
    if out_channels_last:
        sim_out.shape = (nyi, nyj, nf)
        sim_out = np.transpose(sim_out, (2, 0, 1))
    else:
        sim_out.shape = (nf, nyi, nyj)

    out_max = sim_out.max()

    # --- plot results
    rows = 1
    cols = 2

    ax = plt.subplot(rows, cols, 1)
    plt.hist(sim_out.ravel(), bins=11)

    ax = plt.subplot(rows, cols, 2)
    tile(sim_out, vmin=0, vmax=out_max, grid=True, ax=ax)

    print("sim_out:\n%r" % (sim_out[:, :, 0],))

    # ref_out determined by emulator running code known to work
    if nc == 1:
        ref_out = np.array([[0.06, 0.02],
                            [0.055, 0.],
                            [0.0825, 0.0225],
                            [0.125, 0.04]])
    elif nc == 2:
        ref_out = np.array([[0.0975, 0.02],
                            [0.0825, 0.02],
                            [0.125, 0.055],
                            [0.1675, 0.0825]])
    assert allclose(sim_out[:, :, 0], ref_out, rtol=0, atol=1e-7)


def test_conv2d_weights(request, plt, seed, rng, allclose):
    pop_type = 32
    out_channels_last = False

    # load data
    with open(os.path.join(test_dir, 'mnist10.pkl'), 'rb') as f:
        test10 = pickle.load(f)

    test_x = test10[0][0].reshape(28, 28)
    test_x = test_x[3:24, 3:24]
    test_x = 1.999 * test_x - 0.999

    filters = Gabor(freq=Uniform(0.5, 1)).generate(8, (7, 7), rng=rng)
    sti, stj = 2, 2
    tau_rc = 0.02
    tau_ref = 0.002
    tau_s = 0.005
    dt = 0.001

    encode_type = nengo.SpikingRectifiedLinear()
    encode_gain = 1. / dt
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
    inp_shape = ImageShape.from_shape(inp_biases.shape, channels_last=False)

    kernel = np.array([filters, -filters])  # two channels, pos and neg
    kernel = np.transpose(kernel, (0, 2, 3, 1))
    conv2d_transform = Conv2D.from_kernel(
        kernel, inp_shape, strides=(sti, stj),
        output_channels_last=out_channels_last)

    ni, nj, nk = inp_shape.shape(channels_last=True)
    out_size = ref_out.size
    nf, nyi, nyj = ref_out.shape
    assert out_size <= 1024

    model = loihi_cx.CxModel()

    # input group
    inp = loihi_cx.CxGroup(inp_shape.size, label='inp')
    assert inp.n <= 1024
    inp.configure_relu()
    inp.bias[:] = inp_biases.ravel()

    inp_ax = loihi_cx.CxAxons(inp_shape.n_pixels, label='inp_ax')
    inp_ax.set_axon_map(inp_shape.pixel_idxs(), inp_shape.channel_idxs())
    inp.add_axons(inp_ax)

    model.add_group(inp)

    # conv group
    neurons = loihi_cx.CxGroup(out_size, label='neurons')
    assert neurons.n <= 1024
    neurons.configure_lif(tau_rc=tau_rc, tau_ref=tau_ref, dt=dt)
    neurons.configure_filter(tau_s, dt=dt)
    neurons.bias[:] = neuron_bias

    synapses = loihi_cx.CxSynapses(inp_shape.n_pixels, label='synapses')
    weights, indices, axon_to_weight_map, cx_bases = conv2d_loihi_weights(
        conv2d_transform)
    synapses.set_population_weights(
        weights, indices, axon_to_weight_map, cx_bases, pop_type=pop_type)

    neurons.add_synapses(synapses)

    out_probe = loihi_cx.CxProbe(target=neurons, key='s')
    neurons.add_probe(out_probe)

    inp_ax.target = synapses
    model.add_group(neurons)

    # simulation
    model.discretize()

    n_steps = int(pres_time / dt)
    target = request.config.getoption("--target")
    if target == 'loihi':
        with LoihiSimulator(model, use_snips=False, seed=seed) as sim:
            sim.run_steps(n_steps)
            sim_out = sim.get_probe_output(out_probe)
    else:
        with CxSimulator(model, seed=seed) as sim:
            sim.run_steps(n_steps)
            sim_out = sim.get_probe_output(out_probe)

    sim_out = np.sum(sim_out, axis=0) / pres_time
    if out_channels_last:
        sim_out.shape = (nyi, nyj, nf)
        sim_out = np.transpose(sim_out, (2, 0, 1))
    else:
        sim_out.shape = (nf, nyi, nyj)

    out_max = max(ref_out.max(), sim_out.max())

    # --- plot results
    rows = 2
    cols = 2

    ax = plt.subplot(rows, cols, 1)
    tile(filters, cols=8, ax=ax)

    ax = plt.subplot(rows, cols, 2)
    tile(ref_out, vmin=0, vmax=out_max, cols=8, ax=ax)

    ax = plt.subplot(rows, cols, 3)
    plt.hist(ref_out.ravel(), bins=31)
    plt.hist(sim_out.ravel(), bins=31)

    ax = plt.subplot(rows, cols, 4)
    # tile(sim_out, vmin=0, vmax=1, cols=8, ax=ax)
    tile(sim_out, vmin=0, vmax=out_max, cols=8, ax=ax)

    assert allclose(sim_out, ref_out, atol=10, rtol=1e-3)


@pytest.mark.parametrize('channels', [1, 2])
def test_conv_connection(channels, Simulator, seed, rng, plt, allclose):
    # channels_last = True
    channels_last = False
    if channels > 1:
        pytest.xfail("Cannot send population spikes to chip")

    # load data
    with open(os.path.join(test_dir, 'mnist10.pkl'), 'rb') as f:
        test10 = pickle.load(f)

    test_x = test10[0][0].reshape(28, 28)
    test_x = 1.999 * test_x - 0.999  # range (-1, 1)
    test_x = test_x[:, :, None]  # single channel
    input_shape = ImageShape(test_x.shape[0], test_x.shape[1], channels,
                             channels_last=channels_last)

    filters = Gabor(freq=Uniform(0.5, 1)).generate(8, (7, 7), rng=rng)
    filters = filters[None, :, :, :]  # single channel
    filters = np.transpose(filters, (0, 2, 3, 1))  # filters last
    strides = (2, 2)
    tau_rc = 0.02
    tau_ref = 0.002
    tau_s = 0.005
    dt = 0.001

    neuron_type = LoihiLIF(tau_rc=tau_rc, tau_ref=tau_ref)

    pres_time = 1.0

    with nengo.Network(seed=seed) as model:
        nengo_loihi.add_params(model)

        u = nengo.Node(nengo.processes.PresentInput(
            [test_x.ravel()], pres_time), label='u')

        a = nengo.Ensemble(input_shape.size, 1,
                           neuron_type=LoihiSpikingRectifiedLinear(),
                           max_rates=nengo.dists.Choice([40/channels]),
                           intercepts=nengo.dists.Choice([0]),
                           label='a')
        model.config[a].on_chip = False

        if channels == 1:
            nengo.Connection(u, a.neurons, transform=1, synapse=None)
        elif channels == 2:
            # encode image into spikes using two channels (on/off)
            if input_shape.channels_last:
                nengo.Connection(u, a.neurons[0::2], transform=1, synapse=None)
                nengo.Connection(u, a.neurons[1::2], transform=-1,
                                 synapse=None)
            else:
                k = input_shape.rows * input_shape.cols
                nengo.Connection(u, a.neurons[:k], transform=1, synapse=None)
                nengo.Connection(u, a.neurons[k:], transform=-1, synapse=None)

            filters = np.vstack([filters, -filters])
        else:
            raise ValueError("Test not configured for more than two channels")

        conv2d_transform = Conv2D.from_kernel(
            filters, input_shape, strides=strides)
        output_shape = conv2d_transform.output_shape

        gain, bias = neuron_type.gain_bias(max_rates=100, intercepts=0)
        gain = gain * 0.01  # account for `a` max_rates
        b = nengo.Ensemble(output_shape.size, 1,
                           neuron_type=neuron_type,
                           gain=nengo.dists.Choice([gain[0]]),
                           bias=nengo.dists.Choice([bias[0]]),
                           label='b')
        nengo.Connection(
            a.neurons, b.neurons, synapse=tau_s, transform=conv2d_transform)

        bp = nengo.Probe(b.neurons)

    with nengo.Simulator(model, dt=dt, optimize=False) as sim:
        sim.run(pres_time)
    ref_out = sim.data[bp].mean(axis=0).reshape(output_shape.shape())

    # Currently, default TensorFlow does not support channels first in conv
    use_nengo_dl = nengo_dl is not None and channels_last
    ndl_out = np.zeros_like(ref_out)
    if use_nengo_dl:
        with nengo_dl.Simulator(model, dt=dt) as sim:
            sim.run(pres_time)
        ndl_out = sim.data[bp].mean(axis=0).reshape(output_shape.shape())

    with nengo_loihi.Simulator(model, dt=dt, target='simreal') as sim:
        sim.run(pres_time)
    real_out = sim.data[bp].mean(axis=0).reshape(output_shape.shape())

    with Simulator(model, dt=dt) as sim:
        sim.run(pres_time)
    sim_out = sim.data[bp].mean(axis=0).reshape(output_shape.shape())

    if not output_shape.channels_last:
        ref_out = np.transpose(ref_out, (1, 2, 0))
        ndl_out = np.transpose(ndl_out, (1, 2, 0))
        real_out = np.transpose(real_out, (1, 2, 0))
        sim_out = np.transpose(sim_out, (1, 2, 0))

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

    if use_nengo_dl:
        assert allclose(ndl_out, ref_out, atol=1e-5, rtol=1e-5)
    assert allclose(real_out, ref_out, atol=1, rtol=1e-3)
    assert allclose(sim_out, ref_out, atol=10, rtol=1e-3)


@pytest.mark.xfail  # Pop spikes not yet sent to board
@pytest.mark.parametrize('channels_last', [True, False])
def test_conv_input(channels_last, Simulator, plt, allclose):
    input_shape = ImageShape(4, 4, 1, channels_last=channels_last)
    seed = 3  # fix seed to do the same computation for both channel positions
    rng = np.random.RandomState(seed+1)

    with nengo.Network(seed=seed) as net:
        nengo_loihi.add_params(net)

        a = nengo.Node(rng.uniform(0, 1, size=input_shape.size))

        nc = 2
        kernel = np.array([1., -1.]).reshape((1, 1, 1, nc))
        transform = nengo_loihi.Conv2D.from_kernel(kernel, input_shape)
        b = nengo.Ensemble(transform.output_shape.size, 1,
                           neuron_type=nengo.SpikingRectifiedLinear(),
                           max_rates=nengo.dists.Choice([50]),
                           intercepts=nengo.dists.Choice([0]))
        net.config[b].on_chip = False
        nengo.Connection(a, b.neurons, transform=transform)
        output_shape = transform.output_shape

        nf = 4
        kernel = rng.uniform(-0.005, 0.005, size=(nc, 3, 3, nf))
        transform = nengo_loihi.Conv2D.from_kernel(kernel, output_shape)
        c = nengo.Ensemble(transform.output_shape.size, 1,
                           neuron_type=nengo.LIF(),
                           max_rates=nengo.dists.Choice([100]),
                           intercepts=nengo.dists.Choice([0]))
        nengo.Connection(b.neurons, c.neurons, transform=transform)
        output_shape = transform.output_shape

        p = nengo.Probe(c.neurons)

    with nengo.Simulator(net, optimize=False) as sim:
        sim.run(1.0)

    with Simulator(net, seed=seed) as sim_loihi:
        sim_loihi.run(1.0)

    p0 = np.sum(sim.data[p] > 0, axis=0).reshape(output_shape.shape())
    p1 = np.sum(sim_loihi.data[p] > 0, axis=0).reshape(output_shape.shape())
    if not output_shape.channels_last:
        p0 = np.transpose(p0, (1, 2, 0))
        p1 = np.transpose(p1, (1, 2, 0))

    plt.plot(p0.ravel(), 'k')
    plt.plot(p1.ravel(), 'b--')

    # loihi spikes are not exactly the same, but should be close-ish
    assert allclose(p0, p1, rtol=0.15, atol=1)


def test_conv_split(Simulator, rng, plt, allclose):
    channels_last = False

    # load data
    with open(os.path.join(test_dir, 'mnist10.pkl'), 'rb') as f:
        test10 = pickle.load(f)

    input_shape = ImageShape(28, 28, 1, channels_last=channels_last)
    test_x = test10[0][0].reshape(input_shape.shape(channels_last=True))
    if not input_shape.channels_last:
        test_x = np.transpose(test_x, (2, 0, 1))

    n_filters = 8
    kernel_size = (7, 7)
    kernel = Gabor(freq=Uniform(0.5, 1)).generate(
        n_filters, kernel_size, rng=rng)
    kernel = kernel[None, :, :, :]  # single channel
    kernel = np.transpose(kernel, (0, 2, 3, 1))  # filters last
    strides = (2, 2)

    seed = 3  # fix seed to do the same computation for both channel positions
    rng = np.random.RandomState(seed+1)

    with nengo.Network(seed=seed) as net:
        nengo_loihi.add_params(net)

        a = nengo.Node(test_x.ravel())

        # --- make population to turn image into spikes
        nc = 1
        in_kernel = np.array([1.]).reshape((1, 1, 1, nc))
        transform = nengo_loihi.Conv2D.from_kernel(in_kernel, input_shape)
        b = nengo.Ensemble(transform.output_shape.size, 1,
                           neuron_type=nengo.SpikingRectifiedLinear(),
                           max_rates=nengo.dists.Choice([50]),
                           intercepts=nengo.dists.Choice([0]))
        net.config[b].on_chip = False
        nengo.Connection(a, b.neurons, transform=transform)
        in_shape = transform.output_shape

        transform = nengo_loihi.Conv2D.from_kernel(
            kernel, in_shape, strides=strides)
        out_shape = transform.output_shape
        split_slices = out_shape.split_channels(max_size=1024, max_channels=4)

        # --- make convolution population, split across ensembles
        cc = []
        cp = []
        out_shapes = []
        xslice = ImageSlice(in_shape)
        for yslice in split_slices:
            transform_xy = split_transform(transform, xslice, yslice)
            out_shapes.append(transform_xy.output_shape)
            c = nengo.Ensemble(transform_xy.output_shape.size, 1,
                               neuron_type=nengo.LIF(),
                               max_rates=nengo.dists.Choice([15]),
                               intercepts=nengo.dists.Choice([0]))
            nengo.Connection(b.neurons, c.neurons, transform=transform_xy)
            cc.append(c)
            cp.append(nengo.Probe(c.neurons))

    with nengo.Simulator(net, optimize=False) as sim_nengo:
        sim_nengo.run(1.0)

    with Simulator(net, seed=seed) as sim_loihi:
        if "loihi" in sim_loihi.sims:
            sim_loihi.sims["loihi"].snip_max_spikes_per_step = 100
        sim_loihi.run(1.0)

    nengo_out = []
    loihi_out = []
    for p, out_shape_i in zip(cp, out_shapes):
        nengo_out.append(
            (sim_nengo.data[p] > 0).sum(axis=0).reshape(out_shape_i.shape()))
        loihi_out.append(
            (sim_loihi.data[p] > 0).sum(axis=0).reshape(out_shape_i.shape()))

    if channels_last:
        nengo_out = np.concatenate(nengo_out, axis=2)
        loihi_out = np.concatenate(loihi_out, axis=2)

        # put channels first to display them separately
        nengo_out = np.transpose(nengo_out, (2, 0, 1))
        loihi_out = np.transpose(loihi_out, (2, 0, 1))
    else:
        nengo_out = np.concatenate(nengo_out, axis=0)
        loihi_out = np.concatenate(loihi_out, axis=0)

    out_max = np.maximum(nengo_out.max(), loihi_out.max())

    # --- plot results
    rows = 2
    cols = 3

    ax = plt.subplot(rows, cols, 1)
    imshow(test_x[0, :, :], vmin=0, vmax=1, ax=ax)

    ax = plt.subplot(rows, cols, 2)
    tile(np.transpose(kernel[0], (2, 0, 1)), cols=8, ax=ax)

    ax = plt.subplot(rows, cols, 3)
    plt.hist(nengo_out.ravel(), bins=31)
    plt.hist(loihi_out.ravel(), bins=31)

    ax = plt.subplot(rows, cols, 4)
    tile(nengo_out, vmin=0, vmax=out_max, cols=8, ax=ax)

    ax = plt.subplot(rows, cols, 6)
    tile(loihi_out, vmin=0, vmax=out_max, cols=8, ax=ax)

    assert allclose(loihi_out, nengo_out, atol=0.05*out_max, rtol=0.15)
