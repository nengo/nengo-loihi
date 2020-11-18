import os
import pickle

import nengo
import numpy as np
import pytest
import scipy.signal
from nengo.dists import Choice, Uniform
from nengo.exceptions import ValidationError
from nengo_extras.matplotlib import imshow, tile
from nengo_extras.vision import Gabor

import nengo_loihi
from nengo_loihi import conv
from nengo_loihi.block import Axon, LoihiBlock, Synapse
from nengo_loihi.builder import Model
from nengo_loihi.builder.discretize import discretize_model
from nengo_loihi.emulator import EmulatorInterface
from nengo_loihi.hardware import HardwareInterface
from nengo_loihi.hardware.allocators import RoundRobin
from nengo_loihi.neurons import LoihiLIF, LoihiSpikingRectifiedLinear, loihi_rates
from nengo_loihi.probe import LoihiProbe
from nengo_loihi.tests import require_partition

home_dir = os.path.dirname(nengo_loihi.__file__)
test_dir = os.path.join(home_dir, "tests")


def make_shape(spatial_shape, n_channels, channels_last):
    s = tuple(spatial_shape)
    c = (n_channels,)
    return s + c if channels_last else c + s


def make_channel_shape(spatial_shape, n_channels, channels_last):
    shape = make_shape(spatial_shape, n_channels, channels_last)
    return nengo.transforms.ChannelShape(shape, channels_last=channels_last)


@pytest.mark.parametrize(
    "pop_type, channels_last, nc",
    [(16, True, 2), (32, True, 1), (32, True, 2), (32, False, 1), (32, False, 2)],
)
def test_pop_tiny(pop_type, channels_last, nc, request, plt, seed, allclose):
    tau_rc = 0.02
    tau_ref = 0.001
    tau_s = 0.0
    dt = 0.001

    neuron_bias = 1.0

    pres_time = 0.4

    sti, stj = 1, 1

    if nc == 1:
        filters = np.array(
            [
                [-0.5, 2.0, -0.25],
                [-0.75, 2.0, -1.0],
                [-0.5, 3.0, -0.5],
                [-1.0, 6.0, -0.25],
            ]
        ).reshape((1, 4, 1, 3))

        inp_biases = np.array([[1, 5, 1], [2, 1, 2]])
        inp_biases = inp_biases[:, :, None]
    elif nc == 2:
        filters = np.array(
            [
                [
                    [-0.5, 2.0, -0.2],
                    [-0.7, 2.0, -1.0],
                    [-0.5, 3.0, -0.5],
                    [-1.0, 6.0, -0.2],
                ],
                [
                    [-1.0, 2.0, -1.0],
                    [-0.5, 2.0, -0.5],
                    [-0.8, 3.0, -0.2],
                    [-1.0, 4.0, -0.2],
                ],
            ]
        ).reshape((2, 4, 1, 3))

        inp_biases = np.array([[[1, 5, 1], [2, 1, 2]], [[0, 3, 1], [4, 2, 1]]])
        inp_biases = np.transpose(inp_biases, (1, 2, 0))

    # rearrange to (kernel_rows, kernel_cols, in_channels, out_channels)
    filters = np.transpose(filters, (2, 3, 0, 1))

    inp_biases = inp_biases / (inp_biases.max() + 0.001)

    # --- compute nengo_loihi outputs
    ni, nj, nk = inp_biases.shape
    si, sj, nc, nf = filters.shape
    nij = ni * nj
    nyi = 1 + (ni - si) // sti
    nyj = 1 + (nj - sj) // stj
    out_size = nyi * nyj * nf
    assert out_size <= 1024

    model = Model()

    # input block
    inp = LoihiBlock(ni * nj * nk, label="inp")
    model.add_block(inp)

    assert inp.n_neurons <= 1024
    inp.compartment.configure_relu()
    inp.compartment.bias[:] = inp_biases.ravel()

    inp_ax = Axon(nij, label="inp_ax")

    # we always compute the pixel/channel idxs with channels_last=True
    # (not sure why?), and then set it to the correct value afterwards
    inp_shape = make_channel_shape((ni, nj), nk, channels_last=True)
    inp_ax.set_compartment_axon_map(
        target_axons=conv.pixel_idxs(inp_shape), atoms=conv.channel_idxs(inp_shape)
    )
    inp_shape.shape = make_shape((ni, nj), nk, channels_last)
    inp_shape.channels_last = channels_last

    inp.add_axon(inp_ax)

    # conv block
    neurons = LoihiBlock(out_size, label="neurons")
    model.add_block(neurons)

    assert neurons.n_neurons <= 1024
    neurons.compartment.configure_lif(tau_rc=tau_rc, tau_ref=tau_ref, dt=dt)
    neurons.compartment.configure_filter(tau_s, dt=dt)
    neurons.compartment.bias[:] = neuron_bias

    synapse = Synapse(np.prod(inp_shape.spatial_shape), label="synapse")
    conv2d_transform = nengo.Convolution(
        nf,
        inp_shape,
        strides=(sti, stj),
        channels_last=channels_last,
        init=filters,
        kernel_size=(1, 3),
    )
    weights, indices, axon_to_weight_map, bases = conv.conv2d_loihi_weights(
        conv2d_transform
    )
    synapse.set_population_weights(
        weights, indices, axon_to_weight_map, bases, pop_type=pop_type
    )
    neurons.add_synapse(synapse)

    out_probe = LoihiProbe(target=neurons, key="spiked")
    model.add_probe(out_probe)

    inp_ax.target = synapse

    # simulation
    discretize_model(model)

    n_steps = int(pres_time / dt)
    target = request.config.getoption("--target")
    if target == "loihi":
        with HardwareInterface(model, use_snips=False, seed=seed) as sim:
            sim.run_steps(n_steps)
            sim_out = sim.get_probe_output(out_probe)
    else:
        with EmulatorInterface(model, seed=seed) as sim:
            sim.run_steps(n_steps)
            sim_out = sim.get_probe_output(out_probe)

    sim_out = np.sum(sim_out, axis=0) * (dt / pres_time)
    sim_out.shape = make_shape((nyi, nyj), nf, channels_last)
    if channels_last:
        sim_out = np.transpose(sim_out, (2, 0, 1))

    out_max = sim_out.max()

    # --- plot results
    rows = 1
    cols = 2

    ax = plt.subplot(rows, cols, 1)
    plt.hist(sim_out.ravel(), bins=11)

    ax = plt.subplot(rows, cols, 2)
    tile(sim_out, vmin=0, vmax=out_max, grid=True, ax=ax)

    # ref_out determined by emulator running code known to work
    if nc == 1:
        ref_out = np.array(
            [[0.06, 0.02], [0.055, 0.0], [0.0825, 0.0225], [0.125, 0.04]]
        )
    elif nc == 2:
        ref_out = np.array(
            [[0.0975, 0.02], [0.0825, 0.02], [0.125, 0.055], [0.2475, 0.0825]]
        )
    assert allclose(sim_out[:, :, 0], ref_out, rtol=0, atol=1e-7)


@pytest.mark.parametrize("channels_last", (True, False))
def test_conv2d_weights(channels_last, request, plt, seed, rng, allclose):
    # with NxSDK 0.9.8, only Nahuku32 is working with multi-chip SNIPs
    require_partition(
        "nahuku32",
        request=request,
        lmt_options="--skip-power=1",
        action="fail" if nengo_loihi.version.dev is None else "skip",
    )

    def loihi_rates_n(neuron_type, x, gain, bias, dt):
        """Compute Loihi rates on higher dimensional inputs"""
        y = x.reshape((-1, x.shape[-1]))
        gain = np.asarray(gain)
        bias = np.asarray(bias)
        if gain.ndim == 0:
            gain = gain * np.ones(x.shape[-1])
        if bias.ndim == 0:
            bias = bias * np.ones(x.shape[-1])
        rates = loihi_rates(neuron_type, y, gain, bias, dt)
        return rates.reshape(x.shape)

    target = request.config.getoption("--target")

    pop_type = 32

    # load data
    with open(os.path.join(test_dir, "mnist10.pkl"), "rb") as f:
        test10 = pickle.load(f)

    test_x = test10[0][0].reshape((28, 28))
    test_x = test_x[3:24, 3:24]
    test_x = 1.999 * test_x - 0.999

    filters = Gabor(freq=Uniform(0.5, 1)).generate(8, (7, 7), rng=rng)
    sti, stj = 2, 2
    tau_rc = 0.02
    tau_ref = 0.002
    tau_s = 0.005
    dt = 0.001

    encode_type = nengo.SpikingRectifiedLinear()
    encode_gain = 1.0 / dt
    encode_bias = 0.0
    neuron_type = nengo.LIF(tau_rc=tau_rc, tau_ref=tau_ref)
    neuron_gain = 1.0
    neuron_bias = 1.0

    pres_time = 0.2

    # --- compute ideal outputs
    def conv_pm(x, kernel):
        y0 = scipy.signal.correlate2d(x[0], kernel, mode="valid")[::sti, ::stj]
        y1 = scipy.signal.correlate2d(x[1], kernel, mode="valid")[::sti, ::stj]
        return [y0, -y1]

    ref_out = np.array([test_x, -test_x])
    ref_out = loihi_rates_n(encode_type, ref_out, encode_gain, encode_bias, dt)
    ref_out = ref_out / encode_gain
    ref_out = np.array([conv_pm(ref_out, kernel) for kernel in filters])
    ref_out = ref_out.sum(axis=1)  # sum positive and negative parts
    ref_out = loihi_rates_n(neuron_type, ref_out, neuron_gain, neuron_bias, dt)

    # --- compute nengo_loihi outputs
    inp_biases = np.stack([test_x, -test_x], axis=-1 if channels_last else 0)
    inp_shape = nengo.transforms.ChannelShape(
        inp_biases.shape, channels_last=channels_last
    )

    kernel = np.array([filters, -filters])  # two channels, pos and neg
    kernel = np.transpose(kernel, (2, 3, 0, 1))
    conv2d_transform = nengo.Convolution(
        8,
        inp_shape,
        strides=(sti, stj),
        channels_last=channels_last,
        kernel_size=(7, 7),
        init=kernel,
    )

    out_size = ref_out.size
    nf, nyi, nyj = ref_out.shape
    assert out_size <= 1024

    model = Model()

    # input block
    inp = LoihiBlock(inp_shape.size, label="inp")
    model.add_block(inp)

    assert inp.n_neurons <= 1024
    inp.compartment.configure_relu()
    inp.compartment.bias[:] = inp_biases.ravel()

    inp_ax = Axon(np.prod(inp_shape.spatial_shape), label="inp_ax")
    inp_ax.set_compartment_axon_map(
        target_axons=conv.pixel_idxs(inp_shape), atoms=conv.channel_idxs(inp_shape)
    )
    inp.add_axon(inp_ax)

    # conv block
    neurons = LoihiBlock(out_size, label="neurons")
    model.add_block(neurons)

    assert neurons.n_neurons <= 1024
    neurons.compartment.configure_lif(tau_rc=tau_rc, tau_ref=tau_ref, dt=dt)
    neurons.compartment.configure_filter(tau_s, dt=dt)
    neurons.compartment.bias[:] = neuron_bias

    synapse = Synapse(np.prod(inp_shape.spatial_shape), label="synapse")
    weights, indices, axon_to_weight_map, bases = conv.conv2d_loihi_weights(
        conv2d_transform
    )
    synapse.set_population_weights(
        weights, indices, axon_to_weight_map, bases, pop_type=pop_type
    )

    neurons.add_synapse(synapse)

    out_probe = LoihiProbe(target=neurons, key="spiked")
    model.add_probe(out_probe)

    inp_ax.target = synapse

    # simulation
    discretize_model(model)

    n_steps = int(pres_time / dt)
    if target == "loihi":
        with HardwareInterface(
            model, use_snips=False, seed=seed, allocator=RoundRobin()
        ) as sim:
            sim.run_steps(n_steps)
            sim_out = sim.get_probe_output(out_probe)
    else:
        with EmulatorInterface(model, seed=seed) as sim:
            sim.run_steps(n_steps)
            sim_out = sim.get_probe_output(out_probe)

    sim_out = np.sum(sim_out, axis=0) / pres_time
    sim_out.shape = make_shape((nyi, nyj), nf, channels_last)
    if channels_last:
        sim_out = np.transpose(sim_out, (2, 0, 1))

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

    assert allclose(sim_out, ref_out, atol=12, rtol=1e-3)


@pytest.mark.parametrize("channels", [1, 2])
@pytest.mark.parametrize("channels_last", (True, False))
def test_conv_connection(channels, channels_last, Simulator, seed, rng, plt, allclose):
    # load data
    with open(os.path.join(test_dir, "mnist10.pkl"), "rb") as f:
        test10 = pickle.load(f)

    test_x = test10[0][0].reshape((28, 28))
    test_x = 1.999 * test_x - 0.999  # range (-1, 1)
    input_shape = make_channel_shape(test_x.shape, channels, channels_last)

    filters = Gabor(freq=Uniform(0.5, 1)).generate(8, (7, 7), rng=rng)
    filters = filters[None, :, :, :]  # single channel
    filters = np.transpose(filters, (2, 3, 0, 1))
    strides = (2, 2)
    tau_rc = 0.02
    tau_ref = 0.002
    tau_s = 0.005

    neuron_type = LoihiLIF(tau_rc=tau_rc, tau_ref=tau_ref)

    pres_time = 0.1

    with nengo.Network(seed=seed) as model:
        nengo_loihi.add_params(model)

        u = nengo.Node(test_x.ravel(), label="u")

        a = nengo.Ensemble(
            input_shape.size,
            1,
            neuron_type=LoihiSpikingRectifiedLinear(),
            max_rates=nengo.dists.Choice([40 / channels]),
            intercepts=nengo.dists.Choice([0]),
            label="a",
        )
        model.config[a].on_chip = False

        if channels == 1:
            nengo.Connection(u, a.neurons, transform=1, synapse=None)
        elif channels == 2:
            # encode image into spikes using two channels (on/off)
            if input_shape.channels_last:
                nengo.Connection(u, a.neurons[0::2], transform=1, synapse=None)
                nengo.Connection(u, a.neurons[1::2], transform=-1, synapse=None)
            else:
                k = input_shape.spatial_shape[0] * input_shape.spatial_shape[1]
                nengo.Connection(u, a.neurons[:k], transform=1, synapse=None)
                nengo.Connection(u, a.neurons[k:], transform=-1, synapse=None)

            filters = np.concatenate([filters, -filters], axis=2)
        else:
            raise ValueError("Test not configured for more than two channels")

        conv2d_transform = nengo.Convolution(
            8,
            input_shape,
            strides=strides,
            kernel_size=(7, 7),
            channels_last=channels_last,
            init=filters,
        )

        output_shape = conv2d_transform.output_shape

        gain, bias = neuron_type.gain_bias(max_rates=100, intercepts=0)
        gain = gain * 0.01  # account for `a` max_rates
        b = nengo.Ensemble(
            output_shape.size,
            1,
            neuron_type=neuron_type,
            gain=nengo.dists.Choice([gain[0]]),
            bias=nengo.dists.Choice([bias[0]]),
            label="b",
        )
        nengo.Connection(
            a.neurons, b.neurons, synapse=tau_s, transform=conv2d_transform
        )

        bp = nengo.Probe(b.neurons)

    with nengo.Simulator(model, optimize=False) as sim_nengo:
        sim_nengo.run(pres_time)
    ref_out = sim_nengo.data[bp].mean(axis=0).reshape(output_shape.shape)

    with Simulator(model, target="simreal") as sim_emu:
        sim_emu.run(pres_time)
    emu_out = sim_emu.data[bp].mean(axis=0).reshape(output_shape.shape)

    with Simulator(
        model, hardware_options={"snip_max_spikes_per_step": 800}
    ) as sim_loihi:
        sim_loihi.run(pres_time)
    sim_out = sim_loihi.data[bp].mean(axis=0).reshape(output_shape.shape)

    if not output_shape.channels_last:
        ref_out = np.transpose(ref_out, (1, 2, 0))
        emu_out = np.transpose(emu_out, (1, 2, 0))
        sim_out = np.transpose(sim_out, (1, 2, 0))

    out_max = max(ref_out.max(), emu_out.max(), sim_out.max())

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
    tile(np.transpose(emu_out, (2, 0, 1)), vmin=0, vmax=out_max, cols=8, ax=ax)

    ax = plt.subplot(rows, cols, 6)
    tile(np.transpose(sim_out, (2, 0, 1)), vmin=0, vmax=out_max, cols=8, ax=ax)

    assert allclose(emu_out, ref_out, atol=10, rtol=1e-3)
    assert allclose(sim_out, ref_out, atol=10, rtol=1e-3)


@pytest.mark.parametrize("channels_last", [True, False])
def test_conv_input(channels_last, Simulator, plt, allclose):
    input_shape = make_channel_shape((4, 4), 1, channels_last=channels_last)
    seed = 3  # fix seed to do the same computation for both channel positions
    rng = np.random.RandomState(seed + 1)

    with nengo.Network(seed=seed) as net:
        nengo_loihi.add_params(net)

        a = nengo.Node(rng.uniform(0, 1, size=input_shape.size))

        nc = 2
        kernel = np.array([1.0, -1.0]).reshape((1, 1, 1, nc))
        transform = nengo.Convolution(
            nc,
            input_shape,
            channels_last=channels_last,
            init=kernel,
            kernel_size=(1, 1),
        )
        b = nengo.Ensemble(
            transform.output_shape.size,
            1,
            neuron_type=nengo.SpikingRectifiedLinear(),
            max_rates=nengo.dists.Choice([50]),
            intercepts=nengo.dists.Choice([0]),
        )
        net.config[b].on_chip = False
        nengo.Connection(a, b.neurons, transform=transform)
        output_shape = transform.output_shape

        nf = 4
        kernel = rng.uniform(-0.005, 0.005, size=(3, 3, nc, nf))
        transform = nengo.Convolution(
            nf,
            output_shape,
            channels_last=channels_last,
            init=-kernel,
            kernel_size=(3, 3),
        )
        c = nengo.Ensemble(
            transform.output_shape.size,
            1,
            neuron_type=nengo.LIF(),
            max_rates=nengo.dists.Choice([100]),
            intercepts=nengo.dists.Choice([0]),
        )
        nengo.Connection(b.neurons, c.neurons, transform=transform)
        output_shape = transform.output_shape

        p = nengo.Probe(c.neurons)

    with nengo.Simulator(net, optimize=False) as sim:
        sim.run(1.0)

    with Simulator(net, seed=seed) as sim_loihi:
        sim_loihi.run(1.0)

    p0 = np.sum(sim.data[p] > 0, axis=0).reshape(output_shape.shape)
    p1 = np.sum(sim_loihi.data[p] > 0, axis=0).reshape(output_shape.shape)
    if not channels_last:
        p0 = np.transpose(p0, (1, 2, 0))
        p1 = np.transpose(p1, (1, 2, 0))

    plt.plot(p0.ravel(), "k")
    plt.plot(p1.ravel(), "b--")

    # loihi spikes are not exactly the same, but should be close-ish
    assert allclose(p0, p1, rtol=0.15, atol=1)


@pytest.mark.parametrize("precompute", [False, True])  # noqa: C901
@pytest.mark.parametrize("channels_last, pop_type", [(True, 16), (False, 32)])
def test_conv_deepnet(
    channels_last, pop_type, precompute, Simulator, request, rng, seed, plt, allclose
):
    """Run a convolutional network with two layers on the chip.

    Checks that network with block splitting on the target matches one without
    on the emulator.
    """

    # if request.config.getoption("--target") == "loihi":
    #     if (
    #         pop_type == 32
    #         and nxsdk_version is not None
    #         and nxsdk_version < parse_version("0.9.5.dev0")
    #     ):
    #         pytest.skip("Pop32 multichip test requires NxSDK >= 0.9.5")
    #     elif pop_type == 16:
    #         # multichip pop_type = 16 works only on nahuku32 board currently
    #         require_partition(
    #             "nahuku32",
    #             lmt_options="--skip-power=1",
    #             action="fail" if nengo_loihi.version.dev is None else "skip",
    #         )

    # with NxSDK 0.9.8, only Nahuku32 is working with multi-chip SNIPs
    require_partition(
        "nahuku32",
        request=request,
        lmt_options="--skip-power=1",
        action="fail" if nengo_loihi.version.dev is None else "skip",
    )

    def conv_layer(
        x, input_shape, array_init=None, label=None, conn_args=None, **conv_args
    ):
        conn_args = {} if conn_args is None else conn_args

        if array_init is not None:
            assert all(a not in conv_args for a in ("init", "kernel_size", "n_filters"))
            assert array_init.ndim == 4
            conv_args["init"] = array_init
            conv_args["kernel_size"] = array_init.shape[:2]
            assert array_init.shape[2] == input_shape.n_channels
            conv_args["n_filters"] = array_init.shape[3]

        conv = nengo.Convolution(input_shape=input_shape, **conv_args)

        # add an ensemble to implement the activation function
        layer = nengo.Ensemble(conv.output_shape.size, 1, label=label)

        # connect up the input object to the new layer
        conn = nengo.Connection(x, layer.neurons, transform=conv, **conn_args)

        return layer, conv, conn

    channels = 1
    n_filters0 = 1
    n_filters1 = 4
    n_filters2 = 4

    # load data
    with open(os.path.join(test_dir, "mnist10.pkl"), "rb") as f:
        test10 = pickle.load(f)

    test_x = test10[0][0].reshape(28, 28)  # range (0, 1)
    input_shape = make_channel_shape(test_x.shape, channels, channels_last)

    filters0 = np.ones((1, 1, channels, n_filters0))

    # use Gabor filters for first layer
    filters1 = Gabor(
        freq=Uniform(0.5, 1), sigma_x=Choice([0.9]), sigma_y=Choice([0.9])
    ).generate(n_filters1, (7, 7), rng=rng)
    assert n_filters0 == 1
    filters1 = filters1[None, :, :, :]  # single channel
    filters1 = np.transpose(filters1, (2, 3, 0, 1))  # rows, cols, in_chan, out_chan

    # use random combinations of first-layer channels in 1x1 convolution
    filters2 = rng.uniform(-0.2, 1, size=(n_filters1, n_filters2)).clip(0, None)
    filters2 *= 2 / filters2.sum(axis=0, keepdims=True)  # each filter sums to 2
    filters2 = filters2[None, None, :, :]  # rows, cols, in_chan, out_chan

    tau_s = 0.001
    max_rate = 100
    amp = 1 / max_rate
    f_split = 2 if pop_type == 32 else 4

    # use Loihi neuron type so Nengo sim mimics Loihi neuron effects
    neuron_type = LoihiSpikingRectifiedLinear(amplitude=amp)

    pres_time = 0.2

    with nengo.Network(seed=seed) as net:
        nengo_loihi.add_params(net)

        net.config[nengo.Ensemble].neuron_type = neuron_type
        net.config[nengo.Ensemble].max_rates = Choice([max_rate])
        net.config[nengo.Ensemble].intercepts = Choice([0])
        net.config[nengo.Connection].synapse = tau_s

        u = nengo.Node(test_x.ravel(), label="u")

        layer0, conv0, conn0 = conv_layer(
            u,
            input_shape=input_shape,
            array_init=filters0,
            strides=(1, 1),
            channels_last=channels_last,
            label="layer0",
            conn_args=dict(synapse=None),
        )
        net.config[layer0].on_chip = False

        layer1, conv1, conn1 = conv_layer(
            layer0.neurons,
            input_shape=conv0.output_shape,
            array_init=filters1,
            strides=(2, 2),
            channels_last=channels_last,
            label="layer1",
        )
        net.config[layer1].block_shape = nengo_loihi.BlockShape(
            make_shape((4, 4), f_split, channels_last), conv1
        )
        net.config[conn1].pop_type = pop_type

        layer2, conv2, conn2 = conv_layer(
            layer1.neurons,
            input_shape=conv1.output_shape,
            array_init=filters2,
            strides=(1, 1),
            channels_last=channels_last,
            label="layer2",
        )
        net.config[layer2].block_shape = nengo_loihi.BlockShape(
            make_shape((4, 4), f_split, channels_last), conv2
        )
        net.config[conn2].pop_type = pop_type

        output_p = nengo.Probe(layer2.neurons)
        output_shape = conv2.output_shape

    with nengo.Simulator(net, optimize=False) as sim_nengo:
        sim_nengo.run(pres_time)
        ref_out = (sim_nengo.data[output_p] > 0).sum(axis=0).reshape(output_shape.shape)

    with Simulator(net, target="sim") as sim_emu:
        sim_emu.run(pres_time)
        emu_out = (sim_emu.data[output_p] > 0).sum(axis=0).reshape(output_shape.shape)

    with Simulator(
        net,
        precompute=precompute,
        hardware_options={
            "allocator": RoundRobin(),
            "snip_max_spikes_per_step": 800,
        },
    ) as sim_loihi:
        sim_loihi.run(pres_time)
        sim_out = (sim_loihi.data[output_p] > 0).sum(axis=0).reshape(output_shape.shape)

    out_max = ref_out.max()
    ref_out = ref_out / out_max
    emu_out = emu_out / out_max
    sim_out = sim_out / out_max

    if channels_last:
        # channels first, to display channels in separate plots
        ref_out = np.transpose(ref_out, (2, 0, 1))
        emu_out = np.transpose(emu_out, (2, 0, 1))
        sim_out = np.transpose(sim_out, (2, 0, 1))

    # --- plot results
    rows = 2
    cols = 3

    ax = plt.subplot(rows, cols, 1)
    imshow(test_x, vmin=0, vmax=1, ax=ax)

    ax = plt.subplot(rows, cols, 2)
    tile(np.transpose(filters1, (2, 3, 0, 1))[0], rows=2, cols=2, grid=True, ax=ax)

    ax = plt.subplot(rows, cols, 3)
    plt.hist((ref_out.ravel(), emu_out.ravel(), sim_out.ravel()), bins=21)

    ax = plt.subplot(rows, cols, 4)
    tile(ref_out, rows=2, cols=2, grid=True, ax=ax)

    ax = plt.subplot(rows, cols, 5)
    tile(emu_out, rows=2, cols=2, grid=True, ax=ax)

    ax = plt.subplot(rows, cols, 6)
    tile(sim_out, rows=2, cols=2, grid=True, ax=ax)

    assert allclose(sim_out, ref_out, atol=0.15, rtol=1e-3)
    # The emulator and hardware usually match almost perfectly. However, for this test
    # with pop_type=16 and precompute=False, timing appears to change slightly on the
    # input spikes, and we get a difference. So we've loosened the tolerances.
    assert allclose(sim_out, emu_out, atol=0.08, rtol=1e-3)


def test_conv_split(Simulator, rng, plt, allclose):
    channels_last = False

    # load data
    with open(os.path.join(test_dir, "mnist10.pkl"), "rb") as f:
        test10 = pickle.load(f)

    input_shape = make_channel_shape((28, 28), 1, channels_last)

    n_filters = 8
    kernel_size = (7, 7)
    kernel = Gabor(freq=Uniform(0.5, 1)).generate(n_filters, kernel_size, rng=rng)
    kernel = kernel[None, :, :, :]  # single channel
    kernel = np.transpose(kernel, (2, 3, 0, 1))
    strides = (2, 2)

    seed = 3  # fix seed to do the same computation for both channel positions

    with nengo.Network(seed=seed) as net:
        nengo_loihi.add_params(net)

        a = nengo.Node(test10[0][0].ravel())

        # --- make population to turn image into spikes
        nc = 1
        in_kernel = np.array([1.0]).reshape((1, 1, 1, nc))
        transform = nengo.Convolution(
            1,
            input_shape,
            kernel_size=(1, 1),
            init=in_kernel,
            channels_last=channels_last,
        )
        b = nengo.Ensemble(
            transform.output_shape.size,
            1,
            neuron_type=nengo.SpikingRectifiedLinear(),
            max_rates=nengo.dists.Choice([50]),
            intercepts=nengo.dists.Choice([0]),
        )
        net.config[b].on_chip = False
        nengo.Connection(a, b.neurons, transform=transform)
        in_shape = transform.output_shape

        transform = nengo.Convolution(
            n_filters,
            in_shape,
            kernel_size=kernel_size,
            strides=strides,
            init=kernel,
            channels_last=channels_last,
        )
        out_shape = transform.output_shape
        split_slices = conv.split_channels(out_shape, max_size=1024, max_channels=4)

        # --- make convolution population, split across ensembles
        cc = []
        cp = []
        out_shapes = []
        xslice = conv.ImageSlice(in_shape)
        for yslice in split_slices:
            transform_xy = conv.split_transform(transform, xslice, yslice)
            out_shapes.append(transform_xy.output_shape)
            c = nengo.Ensemble(
                transform_xy.output_shape.size,
                1,
                neuron_type=nengo.LIF(),
                max_rates=nengo.dists.Choice([15]),
                intercepts=nengo.dists.Choice([0]),
            )
            nengo.Connection(b.neurons, c.neurons, transform=transform_xy)
            cc.append(c)
            cp.append(nengo.Probe(c.neurons))

    simtime = 0.3

    with nengo.Simulator(net, optimize=False) as sim_nengo:
        sim_nengo.run(simtime)

    hw_opts = dict(snip_max_spikes_per_step=100)
    with Simulator(net, seed=seed, hardware_options=hw_opts) as sim_loihi:
        sim_loihi.run(simtime)

    nengo_out = []
    loihi_out = []
    for p, out_shape_i in zip(cp, out_shapes):
        nengo_out.append((sim_nengo.data[p] > 0).sum(axis=0).reshape(out_shape_i.shape))
        loihi_out.append((sim_loihi.data[p] > 0).sum(axis=0).reshape(out_shape_i.shape))

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
    imshow(test10[0][0].reshape((28, 28)), vmin=0, vmax=1, ax=ax)

    ax = plt.subplot(rows, cols, 2)
    tile(np.transpose(kernel[0], (2, 0, 1)), cols=8, ax=ax)

    ax = plt.subplot(rows, cols, 3)
    plt.hist(nengo_out.ravel(), bins=31)
    plt.hist(loihi_out.ravel(), bins=31)

    ax = plt.subplot(rows, cols, 4)
    tile(nengo_out, vmin=0, vmax=out_max, cols=8, ax=ax)

    ax = plt.subplot(rows, cols, 6)
    tile(loihi_out, vmin=0, vmax=out_max, cols=8, ax=ax)

    assert allclose(loihi_out, nengo_out, atol=0.15 * out_max, rtol=0.15)


@pytest.mark.parametrize("on_chip", [True, False])
def test_conv_preslice(on_chip, Simulator, plt):
    conv2d = pytest.importorskip("nengo._vendor.npconv2d.conv2d")

    kernel = np.array([[-1, 2, -1], [-1, 2, -1], [-1, 2, -1]], dtype=float)
    kernel /= kernel.max()

    image = np.array(
        [
            [1, 2, 1, 2, 0],
            [2, 3, 2, 1, 1],
            [1, 2, 1, 2, 3],
            [2, 3, 2, 1, 1],
            [1, 2, 1, 2, 0],
        ],
        dtype=float,
    )
    image /= image.max()

    image2 = np.column_stack([c * x for c in image.T for x in (1, -1)])

    input_gain = 149.0

    neuron_type = nengo.SpikingRectifiedLinear()
    loihi_neuron = LoihiSpikingRectifiedLinear()
    layer0_neuron = loihi_neuron if on_chip else neuron_type

    y_ref = layer0_neuron.rates(image.ravel(), input_gain, 0)
    y_ref = conv2d.conv2d(
        y_ref.reshape((1, 5, 5, 1)), kernel.reshape((3, 3, 1, 1)), pad="VALID"
    )
    y_ref = loihi_neuron.rates(y_ref.ravel(), 1.0, 0.0).reshape((3, 3))

    with nengo.Network() as net:
        nengo_loihi.add_params(net)

        u = nengo.Node(image2.ravel())
        a = nengo.Ensemble(
            50,
            1,
            neuron_type=neuron_type,
            gain=nengo.dists.Choice([input_gain]),
            bias=nengo.dists.Choice([0]),
        )
        net.config[a].on_chip = on_chip

        transform = nengo.Convolution(
            n_filters=1, input_shape=(5, 5, 1), init=kernel.reshape((3, 3, 1, 1))
        )

        b = nengo.Ensemble(
            transform.output_shape.size,
            1,
            neuron_type=neuron_type,
            gain=nengo.dists.Choice([1]),
            bias=nengo.dists.Choice([0]),
        )

        nengo.Connection(u, a.neurons, synapse=None)
        nengo.Connection(a.neurons[::2], b.neurons, transform=transform)
        bp = nengo.Probe(b.neurons, synapse=nengo.Alpha(0.02))

    with Simulator(net) as sim:
        assert sim.precompute is True
        sim.run(0.3)

    y_ref = y_ref / input_gain
    y = sim.data[bp][-1].reshape((3, -1)) / input_gain

    plt.subplot(121)
    plt.imshow(y_ref)
    plt.colorbar()
    plt.subplot(122)
    plt.imshow(y)
    plt.colorbar()

    assert np.allclose(y, y_ref, atol=0.02, rtol=0.1)


def test_conv_onchip(Simulator, plt):
    """Tests a fully on-chip conv connection. """
    conv2d = pytest.importorskip("nengo._vendor.npconv2d.conv2d")

    kernel = np.array([[-1, 2, -1], [-1, 2, -1], [-1, 2, -1]], dtype=float)
    kernel /= kernel.max()

    image = np.array(
        [
            [1, 2, 1, 2, 0],
            [2, 3, 2, 1, 1],
            [1, 2, 1, 2, 3],
            [2, 3, 2, 1, 1],
            [1, 2, 1, 2, 0],
        ],
        dtype=float,
    )
    image /= image.max()

    input_scale = 119.0
    bias = input_scale * image.ravel()

    neuron_type = nengo.SpikingRectifiedLinear()

    y_ref = LoihiSpikingRectifiedLinear().rates(image.ravel(), input_scale, 0)
    y_ref = conv2d.conv2d(
        y_ref.reshape((1, 5, 5, 1)), kernel.reshape((3, 3, 1, 1)), pad="VALID"
    )
    y_ref = LoihiSpikingRectifiedLinear().rates(y_ref.ravel(), 1.0, 0.0).reshape((3, 3))

    with nengo.Network() as net:
        a = nengo.Ensemble(
            bias.size,
            1,
            neuron_type=neuron_type,
            gain=nengo.dists.Choice([0]),
            bias=bias,
        )

        transform = nengo.Convolution(
            n_filters=1, input_shape=(5, 5, 1), init=kernel.reshape((3, 3, 1, 1))
        )

        b = nengo.Ensemble(
            transform.output_shape.size,
            1,
            neuron_type=neuron_type,
            gain=nengo.dists.Choice([1]),
            bias=nengo.dists.Choice([0]),
        )

        nengo.Connection(a.neurons, b.neurons, transform=transform)
        bp = nengo.Probe(b.neurons, synapse=nengo.Alpha(0.02))

    with Simulator(net) as sim:
        sim.run(0.3)

    y_ref = y_ref / input_scale
    y = sim.data[bp][-1].reshape((3, -1)) / input_scale

    plt.subplot(121)
    plt.imshow(y_ref)
    plt.colorbar()
    plt.subplot(122)
    plt.imshow(y)
    plt.colorbar()

    assert np.allclose(y, y_ref, atol=0.02, rtol=0.1)


def test_conv_overlap_input(Simulator, plt):
    """Tests a fully on-chip conv connection. """
    conv2d = pytest.importorskip("nengo._vendor.npconv2d.conv2d")

    kernel = np.array([[-1, 2, -1], [-1, 2, -1], [-1, 2, -1]], dtype=float)
    kernel /= kernel.max()

    image = np.array(
        [
            [1, 2, 1, 2, 0],
            [2, 3, 2, 1, 1],
            [1, 2, 1, 2, 3],
            [2, 3, 2, 1, 1],
            [1, 2, 1, 2, 0],
        ],
        dtype=float,
    )
    image /= image.max()

    input_scale = 119.0
    bias = input_scale * image.ravel()

    neuron_type = nengo.SpikingRectifiedLinear()

    y_ref = LoihiSpikingRectifiedLinear().rates(image.ravel(), input_scale, 0)
    y_ref = conv2d.conv2d(
        y_ref.reshape((1, 5, 5, 1)), kernel.reshape((3, 3, 1, 1)), pad="VALID"
    )
    y_ref = LoihiSpikingRectifiedLinear().rates(y_ref.ravel(), 1.0, 0.0).reshape((3, 3))

    with nengo.Network() as net:
        a = nengo.Ensemble(
            bias.size,
            1,
            neuron_type=neuron_type,
            gain=nengo.dists.Choice([0]),
            bias=bias,
        )

        transform = nengo.Convolution(
            n_filters=1, input_shape=(4, 5, 1), init=kernel.reshape((3, 3, 1, 1))
        )

        b0 = nengo.Ensemble(
            transform.output_shape.size,
            1,
            neuron_type=neuron_type,
            gain=nengo.dists.Choice([1]),
            bias=nengo.dists.Choice([0]),
        )
        b1 = nengo.Ensemble(
            transform.output_shape.size,
            1,
            neuron_type=neuron_type,
            gain=nengo.dists.Choice([1]),
            bias=nengo.dists.Choice([0]),
        )

        nengo.Connection(a.neurons[:20], b0.neurons, transform=transform)
        nengo.Connection(a.neurons[5:], b1.neurons, transform=transform)
        b0p = nengo.Probe(b0.neurons, synapse=nengo.Alpha(0.02))
        b1p = nengo.Probe(b1.neurons, synapse=nengo.Alpha(0.02))

    with Simulator(net) as sim:
        sim.run(0.3)

    y_ref = y_ref / input_scale
    y0 = sim.data[b0p][-1].reshape((2, -1)) / input_scale
    y1 = sim.data[b1p][-1].reshape((2, -1)) / input_scale

    plt.subplot(131)
    plt.imshow(y_ref)
    plt.colorbar()
    plt.subplot(132)
    plt.imshow(b0)
    plt.colorbar()
    plt.subplot(133)
    plt.imshow(b1)
    plt.colorbar()

    assert np.allclose(y0, y_ref[:2], atol=0.02, rtol=0.1)
    assert np.allclose(y1, y_ref[1:], atol=0.02, rtol=0.1)


@pytest.mark.target_loihi
@pytest.mark.parametrize("on_chip", [True, False])
@pytest.mark.parametrize("precompute", [True, False])
@pytest.mark.parametrize(
    "pop_type, channels_last, n_filters0, n_filters1",
    [
        (16, True, 4, 4),
        (32, True, 4, 4),
        (32, False, 4, 4),
        (16, True, 36, 36),
        (32, True, 34, 36),
        (32, False, 37, 33),
    ],
)
def test_chip_population_axons(
    on_chip, precompute, pop_type, channels_last, n_filters0, n_filters1, Simulator, rng
):
    """Check that all types of population axons work as inputs or between cores.

    Also, on the chip, dummy axons were still having an effect. Check this is fixed.
    """

    def conv_layer(input=None, label=None, **kwargs):
        conv = nengo.Convolution(**kwargs)
        layer = nengo.Ensemble(conv.output_shape.size, 1, label=label)
        conn = (
            nengo.Connection(input, layer.neurons, transform=conv)
            if input is not None
            else None
        )
        return layer, conv, conn

    if pop_type == 16 and not channels_last:
        pytest.skip("pop16 axons not compatible with single-compartment shifts")
    if pop_type == 16 and (n_filters0 > 32 or n_filters1 > 32):
        # see ``test_pop16_extra_atom_bits_error`` below
        pytest.skip("extra atom bits for pop16 axons not yet implemented in NxSDK")

    max_rate = 100
    amp = 1 / max_rate

    # 6 x 6 input will have one unused pixel at edge with 3 x 3 kernel and stride 2
    input_shape = (6, 6, 1) if channels_last else (1, 6, 6)
    input_shape = nengo.transforms.ChannelShape(
        input_shape, channels_last=channels_last
    )
    X = rng.uniform(0.2, 1, size=input_shape.shape)
    kernel0 = rng.uniform(0.2, 1, size=(1, 1, 1, n_filters0))
    kernel1 = rng.uniform(0.1, 0.5, size=(3, 3, n_filters0, n_filters1))

    with nengo.Network(seed=0) as net:
        nengo_loihi.add_params(net)
        net.config[nengo.Ensemble].neuron_type = nengo.SpikingRectifiedLinear(
            amplitude=amp
        )
        net.config[nengo.Ensemble].max_rates = nengo.dists.Choice([max_rate])
        net.config[nengo.Ensemble].intercepts = nengo.dists.Choice([0])
        net.config[nengo.Connection].synapse = 0.005

        inp = nengo.Node(X.ravel()) if not on_chip else None

        # first layer is off-chip to translate the inputs into spikes
        layer0, conv0, _ = conv_layer(
            input=inp,
            n_filters=n_filters0,
            input_shape=input_shape,
            channels_last=channels_last,
            kernel_size=(1, 1),
            init=kernel0,
            label="layer0",
        )

        net.config[layer0].on_chip = on_chip
        if on_chip:
            assert kernel0.shape[:2] == (1, 1)
            w = kernel0[0, 0]
            Y = X.dot(w) if channels_last else np.tensordot(w.T, X, axes=1)
            layer0.gain = nengo.dists.Choice([0.0])
            layer0.bias = Y.ravel() * max_rate

        layer1, conv1, conn1 = conv_layer(
            input=layer0.neurons,
            n_filters=n_filters1,
            input_shape=conv0.output_shape,
            channels_last=channels_last,
            kernel_size=(3, 3),
            strides=(2, 2),
            init=kernel1,
            label="layer1",
        )
        net.config[conn1].pop_type = pop_type

        probe = nengo.Probe(layer1.neurons)

    sim_time = 0.1
    with Simulator(net, target="sim") as emulator:
        emulator.run(sim_time)

    with Simulator(net, target="loihi", precompute=precompute) as loihi:
        loihi.run(sim_time)

    assert np.all(emulator.data[probe].sum(axis=0) > 0)
    assert np.array_equal(loihi.data[probe], emulator.data[probe])


def test_pop16_extra_atom_bits_error(request, Simulator):
    """pop16 extra atom bits currently not supported on NxSDK (current as of 0.9.8)

    When this feature is enabled on NxSDK, testing for this should be enabled in
    ``test_chip_population_axons``.
    """
    with nengo.Network() as net:
        nengo_loihi.add_params(net)
        net.config[nengo.Ensemble].gain = nengo.dists.Choice([1])
        net.config[nengo.Ensemble].bias = nengo.dists.Choice([0])

        a = nengo.Ensemble(36, 1)
        b = nengo.Ensemble(36, 1)
        conn = nengo.Connection(
            a.neurons,
            b.neurons,
            transform=nengo.Convolution(36, input_shape=(1, 1, 36), kernel_size=(1, 1)),
        )
        net.config[conn].pop_type = 16

    with (
        pytest.raises(NotImplementedError, match="Using more than 32 'populations'")
        if request.config.getoption("--target") == "loihi"
        else pytest.warns(Warning, match="Using more than 32 'populations'")
    ):
        with Simulator(net):
            pass


def test_conv_gain(Simulator):
    with nengo.Network() as net:
        a = nengo.Ensemble(16, 1)
        b = nengo.Ensemble(4, 1)
        nengo.Connection(
            a.neurons, b.neurons, transform=nengo.Convolution(1, (4, 4, 1))
        )

    with pytest.raises(ValidationError, match="must have the same gain"):
        with Simulator(net):
            pass


def test_conv_non_lowpass(Simulator):
    k = 10
    d = 5
    with nengo.Network() as model:
        a = nengo.Ensemble(n_neurons=k ** 2, dimensions=k)

        x = nengo.Ensemble(n_neurons=d, dimensions=d, gain=np.ones(d), bias=np.ones(d))

        conv = nengo.Convolution(
            n_filters=d, input_shape=(k, k, 1), strides=(1, 1), kernel_size=(k, k)
        )
        assert conv.size_in == k ** 2
        assert conv.size_out == d

        nengo.Connection(
            a.neurons, x.neurons, transform=conv, synapse=nengo.Alpha(0.005)
        )

    with pytest.raises(NotImplementedError, match="non-Lowpass synapses"):
        with Simulator(model):
            pass


def test_imageslice_api():
    imageshape = nengo.transforms.ChannelShape((5, 6, 8))
    imageslice = conv.ImageSlice(
        imageshape,
        row_slice=slice(None, None, 2),
        col_slice=slice(1, 4),
        channel_slice=slice(1, None, 2),
    )
    assert not imageslice.channel_slice_only()
    assert imageslice.row_idxs() == [0, 2, 4]
    assert imageslice.col_idxs() == [1, 2, 3]
    assert imageslice.channel_idxs() == [1, 3, 5, 7]

    imageslice = conv.ImageSlice(imageshape, channel_slice=slice(2, None, 2))
    assert imageslice.channel_slice_only()
    assert imageslice.row_idxs() == list(range(5))
    assert imageslice.col_idxs() == list(range(6))
    assert imageslice.channel_idxs() == [2, 4, 6]

    with pytest.raises(ValidationError, match="must be 2-D ChannelShape"):
        conv.ImageSlice(nengo.transforms.ChannelShape((5, 6, 7, 8)))


def test_split_transform(rng):
    n = 8
    shape0 = nengo.transforms.ChannelShape((1, 1, n))
    slice0 = conv.ImageSlice(shape0, channel_slice=slice(1, None, 2))
    shape1 = nengo.transforms.ChannelShape((1, 1, n))
    slice1 = conv.ImageSlice(shape1, channel_slice=slice(0, None, 3))

    shape8 = nengo.transforms.ChannelShape((4, 1, 2))
    slice8 = conv.ImageSlice(shape8, row_slice=slice(1, 3))
    assert np.prod(shape8.shape) == n

    transform = rng.uniform(-1, 1, size=(n, n))
    assert np.array_equal(conv.split_transform(transform), transform)
    assert np.array_equal(
        conv.split_transform(transform, in_slice=slice0),
        transform[:, slice0.channel_slice],
    )
    assert np.array_equal(
        conv.split_transform(transform, out_slice=slice1),
        transform[slice1.channel_slice, :],
    )
    assert np.array_equal(
        conv.split_transform(transform, in_slice=slice0, out_slice=slice1),
        transform[slice1.channel_slice, slice0.channel_slice],
    )

    with pytest.raises(AssertionError):
        conv.split_transform(transform, in_slice=slice8)
    with pytest.raises(AssertionError):
        conv.split_transform(transform, out_slice=slice8)
