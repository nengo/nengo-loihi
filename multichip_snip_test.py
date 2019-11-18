import nengo
import numpy as np
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from nengo.transforms import ChannelShape
from nengo_loihi.hardware.allocators import RoundRobin, OneToOne, GreedyChip
import nengo_loihi
import logging

#logging.basicConfig(level=logging.DEBUG)
#logging.getLogger().setLevel(logging.DEBUG)

rng = np.random.RandomState(0)

input_size = (64, 64)
dt = 0.001  # simulation timestep
presentation_time = 0.1  # input presentation time
max_rate = 1000  # neuron firing rates
# neuron spike amplitude (scaled so that the overall output is ~1)
amp = 1 / max_rate
# input image shape
input_shape = nengo.transforms.ChannelShape((64, 64, 1), channels_last=True)
n_presentations = 1
presentation_steps = int(round(presentation_time / dt))

data_kwargs = dict(rng=rng)
testdata = np.ones([1, 64, 64])

pres_imgs = testdata[:n_presentations]
input_fn = nengo.processes.PresentInput(
    pres_imgs.reshape(pres_imgs.shape[0], -1), presentation_time=presentation_time
)


def conv_layer(
    x,
    n_filters,
    input_shape,
    kernel_size=(3, 3),
    strides=(1, 1),
    init=None,
    activation=True,
    label=None,
):
    # create a Conv2D transform with the given arguments
    if init is None:
        r = 0.4 / np.sqrt(np.prod(kernel_size) * input_shape.n_channels)
        if label == "layer1":
            r = r * 0.5
        init = nengo.dists.Uniform(-r * 0.5, r)
        print(init)
    conv = nengo.Convolution(
        n_filters=n_filters,
        input_shape=input_shape,
        kernel_size=kernel_size,
        strides=strides,
        init=init,
        channels_last=True,
    )

    # add an ensemble to implement the activation function
    layer = nengo.Ensemble(conv.output_shape.size, 1, label=label)

    # connect up the input object to the new layer
    conn = nengo.Connection(x, layer.neurons, transform=conv)

    # print out the shape information for our new layer
    print("LAYER")
    print(conv.input_shape.shape, "->", conv.output_shape.shape)

    return layer, conv, conn


with nengo.Network(seed=0) as net:
    # set up the default parameters for ensembles/connections
    nengo_loihi.add_params(net)
    net.config[nengo.Ensemble].neuron_type = nengo.SpikingRectifiedLinear(amplitude=amp)
    net.config[nengo.Ensemble].max_rates = nengo.dists.Choice([max_rate])
    net.config[nengo.Ensemble].intercepts = nengo.dists.Choice([0])
    net.config[nengo.Connection].synapse = 0.005

    input_shape = ChannelShape(input_size + (1,), channels_last=True)
    input_fn = [0] * input_shape.size if input_fn is None else input_fn
    inp = nengo.Node(input_fn, label="input")

    # the output node provides the 10-dimensional classification
    out = nengo.Node(size_in=10, label="output")
    # build parallel copies of the network
    layer, conv, conn = conv_layer(
        x=inp,
        n_filters=1,
        input_shape=input_shape,
        kernel_size=(1, 1),
        init=np.ones((1, 1, 1, 1)),
    )
    # first layer is off-chip to translate the images into spikes
    net.config[layer.neurons.ensemble].on_chip = False
    layer1, conv, conn1 = conv_layer(
        x=layer.neurons,
        n_filters=12,
        input_shape=conv.output_shape,
        strides=(1, 1),
        label="layer1",
    )
    net.config[layer1].split_full_shape = conv.output_shape.shape
    net.config[layer1].split_shape = (2, 21, 12)
    net.config[conn1].pop_type = 16
    layer3, conv, conn3 = conv_layer(
        x=layer1.neurons,
        n_filters=12,
        input_shape=conv.output_shape,
        strides=(2, 2),
        label="layer3",
    )
    net.config[layer3].split_full_shape = conv.output_shape.shape
    net.config[layer3].split_shape = (8, 4, 12)
    net.config[conn3].pop_type = 16

    layer4, conv, conn4 = conv_layer(
        x=layer3.neurons,
        n_filters=2,
        input_shape=conv.output_shape,
        strides=(1, 1),
        label="layer4",
    )
    # net.config[layer4].split_full_shape = conv.output_shape.shape
    # net.config[layer4].split_shape = (5,8,2)
    # net.config[conn4].pop_type = 16
    # layer5, conv, conn5 = conv_layer(x=layer4.neurons, n_filters=24, input_shape=conv.output_shape,
    #        strides=(1, 1),label='layer5')
    # net.config[layer5].split_full_shape = conv.output_shape.shape
    # net.config[layer5].split_shape = (9,9,12)
    # net.config[conn5].pop_type = 16
    # layer6, conv, conn6 = conv_layer(x=layer5.neurons, n_filters=2, input_shape=conv.output_shape,
    #        strides=(2, 2),label='layer6')

    # nengo.Connection(layer6.neurons, out, transform=nengo_dl.dists.Glorot())
    # voltage = nengo.Probe(layer6.neurons)
    # out_p = nengo.Probe(out)
    # layer1 = nengo.Probe(layer1.neurons)#, synapse=nengo.Alpha(0.05))
    # layer3 = nengo.Probe(layer3.neurons)#, synapse=nengo.Alpha(0.05))
    layer4 = nengo.Probe(layer4.neurons)  # , synapse=nengo.Alpha(0.05))
    # layer5 = nengo.Probe(layer5.neurons)#, synapse=nengo.Alpha(0.05))
    # layer6 = nengo.Probe(layer6.neurons)#, synapse=nengo.Alpha(0.05))
    # out_p_filt = nengo.Probe(layer6.neurons)

probe_name = "layer4"

sim_args = dict(dt=dt, precompute=False, dismantle=True,)
#hardware_options = {"allocator": GreedyChip(n_chips=2, cores_per_chip=126)}
hardware_options = dict(
    snip_max_spikes_per_step=10000,
    #allocator=GreedyChip(n_chips=2, cores_per_chip=126),
    allocator=RoundRobin(n_chips=2),
)

# --- Run in emulator
emu_val = {}
#with nengo_loihi.Simulator(net, target="sim", **sim_args) as sim_emu:
#    # run the simulation on Loihi
#    sim_emu.run(n_presentations * presentation_time)
#
#    # check classification error
#    step = int(presentation_time / dt)
#    # emu_val['layer1'] = sim_emu.data[layer1][0:presentation_steps-1]
#    # emu_val['layer3'] = sim_emu.data[layer3][0:presentation_steps-1]
#    emu_val["layer4"] = sim_emu.data[layer4][0 : presentation_steps - 1]
#    # emu_val['layer5'] = sim_emu.data[layer5][0:presentation_steps-1]
#    # emu_val['layer6'] = sim_emu.data[layer6][0:presentation_steps-1]

if probe_name in emu_val:
    emu_output = emu_val[probe_name]
    np.savetxt("emu_" + probe_name + ".txt", emu_output, fmt="%.2f")

# --- Run on Loihi
loihi_val = {}
with nengo_loihi.Simulator(
    net, target="loihi", hardware_options=hardware_options, **sim_args
) as sim_loi:
    # if running on Loihi, increase the max input spikes per step
    if "loihi" in sim_loi.sims:
        sim_loi.sims["loihi"].snip_max_spikes_per_step = 9000

    # run the simulation on Loihi
    sim_loi.run(n_presentations * presentation_time)

    # check classification error
    step = int(presentation_time / dt)
    # loihi_val['layer1'] = sim_loi.data[layer1][0:presentation_steps-1]
    # loihi_val['layer3'] = sim_loi.data[layer3][0:presentation_steps-1]
    loihi_val["layer4"] = sim_loi.data[layer4][0 : presentation_steps - 1]
    # loihi_val['layer5'] = sim_loi.data[layer5][0:presentation_steps-1]
    # loihi_val['layer6'] = sim_loi.data[layer6][0:presentation_steps-1]

spikes = np.array(loihi_val[probe_name]).reshape(
    (n_presentations, presentation_steps - 1, -1)
)
rates = (spikes > 0).sum(axis=1) / presentation_time

print("average rate %f" % rates.mean())

for layer in sorted(list(emu_val)):
    x = emu_val[layer]
    print(np.shape(x))
    spikes = np.array(x).reshape((n_presentations, presentation_steps - 1, -1))
    rates = (spikes > 0).sum(axis=1) / presentation_time
    print(rates.shape)
    print(rates.mean())
    plt.hist(rates[0, :], bins=20, alpha=0.75)
    plt.title("rate histogram for " + str(layer))
    plt.savefig(str(layer) + ".pdf")
    plt.close()

if probe_name in loihi_val:
    loihi_output = loihi_val[probe_name]
    np.savetxt("loihi_" + probe_name + ".txt", loihi_output, fmt="%.2f")


print("Difference between emu and loihi %f" % np.sum(np.abs(emu_output - loihi_output)))

np.savetxt("loihi_" + probe_name + ".txt", loihi_output, fmt="%.2f")
np.savetxt("diff_" + probe_name + ".txt", np.abs(emu_output - loihi_output), fmt="%.2f")
