import pickle
import warnings

import matplotlib.pyplot as plt
import nengo
from nengo._vendor.npconv2d.conv2d import conv2d
import nengo_loihi
import numpy as np
import PIL as pil

warnings.filterwarnings("ignore", category=FutureWarning)

n_dim = 5
dt = 0.001

with open("mnist10.pkl", "rb") as fh:
    images, labels = pickle.load(fh)

image = images[1].reshape(28, 28)
image2 = pil.Image.fromarray(image, mode="F")
image2 = image2.resize((64, 64))
image = np.array(image2)

image = 2 * image - 1  # range [-1, 1]

image_shape = nengo.transforms.ChannelShape((64, 64, 1), channels_last=True)

kernel = np.array(
    [
        [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
        [[-1, -1, -1], [2, 2, 2], [-1, -1, -1]],
        [[-1, 2, -1], [-1, 2, -1], [-1, 2, -1]],
        [[0, -1, 0], [-1, 4, -1], [0, -1, 0]],
    ],
    dtype=np.float32,
).reshape((4, 3, 3, 1))
kernel = np.transpose(kernel, (1, 2, 3, 0))
kernel /= np.abs(kernel).sum(axis=(0, 1, 2), keepdims=True)
n_filters = kernel.shape[-1]

for k in range(kernel.shape[-1]):
    print(kernel[..., 0, k])

if 0:
    # single-channel
    n_channels = 1
    enc_kernel = np.array([1]).reshape((1, 1, 1, n_channels))
else:
    # double-channel
    n_channels = 2
    enc_kernel = np.array([1, -1]).reshape((1, 1, 1, n_channels))
    kernel = np.concatenate([kernel, -kernel], axis=2)

relu = lambda x: np.maximum(x, 0)
enc_images = conv2d(image[None, :, :, None], enc_kernel, pad="VALID")[0]
enc_images = relu(enc_images)
ref_outputs = conv2d(enc_images[None, ...], kernel, pad="VALID")[0]
ref_outputs = relu(ref_outputs)

for k in range(ref_outputs.shape[-1]):
    print(ref_outputs[:5, :5, k])

if 0:
    rows = 1
    cols = 1 + n_filters
    plt.subplot(rows, cols, 1)
    plt.imshow(image)

    for k in range(n_filters):
        plt.subplot(rows, cols, 2 + k)
        plt.imshow(ref_outputs[:, :, k])

    plt.show()
    assert 0


def input_f(t):
    return image.ravel()


with nengo.Network(seed=0) as net:
    nengo_loihi.add_params(net)

    u = nengo.Node(input_f, label="u")

    transform0 = nengo.Convolution(
        n_filters=n_channels,
        input_shape=image_shape,
        kernel_size=(1, 1),
        init=enc_kernel,
    )
    enc = nengo.Ensemble(
        transform0.output_shape.size,
        n_dim,
        label="a",
        neuron_type=nengo.SpikingRectifiedLinear(),
        max_rates=nengo.dists.Choice([1000]),
        intercepts=nengo.dists.Choice([0]),
    )
    net.config[enc].on_chip = False
    nengo.Connection(u, enc.neurons, transform=transform0, label="u-enc")

    transform = nengo.Convolution(
        n_filters=n_filters, input_shape=transform0.output_shape, init=kernel * dt
    )
    a = nengo.Ensemble(
        transform.output_shape.size,
        n_dim,
        label="a",
        neuron_type=nengo.SpikingRectifiedLinear(),
        max_rates=nengo.dists.Choice([500]),
        intercepts=nengo.dists.Choice([0]),
    )
    net.config[a].split_full_shape = transform.output_shape.shape
    net.config[a].split_shape = (16, 16, 2)
    nengo.Connection(enc.neurons, a.neurons, transform=transform, label="enc-a")

    # b = nengo.Ensemble(1000, n_dim, radius=1.3, label="b")
    # nengo.Connection(a, b, label="a-b")

    up = nengo.Probe(u)
    ap = nengo.Probe(a.neurons, synapse=nengo.Alpha(0.01))
    # bp = nengo.Probe(b.neurons, synapse=nengo.Alpha(0.01))


def ind_string(inds):
    d = np.diff(inds)
    if np.all(d == d[0]):
        return "slice(%d, %d, %d)" % (inds[0], inds[-1], d[0])
    else:
        return str(inds)


with nengo_loihi.Simulator(net, dt=dt) as sim:
    for input in sim.model.inputs:
        print("Input %s: %d" % (input.label, input.n_neurons))
        for axon in input.axons:
            print(
                "  Axon %s: target %s: %d"
                % (axon.label, axon.target.label, axon.n_axons)
            )
    for block in sim.model.blocks:
        print("Block %s: %d" % (block.label, block.compartment.n_compartments))
        for synapse in block.synapses:
            print("  Synapse %s: %d" % (synapse.label, synapse.n_axons))
        for axon in block.axons:
            print(
                "  Axon %s: target %s: %d"
                % (axon.label, axon.target.label, axon.n_axons)
            )
            print("    inds: %s" % ind_string(axon.compartment_map))

    # sim.run(1.0)
    sim.run(0.2)

nt = len(sim.trange())
a_outputs = sim.data[ap].reshape((nt, 62, 62, n_filters))
print(a_outputs.max())

# spikes = sim.data[anp]
# spike_counts = (spikes > 0).sum(axis=0)
# print(spike_counts[0::10])
# print(spike_counts[1::10])

plt.figure()

rows = 2
cols = 1 + n_filters
plt.subplot(rows, cols, 1)
plt.imshow(image)

for k in range(n_filters):
    plt.subplot(rows, cols, 1 + k + 1)
    plt.imshow(ref_outputs[:, :, k])

    plt.subplot(rows, cols, 1 * cols + 1 + k + 1)
    plt.imshow(a_outputs[-1, :, :, k], vmin=0, vmax=a_outputs[-1, :, :, k].max())
    print("Output channel %d: max: %0.3f" % (k, a_outputs[-1, :, :, k].max()))

    print(a_outputs[-1, :5, :5, k])

plt.show()
