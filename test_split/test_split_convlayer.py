import os
import pickle
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

import jinja2
import matplotlib.pyplot as plt
import nengo
from nengo._vendor.npconv2d.conv2d import conv2d
import nengo_loihi
import numpy as np
import PIL as pil

this_dir = os.path.dirname(os.path.realpath(__file__))

SNIP_PROBE = False

n_dim = 5
dt = 0.001

with open("../nengo_loihi/tests/mnist10.pkl", "rb") as fh:
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

# for k in range(kernel.shape[-1]):
#     print(kernel[..., 0, k])

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
    # print(ref_outputs[:5, :5, k])
    print(np.round(ref_outputs[::10, ::10, k] / ref_outputs[:, :, k].max(), 3))

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

    up = nengo.Probe(u)

    output_ensemble = a
    if not SNIP_PROBE:
        ap = nengo.Probe(a.neurons, synapse=nengo.Alpha(0.01))


def ind_string(inds):
    d = np.diff(inds)
    if np.all(d == d[0]):
        return "slice(%d, %d, %d)" % (inds[0], inds[-1], d[0])
    else:
        return str(inds)


#with nengo_loihi.Simulator(net, dt=dt, precompute=True, dismantle=True) as sim:
#with nengo_loihi.Simulator(net, dt=dt, dismantle=True) as sim:
with nengo_loihi.Simulator(net, dt=dt, dismantle=True, precompute=False) as sim:

    # for input in sim.model.inputs:
    #     print("Input %s: %d" % (input.label, input.n_neurons))
    #     for axon in input.axons:
    #         print(
    #             "  Axon %s: target %s: %d"
    #             % (axon.label, axon.target.label, axon.n_axons)
    #         )
    # for block in sim.model.blocks:
    #     print("Block %s: %d" % (block.label, block.compartment.n_compartments))
    #     for synapse in block.synapses:
    #         print("  Synapse %s: %d" % (synapse.label, synapse.n_axons))
    #     for axon in block.axons:
    #         print(
    #             "  Axon %s: target %s: %d"
    #             % (axon.label, axon.target.label, axon.n_axons)
    #         )
    #         print("    inds: %s" % ind_string(axon.compartment_map))

    output_block = sim.model.objs[output_ensemble]["out"]
    output_blocks = list(sim.dismantle_blockmap.get(output_block, [output_block]))
    del output_block

    if SNIP_PROBE:
        snip_file_base = "test_split_convlayer_io"
        output_file_name = "test_split_convlayer_output.bin"
        steps_per_write = 10

        board = sim.sims["loihi"].board
        nxsdk_board = sim.sims["loihi"].nxsdk_board

        refract_delay = output_blocks[0].compartment.refract_delay[0]
        assert all(
            (block.compartment.refract_delay == refract_delay).all()
            for block in output_blocks
        )

        output_core_ids = []
        output_core_neurons = []
        for block in output_blocks:
            chip_idx, core_idx, block_idx, _, _ = board.find_block(block)
            assert block_idx == 0
            nxsdk_core = nxsdk_board.n2Chips[chip_idx].n2Cores[core_idx]
            output_core_ids.append(nxsdk_core.id)
            output_core_neurons.append(block.n_neurons)

        n_outputs = sum(output_core_neurons)

        # --- make snip
        env = jinja2.Environment(
            trim_blocks=True,
            loader=jinja2.FileSystemLoader(this_dir),
            keep_trailing_newline=True,
        )
        template = env.get_template(snip_file_base + ".c.template")
        code = template.render(
            steps_per_write=steps_per_write,
            output_file_name=os.path.join(this_dir, output_file_name),
            n_output_cores=len(output_core_ids),
            output_core_ids="{%s}" % ",".join(str(x) for x in output_core_ids),
            output_core_neurons="{%s}" % ",".join(str(x) for x in output_core_neurons),
            n_outputs=n_outputs,
            spike_voltage=refract_delay * 128,
        )

        cPath = os.path.join(this_dir, snip_file_base + ".c")
        with open(cPath, "w") as f:
            f.write(code)

        includeDir = this_dir
        funcName = "runMgmt"
        guardName = "doRunMgmt"
        phase = "mgmt"
        nxsdk_board.io_snip = nxsdk_board.createProcess(
            "runMgmt", cPath, includeDir, funcName, guardName, phase
        )

    # sim.run(1.0)
    sim.run(0.2)

nt = len(sim.trange())

if not SNIP_PROBE:
    a_outputs = sim.data[ap].reshape((nt, 62, 62, n_filters))
    a_outputs = a_outputs[-10:].mean(axis=0)
else:
    with open(output_file_name, "rb") as fh:
        output_bytes = []
        while True:
            line = fh.read(n_outputs)
            if len(line) > 0:
                assert len(line) == n_outputs
                output_bytes.append(line)
            else:
                break

    a_outputs = np.array(
        [np.frombuffer(b, dtype=np.uint8) for b in output_bytes], dtype=np.float32
    )

    # take mean of last 50 timesteps, because
    n = 50
    a_outputs = a_outputs[-n // steps_per_write :].mean(axis=0) / steps_per_write

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
    a_max_k = a_outputs[:, :, k].max()
    plt.imshow(a_outputs[:, :, k], vmin=0, vmax=a_max_k)
    print("Output channel %d: max: %0.3f" % (k, a_max_k))

    # print(a_outputs[:5, :5, k])
    print(np.round(a_outputs[::10, ::10, k] / a_max_k, 3))

plt.savefig("test_split_convlayer.png")
plt.show()
