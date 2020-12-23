import matplotlib.pyplot as plt
import nengo
import numpy as np

import nengo_loihi
from nengo_loihi.inputs import DVSFileChipNode

with nengo.Network() as net:
    u = DVSFileChipNode(filename="davis240c-5sec-handmove.aedat")
    h = u.height
    w = u.width
    p = u.polarity

    split_h = 4
    split_w = 4
    hh = h // split_h
    ww = w // split_w
    print((hh, ww))
    ensembles = []
    for i in range(split_h):
        ensembles_i = []
        for j in range(split_w):

            slice0 = (
                w * p * np.arange(h)[hh * i : hh * (i + 1), None, None]
                + p * np.arange(w)[None, ww * j : ww * (j + 1), None]
                + np.arange(p)[None, None, :]
            ).ravel()
            transform0 = nengo.Convolution(
                n_filters=1,
                input_shape=(2, hh, ww),
                channels_last=False,
                kernel_size=(5, 5),
                strides=(5, 5),
                padding="valid",
                # scale transform by dt, because we have 1/dt magnitude spikes
                init=nengo.dists.Choice([0.001 / (5 * 5)]),
            )
            out_shape = transform0.output_shape

            a = nengo.Ensemble(
                n_neurons=transform0.size_out,
                dimensions=1,
                neuron_type=nengo.neurons.SpikingRectifiedLinear(),
                max_rates=nengo.dists.Choice([500]),
                intercepts=nengo.dists.Choice([0]),
            )
            nengo.Connection(u[slice0], a.neurons, transform=transform0)

            ensembles_i.append(a)
        ensembles.append(ensembles_i)

    probes = [[nengo.Probe(e.neurons, synapse=0.03) for e in ei] for ei in ensembles]

with nengo_loihi.Simulator(net) as sim:
    sim.run(2.0)


for i in range(1000, 2000, 100):
    plt.figure()

    image = np.block(
        [[sim.data[p][i].reshape(out_shape.spatial_shape) for p in pi] for pi in probes]
    )

    print(image[::2, ::2])

    plt.imshow(image, vmin=-0.05, vmax=0.05)
    plt.title("Step %d" % (i))
    plt.savefig("dvs-file-step-%d.png" % i)

# plt.show()
