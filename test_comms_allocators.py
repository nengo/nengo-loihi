import nengo
from nengo.transforms import ChannelShape
import nengo_loihi
from nengo_loihi.hardware.allocators import (
    Greedy,
    GreedyComms,
    RoundRobin,
    measure_interchip_conns,
)
import numpy as np


def conv_layer(x, n_filters, input_shape, strides=1, label="conv", **kwargs):
    transform = nengo.Convolution(
        n_filters,
        input_shape,
        strides=(strides, strides),
        padding="same",
        channels_last=True,
        **kwargs,
    )
    ens = nengo.Ensemble(
        transform.output_shape.size,
        1,
        label=label,
    )
    if x is not None:
        nengo.Connection(x, ens.neurons, transform=transform)

    return ens, transform


def make_conv_network(size=None, channels=3):

    with nengo.Network(seed=0) as net:
        # the input node that will be used to feed in input images
        nengo_loihi.add_params(net)
        net.config[nengo.Ensemble].neuron_type = nengo.SpikingRectifiedLinear()
        net.config[nengo.Ensemble].max_rates = nengo.dists.Choice([100])
        net.config[nengo.Ensemble].intercepts = nengo.dists.Choice([0])
        net.config[nengo.Connection].synapse = None

        input_shape = ChannelShape((size, size) + (channels,), channels_last=True)

        input_fn = [0] * input_shape.size
        inp = nengo.Node(input_fn, label="input")

        x, transform = conv_layer(
            inp, 4, input_shape, kernel_size=(1, 1), label="to_spikes"
        )
        net.config[x].on_chip = False

        x, transform = conv_layer(
            x.neurons,
            n_filters=16,
            input_shape=transform.output_shape,
            kernel_size=(3, 3),
            strides=2,
            label="layer1",
        )
        net.config[x].block_shape = nengo_loihi.BlockShape((4, 4, 2), transform)

        x, transform = conv_layer(
            x.neurons,
            n_filters=16,
            input_shape=transform.output_shape,
            kernel_size=(3, 3),
            label="layer2",
        )
        net.config[x].block_shape = nengo_loihi.BlockShape((4, 4, 2), transform)

        x, transform = conv_layer(
            x.neurons,
            n_filters=32,
            input_shape=transform.output_shape,
            kernel_size=(3, 3),
            label="layer3",
        )
        net.config[x].block_shape = nengo_loihi.BlockShape((4, 4, 8), transform)

    return net


def make_pairs_network(n_pairs=50, factor=20):
    n_neurons = 100
    dim = 1
    gain = np.ones((n_neurons))
    neuron_type = nengo_loihi.LoihiSpikingRectifiedLinear(amplitude=1)
    transform = np.eye(n_neurons)
    rng = np.random.RandomState(1)

    with nengo.Network(seed=0) as net:
        for n in range(n_pairs):
            bias = rng.uniform(0.1, 1.0, size=(n_neurons)) * 10 * factor

            x = nengo.Ensemble(
                n_neurons,
                dim,
                gain=gain,
                bias=bias,
                neuron_type=neuron_type,
                label="a%d" % n,
            )
            y = nengo.Ensemble(
                n_neurons,
                dim,
                neuron_type=neuron_type,
                gain=gain,
                bias=np.zeros((n_neurons)),
                label="b%d" % n,
            )
            nengo.Connection(
                x.neurons,
                y.neurons,
                transform=transform,
                synapse=0.001,
            )

    return net


rng = np.random.RandomState(0)

net = make_conv_network(size=16)
# net = make_pairs_network(n_pairs=50)

n_chips = 2
# n_chips = 4

# cores_per_chip = 32
cores_per_chip = 64

with nengo_loihi.Simulator(net, target="sim") as sim:
    # --- make up firing rates
    block_rates = {}
    for block in sim.model.blocks:
        r = 100 if "layer1" in block.label else 10
        block_rates[block] = r * np.ones(block.compartment.n_compartments)
        # block_rates[block] = rng.uniform(0, r, size=block.compartment.n_compartments)

    # allocator = Greedy(cores_per_chip=cores_per_chip)
    # allocator = RoundRobin()
    allocator = GreedyComms(cores_per_chip=cores_per_chip)
    # allocator = GreedyComms(cores_per_chip=cores_per_chip, block_rates=block_rates)

    board = allocator(sim.model, n_chips=n_chips)

    for chip_idx, chip in enumerate(board.chips):
        print("Chip %d:" % (chip_idx,))

        block_labels = []
        for core_idx, core in enumerate(chip.cores):
            for block_idx, block in enumerate(core.blocks):
                block_labels.append(block.label)

        print(sorted(block_labels))

    stats = measure_interchip_conns(board)
    print(
        "Interchip: %0.2f, Intrachip: %0.2f" % (stats["interchip"], stats["intrachip"])
    )

    stats = measure_interchip_conns(board, block_rates=block_rates)
    print(
        "Interchip: %0.2f, Intrachip: %0.2f" % (stats["interchip"], stats["intrachip"])
    )

    print("Interchip pairs:")
    for a, b in stats["interchip_pairs"]:
        print("  %s: %s" % (a.label, b.label))
