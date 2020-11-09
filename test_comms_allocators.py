import nengo
from nengo.transforms import ChannelShape
import nengo_loihi
from nengo_loihi.hardware.allocators import Greedy, GreedyComms


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


def make_network(size=None, channels=3):

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
        net.config[x].block_shape = nengo_loihi.BlockShape(
            (4, 4, 2), transform.input_shape.shape
        )

        x, transform = conv_layer(
            x.neurons,
            n_filters=16,
            input_shape=transform.output_shape,
            kernel_size=(3, 3),
            label="layer2",
        )
        net.config[x].block_shape = nengo_loihi.BlockShape(
            (4, 4, 2), transform.input_shape.shape
        )

    return net


net = make_network(size=16)

n_chips = 2
# n_chips = 4
cores_per_chip = 32
# allocator = Greedy(cores_per_chip=cores_per_chip)
allocator = GreedyComms(cores_per_chip=cores_per_chip)


with nengo_loihi.Simulator(net) as sim:
    board = allocator(sim.model, n_chips=n_chips)

    for chip_idx, chip in enumerate(board.chips):
        print("Chip %d:" % (chip_idx,))

        block_labels = []
        for core_idx, core in enumerate(chip.cores):
            for block_idx, block in enumerate(core.blocks):
                block_labels.append(block.label)

        print(block_labels)
