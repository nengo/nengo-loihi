# TODO: add some more advanced discretization logic, or modify the training
# in some way that will result in weights more amenable to discretization

# TODO: I believe the performance used to be better, (currently around 12%
# error) and was negatively impacted by some recent change, but need to
# track that down

import collections
import gzip
import os
import pickle
from urllib.request import urlretrieve
import zipfile

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

import nengo
import nengo_dl
import nengo_loihi

# load mnist dataset
if not os.path.exists('mnist.pkl.gz'):
    urlretrieve('http://deeplearning.net/data/mnist/mnist.pkl.gz',
                'mnist.pkl.gz')

with gzip.open('mnist.pkl.gz') as f:
    train_data, _, test_data = pickle.load(f, encoding="latin1")
train_data = list(train_data)
test_data = list(test_data)
for data in (train_data, test_data):
    one_hot = np.zeros((data[0].shape[0], 10))
    one_hot[np.arange(data[0].shape[0]), data[1]] = 1
    data[1] = one_hot


def pseudo_conv(input, input_shape, kernel_shape, kernel_stride, n_filters):
    """Create a set of ensembles with sparsely tiled connections from the
    input."""

    input_inds = np.reshape(np.arange(len(input)), input_shape)

    row_range = np.arange(0, input_shape[0] - kernel_shape[0] + 1,
                          kernel_stride[0])
    col_range = np.arange(0, input_shape[1] - kernel_shape[1] + 1,
                          kernel_stride[1])
    output = nengo.Node(size_in=len(row_range) * len(col_range) * n_filters)
    ensembles = []
    for i, row in enumerate(row_range):
        for j, col in enumerate(col_range):
            ens = nengo.Ensemble(n_filters, 1).neurons
            ensembles.append(ens)

            input_idxs = np.ravel(input_inds[
                                  row:row + kernel_shape[0],
                                  col:col + kernel_shape[1]])

            nengo.Connection(input[input_idxs], ens,
                             transform=nengo_dl.dists.He())

            output_idx = (i * len(col_range) + j) * n_filters
            c = nengo.Connection(
                ens, output[output_idx:output_idx + n_filters])

            # set connections to the passthrough nodes non-trainable
            conf = nengo.Config.context[-1]
            conf[c].trainable = False

    return output, ensembles


# build the network
with nengo.Network(seed=0) as net:
    # set up default parameters
    net.config[nengo.Ensemble].neuron_type = nengo.LIFRate(
        amplitude=0.01)
    net.config[nengo.Ensemble].max_rates = nengo.dists.Choice([100])
    net.config[nengo.Ensemble].intercepts = nengo.dists.Choice([0])
    net.config[nengo.Connection].synapse = None
    nengo_dl.configure_settings(trainable=None)
    nengo_loihi.add_params(net)

    # input node
    presentation_time = 0.1
    inp = nengo.Node(
        nengo.processes.PresentInput(test_data[0], presentation_time),
        size_in=0, size_out=28 * 28)

    # convolutional layer
    conv_layer, ens = pseudo_conv(inp, (28, 28, 1), (7, 7), (3, 3), 64)

    # dense layer
    dense_layer = nengo.Ensemble(128, 1).neurons
    nengo.Connection(conv_layer, dense_layer, transform=nengo_dl.dists.He())
    # note: we could connect directly ensemble-to-ensemble (rather than
    # going through a passthrough node), but we run out of synapse memory
    # for e in ens:
    #     nengo.Connection(e, dense_layer, transform=nengo_dl.dists.He())

    # linear readout
    out = nengo.Node(label='out', size_in=10)
    nengo.Connection(dense_layer, out, transform=nengo_dl.dists.He())

    out_p = nengo.Probe(out)

    # debugging probes
    # inp_p = nengo.Probe(inp, label="input")
    # conv_p = nengo.Probe(conv_layer, label="conv")
    # ens_p = nengo.Probe(ens[0], label="ens")
    # dense_p = nengo.Probe(dense_layer, label="dense")

# set up training/test data
train_inputs = {inp: train_data[0][:, None, :]}
train_targets = {out_p: train_data[1][:, None, :]}
test_inputs = {inp: test_data[0][:, None, :]}
test_targets = {out_p: test_data[1][:, None, :]}


def crossentropy(outputs, targets):
    """Cross-entropy loss function (for training)."""
    return tf.nn.softmax_cross_entropy_with_logits_v2(logits=outputs,
                                                      labels=targets)


def classification_error(outputs, targets):
    """Classification error function (for testing)."""
    return 100 * tf.reduce_mean(
        tf.cast(tf.not_equal(tf.argmax(outputs[:, -1], axis=-1),
                             tf.argmax(targets[:, -1], axis=-1)),
                tf.float32))


# train our network in NengoDL
with nengo_dl.Simulator(net, minibatch_size=256) as sim:
    print("error before training: %.2f%%" %
          sim.loss(test_inputs, test_targets, classification_error))

    # run training
    sim.train(train_inputs, train_targets,
              tf.train.RMSPropOptimizer(learning_rate=0.001),
              objective=crossentropy,
              n_epochs=10)

    print("error after training: %.2f%%" %
          sim.loss(test_inputs, test_targets, classification_error))

    # store trained parameters back into the network
    sim.freeze_params(net)

# convert neurons to spiking LIF and add synapse to output probe
for ens in net.all_ensembles:
    ens.neuron_type = nengo.LIF(amplitude=0.01)
out_p.synapse = 0.02


def plot_results(sim):
    """Output results from the given Simulator."""

    # classification error
    data = np.reshape(sim.data[out_p],
                      (-1, int(presentation_time / sim.dt), 10))
    print("error: %.2f%%" % (100 * np.mean(
        np.argmax(data[:, -1], axis=-1) !=
        np.argmax(test_data[1][:data.shape[0]], axis=-1))))

    # plot some examples
    n_examples = 5
    f, axes = plt.subplots(2, n_examples)
    for i in range(n_examples):
        axes[0][i].imshow(np.reshape(test_data[0][i], (28, 28)))

        axes[1][i].plot(data[i])
        if i == 0:
            axes[1][i].legend([str(i) for i in range(10)], loc="upper left")
        axes[1][i].set_xlabel("time")
        axes[1][i].set_title(str(np.argmax(data[i, -1])))

    # for p in (inp_p, conv_p, ens_p, dense_p):
    #     print(p)
    #     data = sim.data[p][:int(presentation_time / sim.dt)]
    #     print(np.min(data), np.mean(data), np.max(data))
    #
    #     rates = np.sum(data > 0, axis=0) / presentation_time
    #     print(np.min(rates), np.mean(rates), np.max(rates))
    #
    #     plt.figure()
    #     plt.plot(data)
    #     plt.title(p.label)


# run in default nengo simulator
print("NENGO")
n_test = 200
with nengo.Simulator(net) as sim:
    sim.run(presentation_time * n_test)

plot_results(sim)

# run in nengo_loihi simulator
print("NENGO_LOIHI")
with nengo_loihi.Simulator(net, precompute=False) as sim:
    sim.run(presentation_time * n_test)

plot_results(sim)

plt.show()
