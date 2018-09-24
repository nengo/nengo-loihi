from collections import defaultdict
import pickle

import matplotlib.pyplot as plt
import numpy as np
import nengo
import nengo_dl
import tensorflow as tf

import nengo_loihi

with open("cifar-data/batches.meta", "rb") as f:
    label_names = pickle.load(f, encoding="bytes")[b"label_names"]


def load(filenames):
    new_data = defaultdict(list)

    for filename in filenames:
        with open(filename, "rb") as f:
            data = pickle.load(f, encoding="bytes")
            for k in [b"data", b"labels"]:
                new_data[k.decode("utf-8")].append(data[k])

    for k, v in new_data.items():
        if k == "data":
            new_data[k] = np.concatenate(v, axis=0)[:, None, :] / 255 - 0.5
        else:
            labels = np.concatenate(v, axis=0)
            targets = np.zeros((labels.shape[0], 10))
            targets[np.arange(labels.shape[0]), labels] = 1
            new_data[k] = targets[:, None, :]

    return new_data


train_data = load(["cifar-data/data_batch_1", "cifar-data/data_batch_2",
                   "cifar-data/data_batch_3", "cifar-data/data_batch_4"])
test_data = load(["cifar-data/data_batch_5"])


with nengo.Network() as net:
    # def conv_layer(x, *args, activation="relu", **kwargs):
    #     conv = nengo_loihi.Conv2D(*args, **kwargs)
    #     if activation is None:
    #         layer = nengo.Node(size_in=conv.output_shape.size)
    #     else:
    #         layer = nengo.Ensemble(conv.output_shape.size, 1).neurons
    #     nengo.Connection(x, layer, transform=conv)
    #
    #     print("LAYER")
    #     print(conv.input_shape.shape(), "->", conv.output_shape.shape())
    #
    #     return layer, conv

    def conv_layer(x, filters, input_shape, activation=tf.nn.relu,
                   kernel_size=3, **kwargs):
        layer, conn = nengo_dl.tensor_layer(
            x, tf.layers.conv2d, shape_in=input_shape.shape(),
            kernel_size=kernel_size, filters=filters,
            activation=activation, return_conn=True,
            data_format=("channels_last" if input_shape.channels_last else
                         "channels_first"),
            **kwargs)
        net.config[conn].trainable = True if activation is None else False

        conv = nengo_loihi.Conv2D(filters, input_shape,
                                  kernel_size=kernel_size, **kwargs)

        print("LAYER")
        print(conv.input_shape.shape(), "->", conv.output_shape.shape())

        return layer, conv


    nengo_loihi.add_params(net)
    nengo_dl.configure_settings(trainable=None)
    net.config[nengo.Ensemble].neuron_type = nengo.RectifiedLinear()
    net.config[nengo.Ensemble].gain = nengo.dists.Choice([1])
    net.config[nengo.Ensemble].bias = nengo.dists.Choice([0])
    net.config[nengo.Connection].synapse = None

    input = nengo.Node(np.zeros(32 * 32 * 3))

    layer, conv = conv_layer(
        input, 6, nengo_loihi.conv.ImageShape.from_shape((3, 32, 32),
                                                         channels_last=False),
        kernel_size=1)
    # net.config[layer.ensemble].on_chip = False

    layer, conv = conv_layer(layer, 64, conv.output_shape, strides=2)
    for _ in range(3):
        layer, conv = conv_layer(layer, 96, conv.output_shape)
    layer, conv = conv_layer(layer, 96, conv.output_shape, kernel_size=1)
    layer, conv = conv_layer(layer, 10, conv.output_shape, kernel_size=1,
                             activation=None)

    # sum across rows/cols (average pooling)
    transform = np.zeros((10,) + conv.output_shape.shape())
    for i in range(10):
        transform[i, i, :, :] = 1 / conv.output_shape.n_pixels
    transform = np.reshape(transform, (-1, 10))
    output1 = nengo.Node(size_in=10)
    conn = nengo.Connection(layer, output1, transform=transform.T)
    net.config[conn].trainable = True

    output = nengo.Probe(output1)


def class_err(outputs, targets):
    return 100 * tf.reduce_mean(
        tf.cast(tf.not_equal(tf.argmax(outputs, axis=-1),
                             tf.argmax(targets, axis=-1)),
                tf.float32))


def objective(outputs, targets):
    return tf.nn.softmax_cross_entropy_with_logits_v2(
        labels=targets, logits=outputs)


with nengo_dl.Simulator(net, minibatch_size=64,
                        tensorboard="./tensorboard") as sim:
    print("pre err", sim.loss(
        {input: test_data["data"]}, {output: test_data["labels"]}, class_err))

    sim.train({input: train_data["data"]}, {output: train_data["labels"]},
              tf.train.RMSPropOptimizer(1e-3), objective=objective,
              n_epochs=10, summaries=["loss"])

    print("post err", sim.loss(
        {input: test_data["data"]}, {output: test_data["labels"]}, class_err))
