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
from nengo.utils.compat import is_iterable
import nengo_dl
from nengo_dl import SoftLIFRate

import nengo_loihi
from nengo_loihi.conv import Conv2D


class TfConv2d(object):
    KERNEL_IDX = 0

    def __init__(self, name, n_filters, kernel_size=(3, 3), strides=(1, 1)):
        self.name = name
        self.n_filters = n_filters
        self.kernel_size = kernel_size if is_iterable(kernel_size) else (
            kernel_size, kernel_size)
        self.strides = strides if is_iterable(strides) else (strides, strides)
        self.padding = 'VALID'

        self.shape_in = None

        self.kernel = None

    def output_shape(self, input_shape):
        conv2d = Conv2D(
            self.n_filters,
            input_shape,
            kernel_size=self.kernel_size,
            strides=self.strides,
        )
        return conv2d.output_shape.shape(channels_last=True)

    def pre_build(self, shape_in, shape_out):
        assert self.shape_in is not None
        assert np.prod(self.shape_in) == shape_in[1]
        ni, nj, nc = self.shape_in
        nf = self.n_filters
        si, sj = self.kernel_size
        self.kernel = tf.get_variable('kernel_%s' % self.name,
                                      shape=(si, sj, nc, nf),
                                      initializer=None)

    def __call__(self, t, x):
        batch_size = x.get_shape()[0].value
        x = tf.reshape(x, (batch_size,) + self.shape_in)
        strides = (1,) + self.strides + (1,)  # 1 for examples, channels
        y = tf.nn.conv2d(x, self.kernel, strides, self.padding,
                         data_format='NHWC')
        return tf.reshape(y, (batch_size, -1))

    def save_params(self, sess):
        self.kernel_value = sess.run(self.kernel)

    def transform(self):
        assert self.shape_in is not None
        kernel = self.kernel_value
        kernel = np.transpose(kernel, (2, 0, 1, 3))  # (nc, si, sj, nf)
        return Conv2D.from_kernel(kernel, self.shape_in,
                                  strides=self.strides,
                                  mode=self.padding.lower(),
                                  correlate=True,
                                  output_channels_last=True)


class TfDense(object):
    def __init__(self, name, n_outputs):
        self.name = name
        self.n_outputs = n_outputs
        self.shape_in = None

    def output_shape(self, input_shape):
        return (self.n_outputs,)

    def pre_build(self, shape_in, shape_out):
        assert shape_out[1] == self.n_outputs
        n_inputs = shape_in[1]
        n_outputs = self.n_outputs
        self.weights = tf.get_variable('weights_%s' % self.name,
                                       shape=(n_inputs, n_outputs))

    def __call__(self, t, x):
        x = tf.reshape(x, (x.shape[0], -1))
        return tf.tensordot(x, self.weights, axes=[[1], [0]])

    def save_params(self, sess):
        self.weights_value = sess.run(self.weights)

    def transform(self):
        weights = self.weights_value
        return weights.T


def crossentropy(outputs, targets):
    """Cross-entropy loss function (for training)."""
    return tf.nn.softmax_cross_entropy_with_logits_v2(
        logits=outputs, labels=targets)


def classification_error(outputs, targets):
    """Classification error function (for testing)."""
    return 100 * tf.reduce_mean(
        tf.cast(tf.not_equal(tf.argmax(outputs[:, -1], axis=-1),
                             tf.argmax(targets[:, -1], axis=-1)),
                tf.float32))


def has_checkpoint(checkpoint_base):
    checkpoint_dir, checkpoint_name = os.path.split(checkpoint_base)
    if not os.path.exists(checkpoint_dir):
        return False

    files = os.listdir(checkpoint_dir)
    files = [f for f in files if f.startswith(checkpoint_name)]
    return len(files) > 0


checkpoint_base = './checkpoints/mnist_convnet'
RETRAIN = False

amp = 0.01

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


neuron_type = SoftLIFRate(amplitude=amp, sigma=0.01)
layer_dicts = [
    # dict(layer_func=tf.layers.conv2d, neuron_type=nengo.RectifiedLinear(), filters=2, kernel_size=1),
    # dict(layer_func=tf.layers.conv2d, neuron_type=neuron_type, filters=32, kernel_size=3),
    # dict(layer_func=tf.layers.conv2d, neuron_type=neuron_type, filters=64, kernel_size=3, strides=2),
    # dict(layer_func=tf.layers.conv2d, neuron_type=neuron_type, filters=128, kernel_size=3, strides=2),
    # dict(layer_func=tf.layers.dense, units=10),
    dict(layer_func=TfConv2d('layer1', 2, kernel_size=1), neuron_type=nengo.RectifiedLinear()),
    dict(layer_func=TfConv2d('layer2', 32, kernel_size=3), neuron_type=neuron_type),
    dict(layer_func=TfConv2d('layer3', 64, kernel_size=3, strides=2), neuron_type=neuron_type),
    dict(layer_func=TfConv2d('layer4', 128, kernel_size=3, strides=2), neuron_type=neuron_type),
    dict(layer_func=TfDense('layer5', 10)),
]

# build the network
with nengo.Network(seed=0) as net:
    net.config[nengo.Ensemble].max_rates = nengo.dists.Choice([100])
    net.config[nengo.Ensemble].intercepts = nengo.dists.Choice([0])
    net.config[nengo.Connection].synapse = None

    nengo_dl.configure_settings(trainable=False)

    # the input node that will be used to feed in input images
    inp = nengo.Node([0] * 28 * 28, label='input')

    layers = []
    shape_in = (28, 28, 1)
    x = inp
    for layer_dict in layer_dicts:
        layer_dict = dict(layer_dict)  # so we can pop
        layer_func = layer_dict.pop('layer_func', None)
        layer_neuron = layer_dict.pop('neuron_type', None)

        shape_out = None
        fn_layer = None
        neuron_layer = None

        if layer_func is not None:
            assert shape_in
            layer_func.shape_in = shape_in
            shape_out = layer_func.output_shape(shape_in)

            size_in = np.prod(shape_in) if shape_in else x.size_out
            size_out = np.prod(shape_out) if shape_out else size_in
            y = nengo_dl.TensorNode(layer_func,
                                    size_in=size_in, size_out=size_out)
            nengo.Connection(x, y)
            x = y
            fn_layer = x

        if layer_neuron is not None:
            y = nengo.Ensemble(x.size_out, 1, neuron_type=layer_neuron).neurons
            nengo.Connection(x, y)
            x = y
            neuron_layer = x

        shape_in = shape_out
        layers.append((fn_layer, neuron_layer))

    out_p = nengo.Probe(x)
    out_p_filt = nengo.Probe(x, synapse=0.1)


# set up training/test data
train_inputs = {inp: train_data[0][:, None, :]}
train_targets = {out_p: train_data[1][:, None, :]}
test_inputs = {inp: test_data[0][:, None, :]}
test_targets = {out_p: test_data[1][:, None, :]}

# train our network in NengoDL
with nengo_dl.Simulator(net, minibatch_size=256) as sim:
    if not RETRAIN and has_checkpoint(checkpoint_base):
        sim.load_params(checkpoint_base)

    else:
        print("error before training: %.2f%%" %
              sim.loss(test_inputs, test_targets, classification_error))

        # run training
        sim.train(train_inputs, train_targets,
                  tf.train.RMSPropOptimizer(learning_rate=0.001),
                  objective=crossentropy,
                  n_epochs=1)
        sim.save_params(checkpoint_base)

    print("error after training: %.2f%%" %
          sim.loss(test_inputs, test_targets, classification_error))

    # store trained parameters back into the network
    for fn_layer, _ in layers:
        fn_layer.tensor_func.save_params(sim.sess)

del net  # so we don't accidentally use it
del x

with nengo.Network() as nengo_net:
    nengo_net.config[nengo.Ensemble].max_rates = nengo.dists.Choice([100])
    nengo_net.config[nengo.Ensemble].intercepts = nengo.dists.Choice([0])

    nengo_net.config[nengo.Connection].synapse = 0.005
    # nengo_net.config[nengo.Connection].synapse = nengo.Alpha(0.003)

    # presentation_time = 0.001
    presentation_time = 0.2
    inp = nengo.Node(
        nengo.processes.PresentInput(test_data[0], presentation_time),
        size_in=0, size_out=28 * 28)
    inp_p = nengo.Probe(inp)
    x = inp

    for fn_layer, neuron_layer in layers:
        # --- create neuron layer
        if neuron_layer is not None:
            layer_neurons = neuron_layer.ensemble.neuron_type
            if isinstance(layer_neurons, SoftLIFRate):
                layer_neurons = nengo.LIF(tau_rc=layer_neurons.tau_rc,
                                          tau_ref=layer_neurons.tau_ref,
                                          amplitude=layer_neurons.amplitude)
            elif isinstance(layer_neurons, nengo.RectifiedLinear):
                layer_neurons = nengo.SpikingRectifiedLinear(
                                          amplitude=layer_neurons.amplitude)
            else:
                raise ValueError("Unsupported neuron type %s" % layer_neurons)

            y = nengo.Ensemble(neuron_layer.size_out, 1,
                               neuron_type=layer_neurons).neurons
        else:
            y = nengo.Node(size_in=fn_layer.size_out)

        # --- create function layer
        transform = fn_layer.tensor_func.transform()
        nengo.Connection(x, y, transform=transform)
        x = y

    out_p = nengo.Probe(x, synapse=nengo.Alpha(0.01))


n_presentations = 5
# with nengo.Simulator(nengo_net, dt=0.001, optimize=False, progress_bar=False) as sim:
with nengo_loihi.Simulator(nengo_net, dt=0.001, precompute=True) as sim:
    sim.run(n_presentations * presentation_time)


# --- fancy plots
plt.figure()

plt.subplot(2, 1, 1)
images = test_data[0].reshape(-1, 28, 28, 1)
ni, nj, nc = images[0].shape
allimage = np.zeros((ni, nj*n_presentations, nc), dtype=images.dtype)
for i, image in enumerate(images[:n_presentations]):
    allimage[:, i*nj:(i + 1)*nj] = image
if allimage.shape[-1] == 1:
    allimage = allimage[:, :, 0]
plt.imshow(allimage, aspect='auto', interpolation='none', cmap='gray')

plt.subplot(2, 1, 2)
plt.plot(sim.trange(), sim.data[out_p])
plt.legend(['%d' % i for i in range(10)], loc='best')

plt.show()
