"""
NOTES:
- Occasionally, the training for the original network can fail to converge
  (the loss stops going down at the start of training, and remains around 500).
  I believe this is due to bad random weights chosen for the initial kernels.
  In this case, simply restart the training and it should work.
"""

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
from nengo_loihi.conv import Conv2D, ImageShape, ImageSlice, split_transform


class TfConv2d(object):
    KERNEL_IDX = 0

    def __init__(self, name, n_filters, kernel_size=(3, 3), strides=(1, 1),
                 initializer=None):
        self.name = name
        self.n_filters = n_filters
        self.kernel_size = kernel_size if is_iterable(kernel_size) else (
            kernel_size, kernel_size)
        self.strides = strides if is_iterable(strides) else (strides, strides)
        self.padding = 'VALID'
        self.initializer = initializer

        self.kernel = None
        self.shape_in = None

    def output_shape(self, input_shape=None):
        if input_shape is None:
            assert self.shape_in is not None
            input_shape = self.shape_in
        conv2d = Conv2D(
            self.n_filters,
            input_shape,
            kernel_size=self.kernel_size,
            strides=self.strides,
        )
        return conv2d.output_shape

    def pre_build(self, shape_in, shape_out):
        assert isinstance(self.shape_in, ImageShape)
        assert shape_in[1] == self.shape_in.size
        ni, nj, nc = self.shape_in.shape(channels_last=True)
        nf = self.n_filters
        si, sj = self.kernel_size
        self.kernel = tf.get_variable('kernel_%s' % self.name,
                                      shape=(si, sj, nc, nf),
                                      initializer=self.initializer)

    def __call__(self, t, x):
        batch_size = x.get_shape()[0].value
        x = tf.reshape(x, (batch_size,) + self.shape_in.shape())

        # make strides 1 for examples, channels
        channels_last = self.shape_in.channels_last
        strides = ((1,) + self.strides + (1,) if channels_last else
                   (1, 1) + self.strides)

        data_format = 'NHWC' if channels_last else 'NCHW'
        y = tf.nn.conv2d(x, self.kernel, strides, self.padding,
                         data_format=data_format)
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
                                  correlate=True)


class TfDense(object):
    def __init__(self, name, n_outputs):
        self.name = name
        self.n_outputs = n_outputs

        self.weights = None
        self.shape_in = None

    def output_shape(self, input_shape):
        return ImageShape(1, 1, self.n_outputs, channels_last=True)

    def pre_build(self, shape_in, shape_out):
        assert isinstance(self.shape_in, ImageShape)
        assert shape_in[1] == self.shape_in.size
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
# RETRAIN = True

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

channels_last = False
input_shape = ImageShape(28, 28, 1, channels_last=channels_last)
test_images = test_data[0].reshape((-1,) + input_shape.shape(channels_last=True))
if not channels_last:
    test_images = np.transpose(test_images, (0, 3, 1, 2))


neuron_type = SoftLIFRate(amplitude=amp, sigma=0.01)
layer_dicts = [
    # dict(layer_func=TfConv2d('layer1', 2, kernel_size=1), neuron_type=nengo.RectifiedLinear(), on_chip=False),
    # dict(layer_func=TfConv2d('layer2', 64, kernel_size=3, strides=2), neuron_type=neuron_type),
    # dict(layer_func=TfConv2d('layer3', 128, kernel_size=3, strides=2), neuron_type=neuron_type),
    # dict(layer_func=TfConv2d('layer4', 256, kernel_size=3, strides=2), neuron_type=neuron_type),
    # dict(layer_func=TfDense('layer_out', 10)),
    dict(layer_func=TfConv2d('layer1', 1, kernel_size=1, initializer=tf.constant_initializer(1)),
         neuron_type=nengo.RectifiedLinear(), on_chip=False),
    # ^ Has to be one channel input for now since we can't send pop spikes to chip
    dict(layer_func=TfConv2d('layer2', 32, kernel_size=3, strides=2), neuron_type=neuron_type),
    dict(layer_func=TfConv2d('layer3', 64, kernel_size=3, strides=2), neuron_type=neuron_type),
    dict(layer_func=TfConv2d('layer4', 128, kernel_size=3, strides=2), neuron_type=neuron_type),
    dict(layer_func=TfDense('layer_out', 10)),
]

# build the network
with nengo.Network(seed=0) as net:
    net.config[nengo.Ensemble].max_rates = nengo.dists.Choice([100])
    net.config[nengo.Ensemble].intercepts = nengo.dists.Choice([0])
    net.config[nengo.Connection].synapse = None

    nengo_dl.configure_settings(trainable=False)

    # the input node that will be used to feed in input images
    inp = nengo.Node([0] * input_shape.size, label='input')

    layers = []
    shape_in = input_shape
    x = inp
    for layer_dict in layer_dicts:
        layer_dict = dict(layer_dict)  # so we can pop
        layer_func = layer_dict.pop('layer_func', None)
        layer_neuron = layer_dict.pop('neuron_type', None)
        on_chip = layer_dict.pop('on_chip', True)

        shape_out = None
        fn_layer = None
        neuron_layer = None

        if layer_func is not None:
            assert shape_in
            layer_func.shape_in = shape_in
            shape_out = layer_func.output_shape(shape_in)

            size_in = shape_in.size if shape_in else x.size_out
            size_out = shape_out.size if shape_out else size_in
            y = nengo_dl.TensorNode(layer_func,
                                    size_in=size_in, size_out=size_out,
                                    label=layer_func.name)
            nengo.Connection(x, y)
            x = y
            fn_layer = x

        if layer_neuron is not None:
            y = nengo.Ensemble(x.size_out, 1, neuron_type=layer_neuron).neurons
            y.image_shape = shape_out if shape_out is not None else shape_in
            y.on_chip = on_chip
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
                  n_epochs=10)
        sim.save_params(checkpoint_base)

        print("error after training: %.2f%%" %
              sim.loss(test_inputs, test_targets, classification_error))

    # store trained parameters back into the network
    for fn_layer, _ in layers:
        fn_layer.tensor_func.save_params(sim.sess)

del net  # so we don't accidentally use it
del x

# --- Spiking network
with nengo.Network() as nengo_net:
    nengo_loihi.add_params(nengo_net)  # allow setting on_chip

    nengo_net.config[nengo.Ensemble].max_rates = nengo.dists.Choice([100])
    nengo_net.config[nengo.Ensemble].intercepts = nengo.dists.Choice([0])

    nengo_net.config[nengo.Connection].synapse = 0.005

    presentation_time = 0.1
    inp = nengo.Node(
        nengo.processes.PresentInput(
            test_images.reshape(test_images.shape[0], -1), presentation_time),
        size_in=0, size_out=test_images[0].size, label='input')
    inp_p = nengo.Probe(inp)
    xx = [inp]
    xslices = [ImageSlice(input_shape)]

    for fn_layer, neuron_layer in layers:
        func = fn_layer.tensor_func
        name = func.name

        # --- create neuron layer
        yy = []
        yslices = []
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

            image_shape = neuron_layer.image_shape
            if not neuron_layer.on_chip:
                y = nengo.Ensemble(
                    image_shape.size, 1,
                    neuron_type=layer_neurons,
                    label="%s" % (func.name,))
                nengo_net.config[y].on_chip = False
                yy.append(y.neurons)
                yslices.append(ImageSlice(image_shape))
            else:
                split_slices = image_shape.split_channels(
                    max_size=1024, max_channels=16)
                for image_slice in split_slices:
                    assert image_slice.size <= 1024
                    idxs = image_slice.channel_idxs()
                    y = nengo.Ensemble(
                        image_slice.size, 1,
                        neuron_type=layer_neurons,
                        label="%s_%d:%d" % (func.name, min(idxs), max(idxs)))
                    yy.append(y.neurons)
                    yslices.append(image_slice)
        else:
            output_shape = func.output_shape(func.shape_in)
            assert output_shape.size == fn_layer.size_out
            if 0:  # node works on emulator, on Loihi we need something on-chip
                y = nengo.Node(size_in=output_shape.size, label=func.name)
                yy.append(y)
                yslices.append(ImageSlice(output_shape))
                out_p = nengo.Probe(y, synapse=nengo.Alpha(0.01))
            else:
                min_range = -10
                max_range = 0
                max_rate = 300.
                gain = max_rate / (max_range - min_range)
                bias = -gain * min_range
                y = nengo.Ensemble(output_shape.size, 1, label=func.name,
                                   neuron_type=nengo.SpikingRectifiedLinear(),
                                   gain=nengo.dists.Choice([gain]),
                                   bias=nengo.dists.Choice([bias]))
                yy.append(y.neurons)
                yslices.append(ImageSlice(output_shape))
                out_p = nengo.Probe(y.neurons, synapse=nengo.Alpha(0.01))

        assert len(yy) == len(yslices)

        # --- create function layer
        transform = func.transform()
        for xi, (x, xslice) in enumerate(zip(xx, xslices)):
            for yi, (y, yslice) in enumerate(zip(yy, yslices)):
                transform_xy = split_transform(transform, xslice, yslice)
                nengo.Connection(x, y, transform=transform_xy)

        xx = yy
        xslices = yslices

    # out_p = nengo.Probe(y, synapse=nengo.Alpha(0.01))

n_presentations = 5
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

plt.savefig('mnist_convnet_%s.png' % sim.target)
# plt.show()
