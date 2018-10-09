import argparse
import collections
from functools import partial
import gzip
import os
import pickle
from urllib.request import urlretrieve
import tempfile
import zipfile

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

import nengo
from nengo.utils.compat import is_iterable
import nengo_dl
from nengo_dl import SoftLIFRate

import nengo_loihi
from nengo_loihi.conv import (
    Conv2D, ImageShape, ImageSlice, ImageShifter, split_transform)

parser = argparse.ArgumentParser(description="mnist_convnet")
parser.add_argument('--retrain', action='store_true')
parser.add_argument('key', nargs='?', default='small',
                    help="Key for the network architecture to use")
args = parser.parse_args()


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

    def __str__(self):
        return '%s(%s)' % (type(self).__name__, self.name)

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
    return tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(
        logits=outputs, labels=targets))


def classification_error(outputs, targets):
    """Classification error function (for testing)."""
    return 100 * tf.reduce_mean(
        tf.cast(tf.not_equal(tf.argmax(outputs[:, -1], axis=-1),
                             tf.argmax(targets[:, -1], axis=-1)),
                tf.float32))


def percentile_rate_l2_loss(x, y, weight=1.0, target=0.0, percentile=99.):
    # x axes are (batch examples, time (==1), neurons)
    assert len(x.shape) == 3
    rates = tf.contrib.distributions.percentile(x, percentile, axis=(0, 1))
    return weight * tf.nn.l2_loss(rates - target)


def percentile_l2_loss_range(x, y, weight=1.0, min=0.0, max=np.inf,
                             percentile=99.):
    # x axes are (batch examples, time (==1), neurons)
    assert len(x.shape) == 3
    neuron_p = tf.contrib.distributions.percentile(x, percentile, axis=(0, 1))
    low_error = tf.maximum(0.0, min - neuron_p)
    high_error = tf.maximum(0.0, neuron_p - max)
    return weight * tf.nn.l2_loss(low_error + high_error)


def has_checkpoint(checkpoint_base):
    checkpoint_dir, checkpoint_name = os.path.split(checkpoint_base)
    if not os.path.exists(checkpoint_dir):
        return False

    files = os.listdir(checkpoint_dir)
    files = [f for f in files if f.startswith(checkpoint_name)]
    return len(files) > 0


def get_layer_rates(sim, input_data, rate_probes, amplitude=None):
    '''Collect firing rates on internal layers'''
    assert len(input_data) == 1
    in_p, in_x = next(iter(input_data.items()))
    assert in_x.ndim == 3
    n_steps = in_x.shape[1]

    tmpdir = tempfile.TemporaryDirectory()
    sim.save_params(os.path.join(tmpdir.name, "tmp"),
                    include_local=True, include_global=False)

    sim.run_steps(n_steps,
                       input_feeds=input_data,
                       progress_bar=False)

    rates = [sim.data[p] for p in rate_probes]
    if amplitude is not None:
        rates = [rate / amplitude for rate in rates]

    sim.load_params(os.path.join(tmpdir.name, "tmp"),
                    include_local=True, include_global=False)
    tmpdir.cleanup()

    return rates


def get_outputs(sim, input_data, output_probe):
    assert len(input_data) == 1
    in_p, in_x = next(iter(input_data.items()))
    assert in_x.ndim == 3
    n_steps = in_x.shape[1]

    tmpdir = tempfile.TemporaryDirectory()
    sim.save_params(os.path.join(tmpdir.name, "tmp"),
                    include_local=True, include_global=False)

    sim.run_steps(n_steps,
                  input_feeds=input_data,
                  progress_bar=False)

    outs = sim.data[output_probe]

    sim.load_params(os.path.join(tmpdir.name, "tmp"),
                    include_local=True, include_global=False)
    tmpdir.cleanup()

    return outs


# --- load dataset
from nengo_extras.data import load_cifar10, one_hot_from_labels
(X_train, y_train), (X_test, y_test), label_names = load_cifar10(label_names=True)
X_train = X_train.reshape(-1, 3, 32, 32).astype('float32')
X_test = X_test.reshape(-1, 3, 32, 32).astype('float32')

# basic normalize to [0, 1]
# X_train = X_train / 255.
# X_test = X_test / 255.

# basic normalize to [-1, 1]
X_train = (X_train - 127.5) / 127.5
X_test = (X_test - 127.5) / 127.5

n_classes = len(label_names)
T_train = one_hot_from_labels(y_train, n_classes)
T_test = one_hot_from_labels(y_test, n_classes)

channels_last = False
X_shape = ImageShape(32, 32, 3, channels_last=channels_last)
if channels_last:
    X_train = np.transpose(X_train, (0, 2, 3, 1))
    X_test = np.transpose(X_test, (0, 2, 3, 1))

X_min = X_train.min()
X_max = X_train.max()
print("X range: %0.3f, %0.3f" % (X_min, X_max))

# data params and data augmentation
minibatch_size = 256
shifter = ImageShifter(X_shape, shift=3, flip=True,
                       rng=np.random.RandomState(1))
input_shape = shifter.output_shape

# --- specify network parameters
checkpoint_base = './checkpoints/cifar_%s' % args.key

max_rate = 100
amp = 1. / max_rate
rate_reg = 1e-2
rate_target = max_rate * amp  # must be in amplitude scaled units

neuron_type = SoftLIFRate(amplitude=amp, sigma=0.01)
# neuron_type = nengo.RectifiedLinear(amplitude=amp)
if args.key == 'small':
    layer_dicts = [
        dict(layer_func=TfConv2d('layer1', 1, kernel_size=1,
                                 initializer=tf.constant_initializer(1)),
             neuron_type=nengo.RectifiedLinear(amplitude=amp),
             on_chip=False, no_min_rate=True),
        # ^ Has to be one channel input for now since we can't send pop spikes to chip
        dict(layer_func=TfConv2d('layer2', 16, strides=2), neuron_type=neuron_type),
        dict(layer_func=TfDense('layer_out', 10)),
    ]
elif args.key == 'med':
    layer_dicts = [
        dict(layer_func=TfConv2d('layer1', 1, kernel_size=1,
                                 initializer=tf.constant_initializer(1)),
             neuron_type=nengo.RectifiedLinear(amplitude=amp),
             on_chip=False, no_min_rate=True),
        # ^ Has to be one channel input for now since we can't send pop spikes to chip
        dict(layer_func=TfConv2d('layer2', 16, strides=2), neuron_type=neuron_type),
        dict(layer_func=TfConv2d('layer3', 32, strides=2), neuron_type=neuron_type),
        dict(layer_func=TfDense('layer_out', 10)),
    ]
elif args.key == 'large':
    layer_dicts = [
        # dict(layer_func=TfConv2d('layer1', 1, kernel_size=1,
        #                          initializer=tf.constant_initializer(1)),
        #      neuron_type=nengo.RectifiedLinear(amplitude=amp),
        #      on_chip=False, no_min_rate=True),
        # # ^ Has to be one channel input for now since we can't send pop spikes to chip
        dict(layer_func=TfConv2d('layer1', 3, kernel_size=1),
             neuron_type=nengo.RectifiedLinear(amplitude=amp),
             on_chip=False, no_min_rate=True),
        dict(layer_func=TfConv2d('layer2', 32, strides=2), neuron_type=neuron_type),
        dict(layer_func=TfConv2d('layer3', 96, strides=2), neuron_type=neuron_type),
        dict(layer_func=TfConv2d('layer4', 256, strides=2), neuron_type=neuron_type),
        dict(layer_func=TfDense('layer_out', 10)),
    ]
elif args.key == 'demolike':
    layer_dicts = [
        dict(layer_func=TfConv2d('layer1', 4, kernel_size=1),
             neuron_type=nengo.RectifiedLinear(amplitude=amp),
             on_chip=False, no_min_rate=True),
        dict(layer_func=TfConv2d('layer2', 64, strides=2), neuron_type=neuron_type),
        dict(layer_func=TfConv2d('layer3', 96), neuron_type=neuron_type),
        dict(layer_func=TfConv2d('layer4', 128, strides=2), neuron_type=neuron_type),
        dict(layer_func=TfConv2d('layer5', 128), neuron_type=neuron_type),
        dict(layer_func=TfConv2d('layer6', 128, kernel_size=1), neuron_type=neuron_type),
        dict(layer_func=TfDense('layer_out', 10)),
    ]
elif args.key == 'demolike2':
    layer_dicts = [
        dict(layer_func=TfConv2d('layer1', 4, kernel_size=1,
                                 initializer=tf.constant_initializer(0.33)),
             neuron_type=nengo.RectifiedLinear(amplitude=amp),
             on_chip=False, no_min_rate=True),
        dict(layer_func=TfConv2d('layer2', 64, strides=2), neuron_type=neuron_type),
        dict(layer_func=TfConv2d('layer3', 96), neuron_type=neuron_type),
        dict(layer_func=TfConv2d('layer4', 96), neuron_type=neuron_type),
        dict(layer_func=TfConv2d('layer5', 96), neuron_type=neuron_type),
        dict(layer_func=TfConv2d('layer6', 96, kernel_size=1), neuron_type=neuron_type),
        dict(layer_func=TfDense('layer_out', 10)),
    ]
else:
    raise ValueError("Unrecognized architecture key %r" % (args.key,))

# --- build the nengo_dl network
print("Building nengo_dl network %r (retrain=%r)" % (args.key, args.retrain))

objective = {}
rate_probes = collections.OrderedDict()
with nengo.Network(seed=0) as net:
    net.config[nengo.Ensemble].max_rates = nengo.dists.Choice([max_rate])
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
        no_min_rate = layer_dict.pop('no_min_rate', False)

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

            yp = nengo.Probe(y)
            if no_min_rate:
                objective[yp] = partial(
                    percentile_l2_loss_range, weight=10*rate_reg,
                    min=0, max=rate_target, percentile=99.9)
            else:
                objective[yp] = partial(
                    percentile_l2_loss_range, weight=rate_reg,
                    min=0.75*rate_target, max=rate_target, percentile=99.9)
            rate_probes[layer_func] = yp

        shape_in = shape_out
        layers.append((fn_layer, neuron_layer))
        print("Added layer %s %s shape=%s size=%s" % (
            layer_func, layer_neuron, shape_out, shape_out.size))

    out_p = nengo.Probe(x)
    out_p_filt = nengo.Probe(x, synapse=0.1)

objective[out_p] = crossentropy

# set up training/test data
train_inputs = {inp: shifter.center(X_train).reshape(X_train.shape[0], 1, -1)}
train_targets = {out_p: T_train.reshape(T_train.shape[0], 1, -1)}
test_inputs = {inp: shifter.center(X_test).reshape(X_test.shape[0], 1, -1)}
test_targets = {out_p: T_test.reshape(T_test.shape[0], 1, -1)}
rate_inputs = {inp: shifter.center(X_train[:minibatch_size]).reshape(
    minibatch_size, 1, -1)}

# train our network in NengoDL
with nengo_dl.Simulator(net, minibatch_size=minibatch_size) as sim:
    if not args.retrain and has_checkpoint(checkpoint_base):
        sim.load_params(checkpoint_base)

    else:
        print("Test error before training: %.2f%%" %
              sim.loss(test_inputs, test_targets, classification_error))

        # run training
        n_epochs = 30
        optimizer = tf.train.RMSPropOptimizer(learning_rate=0.001)
        for _ in range(n_epochs):
            train_augmented = {
                inp: shifter.augment(X_train).reshape(X_train.shape[0], 1, -1)}
            sim.train(train_augmented, train_targets, optimizer,
                      objective=objective, n_epochs=1)

        sim.save_params(checkpoint_base)

        print("Train error after training: %.2f%%" %
              sim.loss(train_inputs, train_targets, classification_error))
        print("Test error after training: %.2f%%" %
              sim.loss(test_inputs, test_targets, classification_error))

        rates = get_layer_rates(sim, rate_inputs, rate_probes.values(),
                                amplitude=amp)
        for layer_func, rate in zip(rate_probes, rates):
            print("%s rate: mean=%0.3f, 99th: %0.3f" % (
                layer_func, rate.mean(), np.percentile(rate, 99)))

    # compute output range
    outs = get_outputs(sim, rate_inputs, out_p)
    print("Output range: min=%0.3f, 1st=%0.3f, 99th=%0.3f, max=%0.3f" % (
        outs.min(), np.percentile(outs, 1), np.percentile(outs, 99), outs.max()))
    ann_out_min = np.percentile(outs, 1)
    ann_out_max = np.percentile(outs, 99)

    # store trained parameters back into the network
    for fn_layer, _ in layers:
        fn_layer.tensor_func.save_params(sim.sess)

del net  # so we don't accidentally use it
del x

# --- Spiking network
presentation_images = shifter.center(X_test).reshape(X_test.shape[0], -1)
presentation_time = 0.2
present_images = nengo.processes.PresentInput(
    presentation_images, presentation_time)

with nengo.Network() as nengo_net:
    nengo_loihi.add_params(nengo_net)  # allow setting on_chip

    nengo_net.config[nengo.Ensemble].max_rates = nengo.dists.Choice([100])
    nengo_net.config[nengo.Ensemble].intercepts = nengo.dists.Choice([0])

    # nengo_net.config[nengo.Connection].synapse = 0.005
    nengo_net.config[nengo.Connection].synapse = 0.01

    inp = nengo.Node(present_images, label='input')
    inp_p = nengo.Probe(inp)
    xx = [inp]
    xslices = [ImageSlice(input_shape)]
    cores_used = 0

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
                    max_size=1024, max_channels=8)
                for image_slice in split_slices:
                    assert image_slice.size <= 1024
                    idxs = image_slice.channel_idxs()
                    y = nengo.Ensemble(
                        image_slice.size, 1,
                        neuron_type=layer_neurons,
                        label="%s_%d:%d" % (func.name, min(idxs), max(idxs)))
                    yy.append(y.neurons)
                    yslices.append(image_slice)
                    cores_used += 1
        else:
            output_shape = func.output_shape(func.shape_in)
            assert output_shape.size == fn_layer.size_out
            if 0:  # node works on emulator, on Loihi we need something on-chip
                y = nengo.Node(size_in=output_shape.size, label=func.name)
                yy.append(y)
                yslices.append(ImageSlice(output_shape))
                out_p = nengo.Probe(y, synapse=nengo.Alpha(0.01))
            else:
                # max_rate = 300.
                max_rate = 100.
                gain = max_rate / (ann_out_max - ann_out_min)
                bias = -gain * ann_out_min
                y = nengo.Ensemble(output_shape.size, 1, label=func.name,
                                   neuron_type=nengo.SpikingRectifiedLinear(),
                                   gain=nengo.dists.Choice([gain]),
                                   bias=nengo.dists.Choice([bias]))
                yy.append(y.neurons)
                yslices.append(ImageSlice(output_shape))
                out_p = nengo.Probe(y.neurons, synapse=nengo.Alpha(0.02))
                cores_used += 1

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

print("Used %d cores" % cores_used)

n_presentations = 10
with nengo_loihi.Simulator(nengo_net, dt=0.001, precompute=False) as sim:
    sim.run(n_presentations * presentation_time)

class_output = sim.data[out_p]
steps_per_pres = int(presentation_time / sim.dt)
preds = []
for i in range(0, class_output.shape[0], steps_per_pres):
    c = class_output[i:i + steps_per_pres]
    c = c[int(0.7 * steps_per_pres):]  # take last part
    pred = np.argmax(c.sum(axis=0))
    preds.append(pred)

print("Predictions: %s" % (preds,))
print("Actual:      %s" % (list(y_test[:n_presentations]),))

# --- fancy plots
plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
images = X_test if X_shape.channels_last else np.transpose(X_test, (0, 2, 3, 1))
ni, nj, nc = images[0].shape
allimage = np.zeros((ni, nj*n_presentations, nc))
for i, image in enumerate(images[:n_presentations]):
    allimage[:, i*nj:(i + 1)*nj] = image
if allimage.shape[-1] == 1:
    allimage = allimage[:, :, 0]
allimage = (allimage - X_min) / (X_max - X_min)
plt.imshow(allimage, aspect='auto', interpolation='none', cmap='gray')

plt.subplot(2, 1, 2)
plt.plot(sim.trange(), sim.data[out_p])
plt.legend(label_names, loc='best')

target = sim.target if isinstance(sim, nengo_loihi.Simulator) else 'nengo'
plt.savefig('cifar_%s_%s.png' % (args.key, target))
# plt.show()
