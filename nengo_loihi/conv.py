import collections
import warnings

import numpy as np

import nengo
import nengo.utils.numpy as npext
from nengo.builder import Builder, Signal
from nengo.builder.operator import Copy, DotInc, Reset
from nengo.dists import Distribution, get_samples
from nengo.ensemble import Ensemble
from nengo.exceptions import BuildError, NengoWarning
from nengo.neurons import Direct
from nengo.utils.builder import default_n_eval_points
from nengo.builder.ensemble import BuiltEnsemble, gen_eval_points, get_activities, get_gain_bias

import nengo_dl

import nengo_loihi
from nengo_loihi.builder import INTER_TAU
from nengo_loihi.loihi_cx import CxGroup


class Conv2dEnsemble(nengo.Ensemble):

    conv_encoders = nengo.params.NdarrayParam(
        'conv_encoders', shape=('*', '*', '*', '*'))

    def __init__(self, input_shape, weights, strides=(1, 1), mode='valid', **ens_params):
        ni, nj, nk = input_shape
        nc, si, sj, nf = weights.shape
        sti, stj = strides
        nyi = 1 + (ni - si) // sti
        nyj = 1 + (nj - sj) // stj
        nyk = nf
        assert nk == nc

        self.conv2d_encoders = weights
        self.input_shape = input_shape
        self.output_shape = (nyi, nyj, nf)
        self.strides = strides
        self.mode = mode

        n_neurons = nyi * nyj * nyk
        dimensions = ni * nj * nk
        super(Conv2dEnsemble, self).__init__(
            n_neurons, dimensions, **ens_params)


class Conv2d(nengo.builder.Operator):
    def __init__(self, W, X, Y, x_shape, strides=(1, 1), mode='valid',
                 tag=None):
        super(Conv2d, self).__init__(tag=tag)

        self.W = W
        self.X = X
        self.Y = Y
        self.x_shape = x_shape
        self.strides = strides
        self.mode = mode

        self.sets = []
        self.incs = [Y]
        self.reads = [W, X]
        self.updates = []

    def _descstr(self):
        return 'conv2d(%s, %s) -> %s' % (self.W, self.X, self.Y)

    def make_step(self, signals, dt, rng):
        import scipy.signal
        mode = self.mode
        sti, stj = self.strides

        def conv2d(x, w, sti=sti, stj=stj, mode=mode):
            return scipy.signal.correlate2d(x, w, mode=mode)[::sti, ::stj]

        W = signals[self.W]
        X = signals[self.X]
        Y = signals[self.Y]
        x_shape = self.x_shape
        nc, _, _, nf = W.shape

        # find output shape
        X = X.reshape(x_shape)
        y0 = conv2d(X[:, :, 0], W[0, :, :, 0])
        Y = Y.reshape(y0.shape + (nf,))

        def step_conv2d():
            for f in range(nf):
                for c in range(nc):
                    Y[:, :, f] += conv2d(X[:, :, c], W[c, :, :, f])

        return step_conv2d


@nengo_dl.builder.Builder.register(Conv2d)
class Conv2dBuilder(nengo_dl.builder.OpBuilder):
    def __init__(self, ops, signals, config):
        super(Conv2dBuilder, self).__init__(ops, signals, config)

        assert len(ops) == 1
        self.W_data = signals.combine([op.W for op in ops])
        self.X_data = signals.combine([op.X for op in ops])
        self.Y_data = signals.combine([op.Y for op in ops])

        self.x_shape = ops[0].x_shape
        self.strides = ops[0].strides
        self.mode = ops[0].mode

    def build_step(self, signals):
        import tensorflow as tf
        assert self.mode == 'valid'

        strides = self.strides
        x_shape = self.x_shape

        W = signals.gather(self.W_data)
        X = signals.gather(self.X_data)

        assert W.shape[-1] == 1
        W = W[..., 0]

        X = tf.transpose(X, (1, 0))  # put batch size first

        W = tf.transpose(W, (1, 2, 0, 3))  # (si, sj, nc, nf)
        X = tf.reshape(X, (X.shape[0],) + x_shape)

        Y = tf.nn.convolution(
            input=X,
            filter=W,
            strides=strides,
            padding='VALID',
            data_format='NHWC')

        signals.scatter(self.Y_data, Y, mode='inc')


@nengo.builder.Builder.register(Conv2dEnsemble)
def build_conv_ensemble(model, ens):
    # Create random number generator
    rng = np.random.RandomState(model.seeds[ens])

    eval_points = gen_eval_points(ens, ens.eval_points, rng=rng)

    # Set up signal
    model.sig[ens]['in'] = Signal(np.zeros(ens.dimensions),
                                  name="%s.input" % ens)
    model.add_op(Reset(model.sig[ens]['in']))

    # Build the neurons
    gain, bias, max_rates, intercepts = get_gain_bias(ens, rng)

    if isinstance(ens.neuron_type, Direct):
        model.sig[ens.neurons]['in'] = Signal(
            np.zeros(ens.dimensions), name='%s.neuron_in' % ens)
        model.sig[ens.neurons]['out'] = model.sig[ens.neurons]['in']
        model.add_op(Reset(model.sig[ens.neurons]['in']))
    else:
        model.sig[ens.neurons]['in'] = Signal(
            np.zeros(ens.n_neurons), name="%s.neuron_in" % ens)
        model.sig[ens.neurons]['out'] = Signal(
            np.zeros(ens.n_neurons), name="%s.neuron_out" % ens)
        model.sig[ens.neurons]['bias'] = Signal(
            bias, name="%s.bias" % ens, readonly=True)
        model.add_op(Copy(model.sig[ens.neurons]['bias'],
                          model.sig[ens.neurons]['in']))
        # This adds the neuron's operator and sets other signals
        model.build(ens.neuron_type, ens.neurons)

    # # Set up encoders
    # if isinstance(ens.neuron_type, Direct):
    #     encoders = np.identity(ens.dimensions)
    # elif isinstance(ens.encoders, Distribution):
    #     encoders = get_samples(
    #         ens.encoders, ens.n_neurons, ens.dimensions, rng=rng)
    # else:
    #     encoders = npext.array(ens.encoders, min_dims=2, dtype=np.float64)
    # if ens.normalize_encoders:
    #     encoders /= npext.norm(encoders, axis=1, keepdims=True)
    # encoders = None
    encoders = np.zeros((0, 0))
    model.sig[ens]['encoders'] = Signal(
        encoders, name="%s.encoders" % ens, readonly=True)

    # # Scale the encoders
    # if isinstance(ens.neuron_type, Direct):
    #     scaled_encoders = encoders
    # else:
    #     scaled_encoders = encoders * (gain / ens.radius)[:, np.newaxis]
    scaled_encoders = encoders

    assert np.all(gain == gain[0]), "All gains must be the same"
    scaled_conv2d_encoders = ens.conv2d_encoders * gain[0]

    model.sig[ens]['conv2d_encoders'] = Signal(
        scaled_conv2d_encoders,
        name="%s.scaled_conv2d_encoders" % ens,
        readonly=True)

    # Inject noise if specified
    if ens.noise is not None:
        model.build(ens.noise, sig_out=model.sig[ens.neurons]['in'], inc=True)

    # Create output signal, using built Neurons
    model.add_op(Conv2d(
        model.sig[ens]['conv2d_encoders'],
        model.sig[ens]['in'],
        model.sig[ens.neurons]['in'],
        x_shape=ens.input_shape,
        strides=ens.strides,
        mode=ens.mode,
        tag="%s conv2d encoding" % ens))

    # Output is neural output
    model.sig[ens]['out'] = model.sig[ens.neurons]['out']

    model.params[ens] = BuiltEnsemble(eval_points=eval_points,
                                      encoders=encoders,
                                      intercepts=intercepts,
                                      max_rates=max_rates,
                                      scaled_encoders=scaled_encoders,
                                      gain=gain,
                                      bias=bias)


@nengo_loihi.builder.Builder.register(Conv2dEnsemble)
def build_conv_ensemble_nengo_loihi(model, ens):
    # Create random number generator
    rng = np.random.RandomState(model.seeds[ens])

    eval_points = gen_eval_points(ens, ens.eval_points, rng=rng)

    # Build the neurons
    gain, bias, max_rates, intercepts = get_gain_bias(ens, rng)

    if isinstance(ens.neuron_type, nengo.Direct):
        raise NotImplementedError()
    else:
        group = CxGroup(ens.n_neurons, label='%s' % ens)
        group.bias[:] = bias
        model.build(ens.neuron_type, ens.neurons, group)

    group.configure_filter(INTER_TAU, dt=model.dt, default=True)

    if ens.noise is not None:
        raise NotImplementedError("Ensemble noise not implemented")

    # Set up encoders
    assert np.all(gain == gain[0]), "All gains must be the same"
    encoders = ens.conv2d_encoders
    scaled_encoders = encoders * gain[0]

    model.add_group(group)

    model.objs[ens]['in'] = group
    model.objs[ens]['out'] = group
    model.objs[ens.neurons]['in'] = group
    model.objs[ens.neurons]['out'] = group
    model.params[ens] = BuiltEnsemble(eval_points=eval_points,
                                      encoders=encoders,
                                      intercepts=intercepts,
                                      max_rates=max_rates,
                                      scaled_encoders=scaled_encoders,
                                      gain=gain,
                                      bias=bias)
