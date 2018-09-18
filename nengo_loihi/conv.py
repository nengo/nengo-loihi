import numpy as np

import nengo
from nengo.base import NengoObject
from nengo.builder import Signal
from nengo.builder.connection import BuiltConnection, slice_signal
from nengo.builder.operator import ElementwiseInc, Reset
from nengo.connection import PrePostParam
from nengo.ensemble import Neurons
from nengo.exceptions import BuildError
from nengo.params import Default
from nengo.synapses import Lowpass, SynapseParam

try:
    import nengo_dl
except ImportError:
    nengo_dl = None

import nengo_loihi
from nengo_loihi.loihi_cx import CxGroup, CxSpikeInput, CxSynapses, CxAxons
from nengo_loihi.splitter import ChipReceiveNeurons


class Conv2dConnection(nengo.Connection):

    probeable = ('output', 'input', 'weights')

    pre = PrePostParam('pre', nonzero_size_out=True)
    post = PrePostParam('post', nonzero_size_in=True)
    synapse = SynapseParam('synapse', default=Lowpass(tau=0.005))
    weights = nengo.params.NdarrayParam(
        'weights', shape=('*', '*', '*', '*'))

    @classmethod
    def get_output_shape(cls, input_shape, weight_shape,
                         strides=(1, 1), mode='valid'):
        ni, nj, nk = input_shape
        nc, si, sj, nf = weight_shape
        sti, stj = strides
        nyi = 1 + (ni - si) // sti
        nyj = 1 + (nj - sj) // stj
        assert nk == nc
        return (nyi, nyj, nf)

    def __init__(self, pre, post, input_shape=None, weights=None,
                 strides=(1, 1), mode='valid', synapse=Default,
                 seed=None, label=None):
        NengoObject.__init__(self, label, seed)

        if input_shape is None:
            n = pre.size_out
            input_shape = (n, 1, 1)
        if weights is None:
            _, _, nc = input_shape
            weights = np.ones((nc, 1, 1, 1))

        self.pre = pre
        self.post = post
        self.synapse = synapse
        self.weights = weights

        self.input_shape = input_shape
        self.output_shape = self.get_output_shape(
            input_shape, weights.shape, strides=strides, mode=mode)
        self.strides = strides
        self.mode = mode

        assert post.size_in == np.prod(self.output_shape)

        self.synapse = synapse


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


if nengo_dl is not None:
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

            W = tf.transpose(W, (1, 2, 0, 3))  # (si, sj, nc, nf)
            X = tf.transpose(X, (1, 0))  # put batch size first
            X = tf.reshape(X, (X.shape[0],) + x_shape)

            Y = tf.nn.convolution(
                input=X,
                filter=W,
                strides=strides,
                padding='VALID',
                data_format='NHWC')

            signals.scatter(self.Y_data, Y, mode='inc')


@nengo.builder.Builder.register(Conv2dConnection)
def build_conv2d_connection(model, conn):
    def get_prepost_signal(is_pre):
        target = conn.pre_obj if is_pre else conn.post_obj
        key = 'out' if is_pre else 'in'

        if target not in model.sig:
            raise BuildError("Building %s: the %r object %s is not in the "
                             "model, or has a size of zero."
                             % (conn, 'pre' if is_pre else 'post', target))
        if key not in model.sig[target]:
            raise BuildError(
                "Building %s: the %r object %s has a %r size of zero."
                % (conn, 'pre' if is_pre else 'post', target, key))

        return model.sig[target][key]

    model.sig[conn]['in'] = get_prepost_signal(is_pre=True)
    model.sig[conn]['out'] = get_prepost_signal(is_pre=False)
    assert isinstance(conn.pre_obj, Neurons)
    assert isinstance(conn.post_obj, Neurons)

    signal_size = conn.size_out
    post_slice = conn.post_slice

    weights = conn.weights
    in_signal = model.sig[conn]['in']
    in_signal = slice_signal(model, in_signal, conn.pre_slice)

    # Add operator for applying weights
    model.sig[conn]['weights'] = Signal(
        weights, name="%s.weights" % conn, readonly=True)
    signal = Signal(np.zeros(signal_size), name="%s.weighted" % conn)
    model.add_op(Reset(signal))
    model.add_op(Conv2d(
        model.sig[conn]['weights'],
        in_signal,
        signal,
        x_shape=conn.input_shape,
        strides=conn.strides,
        mode=conn.mode,
        tag="%s.conv2d_weights" % conn))

    # Add operator for filtering
    if conn.synapse is not None:
        signal = model.build(conn.synapse, signal)

    # Store the weighted-filtered output in case we want to probe it
    model.sig[conn]['weighted'] = signal

    # Apply neuron gains
    gains = Signal(model.params[conn.post_obj.ensemble].gain[post_slice],
                   name="%s.gains" % conn)
    model.add_op(ElementwiseInc(
        gains, signal, model.sig[conn]['out'][post_slice],
        tag="%s.gains_elementwiseinc" % conn))

    model.params[conn] = BuiltConnection(eval_points=None,
                                         solver_info=None,
                                         transform=None,
                                         weights=weights)


@nengo_loihi.builder.Builder.register(Conv2dConnection)
def build_conv2d_connection_nengo_loihi(model, conn):
    pre_cx = model.objs[conn.pre_obj]['out']
    post_cx = model.objs[conn.post_obj]['in']
    assert isinstance(pre_cx, (CxGroup, CxSpikeInput))
    assert isinstance(post_cx, CxGroup)

    tau_s = 0.0
    if isinstance(conn.synapse, nengo.synapses.Lowpass):
        tau_s = conn.synapse.tau
    elif conn.synapse is not None:
        raise NotImplementedError("Cannot handle non-Lowpass synapses")

    # --- pre
    assert isinstance(conn.pre_obj, (Neurons, ChipReceiveNeurons))
    assert conn.pre_slice == slice(None)

    weights = conn.weights
    input_shape = conn.input_shape

    # Account for nengo spike height of 1/dt
    weights = weights / model.dt

    if isinstance(conn.pre_obj, ChipReceiveNeurons):
        neuron_type = conn.pre_obj.neuron_type
    elif isinstance(conn.pre_obj, Neurons):
        neuron_type = conn.pre_obj.ensemble.neuron_type

    if neuron_type is not None and hasattr(neuron_type, 'amplitude'):
        weights = weights * neuron_type.amplitude

    # --- post
    assert isinstance(conn.post_obj, Neurons)
    assert conn.post_slice == slice(None)

    gain = model.params[conn.post_obj.ensemble].gain
    assert np.all(gain == gain[0]), "All gains must be the same"
    weights = weights * gain[0]

    ni, nj, nk = input_shape
    synapses = CxSynapses(ni * nj, label="conv2d_weights")
    synapses.set_conv2d_weights(
        weights, input_shape, strides=conn.strides, mode=conn.mode)
    post_cx.add_synapses(synapses)
    model.objs[conn]['weights'] = synapses

    ax = CxAxons(ni * nj, label="conv2d_weights")
    ax.target = synapses
    ax.cx_to_axon_map = np.arange(ni * nj * nk) // nk
    ax.cx_atoms = np.arange(ni * nj * nk) % nk
    pre_cx.add_axons(ax)

    post_cx.configure_filter(tau_s, dt=model.dt)

    model.params[conn] = BuiltConnection(
        eval_points=None,
        solver_info=None,
        transform=None,
        weights=weights)
