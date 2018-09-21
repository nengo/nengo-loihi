import numpy as np

import nengo
from nengo import Connection, Ensemble, Node, Direct
from nengo.builder import Signal
from nengo.builder.connection import BuiltConnection, slice_signal, get_samples
from nengo.builder.operator import (
    ElementwiseInc, Reset, DotInc, Copy, SimPyFunc)
from nengo.dists import Distribution
from nengo.ensemble import Neurons
from nengo.exceptions import BuildError
from nengo.utils.compat import is_iterable, itervalues

try:
    import nengo_dl
except ImportError:
    nengo_dl = None

import nengo_loihi
from nengo_loihi.loihi_cx import CxGroup, CxSpikeInput, CxSynapses, CxAxons
from nengo_loihi.splitter import ChipReceiveNeurons


# TODO: create a generic superclass for distributions/these (since it's a bit
# weird to call this a Distribution)
class Conv2D(Distribution):
    @classmethod
    def get_output_shape(cls, input_shape, weight_shape,
                         strides=(1, 1), mode='valid'):
        assert mode == "valid"

        ni, nj, nk = input_shape
        nc, si, sj, nf = weight_shape
        sti, stj = strides
        nyi = 1 + (ni - si) // sti
        nyj = 1 + (nj - sj) // stj
        assert nk == nc
        return nyi, nyj, nf

    def __init__(self, n_filters, input_shape, kernel_size=3,
                 strides=1, mode="valid", kernel=nengo_dl.dists.Glorot()):
        self.n_filters = n_filters
        self.input_shape = input_shape
        self.kernel_size = kernel_size if is_iterable(kernel_size) else (
            kernel_size, kernel_size)
        self.strides = strides if is_iterable(strides) else (strides, strides)
        self.mode = mode
        self.kernel = kernel

        if self.kernel is not None and not isinstance(self.kernel,
                                                      Distribution):
            assert self.kernel.shape == (
                input_shape[-1], self.kernel_size[0], self.kernel_size[1],
                self.n_filters)

    def sample(self, n, d=None, rng=np.random):
        # note: n/d ignored, redo as part of superclass refactoring

        # return kernel with shape
        # (in_channels, filter_height, filter_width, out_channels)

        # TODO: change the kernel shape to the more standard
        # (filter_height, filter_width, in_channels, out_channels)
        shape = (self.input_shape[-1],) + self.kernel_size + (self.n_filters,)
        return np.reshape(
            get_samples(self.kernel, shape[0], d=np.prod(shape[1:]),
                        rng=rng),
            shape)


class Conv2DInc(nengo.builder.Operator):
    def __init__(self, W, X, Y, x_shape, strides=(1, 1), mode='valid',
                 tag=None):
        super(Conv2DInc, self).__init__(tag=tag)

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
    @nengo_dl.builder.Builder.register(Conv2DInc)
    class Conv2DIncBuilder(nengo_dl.builder.OpBuilder):
        def __init__(self, ops, signals, config):
            super(Conv2DIncBuilder, self).__init__(ops, signals, config)

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


@nengo.builder.Builder.register(nengo.Connection)  # noqa: C901
def build_connection(model, conn):
    """Builds a `.Connection` object into a model.

    A brief summary of what happens in the connection build process,
    in order:

    1. Solve for decoders.
    2. Combine transform matrix with decoders to get weights.
    3. Add operators for computing the function
       or multiplying neural activity by weights.
    4. Call build function for the synapse.
    5. Call build function for the learning rule.
    6. Add operator for applying learning rule delta to weights.

    Some of these steps may be altered or omitted depending on the parameters
    of the connection, in particular the pre and post types.

    Parameters
    ----------
    model : Model
        The model to build into.
    conn : Connection
        The connection to build.

    Notes
    -----
    Sets ``model.params[conn]`` to a `.BuiltConnection` instance.
    """

    # Create random number generator
    rng = np.random.RandomState(model.seeds[conn])

    # Get input and output connections from pre and post
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

    weights = None
    eval_points = None
    solver_info = None
    signal_size = conn.size_out
    post_slice = conn.post_slice

    # Sample transform if given a distribution
    transform = get_samples(
        conn.transform, conn.size_out, d=conn.size_mid, rng=rng)

    # Figure out the signal going across this connection
    in_signal = model.sig[conn]['in']
    if (isinstance(conn.pre_obj, Node) or
            (isinstance(conn.pre_obj, Ensemble) and
             isinstance(conn.pre_obj.neuron_type, Direct))):
        # Node or Decoded connection in directmode
        weights = transform
        sliced_in = slice_signal(model, in_signal, conn.pre_slice)
        if conn.function is None:
            in_signal = sliced_in
        elif isinstance(conn.function, np.ndarray):
            raise BuildError("Cannot use function points in direct connection")
        else:
            in_signal = Signal(np.zeros(conn.size_mid), name='%s.func' % conn)
            model.add_op(SimPyFunc(in_signal, conn.function, None, sliced_in))
    elif isinstance(conn.pre_obj, Ensemble):  # Normal decoded connection
        eval_points, weights, solver_info = model.build(
            conn.solver, conn, rng, transform)
        if conn.solver.weights:
            model.sig[conn]['out'] = model.sig[conn.post_obj.neurons]['in']
            signal_size = conn.post_obj.neurons.size_in
            post_slice = None  # don't apply slice later
    else:
        weights = transform
        in_signal = slice_signal(model, in_signal, conn.pre_slice)

    # Add operator for applying weights
    model.sig[conn]['weights'] = Signal(
        weights, name="%s.weights" % conn, readonly=True)
    signal = Signal(np.zeros(signal_size), name="%s.weighted" % conn)
    model.add_op(Reset(signal))

    if isinstance(conn.transform, Conv2D):
        assert not isinstance(conn.pre_obj, Ensemble)
        model.add_op(Conv2DInc(
            model.sig[conn]["weights"], in_signal, signal,
            conn.transform.input_shape, strides=conn.transform.strides))
    else:
        op = ElementwiseInc if weights.ndim < 2 else DotInc
        model.add_op(op(model.sig[conn]['weights'],
                        in_signal,
                        signal,
                        tag="%s.weights_elementwiseinc" % conn))

    # Add operator for filtering
    if conn.synapse is not None:
        signal = model.build(conn.synapse, signal)

    # Store the weighted-filtered output in case we want to probe it
    model.sig[conn]['weighted'] = signal

    if isinstance(conn.post_obj, Neurons):
        # Apply neuron gains (we don't need to do this if we're connecting to
        # an Ensemble, because the gains are rolled into the encoders)
        gains = Signal(model.params[conn.post_obj.ensemble].gain[post_slice],
                       name="%s.gains" % conn)
        model.add_op(ElementwiseInc(
            gains, signal, model.sig[conn]['out'][post_slice],
            tag="%s.gains_elementwiseinc" % conn))
    else:
        # Copy to the proper slice
        model.add_op(Copy(
            signal, model.sig[conn]['out'], dst_slice=post_slice,
            inc=True, tag="%s" % conn))

    # Build learning rules
    if conn.learning_rule is not None:
        assert conn.transform is not Conv2D

        rule = conn.learning_rule
        rule = [rule] if not is_iterable(rule) else rule
        targets = []
        for r in itervalues(rule) if isinstance(rule, dict) else rule:
            model.build(r)
            targets.append(r.modifies)

        if 'encoders' in targets:
            encoder_sig = model.sig[conn.post_obj]['encoders']
            encoder_sig.readonly = False
        if 'decoders' in targets or 'weights' in targets:
            if weights.ndim < 2:
                raise BuildError(
                    "'transform' must be a 2-dimensional array for learning")
            model.sig[conn]['weights'].readonly = False

    model.params[conn] = BuiltConnection(eval_points=eval_points,
                                         solver_info=solver_info,
                                         transform=transform,
                                         weights=weights)


def build_conv2d_connection(model, conn):
    # Create random number generator
    rng = np.random.RandomState(model.seeds[conn])

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

    weights = get_samples(conn.transform, None, rng=rng)
    input_shape = conn.transform.input_shape

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
        weights, input_shape, strides=conn.transform.strides,
        mode=conn.transform.mode)
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
