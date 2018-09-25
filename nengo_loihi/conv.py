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


class ImageShape(object):
    def __init__(self, rows, cols, channels, channels_last=True):
        self.rows = rows
        self.cols = cols
        self.channels = channels
        self.channels_last = channels_last

    def _channels_last(self, channels_last=None):
        return self.channels_last if channels_last is None else channels_last

    @property
    def n_pixels(self):
        return self.rows * self.cols

    @property
    def size(self):
        return self.rows * self.cols * self.channels

    @classmethod
    def from_shape(cls, shape, channels_last=True):
        if channels_last:
            ni, nj, nc = shape
        else:
            nc, ni, nj = shape
        return cls(ni, nj, nc, channels_last=channels_last)

    def shape(self, channels_last=None):
        if self._channels_last(channels_last):
            return (self.rows, self.cols, self.channels)
        else:
            return (self.channels, self.rows, self.cols)

    def channel_idxs(self, channels_last=None):
        """Return the channel indices (atoms) for this image shape.

        Parameters
        ----------
        channels_last : bool (default: True)
            Whether the output indices should assume the channels are
            first (False) or last (True).
        """
        ni, nj, nc = self.shape(channels_last=True)
        idxs = np.arange(ni * nj * nc, dtype=int)
        return ((idxs % nc) if self._channels_last(channels_last) else
                (idxs // (ni * nj)))

    def pixel_idxs(self, channels_last=None):
        """Return the pixel indices for this image shape.

        Parameters
        ----------
        channels_last : bool (default: True)
            Whether the output indices should assume the channels are
            first (False) or last (True).
        """
        ni, nj, nc = self.shape(channels_last=True)
        idxs = np.arange(ni * nj * nc, dtype=int)
        return ((idxs // nc) if self._channels_last(channels_last) else
                (idxs % (ni * nj)))


# TODO: create a generic superclass for distributions/these (since it's a bit
# weird to call this a Distribution)
class Conv2D(Distribution):
    def __init__(self, n_filters, input_shape, kernel_size=3, strides=1,
                 mode="valid", correlate=True, output_channels_last=None,
                 kernel=nengo_dl.dists.Glorot()):
        self.n_filters = n_filters
        self.input_shape = (
            input_shape if isinstance(input_shape, ImageShape)
            else ImageShape.from_shape(
                input_shape,
                channels_last=output_channels_last if output_channels_last is not None else True))
        self.kernel_size = kernel_size if is_iterable(kernel_size) else (
            kernel_size, kernel_size)
        self.strides = strides if is_iterable(strides) else (strides, strides)
        self.mode = mode
        self.correlate = correlate
        self.kernel = kernel

        assert self.correlate, "correlate==False not implemented"
        if self.kernel is not None and not isinstance(self.kernel,
                                                      Distribution):
            assert self.kernel.shape == self.kernel_shape, "%s %s" % (
                self.kernel.shape, self.kernel_shape)

        # --- compute output shape
        ni = self.input_shape.rows
        nj = self.input_shape.cols
        nc, si, sj, nf = self.kernel_shape
        assert nc == self.input_shape.channels
        assert ni >= si and nj >= sj
        sti, stj = self.strides

        if self.mode == 'valid':
            assert ni >= si and nj >= sj
            nyi = 1 + (ni - si) // sti
            nyj = 1 + (nj - sj) // stj
        else:
            raise NotImplementedError(self.mode)

        if output_channels_last is None:
            output_channels_last = self.input_shape.channels_last
        self.output_shape = ImageShape(
            nyi, nyj, nf, channels_last=output_channels_last)

    @classmethod
    def from_kernel(cls, kernel, input_shape, **kwargs):
        nc, si, sj, nf = kernel.shape
        return cls(nf, input_shape, kernel_size=(si, sj), kernel=kernel,
                   **kwargs)

    def copy(self, kernel=None):
        """Make a copy, with an (optional) new kernel."""
        if kernel is None:
            kernel = self.kernel.copy()

        cls = type(self)
        new = cls.from_kernel(
            kernel, self.input_shape, strides=self.strides, mode=self.mode,
            correlate=self.correlate,
            output_channels_last=self.output_shape.channels_last)
        return new

    @property
    def kernel_shape(self):
        # TODO: change the kernel shape to the more standard
        # (filter_height, filter_width, in_channels, out_channels)

        return (self.input_shape.channels,) + self.kernel_size + (
            self.n_filters,)

    def sample(self, n, d=None, rng=np.random):
        shape = self.kernel_shape
        x = get_samples(self.kernel, shape[0], d=np.prod(shape[1:]), rng=rng)
        return np.reshape(x, shape)


class Conv2DInc(nengo.builder.Operator):
    def __init__(self, W, X, Y, conv2d_transform, tag=None):
        super(Conv2DInc, self).__init__(tag=tag)

        self.W = W
        self.X = X
        self.Y = Y
        self.conv2d_transform = conv2d_transform

        self.sets = []
        self.incs = [Y]
        self.reads = [W, X]
        self.updates = []

    def _descstr(self):
        return 'conv2d(%s, %s) -> %s' % (self.W, self.X, self.Y)

    def make_step(self, signals, dt, rng):
        import scipy.signal
        assert self.conv2d_transform.correlate
        mode = self.conv2d_transform.mode
        sti, stj = self.conv2d_transform.strides
        x_shape = self.conv2d_transform.input_shape
        y_shape = self.conv2d_transform.output_shape

        def conv2d(x, w, sti=sti, stj=stj, mode=mode):
            return scipy.signal.correlate2d(x, w, mode=mode)[::sti, ::stj]

        W = signals[self.W]
        X = signals[self.X]
        Y = signals[self.Y]
        nc, _, _, nf = W.shape

        # find output shape
        X = X.reshape(x_shape.shape())
        if not x_shape.channels_last:
            X = np.transpose(X, (1, 2, 0))

        # y0 = conv2d(X[:, :, 0], W[0, :, :, 0])
        Y = Y.reshape(y_shape.shape())
        if not y_shape.channels_last:
            Y = np.transpose(Y, (1, 2, 0))

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

            self.conv = ops[0].conv2d_transform

            self.W_data = signals.combine([op.W for op in ops])
            self.W_data = self.W_data.reshape(
                (self.conv.input_shape.channels,) + self.conv.kernel_size +
                (self.conv.n_filters,))
            self.X_data = signals.combine([op.X for op in ops])
            self.X_data = self.X_data.reshape(
                self.conv.input_shape.shape())
            self.Y_data = signals.combine([op.Y for op in ops])

        def build_step(self, signals):
            import tensorflow as tf

            W = signals.gather(self.W_data)
            X = signals.gather(self.X_data)

            W = tf.transpose(W, (1, 2, 0, 3))  # (si, sj, nc, nf)
            X = tf.transpose(X, (3, 0, 1, 2))  # put batch size first

            Y = tf.nn.convolution(
                input=X,
                filter=W,
                strides=self.conv.strides,
                padding=self.conv.mode.upper(),
                data_format=("NHWC" if self.conv.input_shape.channels_last else
                             "NCHW"))

            if (self.conv.input_shape.channels_last !=
                    self.conv.output_shape.channels_last):
                raise NotImplementedError()

            Y = tf.transpose(Y, (1, 2, 3, 0))  # move batch back to end

            signals.scatter(self.Y_data, Y, mode="inc")


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
            model.sig[conn]["weights"], in_signal, signal, conn.transform))
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

    assert isinstance(conn.transform, Conv2D)

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

    pop_type = 32  # TODO: pick this
    weights, indices, axon_to_weight_map, cx_bases = conv2d_loihi_weights(
        conn.transform.copy(weights))

    synapses = CxSynapses(input_shape.n_pixels, label="conv2d_weights")
    synapses.set_population_weights(
        weights, indices, axon_to_weight_map, cx_bases, pop_type=pop_type)
    post_cx.add_synapses(synapses)
    model.objs[conn]['weights'] = synapses

    ax = CxAxons(input_shape.n_pixels, label="conv2d_weights")
    ax.target = synapses
    ax.cx_to_axon_map = input_shape.pixel_idxs()
    ax.cx_atoms = input_shape.channel_idxs()
    pre_cx.add_axons(ax)

    post_cx.configure_filter(tau_s, dt=model.dt)

    model.params[conn] = BuiltConnection(
        eval_points=None,
        solver_info=None,
        transform=None,
        weights=weights)


def conv2d_loihi_weights(conv2d_transform):
    # TODO: It appears from my old code that there is an upper limit on
    # CxBase of 256 (bug), so I had to make extra sets of reduntant weights
    # with indices to work around this. If using pop32 axons then I could
    # put the filters as the major index to avoid this that way.
    import itertools

    kernel = conv2d_transform.kernel
    output_channels_last = conv2d_transform.output_shape.channels_last

    ni, nj, nk = conv2d_transform.input_shape.shape(channels_last=True)
    nc, si, sj, nf = conv2d_transform.kernel_shape
    sti, stj = conv2d_transform.strides
    assert nk == nc, "Input channels must equal kernel channels"

    if conv2d_transform.correlate:
        # flip weights to do correlation
        kernel = kernel[:, ::-1, ::-1, :]

    nyi, nyj, _ = conv2d_transform.output_shape.shape(channels_last=True)

    # compute number of used input pixels
    nxi = (nyi - 1)*sti + 1
    nxj = (nyj - 1)*stj + 1

    weights = []
    indices = []
    cx_bases = np.zeros(ni*nj, dtype=int)
    axon_to_weight_map = np.zeros(ni*nj, dtype=int)
    weights_map = {}
    for i, j in itertools.product(range(ni), range(nj)):
        ij = i*nj + j

        # unstrided cx indices that this input axon would map to
        # if strides == 1 and mode == 'full'
        ri0, ri1 = i+1-si, i+1
        rj0, rj1 = j+1-sj, j+1
        ri = np.arange(ri0, ri1)
        rj = np.arange(rj0, rj1)
        # ^ TODO: padding

        wmask_i = (ri >= 0) & (ri < nxi) & (ri % sti == 0)
        wmask_j = (rj >= 0) & (rj < nxj) & (rj % stj == 0)

        if wmask_i.sum() == 0 or wmask_j.sum() == 0:
            # this axon is not needed, so indicate this in cx_bases and skip
            cx_bases[ij] = -2048
            continue

        weight_key = (tuple(wmask_i), tuple(wmask_j))
        if weight_key not in weights_map:
            w = kernel[:, wmask_i[:, None]*wmask_j, :]
            assert w.size == nc * wmask_i.sum() * wmask_j.sum() * nf
            assert w.shape == (nc, wmask_i.sum() * wmask_j.sum(), nf)

            if output_channels_last:
                w = w.reshape(nc, -1)
                inds = (np.zeros((nc, 1, 1, 1), dtype=int) +
                        nyj*nf*np.arange(wmask_i.sum())[:, None, None] +
                        nf*np.arange(wmask_j.sum())[:, None] +
                        np.arange(nf)).reshape(nc, -1)
            else:
                w = np.transpose(w, (0, 2, 1)).reshape(nc, -1)
                inds = (np.zeros((nc, 1, 1, 1), dtype=int) +
                        nyi*nyj*np.arange(nf)[:, None, None] +
                        nyj*np.arange(wmask_i.sum())[:, None] +
                        np.arange(wmask_j.sum())).reshape(nc, -1)

            weights_map[weight_key] = len(weights)
            weights.append(w)
            indices.append(inds)

        axon_to_weight_map[ij] = weights_map[weight_key]

        assert ri[wmask_i][0] % sti == 0, "true if mode == 'valid'"
        yi0 = ri[wmask_i][0] // sti
        yj0 = rj[wmask_j][0] // stj
        if output_channels_last:
            cx_bases[ij] = (yi0*nyj + yj0) * nf
        else:
            cx_bases[ij] = yi0*nyj + yj0

        inds = indices[axon_to_weight_map[ij]]
        assert (cx_bases[ij] + inds < nyi*nyj*nf).all()

    return weights, indices, axon_to_weight_map, cx_bases
