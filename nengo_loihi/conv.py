import copy
import itertools

import nengo
from nengo.builder.connection import BuiltConnection
from nengo.ensemble import Neurons
from nengo.exceptions import ValidationError
import numpy as np

from nengo_loihi.block import Axon, LoihiBlock, Synapse
from nengo_loihi.compat import nengo_transforms
from nengo_loihi.inputs import ChipReceiveNeurons, LoihiInput


class ImageSlice:
    def __init__(self, full_shape, row_slice=slice(None),
                 col_slice=slice(None), channel_slice=slice(None)):
        self.full_shape = full_shape
        self.row_slice = row_slice
        self.col_slice = col_slice
        self.channel_slice = channel_slice

    def channel_slice_only(self):
        return self.row_slice == slice(None) and self.col_slice == slice(None)

    def row_idxs(self):
        return list(range(self.full_shape.rows))[self.row_slice]

    def col_idxs(self):
        return list(range(self.full_shape.cols))[self.col_slice]

    def channel_idxs(self):
        return list(range(self.full_shape.channels))[self.channel_slice]


def split_transform(transform, in_slice=None, out_slice=None):
    a_slice = slice(None)
    b_slice = slice(None)

    if isinstance(transform, nengo_transforms.Convolution):
        if in_slice is not None:
            assert in_slice.channel_slice_only()
            a_slice = in_slice.channel_slice
        if out_slice is not None:
            assert out_slice.channel_slice_only()
            b_slice = out_slice.channel_slice

        assert isinstance(transform.init, np.ndarray), \
            "doesn't work with distributions"
        kernel = transform.init[:, :, a_slice, b_slice]
        rows, cols = transform.input_shape.spatial_shape
        nc = kernel.shape[2]
        input_shape = nengo_transforms.ChannelShape(
            (rows, cols, nc) if transform.channels_last else (nc, rows, cols),
            channels_last=transform.channels_last)
        return nengo_transforms.Convolution(
            kernel.shape[3], input_shape, strides=transform.strides,
            channels_last=transform.channels_last, padding=transform.padding,
            kernel_size=transform.kernel_size, init=kernel)
    else:
        if in_slice is not None:
            assert in_slice.channel_slice_only()
            a_slice = in_slice.flatten().channel_slice
        if out_slice is not None:
            assert out_slice.channel_slice_only()
            b_slice = out_slice.flatten().channel_slice

        return transform[b_slice, a_slice]


def split_channels(shape, max_size=None, max_channels=1024):
    if max_size is not None:
        max_size_channels = max_size // (np.prod(shape.spatial_shape))
        max_channels = min(max_channels, max_size_channels)
    assert max_channels >= 1
    n_split = -(-shape.n_channels // max_channels)  # ceiling division
    nc_per_split = -(-shape.n_channels // n_split)  # ceiling division
    return [ImageSlice(shape, channel_slice=slice(i, i+nc_per_split))
            for i in range(0, shape.n_channels, nc_per_split)]


def channel_idxs(shape):
    """Return the channel indices (atoms) for this image shape.

    Parameters
    ----------
    shape : `nengo.transforms.ChannelShape` (default: True)
        Output shape of convolution
    """
    idxs = np.arange(shape.size, dtype=int)
    return ((idxs % shape.n_channels) if shape.channels_last else
            (idxs // np.prod(shape.spatial_shape)))


def pixel_idxs(shape):
    """Return the pixel indices for this image shape.

    Parameters
    ----------
    shape : `nengo.transforms.ChannelShape` (default: True)
        Output shape of convolution
    """
    idxs = np.arange(shape.size, dtype=int)
    return ((idxs // shape.n_channels) if shape.channels_last else
            (idxs % np.prod(shape.spatial_shape)))


def build_conv2d_connection(model, conn):
    if nengo_transforms is None:
        # It should not be possible to reach this, because this function is
        # only called for a Convolution transform, which can exist only if
        # nengo_transforms exists.
        raise NotImplementedError("Convolution requires newer Nengo")

    if conn.transform.dimensions != 2:
        raise NotImplementedError("nengo-loihi only supports 2D convolution")
    if conn.transform.padding != "valid":
        raise NotImplementedError(
            "nengo-loihi only supports convolution with 'valid' padding")

    # Create random number generator
    rng = np.random.RandomState(model.seeds[conn])

    pre_cx = model.objs[conn.pre_obj]['out']
    post_cx = model.objs[conn.post_obj]['in']
    assert isinstance(pre_cx, (LoihiInput, LoihiBlock))
    assert isinstance(post_cx, LoihiBlock)

    tau_s = 0.0
    if isinstance(conn.synapse, nengo.synapses.Lowpass):
        tau_s = conn.synapse.tau
    elif conn.synapse is not None:
        raise NotImplementedError("Cannot handle non-Lowpass synapses")

    # --- pre
    assert isinstance(conn.pre_obj, (Neurons, ChipReceiveNeurons))
    assert conn.pre_slice == slice(None)

    assert isinstance(conn.transform, nengo_transforms.Convolution)

    weights = conn.transform.sample(rng=rng)
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
    if not np.all(gain == gain[0]):
        # TODO: support this?
        raise ValidationError(
            "All neurons targeted by a Convolution connection must "
            "have the same gain", "gain", obj=conn.post_obj.ensemble)
    weights = weights * gain[0]

    pop_type = 32  # TODO: pick this
    new_transform = copy.copy(conn.transform)
    type(new_transform).init.data[new_transform] = weights
    weights, indices, axon_to_weight_map, cx_bases = conv2d_loihi_weights(
        new_transform)

    synapse = Synapse(np.prod(input_shape.spatial_shape),
                      label="conv2d_weights")
    synapse.set_population_weights(
        weights, indices, axon_to_weight_map, cx_bases, pop_type=pop_type)
    post_cx.add_synapse(synapse)
    model.objs[conn]['weights'] = synapse

    ax = Axon(np.prod(input_shape.spatial_shape), label="conv2d_weights")
    ax.target = synapse
    ax.cx_to_axon_map = pixel_idxs(input_shape)
    ax.cx_atoms = channel_idxs(input_shape)
    pre_cx.add_axon(ax)

    post_cx.compartment.configure_filter(tau_s, dt=model.dt)

    model.params[conn] = BuiltConnection(
        eval_points=None,
        solver_info=None,
        transform=None,
        weights=weights)


def conv2d_loihi_weights(transform):
    # TODO: It appears from that there is an upper limit on
    # CxBase of 256 (bug), so I had to make extra sets of redundant weights
    # with indices to work around this. If using pop32 axons then I could
    # put the filters as the major index to avoid this that way.

    inp_shape = transform.input_shape
    input_rows, input_cols = inp_shape.spatial_shape
    output_rows, output_cols = transform.output_shape.spatial_shape

    # compute number of used input pixels
    ri_max = (output_rows - 1) * transform.strides[0] + 1
    rj_max = (output_cols - 1) * transform.strides[1] + 1

    weights = []
    indices = []
    cx_bases = np.zeros(input_rows * input_cols, dtype=int)
    axon_to_weight_map = np.zeros(input_rows * input_cols, dtype=int)
    weights_map = {}
    for i, j in itertools.product(range(input_rows), range(input_cols)):
        ij = i * input_cols + j

        # unstrided cx indices that this input axon would map to
        # if strides == 1 and mode == 'full'
        ri0, ri1 = i + 1 - transform.kernel_size[0], i + 1
        rj0, rj1 = j + 1 - transform.kernel_size[1], j + 1
        ri = np.arange(ri0, ri1)
        rj = np.arange(rj0, rj1)
        # ^ TODO: padding

        wmask_i = (ri >= 0) & (ri < ri_max) & (ri % transform.strides[0] == 0)
        wmask_j = (rj >= 0) & (rj < rj_max) & (rj % transform.strides[1] == 0)

        if wmask_i.sum() == 0 or wmask_j.sum() == 0:
            # this axon is not needed, so indicate this in cx_bases and skip
            cx_bases[ij] = -1
            continue

        weight_key = (tuple(wmask_i), tuple(wmask_j))
        if weight_key not in weights_map:
            # tranpose kernel to (in_channels, rows, cols, out_channels)
            kernel = np.transpose(transform.init, (2, 0, 1, 3))

            # flip weights to do correlation
            kernel = kernel[:, ::-1, ::-1, :]

            w = kernel[:, wmask_i[:, None] * wmask_j, :]
            assert w.size == (inp_shape.n_channels
                              * wmask_i.sum()
                              * wmask_j.sum()
                              * transform.n_filters)
            assert w.shape == (inp_shape.n_channels,
                               wmask_i.sum() * wmask_j.sum(),
                               transform.n_filters)

            if transform.channels_last:
                w = w.reshape(inp_shape.n_channels, -1)
                inds = (
                    np.zeros((inp_shape.n_channels, 1, 1, 1), dtype=int)
                    + (output_cols * transform.n_filters
                       * np.arange(wmask_i.sum())[:, None, None])
                    + transform.n_filters * np.arange(wmask_j.sum())[:, None]
                    + np.arange(transform.n_filters)
                ).reshape(inp_shape.n_channels, -1)
            else:
                w = np.transpose(w, (0, 2, 1)).reshape(
                    inp_shape.n_channels, -1)
                inds = (
                    np.zeros((inp_shape.n_channels, 1, 1, 1), dtype=int)
                    + (output_rows * output_cols
                       * np.arange(transform.n_filters)[:, None, None])
                    + output_cols * np.arange(wmask_i.sum())[:, None]
                    + np.arange(wmask_j.sum())
                ).reshape(inp_shape.n_channels, -1)

            weights_map[weight_key] = len(weights)
            weights.append(w)
            indices.append(inds)

        axon_to_weight_map[ij] = weights_map[weight_key]

        assert ri[wmask_i][0] % transform.strides[0] == 0, \
            "true if mode == 'valid'"
        yi0 = ri[wmask_i][0] // transform.strides[0]
        yj0 = rj[wmask_j][0] // transform.strides[1]
        if transform.channels_last:
            cx_bases[ij] = (yi0 * output_cols + yj0) * transform.n_filters
        else:
            cx_bases[ij] = yi0 * output_cols + yj0

        inds = indices[axon_to_weight_map[ij]]
        assert (cx_bases[ij] + inds
                < output_rows * output_cols * transform.n_filters).all()

    return weights, indices, axon_to_weight_map, cx_bases
