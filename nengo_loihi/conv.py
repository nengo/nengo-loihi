import itertools

import numpy as np
from nengo.exceptions import ValidationError
from nengo.transforms import ChannelShape, Convolution


class ImageSlice:
    """Represents a slice of a larger image across rows/columns/channels.

    Parameters
    ----------
    full_shape : nengo.transforms.ChannelShape
        Full shape of the image to slice.
    row_slice : slice
        Slice across the image rows.
    col_slice : slice
        Slice across the image columns.
    channel_slice : slice
        Slice across the image channels.
    """

    def __init__(
        self,
        full_shape,
        row_slice=slice(None),
        col_slice=slice(None),
        channel_slice=slice(None),
    ):
        if not (isinstance(full_shape, ChannelShape) and full_shape.dimensions == 2):
            raise ValidationError(
                "must be 2-D ChannelShape (got %r)" % full_shape,
                attr="full_shape",
                obj=self,
            )
        self.full_shape = full_shape
        self.row_slice = row_slice
        self.col_slice = col_slice
        self.channel_slice = channel_slice

    def channel_slice_only(self):
        return self.row_slice == slice(None) and self.col_slice == slice(None)

    def row_idxs(self):
        return list(range(self.full_shape.spatial_shape[0]))[self.row_slice]

    def col_idxs(self):
        return list(range(self.full_shape.spatial_shape[1]))[self.col_slice]

    def channel_idxs(self):
        return list(range(self.full_shape.n_channels))[self.channel_slice]


def split_transform(transform, in_slice=None, out_slice=None):
    a_slice = slice(None)
    b_slice = slice(None)

    if isinstance(transform, Convolution):
        if in_slice is not None:
            assert in_slice.channel_slice_only()
            a_slice = in_slice.channel_slice
        if out_slice is not None:
            assert out_slice.channel_slice_only()
            b_slice = out_slice.channel_slice

        assert isinstance(transform.init, np.ndarray), "doesn't work with distributions"
        kernel = transform.init[:, :, a_slice, b_slice]
        rows, cols = transform.input_shape.spatial_shape
        nc = kernel.shape[2]
        input_shape = ChannelShape(
            (rows, cols, nc) if transform.channels_last else (nc, rows, cols),
            channels_last=transform.channels_last,
        )
        return Convolution(
            kernel.shape[3],
            input_shape,
            strides=transform.strides,
            channels_last=transform.channels_last,
            padding=transform.padding,
            kernel_size=transform.kernel_size,
            init=kernel,
        )
    else:
        if in_slice is not None:
            assert in_slice.channel_slice_only()
            a_slice = in_slice.channel_slice
        if out_slice is not None:
            assert out_slice.channel_slice_only()
            b_slice = out_slice.channel_slice

        return transform[b_slice, a_slice]


def split_channels(shape, max_size=None, max_channels=1024):
    if max_size is not None:
        max_size_channels = max_size // (np.prod(shape.spatial_shape))
        max_channels = min(max_channels, max_size_channels)
    assert max_channels >= 1
    n_split = -(-shape.n_channels // max_channels)  # ceiling division
    nc_per_split = -(-shape.n_channels // n_split)  # ceiling division
    return [
        ImageSlice(shape, channel_slice=slice(i, i + nc_per_split))
        for i in range(0, shape.n_channels, nc_per_split)
    ]


def channel_idxs(shape):
    """Return the channel indices (atoms) for this image shape.

    Parameters
    ----------
    shape : `nengo.transforms.ChannelShape` (default: True)
        Output shape of convolution
    """
    idxs = np.arange(shape.size, dtype=int)
    return (
        (idxs % shape.n_channels)
        if shape.channels_last
        else (idxs // np.prod(shape.spatial_shape))
    )


def pixel_idxs(shape):
    """Return the pixel indices for this image shape.

    Parameters
    ----------
    shape : `nengo.transforms.ChannelShape` (default: True)
        Output shape of convolution
    """
    idxs = np.arange(shape.size, dtype=int)
    return (
        (idxs // shape.n_channels)
        if shape.channels_last
        else (idxs % np.prod(shape.spatial_shape))
    )


def conv2d_loihi_weights(transform):
    assert transform.padding == "valid", "Only 'valid' padding currently implemented"
    assert (
        transform.channels_last == transform.input_shape.channels_last
    ), "Transforms that switch the channel position not yet implemented"

    input_rows, input_cols = transform.input_shape.spatial_shape
    n_channels = transform.input_shape.n_channels
    output_rows, output_cols = transform.output_shape.spatial_shape
    n_filters = transform.n_filters
    n_compartments = output_rows * output_cols * n_filters

    # compute number of used input pixels
    ri_max = (output_rows - 1) * transform.strides[0] + 1
    rj_max = (output_cols - 1) * transform.strides[1] + 1

    weights = []
    indices = []
    # compartment offset (aka. compartment base) for each axon
    offsets = np.zeros(input_rows * input_cols, dtype=int)
    axon_to_weight_map = np.zeros(input_rows * input_cols, dtype=int)
    weights_map = {}
    for i, j in itertools.product(range(input_rows), range(input_cols)):
        ij = i * input_cols + j

        # unstrided compartment indices that this input axon would map to
        # if strides == 1 and mode == 'full'
        ri0, ri1 = i + 1 - transform.kernel_size[0], i + 1
        rj0, rj1 = j + 1 - transform.kernel_size[1], j + 1
        ri = np.arange(ri0, ri1)
        rj = np.arange(rj0, rj1)
        # ^ TODO: padding

        wmask_i = (ri >= 0) & (ri < ri_max) & (ri % transform.strides[0] == 0)
        wmask_j = (rj >= 0) & (rj < rj_max) & (rj % transform.strides[1] == 0)

        if wmask_i.sum() == 0 or wmask_j.sum() == 0:
            # this axon is not needed, so indicate this in offsets and skip
            offsets[ij] = -1
            continue

        assert ri[wmask_i][0] % transform.strides[0] == 0, "true if mode == 'valid'"
        yi0 = ri[wmask_i][0] // transform.strides[0]
        yj0 = rj[wmask_j][0] // transform.strides[1]
        yij0 = yi0 * output_cols + yj0
        offset = yij0 * n_filters if transform.channels_last else yij0

        # There is currently an upper limit on the axon compartment offset of 256.
        # To work around this, we split the offset into two parts, and make extra sets
        # of redundant weights with part of the offset in the indices, as needed.
        axon_offset = offset % 256
        index_offset = offset - axon_offset
        offsets[ij] = axon_offset

        weight_key = (tuple(wmask_i), tuple(wmask_j), index_offset)
        if weight_key not in weights_map:
            # tranpose kernel to (in_channels, rows, cols, out_channels)
            kernel = np.transpose(transform.init, (2, 0, 1, 3))

            # flip weights to do correlation
            kernel = kernel[:, ::-1, ::-1, :]

            w = kernel[:, wmask_i[:, None] * wmask_j, :]
            assert w.shape == (n_channels, wmask_i.sum() * wmask_j.sum(), n_filters)

            # --- determine indices
            # channel inds are zero, since we use same indices for each channel
            channel_inds = np.zeros(n_channels, dtype=int)
            row_inds = np.arange(wmask_i.sum())
            col_inds = np.arange(wmask_j.sum())
            filter_inds = np.arange(n_filters)

            order = [channel_inds, row_inds, col_inds, filter_inds]
            shape = [n_channels, output_rows, output_cols, n_filters]
            if not transform.channels_last:
                # move filters (aka. output channels) before rows/cols
                w = np.transpose(w, (0, 2, 1))
                order = [order[i] for i in (0, 3, 1, 2)]
                shape = [shape[i] for i in (0, 3, 1, 2)]

            n = len(shape)
            strides = [np.prod(shape[i + 1 :]) for i in range(n)]

            # inds[i_0,...,i_{n-1}] = sum_{k=0}^{n-1} strides[k] * order[k][i_k]
            strided_inds = [
                stride * ind.reshape([-1] + [1] * (n - 1 - k))
                for k, (ind, stride) in enumerate(zip(order, strides))
            ]
            inds = sum([index_offset] + strided_inds)

            weights_map[weight_key] = len(weights)
            weights.append(w.reshape(n_channels, -1))
            indices.append(inds.reshape(n_channels, -1))

        axon_to_weight_map[ij] = weights_map[weight_key]

        # check that offset (compartment base) plus index points to a valid compartment
        inds = indices[axon_to_weight_map[ij]]
        assert (offsets[ij] + inds < n_compartments).all()

    return weights, indices, axon_to_weight_map, offsets
