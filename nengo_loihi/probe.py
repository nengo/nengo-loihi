import numpy as np
from nengo.utils.numpy import is_iterable

from nengo_loihi.block import LoihiBlock


class LoihiProbe:
    """Record data from one or more LoihiBlock target states.

    To get the final output of the probe:

    1. The LoihiBlock state given by ``key`` will be collected from each target.
    2. The slices will be applied to the outputs.
    3. The weights will be applied to the sliced outputs.
    4. If the weights change the output shape then weighted outputs are summed.
       If no weights or scalar weights, the weighted outputs are concatenated.
    5. Reindexing is applied to order the outputs correctly (in the case of
       multiple targets with no transformitive weights).
    6. Outputs are filtered using ``synapse``.

    The difference between summing and concatenating based on transformitive weights
    is to facilitate splitting blocks: Splitting a block with weights splits the
    weight matrix across the inputs but not across outputs, thus it is natural to
    sum outputs. Splitting a block with scalar weights results in new blocks with
    scalar weights, thus to get the same size of output as before the individual
    outputs are concatenated.

    Parameters
    ----------
    target : LoihiBlock or list of LoihiBlock
        The block to record values from. Use ``slice`` to record from a subset
        of compartments.
    key : string ('current', 'voltage', 'spiked')
        The compartment attribute to probe.
    slice : list of <slice or list>
        Select a subset of the compartments in each block to record from.
    synapse : nengo.synapses.Synapse
        A synapse to use for filtering the probe.
    weights : np.ndarray
        A linear transformation to apply to the outputs.
    reindexing : np.ndarray
        A list of indices used to reorder the outputs.
    """

    _slice = slice

    def __init__(
        self,
        target=None,
        key=None,
        slice=None,
        weights=None,
        synapse=None,
        reindexing=None,
    ):
        self.key = key
        self.synapse = synapse

        iterable_target = is_iterable(target)
        self.target = (
            [] if target is None else list(target) if iterable_target else [target]
        )
        # targets can be LoihiBlock or None. `Model.add_probe` checks Nones are filled.
        assert all(isinstance(t, (LoihiBlock, type(None))) for t in self.target)

        self.slice = (
            [self._slice(None) for _ in self.target]
            if slice is None
            else slice
            if iterable_target  # a single `slice` can be e.g. a list, so use target
            else [slice]
        )
        assert len(self.slice) == len(self.target)

        self.weights = (
            [None for _ in self.target]
            if weights is None
            else [np.asarray(w) if w is not None else None for w in weights]
            if iterable_target
            else [np.asarray(weights)]
        )
        assert len(self.weights) == len(self.target)

        self.reindexing = reindexing

    @property
    def is_transformed(self):
        # if the weights transform the output shapes, then we sum instead of stack later
        return any(w is not None and w.ndim > 1 for w in self.weights)

    @property
    def output_size(self):
        assert len(self.target) == len(self.slice) == len(self.weights)

        sizes = []
        for block, slice_, weights in zip(self.target, self.slice, self.weights):
            size = block.compartment.n_compartments
            if slice_ != slice(None):
                size = np.arange(size)[slice_].size
            if weights is not None and weights.ndim == 2:
                assert (
                    size == weights.shape[0]
                ), "Sliced compartment size (%d) must match weight input size (%d)" % (
                    size,
                    weights.shape[0],
                )
                size = weights.shape[1]
            sizes.append(size)

        if self.is_transformed:
            assert all(
                size == sizes[0] for size in sizes
            ), "All weights should map to the same shape"
            return sizes[0]
        else:
            return sum(sizes)

    def weight_outputs(self, outputs):
        """Apply weights and reindexing to the target outputs.

        We assume that probe slices have already been applied, since these are typically
        performed when collecting the target outputs

        Parameters
        ----------
        outputs : list of lists or arrays (n_blocks, n_timesteps (optional), n_outputs)
            Outputs of the target blocks. The ``timesteps`` dimension is optional,
            and defaults to 1 if not provided. ``n_outputs`` can be different
            across blocks.

        Returns
        -------
        result : (n_timesteps, n_outputs)
        """
        # `outputs` shape is (blocks in probe, timesteps, outputs in block)
        # probe slices have already been applied to outputs
        assert len(outputs) == len(self.target) == len(self.weights)

        weighted_outputs = []
        for k, output in enumerate(outputs):
            output = np.asarray(output)
            if self.weights[k] is not None:
                output = output.dot(self.weights[k])
            if output.ndim == 1:
                output = output.reshape((1, -1))
            weighted_outputs.append(output)

        n_timesteps = weighted_outputs[0].shape[0]
        assert all(
            out.ndim == 2 and out.shape[0] == n_timesteps for out in weighted_outputs
        ), "All outputs must have the same length of time"

        if self.is_transformed:
            # sum results together
            nc = weighted_outputs[0].shape[1]
            assert all(
                out.shape[1] == nc for out in weighted_outputs
            ), "All weights should map to the same shape"
            result = sum(weighted_outputs)
            assert self.reindexing is None
        else:
            # concatenate results together
            nc = sum(out.shape[1] for out in weighted_outputs)
            result = np.column_stack(weighted_outputs)
            if self.reindexing is not None:
                result = result[..., self.reindexing]

        assert result.shape == (n_timesteps, nc), "%s != (%s, %s)" % (
            result.shape,
            n_timesteps,
            nc,
        )
        return result
