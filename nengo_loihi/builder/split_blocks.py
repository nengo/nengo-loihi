"""Break apart LoihiBlocks too large to fit on a single core. """

import itertools
import logging
from collections import OrderedDict
from collections.abc import Sequence

import numpy as np

from nengo_loihi.block import (
    MAX_COMPARTMENTS,
    MAX_IN_AXONS,
    MAX_OUT_AXONS,
    MAX_SYNAPSE_BITS,
    Axon,
    LoihiBlock,
    Synapse,
)
from nengo_loihi.config import BlockShape
from nengo_loihi.inputs import SpikeInput

logger = logging.getLogger(__name__)


class IndicesList(Sequence):
    """A list that uses an equivalent set for fast existence checking.

    This class is typically used for lists of integer indices. It makes some
    assumptions based on this use case, specifically that ``values`` is ordered
    and contains no duplicates.

    Parameters
    ----------
    values : list
        The values in this list.

    Attributes
    ----------
    list : list
        A Python list representation of the values.
    set : set
        A Python set representation of the values.
    """

    def __init__(self, values):
        self.list = list(values)
        self.set = set(self.list)

    def __contains__(self, val):
        return val in self.set

    def __getitem__(self, key):  # pragma: no cover
        return self.list[key]

    def __iter__(self):
        return iter(self.list)

    def __len__(self):
        return len(self.list)


def array_hash(a):
    """Hash full array (not good for large arrays)"""
    v = a.view()
    v.setflags(write=False)
    return hash(v.data.tobytes())


def ceil_div(a, b):
    return -((-a) // b)


def split_model(model):  # noqa: C901
    """Split blocks in the given model that exceed the hardware constraints.

    Will split any block that has more than the allowable number of compartments,
    input or output axons, or weight memory usage.

    The user can also specify blocks to be split manually via the config system.

    This function modifies the model in-place. References to existing objects will often
    be kept when possible (e.g. blocks that do not need to be split).

    Parameters
    ----------
    model : `nengo_loihi.builder.Model`
        The model whose blocks should be split.

    Returns
    -------
    block_map : {LoihiBlock: list of LoihiBlock}
        A map from blocks in the original model to a list of new blocks that that block
        has been split into.

    Notes
    -----
    - Named synapses (i.e. entries in `LoihiBlock.named_synapses`) will no longer have
      their names after splitting, since we cannot generate unique names when a named
      synapse is split.
    """

    # block map: old blocks -> new blocks -> old compartment inds
    block_map = OrderedDict()

    # synapse_map: old synapses -> new synapses -> old axons they require
    synapse_map = {}

    for old_block in model.blocks:
        new_blocks = split_block(old_block, model.block_shapes)
        block_map[old_block] = new_blocks

        if len(new_blocks) == 1:
            # block has not changed, so no need to change synapses
            assert next(iter(new_blocks)) is old_block
            for old_synapse in old_block.synapses:
                synapse_map[old_synapse] = None
        else:
            # break apart synapses
            for old_synapse in old_block.synapses:
                new_synapse_axons = split_synapse(old_block, old_synapse, new_blocks)
                synapse_map[old_synapse] = new_synapse_axons

    for old_block in model.blocks:
        split_block_axons(old_block, block_map, synapse_map)

    for input in model.inputs:
        split_input_axons(input, block_map, synapse_map)

    for probe in model.probes:
        split_probe(probe, block_map, synapse_map)

    new_blocks = [block for group in block_map.values() for block in group]

    # update model
    # - Inputs remain the same, only their axons have been changed to target new blocks
    model.blocks = OrderedDict(zip(new_blocks, range(len(new_blocks))))

    # update block references in model.objs
    for obj, subobjs in model.objs.items():
        for attr, val in subobjs.items():
            if val in block_map:
                # update blocks (call `list` to just get the keys (blocks) in the dict)
                subobjs[attr] = list(block_map[val])

    return block_map


def split_probe(probe, block_map, synapse_map):
    """Modify probe in place to target new blocks"""
    assert len(probe.target) == len(probe.slice) == len(probe.weights) == 1
    old_block = probe.target[0]
    old_weights = probe.weights[0]
    old_slice = probe.slice[0]
    is_transformed = probe.is_transformed
    is_sliced = old_slice != slice(None)

    new_blocks = block_map[old_block]

    if len(new_blocks) <= 1:
        return  # block was not split, so current probes are fine

    n_inputs = old_block.compartment.n_compartments
    old_comp_ids = np.arange(n_inputs)[old_slice]

    targets = []
    weights = []
    slices = []
    ids = []
    for block, block_comp_ids in new_blocks.items():
        # `comp_ids`: indices relative to old block
        # `comp_idxs` indices relative to new block
        if is_sliced:
            comp_ids = sorted(block_comp_ids.set.intersection(old_comp_ids))
            comp_idxs = np.searchsorted(block_comp_ids.list, comp_ids)
        else:
            comp_ids = block_comp_ids.list
            comp_idxs = np.arange(len(comp_ids))

        if len(comp_idxs) == 0:
            continue  # this new block is not part of this probe

        if not is_sliced:
            new_slice = slice(None)  # we want the whole block
        elif len(comp_idxs) == 1:
            new_slice = slice(comp_idxs[0], comp_idxs[0] + 1, None)
        else:
            diffs = np.unique(np.diff(comp_idxs))
            new_slice = (
                slice(comp_idxs[0], comp_idxs[-1] + 1, diffs[0])
                if len(diffs) == 1
                else comp_idxs
            )

        new_weights = old_weights
        if is_transformed:
            new_weight_inds = (
                np.searchsorted(old_comp_ids, comp_ids) if is_sliced else comp_ids
            )
            new_weights = old_weights[new_weight_inds]

        targets.append(block)
        weights.append(new_weights)
        slices.append(new_slice)
        ids.append(comp_ids)

    ids = np.array([i for ii in ids for i in ii])
    assert ids.shape == old_comp_ids.shape
    assert np.array_equal(np.unique(ids), old_comp_ids)

    if is_transformed or np.array_equal(ids, old_comp_ids):
        # weighted probes don't need reindexing because summed outputs are ordered
        probe.reindexing = None
    else:
        probe.reindexing = np.argsort(ids)

    probe.target = targets
    probe.slice = slices
    probe.weights = weights


def split_block_axons(old_block, block_map, synapse_map):
    new_blocks = block_map[old_block]

    for block, block_comp_ids in new_blocks.items():
        # It is possible that the block has not changed ([old_block] == new_blocks),
        # so we make the list of axons first and then replace `block.axons`.
        new_axons = []
        for old_axon in old_block.axons:
            old_synapse = old_axon.target
            assert isinstance(old_synapse, Synapse)
            new_synapses = synapse_map[old_synapse]
            block_comp_ids = list(block_comp_ids)
            old_axon_idxs = old_axon.map_axon(block_comp_ids)
            old_atoms = old_axon.map_atoms(block_comp_ids)

            axons = split_axon(old_axon, old_axon_idxs, old_atoms, new_synapses)
            new_axons.extend(axons)

        block.axons = new_axons


def split_input_axons(input, block_map, synapse_map):
    assert isinstance(input, SpikeInput), "Need SpikeInput to know its size"
    input_comp_inds = list(range(input.n_neurons))

    new_axons = []
    for old_axon in input.axons:
        old_synapse = old_axon.target
        new_synapses = synapse_map[old_synapse]
        old_axon_idxs = old_axon.map_axon(input_comp_inds)
        old_atoms = old_axon.map_atoms(input_comp_inds)

        new_axons.extend(split_axon(old_axon, old_axon_idxs, old_atoms, new_synapses))

    input.axons = new_axons


def split_axon(old_axon, old_axon_idxs, old_atoms, new_synapses):
    """Split one old axon into multiple axons, each going to one of the new synapses.

    Parameters
    ----------
    old_axon : Axon
        The old axon that we want to split.
    old_axon_idxs : list
        The indices of the target synapse for each axon. If the old block owning the
        old axon has been split, these indices should relate to the compartments in
        the new block that will own the new axon.
    old_atoms : list
        The atoms for each of the old axons.
    new_synapses : dict {Synapse: list of int}
        Map from new synapses to axon ids of the old synapse that the new synapse needs.

    Returns
    -------
    new_axons : list of Axon
        A list of the new axons.
    """
    if new_synapses is None:
        # This indicates that the target synapses have not changed. However, the block
        # that owned the old axon could have split, in which case we need to make sure
        # the new axon uses just the axon indices and atoms pertaining to the
        # compartments in the new block.
        new_axon_idxs = old_axon_idxs
        assert all(i is not None for i in new_axon_idxs)
        new_axon_idxs = np.asarray(new_axon_idxs)
        n_axons = np.unique(new_axon_idxs[new_axon_idxs >= 0]).size

        new_axon = Axon(n_axons)
        new_axon.label = old_axon.label
        new_axon.target = old_axon.target
        new_axon.compartment_map = new_axon_idxs
        new_axon.compartment_atoms = np.asarray(old_atoms)
        return [new_axon]

    # turn the one old axon going to one old synapse, into one new axon per new synapse
    new_axons = []
    for k, (new_synapse, old_synapse_axon_ids) in enumerate(new_synapses.items()):
        # `synapse_axon_map` maps old axons to new axons
        new_synapse_axon_idxs = range(len(old_synapse_axon_ids))
        synapse_axon_map = dict(zip(old_synapse_axon_ids, new_synapse_axon_idxs))

        # `new_axon_idxs` are the new synapse indices that the new axon connects to
        new_axon_idxs = [synapse_axon_map.get(i, -1) for i in old_axon_idxs]
        assert all(i is not None for i in new_axon_idxs)
        new_axon_idxs = np.asarray(new_axon_idxs)
        n_axons = np.unique(new_axon_idxs[new_axon_idxs >= 0]).size
        if n_axons == 0:
            continue

        new_axon = Axon(n_axons)
        if old_axon.label is not None:
            new_axon.label = "%s[%d]" % (old_axon.label, k)
        new_axon.target = new_synapse
        new_axon.compartment_map = new_axon_idxs
        new_axon.compartment_atoms = np.asarray(old_atoms)
        new_axons.append(new_axon)

    return new_axons


def split_block(old_block, block_shapes):
    """Break a block apart into smaller blocks, each able to fit on one core"""
    n_compartments = old_block.compartment.n_compartments
    n_in_axons = sum(synapse.n_axons for synapse in old_block.synapses)
    n_out_axons = sum(axon.axon_slots() for axon in old_block.axons)
    synapse_bits = sum(synapse.bits() for synapse in old_block.synapses)

    if block_shapes.get(old_block, None) is None:
        # break block sequentially
        # TODO: account for compartments having different numbers of synapses/axons/etc.
        # Splitting into blocks where each block has the same number of compartments
        # could leave blocks that have more synapses or axons than allowed. But this
        # is rare, and users can work around it by specifying the split shape manually
        n_split = max(
            (
                ceil_div(n_compartments, MAX_COMPARTMENTS),
                ceil_div(n_in_axons, MAX_IN_AXONS),
                ceil_div(n_out_axons, MAX_OUT_AXONS),
                ceil_div(synapse_bits, MAX_SYNAPSE_BITS),
            )
        )
        block_shapes[old_block] = BlockShape(
            (ceil_div(n_compartments, n_split),), (n_compartments,)
        )
    old_block_shape = block_shapes[old_block]
    assert old_block_shape.ensemble_size == old_block.n_neurons

    # find compartment indices for each new block
    new_block_inds = []
    ranges = [range(0, n, i) for n, i in old_block_shape.zip_dimensions()]
    full_inds = np.arange(old_block_shape.ensemble_size).reshape(
        old_block_shape.ensemble_shape
    )
    for inds0 in itertools.product(*ranges):
        inds1 = np.minimum(inds0 + old_block_shape._shape, old_block_shape._ens_shape)
        indslice = tuple(slice(i0, i1) for i0, i1 in zip(inds0, inds1))
        inds = full_inds[indslice]
        new_block_inds.append(IndicesList(inds.flat))

    assert len(new_block_inds) > 0
    if len(new_block_inds) == 1:
        # if block can fit on one core, just return the current block
        assert new_block_inds[0].set == set(range(n_compartments))
        new_blocks = [old_block]
        return OrderedDict(zip(new_blocks, new_block_inds))

    # break apart block
    new_blocks = []
    for k, inds in enumerate(new_block_inds):
        n_neurons = len(inds)
        new_block = LoihiBlock(n_neurons)
        if old_block.label is not None:
            ind_array = np.array(list(inds))
            d = np.diff(ind_array)
            indstr = (
                "%d:%d:%d" % (ind_array[0], ind_array[-1] + 1, d[0])
                if len(d) > 0 and np.all(d[0] == d)
                else "%d:%d" % (ind_array[0], ind_array[0] + 1)
                if len(ind_array) == 1
                else str(k)
            )
            new_block.label = "%s[%s]" % (old_block.label, indstr)

        for attr in (
            "decay_u",
            "decay_v",
            "refract_delay",
            "vth",
            "bias",
            "enable_noise",
        ):
            # copy whole array to ensure that we maintain dtype
            setattr(
                new_block.compartment,
                attr,
                getattr(old_block.compartment, attr)[list(inds)].copy(),
            )

        for attr in (
            "tau_s",
            "scale_u",
            "scale_v",
            "vmin",
            "vmax",
            "noise_offset",
            "noise_exp",
            "noise_at_membrane",
        ):
            setattr(new_block.compartment, attr, getattr(old_block.compartment, attr))

        new_blocks.append(new_block)

    logger.info(
        "Split block (%d) into (%s)",
        n_compartments,
        ", ".join(
            str(new_block.compartment.n_compartments) for new_block in new_blocks
        ),
    )
    return OrderedDict(zip(new_blocks, new_block_inds))


def split_synapse(old_block, old_synapse, new_blocks):
    """Break a synapse apart to work with new blocks

    Parameters
    ----------
    old_block : LoihiBlock
        The old block that the old synapse belonged to.
    old_synapse : Synapse
        The old synapse to be split.
    new_blocks : OrderedDict(Block: list of int)
        A map from new blocks that ``old_block`` has been split into, to old block
        compartment indices that the new block now represents.

    Returns
    -------
    new_synapses : OrderedDict(Synapse: list of int)
        A map from new synapses that ``old_synapse`` has been split into, to the axon
        indices in the old synapse that the new synapse requires.
    """
    # Either synapse has discrete weights, in which case weight sharing and compartment
    # base offsets are not allowed, or it has population weights and these are allowed.
    # This function may work outside this dichotomy, but has not been tested for that.
    assert (  # discrete weights
        old_synapse.axon_to_weight_map is None
        and old_synapse.axon_compartment_bases is None
        and old_synapse.pop_type == 0
    ) or (  # population weights
        old_synapse.axon_to_weight_map is not None
        and old_synapse.axon_compartment_bases is not None
        and old_synapse.pop_type != 0
    )

    assert all(
        isinstance(w, np.ndarray) for w in old_synapse.weights
    ), "Sparse weights not yet supported"

    # --- collect old input axon information
    old_input_axons = OrderedDict()
    for axon_idx in range(old_synapse.n_axons):
        weight_idx = old_synapse.axon_weight_idx(axon_idx)
        indices = old_synapse.indices[weight_idx]
        assert all(
            np.array_equal(i, indices[0]) for i in indices[1:]
        ), "All atoms must target same indices"
        indices = indices[0]

        base = old_synapse.axon_compartment_base(axon_idx)
        if base is None:
            continue  # this axon is not used

        axon_comp_ids = base + indices
        old_input_axons[axon_idx] = (
            weight_idx,
            base,
            axon_comp_ids,
            set(axon_comp_ids),
        )

    # --- create new synapses, one for each new block that `old_block` split into
    # `new_synapse_axons` maps each new synapse to the old axon ids that it requires
    new_synapse_axons = OrderedDict()
    for k, (block, block_comp_ids) in enumerate(new_blocks.items()):
        # find which compartments in this new block each old axon inputs to
        axon_overlaps = {
            axon_id: block_comp_ids.set.intersection(axon_comp_ids_set)
            for axon_id, (_, _, _, axon_comp_ids_set) in old_input_axons.items()
        }

        # select only the axons that input to at least one compartment in this block
        axon_ids = [
            axon_id for axon_id in old_input_axons if len(axon_overlaps[axon_id]) > 0
        ]
        if len(axon_ids) == 0:
            # Can only happen if this synapse inputted to part of the old block only,
            # and none of the compartments it connected to are in this new block.
            # We currently don't allow connections to subsets of neurons, so don't
            # cover this. But just skipping with `continue` _should_ work.
            continue  # pragma: no cover

        # --- make the new synapse
        new_synapse = Synapse(len(axon_ids))
        if old_synapse.label is not None:
            new_synapse.label = "%s[%d]" % (old_synapse.label, k)
        block.add_synapse(new_synapse)
        new_synapse_axons[new_synapse] = axon_ids

        set_new_synapse_weights(
            old_synapse,
            old_input_axons,
            new_synapse,
            block_comp_ids,
            axon_overlaps,
            axon_ids,
        )

    logger.info(
        "Split synapse (%d) into (%s)",
        old_synapse.n_axons,
        ", ".join(str(synapse.n_axons) for synapse in new_synapse_axons),
    )

    assert all(w.dtype == old_synapse.weights[0].dtype for w in old_synapse.weights)
    for new_synapse in new_synapse_axons:
        assert all(w.dtype == old_synapse.weights[0].dtype for w in new_synapse.weights)

    return new_synapse_axons


def set_new_synapse_weights(
    old_synapse, old_input_axons, new_synapse, block_comp_ids, axon_overlaps, axon_ids
):
    has_shared_weights = old_synapse.axon_to_weight_map is not None

    # --- make the weights for the new synapse
    weights = []
    indices = []
    # weight_idx_map: maps old weight key to new weight idx
    weight_idx_map = {}
    new_axon_weight_map = []
    new_axon_compartment_bases = []

    compartment_map = dict(zip(block_comp_ids, range(len(block_comp_ids))))
    new_block_comp_idxs = IndicesList(range(len(block_comp_ids)))

    # iterate over all old axon ids that will also input to this new synapse
    for old_axon_id in axon_ids:
        old_weight_idx, old_base, old_axon_comp_ids, _ = old_input_axons[old_axon_id]
        all_targets_in_block = len(axon_overlaps[old_axon_id]) == len(old_axon_comp_ids)
        old_weights = old_synapse.weights[old_weight_idx]
        old_indices = old_synapse.indices[old_weight_idx]

        if all_targets_in_block:
            ww, ii = np.array(old_weights), old_indices
            valid_comp_ids = old_axon_comp_ids
        else:
            i_valid = np.array(
                [i in block_comp_ids for i in old_axon_comp_ids], dtype=bool
            )
            ww = old_weights[:, i_valid]
            ii = old_indices[:, i_valid]
            valid_comp_ids = old_axon_comp_ids[i_valid]

        # Map old compartment inds to new compartment inds. Mapping is given by
        # `compartment_map`, preserves order of indices (but is otherwise arbitrary).
        # `new_ii`: the new indices for each weight
        # `new_base`: the new compartment base for the indices
        if not has_shared_weights:
            # assume that if no shared weights, also no compartment base (see above)
            assert old_base == 0
            new_base = 0

            # map old inds directly to new inds
            new_ii = np.array([compartment_map[i] for i in valid_comp_ids])
            new_ii = np.tile(new_ii, (ii.shape[0], 1))
            assert new_ii.shape == ii.shape
        else:
            # map old compartment indices to new. Then figure out new base.
            new_axon_comp_ids = np.array([compartment_map[i] for i in valid_comp_ids])

            # choose the new compartment base `new_base` as the lowest index
            min_i = np.argmin(valid_comp_ids)
            assert min_i == np.argmin(new_axon_comp_ids)
            new_base = new_axon_comp_ids[min_i]
            assert new_base >= 0

            # new_base needs to be < 256 due to chip constraints
            new_base = new_base % 256

            new_ii = new_axon_comp_ids - new_base
            new_ii = np.tile(new_ii, (ii.shape[0], 1))
            assert new_ii.shape == ii.shape

        key = (array_hash(ww), array_hash(new_ii))
        if key not in weight_idx_map:
            # we don't have this key, so add weights/indices to the weight memory
            weight_idx_map[key] = len(weights)
            weights.append(ww)
            indices.append(new_ii)
            assert all(new_base + i in new_block_comp_idxs for i in new_ii.flat)
        else:
            # we have these weights/indices in memory, double check they're the same
            weight_idx = weight_idx_map[key]
            assert np.array_equal(ww, weights[weight_idx])
            assert np.array_equal(new_ii, indices[weight_idx])

        # add the weight memory index and compartment base for this axon
        new_axon_weight_map.append(weight_idx_map[key])
        new_axon_compartment_bases.append(new_base)

    new_synapse._set_weights_indices(
        weights,
        indices,
        weight_dtype=weights[0].dtype,
        compression=old_synapse.synapse_cfg.compression,
    )
    new_synapse.axon_to_weight_map = np.asarray(new_axon_weight_map)
    new_synapse.axon_compartment_bases = np.asarray(new_axon_compartment_bases)
    new_synapse.pop_type = old_synapse.pop_type
