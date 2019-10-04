"""Break apart LoihiBlocks too large to fit on a single core. """

import collections
import itertools

import numpy as np

from nengo_loihi.block import (
    Axon,
    LoihiBlock,
    MAX_COMPARTMENTS,
    MAX_IN_AXONS,
    MAX_OUT_AXONS,
    MAX_SYNAPSE_BITS,
    Synapse,
)
from nengo_loihi.inputs import SpikeInput


class SetList:
    def __init__(self, values):
        self.list = list(values)
        self.set = set(self.list)

    def __contains__(self, val):
        return val in self.set

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


def dismantle_model(model):
    """
    TODO:
    - Respect named synapses (but how can we keep the name, if it splits into multiple
      synapses?)
    """

    inputs = model.inputs
    blocks = model.blocks

    # block map: old blocks -> new blocks -> old compartment inds
    block_map = collections.OrderedDict()

    # synapse_map: old synapses -> new synapses -> old axons they require
    synapse_map = {}

    for old_block in blocks:
        new_blocks = dismantle_block(old_block)
        block_map[old_block] = new_blocks

        if len(new_blocks) == 1:
            # block has not changed, so no need to change synapses
            assert next(iter(new_blocks)) is old_block
            for old_synapse in old_block.synapses:
                synapse_map[old_synapse] = None
        else:
            # break apart synapses
            for old_synapse in old_block.synapses:
                new_synapse_axons = dismantle_synapse(
                    old_block, old_synapse, new_blocks
                )
                synapse_map[old_synapse] = new_synapse_axons

    # TODO: somehow need to do blocks with no inputs from other blocks first,
    # since once we do a block, all axons targeting that block will need to be updated
    for block in blocks:
        dismantle_block_axons(block, block_map, synapse_map)

        # for axon in block.axons:
        #     dismantle_block_axon(block, axon, block_map, synapse_map)

    for input in inputs:
        # leave input the same, but point axons to new blocks
        dismantle_input_axons(input, block_map, synapse_map)

    for probe in model.probes:
        # loihi_probe = model.objs[nengo_probe]["out"]
        dismantle_probe(probe, block_map, synapse_map)

    new_blocks = [block for group in block_map.values() for block in group]

    # update model
    # model.inputs = collections.OrderedDict(zip(new_inputs, range(len(new_inputs))))
    model.blocks = collections.OrderedDict(zip(new_blocks, range(len(new_blocks))))

    # update references within model
    for obj in model.objs:
        # if isinstance(obj, ChipReceiveNode):
        #     obj.spike_input = input_map[obj].spike_input

        # TODO: update probes (need to collect data from multiple blocks)
        pass

    # return new_inputs, new_blocks


def dismantle_probe(probe, block_map, synapse_map):
    """Modify probe in place to target new blocks"""
    assert len(probe.targets) == len(probe.slices) == len(probe.weights) == 1
    old_block = probe.targets[0]
    old_weights = probe.weights[0]
    scalar_weights = not isinstance(old_weights, np.ndarray) or old_weights.size == 1

    new_blocks = block_map[old_block]
    if len(new_blocks) < 2:
        return  # block was not split, so current probes are fine

    # TODO: figure out interaction between probe.reindexing and probe.slice
    assert probe.slices[0] == slice(None)

    targets = []
    inds = []
    weights = []
    slices = []
    for block, block_comp_inds in new_blocks.items():
        block_comp_inds = list(block_comp_inds)
        targets.append(block)
        inds.append(block_comp_inds)
        weights.append(old_weights if scalar_weights else old_weights[block_comp_inds])
        slices.append(slice(None))

    all_inds = np.concatenate(inds)
    if np.array_equal(all_inds, np.arange(all_inds.size)):
        probe.reindexing = None
    # elif probe.weights is not None:
    #     # permute weights
    #     probe.weights = probe.weights[all_inds, :]
    else:
        probe.reindexing = np.argsort(all_inds)

    probe.targets = targets
    probe.slices = slices
    probe.weights = weights


def dismantle_block_axons(old_block, block_map, synapse_map):
    """
    It is possible that the block has not changed ([old_block] == new_blocks), so we
    make the list of axons first and then replace `block.axons`.
    """
    new_blocks = block_map[old_block]

    for block, block_comp_inds in new_blocks.items():
        new_axons = []
        for old_axon in old_block.axons:
            old_synapse = old_axon.target
            assert isinstance(old_synapse, Synapse)
            new_synapses = synapse_map[old_synapse]
            block_comp_inds = list(block_comp_inds)
            old_axon_idxs = old_axon.map_axon(block_comp_inds)
            old_atoms = old_axon.map_atoms(block_comp_inds)

            axons = dismantle_axon(old_axon, old_axon_idxs, old_atoms, new_synapses)
            new_axons.extend(axons)
            # for new_axon in new_axons:
            #     block.add_axon(new_axon)

        block.axons = new_axons


def dismantle_input_axons(input, block_map, synapse_map):
    assert isinstance(input, SpikeInput), "Need SpikeInput to know its size"
    input_comp_inds = list(range(input.n_neurons))

    new_axons = []
    for old_axon in input.axons:
        old_synapse = old_axon.target
        new_synapses = synapse_map[old_synapse]
        old_axon_idxs = old_axon.map_axon(input_comp_inds)
        old_atoms = old_axon.map_atoms(input_comp_inds)

        new_axons.extend(
            dismantle_axon(old_axon, old_axon_idxs, old_atoms, new_synapses)
        )

    input.axons = new_axons


def dismantle_axon(old_axon, old_axon_idxs, old_atoms, new_synapses):
    """
    old_axon_idxs : list
        The indices of the target synapse for each axon. If the old block owning the
        old axon has been split, these indices should relate to the compartments in
        the new block that will own the new axon.
    """
    if new_synapses is None:
        # This indicates that the target synapses have not changed. However, the block
        # that owned the old axon could have split, in which case we need to make sure
        # the new axon uses just the axon indices and atoms pertaining to the
        # compartments in the new block.
        new_axon_idxs = old_axon_idxs
        assert all(i is not None for i in new_axon_idxs)
        n_axons = sum(i >= 0 for i in new_axon_idxs)
        new_axon = Axon(n_axons)
        new_axon.label = old_axon.label
        new_axon.target = old_axon.target
        new_axon.compartment_map = np.asarray(new_axon_idxs)
        new_axon.compartment_atoms = np.asarray(old_atoms)
        return [new_axon]

    new_axons = []
    for k, (synapse, synapse_axon_inds) in enumerate(new_synapses.items()):
        # synapse_axon_inds maps old axons to new axons
        synapse_axon_map = dict(zip(synapse_axon_inds, range(len(synapse_axon_inds))))
        new_axon_idxs = [synapse_axon_map.get(i, -1) for i in old_axon_idxs]
        assert all(i is not None for i in new_axon_idxs)
        n_axons = sum(i >= 0 for i in new_axon_idxs)
        if n_axons == 0:
            continue

        new_axon = Axon(n_axons)
        if old_axon.label is not None:
            new_axon.label = "%s[%d]" % (old_axon.label, k)
        new_axon.target = synapse
        new_axon.compartment_map = np.asarray(new_axon_idxs)
        new_axon.compartment_atoms = np.asarray(old_atoms)
        new_axons.append(new_axon)

    return new_axons


def dismantle_block(old_block):
    """Break a block apart into smaller blocks, each able to fit on one core"""
    n_compartments = old_block.compartment.n_compartments
    n_in_axons = sum(synapse.n_axons for synapse in old_block.synapses)
    n_out_axons = sum(axon.axon_slots() for axon in old_block.axons)
    synapse_bits = sum(synapse.bits() for synapse in old_block.synapses)

    new_block_inds = []
    if old_block.split_shape is not None:
        print("Splitting using shape")
        full_shape = np.asarray(old_block.split_full_shape)
        split_shape = np.asarray(old_block.split_shape)
        assert full_shape is not None and full_shape.ndim == 1
        assert full_shape.shape == split_shape.shape
        ranges = [range(0, n, i) for n, i in zip(full_shape, split_shape)]

        full_inds = np.arange(np.prod(full_shape)).reshape(full_shape)
        for inds0 in itertools.product(*ranges):
            inds1 = np.minimum(inds0 + split_shape, full_shape)
            indslice = tuple(slice(i0, i1) for i0, i1 in zip(inds0, inds1))
            inds = full_inds[indslice]
            new_block_inds.append(SetList(inds.flat))
    else:
        # break block sequentially
        # TODO: account for compartments having different numbers of synapses/axons/etc
        n_split = max(
            (
                ceil_div(n_compartments, MAX_COMPARTMENTS),
                ceil_div(n_in_axons, MAX_IN_AXONS),
                ceil_div(n_out_axons, MAX_OUT_AXONS),
                ceil_div(synapse_bits, MAX_SYNAPSE_BITS),
            )
        )

        compartments_per_block = ceil_div(n_compartments, n_split)
        for i0 in range(0, n_compartments, compartments_per_block):
            i1 = min(i0 + compartments_per_block, n_compartments)
            new_block_inds.append(SetList(range(i0, i1)))

    assert len(new_block_inds) > 0
    if len(new_block_inds) == 1:
        # if block can fit on one core, just return the current block
        # TODO: should we copy it, just so all new blocks are consistently copies?
        assert new_block_inds[0].set == set(range(n_compartments))
        new_blocks = [old_block]
        return collections.OrderedDict(zip(new_blocks, new_block_inds))

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

    print(
        "Split block (%d) into (%s)"
        % (
            n_compartments,
            ", ".join(
                str(new_block.compartment.n_compartments) for new_block in new_blocks
            ),
        )
    )
    new_blocks = collections.OrderedDict(zip(new_blocks, new_block_inds))
    return new_blocks


def dismantle_synapse(old_block, old_synapse, new_blocks):  # noqa: C901
    """Break a synapse apart to work with new blocks

    old_block : LoihiBlock
    old_synapse : Synapse
    new_blocks : OrderedDict(new block -> old block compartment indices)
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
    # has_compartment_bases = old_synapse.axon_compartment_bases is not None
    has_shared_weights = old_synapse.axon_to_weight_map is not None

    assert all(
        isinstance(w, np.ndarray) for w in old_synapse.weights
    ), "Sparse weights not yet supported"

    old_compression = old_synapse.synapse_cfg.compression
    old_input_axons = collections.OrderedDict()
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

        axon_comp_inds = base + indices
        old_input_axons[axon_idx] = (
            weight_idx,
            base,
            axon_comp_inds,
            set(axon_comp_inds),
        )

    new_synapse_axons = collections.OrderedDict()
    for k, (block, block_comp_inds) in enumerate(new_blocks.items()):
        axon_overlaps = {
            axon_idx: block_comp_inds.set.intersection(axon_comp_ind_set)
            for axon_idx, (_, _, _, axon_comp_ind_set) in old_input_axons.items()
        }
        axon_idxs = [
            axon_idx for axon_idx in old_input_axons if len(axon_overlaps[axon_idx]) > 0
        ]
        if len(axon_idxs) == 0:
            continue

        new_synapse = Synapse(len(axon_idxs))
        if old_synapse.label is not None:
            new_synapse.label = "%s[%d]" % (old_synapse.label, k)
        block.add_synapse(new_synapse)
        new_synapse_axons[new_synapse] = axon_idxs

        weights = []
        indices = []
        # weight_idx_map: maps old weight idx or key to new weight idx
        weight_idx_map = {}
        new_axon_weight_map = []
        new_axon_compartment_bases = []

        compartment_map = dict(zip(block_comp_inds, range(len(block_comp_inds))))
        new_block_comp_inds = SetList(range(len(block_comp_inds)))

        for new_axon_idx, old_axon_idx in enumerate(axon_idxs):
            (
                old_weight_idx,
                old_base,
                old_axon_comp_inds,
                old_axon_comp_ind_set,
            ) = old_input_axons[old_axon_idx]
            all_targets_in_block = len(axon_overlaps[old_axon_idx]) == len(
                old_axon_comp_ind_set
            )
            old_weights = old_synapse.weights[old_weight_idx]
            old_indices = old_synapse.indices[old_weight_idx]

            if all_targets_in_block:
                ww, ii = old_weights, old_indices
                valid_comp_inds = old_axon_comp_inds
            else:
                i_valid = np.array(
                    [i in block_comp_inds for i in old_axon_comp_inds], dtype=bool
                )
                ww = old_weights[:, i_valid]
                ii = old_indices[:, i_valid]
                valid_comp_inds = old_axon_comp_inds[i_valid]

            # if not has_shared_weights:
            #     key = old_weight_idx
            # elif all_targets_in_block:
            #     key = old_weight_idx
            # else:
            #     key = hash((array_hash(ww), array_hash(ii)))

            # Map old compartment inds to new compartment inds.
            # Mapping could be arbitrary (given by block_comp_inds).
            if not has_shared_weights:
                # assume that if no shared weights, also no compartment base (see above)
                assert old_base == 0
                new_base = 0

                # map old inds directly to new inds
                new_ii = np.array([compartment_map[i] for i in valid_comp_inds])
                new_ii = np.tile(new_ii, (ii.shape[0], 1))
                assert new_ii.shape == ii.shape
            else:
                # Map old compartment indices to new. Then figure out new base.
                new_axon_comp_inds = np.array(
                    [compartment_map[i] for i in valid_comp_inds]
                )

                # new_base is lowest index
                min_i = np.argmin(valid_comp_inds)
                assert min_i == np.argmin(new_axon_comp_inds)
                new_base = new_axon_comp_inds[min_i]
                assert new_base >= 0

                # new_base needs to be < 256 due to chip constraints
                new_base = new_base % 256

                new_ii = new_axon_comp_inds - new_base
                new_ii = np.tile(new_ii, (ii.shape[0], 1))
                assert new_ii.shape == ii.shape

            key = hash((array_hash(ww), array_hash(new_ii)))
            if key not in weight_idx_map:
                weight_idx_map[key] = len(weights)
                weights.append(ww)
                indices.append(new_ii)
                assert all(new_base + i in new_block_comp_inds for i in new_ii.flat)
            else:
                weight_idx = weight_idx_map[key]
                assert np.array_equal(ww, weights[weight_idx])
                assert np.array_equal(new_ii, indices[weight_idx])

            new_axon_weight_map.append(weight_idx_map[key])
            new_axon_compartment_bases.append(new_base)

        new_synapse._set_weights_indices(
            weights,
            indices,
            weight_dtype=weights[0].dtype,
            compression=old_compression,
        )
        new_synapse.axon_to_weight_map = np.asarray(new_axon_weight_map)
        new_synapse.axon_compartment_bases = np.asarray(new_axon_compartment_bases)
        # new_synapse.axon_compartment_bases = (
        #     new_axon_compartment_bases
        #     if new_axon_compartment_bases[0] is not None
        #     else None
        # )
        new_synapse.pop_type = old_synapse.pop_type

    print(
        "Split synapse (%d) into (%s)"
        % (
            old_synapse.n_axons,
            ", ".join(str(synapse.n_axons) for synapse in new_synapse_axons),
        )
    )

    assert all(w.dtype == old_synapse.weights[0].dtype for w in old_synapse.weights)
    for new_synapse in new_synapse_axons:
        assert all(w.dtype == old_synapse.weights[0].dtype for w in new_synapse.weights)

    # return OrderedDict of new synapses, and which old input axons each of them need
    return new_synapse_axons
