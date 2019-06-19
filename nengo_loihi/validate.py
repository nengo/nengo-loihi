from nengo.exceptions import BuildError
import numpy as np

from nengo_loihi.block import Synapse


def validate_model(model):
    if len(model.blocks) == 0:
        raise BuildError("No neurons marked for execution on-chip. "
                         "Please mark some ensembles as on-chip.")

    for block in model.blocks:
        validate_block(block)


def validate_block(block):
    # -- Compartment
    validate_compartment(block.compartment)

    # -- Axons
    OUT_AXONS_MAX = 4096
    n_axons = sum(a.axon_slots() for a in block.axons)
    if n_axons > OUT_AXONS_MAX:
        raise BuildError("Output axons (%d) exceeded max (%d)" % (
            n_axons, OUT_AXONS_MAX))

    for axon in block.axons:
        validate_axon(axon)

    # -- Synapses
    IN_AXONS_MAX = 4096
    n_axons = sum(s.n_axons for s in block.synapses)
    if n_axons > IN_AXONS_MAX:
        raise BuildError("Input axons (%d) exceeded max (%d)" % (
            n_axons, IN_AXONS_MAX))

    MAX_SYNAPSE_BITS = 16384*64
    synapse_bits = sum(s.bits() for s in block.synapses)
    if synapse_bits > MAX_SYNAPSE_BITS:
        raise BuildError("Total synapse bits (%d) exceeded max (%d)" % (
            synapse_bits, MAX_SYNAPSE_BITS))

    for synapse in block.synapses:
        validate_synapse(synapse)

    # -- Probes
    for probe in block.probes:
        validate_probe(probe)


def validate_compartment(comp):
    N_MAX_COMPARTMENTS = 1024
    if comp.n_compartments > N_MAX_COMPARTMENTS:
        raise BuildError("Number of compartments (%d) exceeded max (%d)" %
                         (comp.n_compartments, N_MAX_COMPARTMENTS))


def validate_axon(axon):
    if isinstance(axon.target, Synapse):
        if axon.compartment_atoms is not None:
            idxs = np.arange(len(axon.compartment_atoms))
            axon_ids = axon.map_axon(idxs)
            for atom, axon_id in zip(axon.compartment_atoms, axon_ids):
                n_populations = axon.target.axon_populations(axon_id)
                assert 0 <= atom < n_populations


def validate_synapse(synapse):
    validate_synapse_cfg(synapse.synapse_cfg)
    if synapse.axon_compartment_bases is not None:
        assert all(-1 <= b < 256 for b in synapse.axon_compartment_bases), (
            "compartment base must be >= 0 and < 256 or -1, indicating unused")
    if synapse.pop_type == 16:
        if synapse.axon_compartment_bases is not None:
            assert all(b % 4 == 0 for b in synapse.axon_compartment_bases)


def validate_synapse_cfg(synapse_cfg):
    assert -7 <= synapse_cfg.weight_exp <= 7
    assert 0 <= synapse_cfg.tag_bits < 4
    assert 0 <= synapse_cfg.delay_bits < 8
    assert 1 <= synapse_cfg.weight_bits < 8
    assert 0 <= synapse_cfg.idx_offset < 16
    assert 0 <= synapse_cfg.idx_mult < 16
    assert 0 <= synapse_cfg.idx_bits < 8
    assert 1 <= synapse_cfg.fanout_type < 4


def validate_probe(probe):
    pass
