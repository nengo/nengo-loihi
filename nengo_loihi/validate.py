from nengo.exceptions import BuildError
import numpy as np

from nengo_loihi.block import Synapse
from nengo_loihi.nxsdk_obfuscation import d


def validate_model(model):
    if len(model.blocks) == 0:
        raise BuildError(
            "No neurons marked for execution on-chip. "
            "Please mark some ensembles as on-chip."
        )

    for block in model.blocks:
        validate_block(block)


def validate_block(block):
    # -- Compartment
    validate_compartment(block.compartment)

    # -- Axons
    OUT_AXONS_MAX = d(b"NDA5Ng==", int)
    n_axons = sum(a.axon_slots() for a in block.axons)
    if n_axons > OUT_AXONS_MAX:
        raise BuildError(
            "Output axons (%d) exceeded max (%d)" % (n_axons, OUT_AXONS_MAX)
        )

    for axon in block.axons:
        validate_axon(axon)

    # -- Synapses
    IN_AXONS_MAX = d(b"NDA5Ng==", int)
    n_axons = sum(s.n_axons for s in block.synapses)
    if n_axons > IN_AXONS_MAX:
        raise BuildError("Input axons (%d) exceeded max (%d)" % (n_axons, IN_AXONS_MAX))

    MAX_SYNAPSE_BITS = d(b"MTA0ODU3Ng==", int)
    synapse_bits = sum(s.bits() for s in block.synapses)
    if synapse_bits > MAX_SYNAPSE_BITS:
        raise BuildError(
            "Total synapse bits (%d) exceeded max (%d)"
            % (synapse_bits, MAX_SYNAPSE_BITS)
        )

    for synapse in block.synapses:
        validate_synapse(synapse)

    # -- Probes
    for probe in block.probes:
        validate_probe(probe)


def validate_compartment(comp):
    N_MAX_COMPARTMENTS = d(b"MTAyNA==", int)
    if comp.n_compartments > N_MAX_COMPARTMENTS:
        raise BuildError(
            "Number of compartments (%d) exceeded max (%d)"
            % (comp.n_compartments, N_MAX_COMPARTMENTS)
        )


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
        min_base = d(b"LTE=", int)
        max_base = d(b"MjU2", int)
        assert all(min_base <= b < max_base for b in synapse.axon_compartment_bases), (
            "compartment base must be >= %d and < %d (-1 indicating unused)"
            % (min_base, max_base)
        )
    if synapse.pop_type == 16:
        if synapse.axon_compartment_bases is not None:
            assert all(
                b % d(b"NA==", int) == 0
                for b in synapse.axon_compartment_bases
                if b >= 0
            )


def validate_synapse_cfg(synapse_cfg):
    assert d(b"LTc=", int) <= synapse_cfg.weight_exp <= d(b"Nw==", int)
    assert d(b"MA==", int) <= synapse_cfg.tag_bits < d(b"NA==", int)
    assert d(b"MA==", int) <= synapse_cfg.delay_bits < d(b"OA==", int)
    assert d(b"MQ==", int) <= synapse_cfg.weight_bits < d(b"OA==", int)
    assert d(b"MA==", int) <= synapse_cfg.idx_offset < d(b"MTY=", int)
    assert d(b"MA==", int) <= synapse_cfg.idx_mult < d(b"MTY=", int)
    assert d(b"MA==", int) <= synapse_cfg.idx_bits < d(b"OA==", int)
    assert d(b"MQ==", int) <= synapse_cfg.fanout_type < d(b"NA==", int)


def validate_probe(probe):
    pass
