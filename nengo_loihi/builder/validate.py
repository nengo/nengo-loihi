import numpy as np
from nengo.exceptions import BuildError

from nengo_loihi.block import (
    MAX_COMPARTMENTS,
    MAX_IN_AXONS,
    MAX_OUT_AXONS,
    MAX_SYNAPSE_BITS,
    Synapse,
)
from nengo_loihi.nxsdk_obfuscation import d


def validate_model(model):
    if len(model.blocks) == 0:
        raise BuildError(
            "No neurons marked for execution on-chip. "
            "Please mark some ensembles as on-chip."
        )

    for block in model.blocks:
        validate_block(block)

    for probe in model.probes:
        validate_probe(probe)


def validate_block(block):
    # -- Compartment
    validate_compartment(block.compartment)

    # -- Axons
    n_axons = sum(a.axon_slots() for a in block.axons)
    if n_axons > MAX_OUT_AXONS:
        raise BuildError(
            f"{block}: Output axons ({n_axons}) exceeded max ({MAX_OUT_AXONS})"
        )

    for axon in block.axons:
        validate_axon(axon)

    # -- Synapses
    n_axons = sum(s.n_axons for s in block.synapses)
    if n_axons > MAX_IN_AXONS:
        raise BuildError(
            f"{block}: Input axons ({n_axons}) exceeded max ({MAX_IN_AXONS})"
        )

    synapse_bits = sum(s.bits() for s in block.synapses)
    if synapse_bits > MAX_SYNAPSE_BITS:
        raise BuildError(
            f"{block}: Total synapse bits ({synapse_bits}) exceeded max "
            f"({MAX_SYNAPSE_BITS})"
        )

    for synapse in block.synapses:
        validate_synapse(synapse)


def validate_compartment(comp):
    if comp.n_compartments > MAX_COMPARTMENTS:
        raise BuildError(
            f"{comp}: Number of compartments ({comp.n_compartments}) exceeded max "
            f"({MAX_COMPARTMENTS})"
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
            f"{synapse}: compartment base must be >= {min_base} and < {max_base} "
            "(-1 indicating unused)"
        )
    if synapse.pop_type == 16:
        if synapse.axon_compartment_bases is not None:
            assert all(b % 4 == 0 for b in synapse.axon_compartment_bases if b >= 0), (
                f"{synapse}: Pop16 axons must have all compartment bases modulo 4: "
                f"{synapse.axon_compartments_bases}"
            )


def validate_synapse_cfg(synapse_cfg):
    assert synapse_cfg.idx_bits >= 0, (
        "Synapse idx_bits is < 0. This likely indicates the target compartment is "
        "too large to fit on a core."
    )
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
