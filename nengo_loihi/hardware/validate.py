import numpy as np

from nengo_loihi.builder.discretize import VTH_MAN_MAX
from nengo_loihi.builder.validate import validate_synapse_cfg
from nengo_loihi.nxsdk_obfuscation import d


def validate_board(board):
    for chip in board.chips:
        validate_chip(chip)


def validate_chip(chip):
    for core in chip.cores:
        validate_core(core)


def validate_core(core):
    # TODO: check these numbers are correct
    assert len(core.compartment_cfgs) <= d(b"MzI=", int)
    assert len(core.vth_cfgs) <= d(b"MTY=", int)
    assert len(core.synapse_cfgs) <= d(b"MTY=", int)
    assert len(core.stdp_pre_cfgs) <= d(b"Mw==", int)

    for cfg in core.compartment_cfgs:
        validate_compartment_cfg(cfg)
    for cfg in core.vth_cfgs:
        validate_vth_cfg(cfg, core=core)
    for cfg in core.synapse_cfgs:
        if cfg is not None:
            validate_synapse_cfg(cfg)
    for cfg in core.stdp_pre_cfgs:
        validate_trace_cfg(cfg)

    for synapse in core.synapse_axons:
        cfg = core.get_synapse_cfg(synapse)
        idxbits = cfg.real_idx_bits
        for i in synapse.indices:
            assert np.all(i >= 0)
            assert np.all(i < 2 ** idxbits)


def validate_compartment_cfg(cfg):
    assert cfg.decay_u >= 0
    assert cfg.decay_u <= cfg.DECAY_U_MAX
    assert cfg.decay_v >= 0
    assert cfg.decay_v <= cfg.DECAY_V_MAX
    assert cfg.refract_delay >= 1
    assert cfg.refract_delay <= cfg.REFRACT_DELAY_MAX
    assert cfg.enable_noise in (0, 1)


def validate_vth_cfg(cfg, core=None):
    assert cfg.vth > 0
    assert cfg.vth <= VTH_MAN_MAX


def validate_stdp_cfg(cfg):
    pass


def validate_stdp_pre_cfg(cfg):
    pass


def validate_trace_cfg(cfg):
    pass
