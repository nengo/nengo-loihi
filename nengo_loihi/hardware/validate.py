import numpy as np

from nengo_loihi.discretize import VTH_MAN_MAX
from nengo_loihi.validate import validate_synapse_fmt


def validate_board(board):
    for chip in board.chips:
        validate_chip(chip)


def validate_chip(chip):
    for core in chip.cores:
        validate_core(core)


def validate_core(core):
    assert len(core.cxProfiles) <= 32  # TODO: check this number
    assert len(core.vthProfiles) <= 16  # TODO: check this number
    assert len(core.synapseFmts) <= 16  # TODO: check this number
    assert len(core.stdpPreCfgs) <= 3

    for cxProfile in core.cxProfiles:
        validate_cx_profile(cxProfile)
    for vthProfile in core.vthProfiles:
        validate_vth_profile(vthProfile, core=core)
    for synapseFmt in core.synapseFmts:
        if synapseFmt is not None:
            validate_synapse_fmt(synapseFmt)
    for traceCfg in core.stdpPreCfgs:
        validate_trace_cfg(traceCfg)

    for synapse in core.synapse_axons:
        synapseFmt = core.get_synapse_fmt(synapse)
        idxbits = synapseFmt.realIdxBits
        for i in synapse.indices:
            assert np.all(i >= 0)
            assert np.all(i < 2**idxbits)


def validate_cx_profile(cx_profile):
    assert cx_profile.decayU >= 0
    assert cx_profile.decayU <= cx_profile.DECAY_U_MAX
    assert cx_profile.decayV >= 0
    assert cx_profile.decayV <= cx_profile.DECAY_V_MAX
    assert cx_profile.refractDelay >= 1
    assert cx_profile.refractDelay <= cx_profile.REFRACT_DELAY_MAX
    assert cx_profile.enableNoise in (0, 1)


def validate_vth_profile(vth_profile, core=None):
    assert vth_profile.vth > 0
    assert vth_profile.vth <= VTH_MAN_MAX
    # if core is not None:
    #     assert vth_profile.realVth < core.dendrite_shared_cfg.v_max


def validate_stdp_profile(stdp_profile):
    pass


def validate_stdp_pre_profile(stdp_pre_profile):
    pass


def validate_trace_cfg(trace_cfg):
    pass
