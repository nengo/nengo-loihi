import numpy as np

VTH_MAN_MAX = 2**17 - 1
VTH_EXP = 6
VTH_MAX = VTH_MAN_MAX * 2**VTH_EXP

BIAS_MAN_MAX = 2**12 - 1
BIAS_EXP_MAX = 2**3 - 1
BIAS_MAX = BIAS_MAN_MAX * 2**BIAS_EXP_MAX


def vth_to_manexp(vth):
    exp = VTH_EXP * np.ones(vth.shape, dtype=np.int32)
    man = np.round(vth / 2**exp).astype(np.int32)
    assert ((man >= 0) & (man <= VTH_MAN_MAX)).all()
    return man, exp


def bias_to_manexp(bias):
    r = np.maximum(np.abs(bias) / BIAS_MAN_MAX, 1)
    exp = np.ceil(np.log2(r)).astype(np.int32)
    man = np.round(bias / 2**exp).astype(np.int32)
    assert ((exp >= 0) & (exp <= BIAS_EXP_MAX)).all()
    assert (np.abs(man) <= BIAS_MAN_MAX).all()
    return man, exp


def discretize(target, value):
    assert target.dtype == np.float32
    new = np.round(value).astype(np.int32)
    target.dtype = np.int32
    target[:] = new
