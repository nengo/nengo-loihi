from __future__ import division

import collections
import warnings

import numpy as np

CX_PROFILES_MAX = 32
VTH_PROFILES_MAX = 8
SYNAPSE_FMTS_MAX = 16

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


def shift(x, s, **kwargs):
    if s < 0:
        return np.right_shift(x, -s, **kwargs)
    else:
        return np.left_shift(x, s, **kwargs)


class CxSlice(object):
    def __init__(self, board_idx, chip_idx, core_idx, cx_i0, cx_i1):
        self.board_idx = board_idx
        self.chip_idx = chip_idx
        self.core_idx = core_idx
        self.cx_i0 = cx_i0
        self.cx_i1 = cx_i1


class Board(object):
    def __init__(self, board_id=1):
        self.board_id = board_id

        self.chips = []
        self.chip_idxs = {}

        self.synapses_index = {}

        self.probe_map = {}

    def validate(self):
        for chip in self.chips:
            chip.validate()

    def _add_chip(self, chip):
        assert chip not in self.chips
        self.chip_idxs[chip] = len(self.chips)
        self.chips.append(chip)

    def new_chip(self):
        chip = Chip(board=self)
        self._add_chip(chip)
        return chip

    def chip_index(self, chip):
        return self.chip_idxs[chip]

    def map_probe(self, cx_probe, n2probe):
        assert cx_probe not in self.probe_map
        self.probe_map[cx_probe] = n2probe

    def index_synapses(self, synapses, chip, core, a0, a1):
        chip_idx = self.chip_index(chip)
        core_idx = chip.core_index(core)
        self.synapses_index[synapses] = (chip_idx, core_idx, a0, a1)

    def find_synapses(self, synapses):
        group = synapses.group
        if group.location == 'core':
            return self.synapses_index[synapses]
        elif group.location == 'cpu':
            raise NotImplementedError("CPU neurons not implemented")
        else:
            raise ValueError("Unrecognized location %r" % group.location)

    def cores(self):
        return (core for chip in self.chips for core in chip.cores)

    def n_chips(self):
        return len(self.chips)

    def n_cores_per_chip(self):
        return [chip.n_cores() for chip in self.chips]

    def n_synapses_per_core(self):
        return [[core.n_synapses() for core in chip.cores]
                for chip in self.chips]


class Chip(object):
    def __init__(self, board):
        self.board = board

        self.cores = []
        self.core_idxs = {}

    def validate(self):
        for core in self.cores:
            core.validate()

    def _add_core(self, core):
        assert core not in self.cores
        self.core_idxs[core] = len(self.cores)
        self.cores.append(core)

    def new_core(self):
        core = Core(chip=self)
        self._add_core(core)
        return core

    def core_index(self, core):
        return self.core_idxs[core]

    def n_cores(self):
        return len(self.cores)


class Core(object):
    def __init__(self, chip):
        self.chip = chip
        self.groups = []

        self.cxProfiles = []
        self.vthProfiles = []
        self.synapseFmts = [None]  # keep index 0 unused

        self.synapse_fmt_idxs = {}  # one synfmt per CxSynapses, for now
        self.synapse_axons = collections.OrderedDict()
        self.synapse_entries = collections.OrderedDict()

    @property
    def board(self):
        return self.chip.board

    def validate(self):
        for cxProfile in self.cxProfiles:
            cxProfile.validate(core=self)
        for vthProfile in self.vthProfiles:
            vthProfile.validate(core=self)
        for synapseFmt in self.synapseFmts:
            if synapseFmt is not None:
                synapseFmt.validate(core=self)
        for synapses in self.synapse_axons:
            synapseFmt = self.get_synapse_fmt(synapses)
            idxbits = synapseFmt.realIdxBits
            for i in synapses.indices:
                assert np.all(i >= 0)
                assert np.all(i < 2**idxbits)

    def iterate_groups(self):
        i0 = 0
        a0 = 0
        for group in self.groups:
            i1 = i0 + group.n
            a1 = a0 + sum(ax.n_axons for ax in group.axons)
            cx_idxs = list(range(i0, i1))
            ax_range = (a0, a1)
            yield group, cx_idxs, ax_range
            i0 = i1
            a0 = a1

    def add_group(self, group):
        self.groups.append(group)

    def add_cx_profile(self, cx_profile):
        self.cxProfiles.append(cx_profile)
        return len(self.cxProfiles) - 1  # index

    def add_vth_profile(self, vth_profile):
        self.vthProfiles.append(vth_profile)
        return len(self.vthProfiles) - 1  # index

    def add_synapse_fmt(self, synapse_fmt):
        self.synapseFmts.append(synapse_fmt)
        return len(self.synapseFmts) - 1  # index

    def n_synapses(self):
        return sum(synapses.size() for group in self.groups
                   for synapses in group.synapses)

    def add_synapses(self, synapses):
        synapse_fmt = synapses.synapse_fmt
        synapse_fmt_idx = self.add_synapse_fmt(synapse_fmt)
        self.synapse_fmt_idxs[synapses] = synapse_fmt_idx

        a0 = 0
        if len(self.synapse_axons) > 0:
            last = next(reversed(self.synapse_axons))
            a0 = self.synapse_axons[last][1]
        a1 = a0 + synapses.n_axons
        self.synapse_axons[synapses] = (a0, a1)
        self.board.index_synapses(synapses, self.chip, self, a0, a1)

        s0 = 0
        if len(self.synapse_entries) > 0:
            last = next(reversed(self.synapse_entries))
            s0 = self.synapse_entries[last][1]
        s1 = s0 + synapses.size()
        self.synapse_entries[synapses] = (s0, s1)

    def add_axons(self, axons):
        pass

    def get_synapse_fmt(self, synapses):
        return self.synapseFmts[self.synapse_fmt_idxs[synapses]]


class CxProfile(object):
    DECAY_U_MAX = 2**12 - 1
    DECAY_V_MAX = 2**12 - 1
    REFRACT_DELAY_MAX = 2**6 - 1

    params = ('decayU', 'decayV', 'refractDelay', 'enableNoise')

    def __init__(self, decayV, decayU, refractDelay, enableNoise):
        self.decayV = decayV
        self.decayU = decayU
        self.refractDelay = refractDelay
        self.enableNoise = enableNoise

    def __eq__(self, obj):
        return isinstance(obj, type(self)) and all(
            self.__dict__[key] == obj.__dict__[key] for key in self.params)

    def __hash__(self):
        return hash(tuple(self.__dict__[key] for key in self.params))

    def validate(self, core=None):
        assert 0 <= self.decayU <= self.DECAY_U_MAX
        assert 0 <= self.decayV <= self.DECAY_V_MAX
        assert 1 <= self.refractDelay <= self.REFRACT_DELAY_MAX
        assert 0 <= self.enableNoise <= 1


class VthProfile(object):
    VTH_MAX = 2**17 - 1

    params = ('vth',)

    def __init__(self, vth):
        self.vth = vth

    def __eq__(self, obj):
        return isinstance(obj, type(self)) and all(
            self.__dict__[key] == obj.__dict__[key] for key in self.params)

    def __hash__(self):
        return hash(tuple(self.__dict__[key] for key in self.params))

    def validate(self, core=None):
        assert 0 < self.vth <= self.VTH_MAX
        # if core is not None:
        #     assert self.realVth < core.dendrite_shared_cfg.v_max


class SynapseFmt(object):
    INDEX_BITS_MAP = [0, 6, 7, 8, 9, 10, 11, 12]
    WEIGHT_BITS_MAP = [0, 1, 2, 3, 4, 5, 6, 8]

    def __init__(self, wgtLimitMant=0, wgtLimitExp=0, wgtExp=0, discMaxWgt=0,
                 learningCfg=0, tagBits=0, dlyBits=0, wgtBits=0,
                 reuseSynData=0, numSynapses=0, cIdxOffset=0, cIdxMult=0,
                 skipBits=0, idxBits=0, synType=0, fanoutType=0,
                 compression=0, stdpProfile=0, ignoreDly=0):
        self.wgtLimitMant = wgtLimitMant
        self.wgtLimitExp = wgtLimitExp
        self.wgtExp = wgtExp
        self.discMaxWgt = discMaxWgt
        self.learningCfg = learningCfg
        self.tagBits = tagBits
        self.dlyBits = dlyBits
        self.wgtBits = wgtBits
        self.reuseSynData = reuseSynData
        self.numSynapses = numSynapses
        self.cIdxOffset = cIdxOffset
        self.cIdxMult = cIdxMult
        self.skipBits = skipBits
        self.idxBits = idxBits
        self.synType = synType
        self.fanoutType = fanoutType
        self.compression = compression
        self.stdpProfile = stdpProfile
        self.ignoreDly = ignoreDly

    @classmethod
    def get_realWgtExp(cls, wgtExp):
        return 6 + wgtExp

    @classmethod
    def get_scale(cls, wgtExp):
        return 2**cls.get_realWgtExp(wgtExp)

    @property
    def realWgtExp(self):
        return self.get_realWgtExp(self.wgtExp)

    @property
    def scale(self):
        return self.get_scale(self.wgtExp)

    def set(self, **kwargs):
        for key, value in kwargs.items():
            assert hasattr(self, key)
            setattr(self, key, value)

    def validate(self, core=None):
        assert -7 <= self.wgtExp <= 7
        assert 0 <= self.tagBits < 4
        assert 0 <= self.dlyBits < 8
        assert 1 <= self.wgtBits < 8
        assert 0 <= self.cIdxOffset < 16
        assert 0 <= self.cIdxMult < 16
        assert 0 <= self.idxBits < 8
        assert 1 <= self.fanoutType < 4

    @property
    def width(self):
        return self.WEIGHT_BITS_MAP[self.wgtBits]

    @property
    def isMixed(self):
        return self.fanoutType == 1

    @property
    def realIdxBits(self):
        return self.INDEX_BITS_MAP[self.idxBits]

    def discretize_weights(self, w, dtype=np.int32):
        s = 8 - self.width + self.isMixed
        m = 2**(8 - s) - 1

        w = np.round(w / 2.**s).clip(-m, m).astype(dtype)
        s2 = s + self.wgtExp
        shift(w, s2, out=w)
        np.left_shift(w, 6, out=w)

        if s2 < 0:
            warnings.warn("Lost %d extra bits in weight rounding" % (-s2,))

        ws = w // self.scale
        assert np.all(ws <= 255) and np.all(ws >= -256)

        return w
