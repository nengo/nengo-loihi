from __future__ import division

import collections
import warnings

import numpy as np

from nengo.exceptions import BuildError

CX_PROFILES_MAX = 32
VTH_PROFILES_MAX = 8
SYNAPSE_FMTS_MAX = 16

VTH_MAN_MAX = 2**17 - 1
VTH_EXP = 6
VTH_MAX = VTH_MAN_MAX * 2**VTH_EXP

BIAS_MAN_MAX = 2**12 - 1
BIAS_EXP_MAX = 2**3 - 1
BIAS_MAX = BIAS_MAN_MAX * 2**BIAS_EXP_MAX

Q_BITS = 21  # number of bits for synapse accumulator
U_BITS = 23  # number of bits for cx input (u)


def overflow_signed(x, bits=7, out=None):
    """Compute overflow on an array of signed integers.

    For example, the Loihi chip uses 23 bits plus sign to represent U.
    We can store them as 32-bit integers, and use this function to compute
    how they would overflow if we only had 23 bits plus sign.

    Parameters
    ----------
    x : array
        Integer values for which to compute values after overflow.
    bits : int
        Number of bits, not including sign, to compute overflow for.
    out : array, optional (Default: None)
        Output array to put computed overflow values in.

    Returns
    -------
    y : array
        Values of x overflowed as would happen with limited bit representation.
    overflowed : array
        Boolean array indicating which values of ``x`` actually overflowed.
    """
    if out is None:
        out = np.array(x)
    else:
        assert isinstance(out, np.ndarray)
        out[:] = x

    assert np.issubdtype(out.dtype, np.integer)

    x1 = np.array(1, dtype=out.dtype)
    smask = np.left_shift(x1, bits)  # mask for the sign bit (2**bits)
    xmask = smask - 1  # mask for all bits <= `bits`

    # find whether we've overflowed
    overflowed = (out < -smask) | (out >= smask)

    zmask = out & smask  # if `out` has negative sign bit, == 2**bits
    out &= xmask  # mask out all bits > `bits`
    out -= zmask  # subtract 2**bits if negative sign bit

    return out, overflowed


def vth_to_manexp(vth):
    exp = VTH_EXP * np.ones(vth.shape, dtype=np.int32)
    man = np.round(vth / 2**exp).astype(np.int32)
    assert (man > 0).all()
    assert (man <= VTH_MAN_MAX).all()
    return man, exp


def bias_to_manexp(bias):
    r = np.maximum(np.abs(bias) / BIAS_MAN_MAX, 1)
    exp = np.ceil(np.log2(r)).astype(np.int32)
    man = np.round(bias / 2**exp).astype(np.int32)
    assert (exp >= 0).all()
    assert (exp <= BIAS_EXP_MAX).all()
    assert (np.abs(man) <= BIAS_MAN_MAX).all()
    return man, exp


def tracing_mag_int_frac(synapses):
    mag = synapses.tracing_mag
    mag = mag / (synapses.size() / 100)

    mag_int = int(mag)
    # TODO: how does mag_frac actually work???
    #  It's the x in x/128, I believe
    mag_frac = int(128 * (mag - mag_int))
    # mag_frac = min(int(round(1./mag_frac)), 128)

    return mag_int, mag_frac


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

    def index_synapses(self, synapses, chip, core, idxs):
        chip_idx = self.chip_index(chip)
        core_idx = chip.core_index(core)
        self.synapses_index[synapses] = (chip_idx, core_idx, idxs)

    def find_synapses(self, synapses):
        group = synapses.group
        if group.location == 'core':
            return self.synapses_index[synapses]
        elif group.location == 'cpu':
            raise NotImplementedError("CPU neurons not implemented")
        else:
            raise BuildError("Unrecognized location %r" % group.location)

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
        self.inputs = []

        self.cxProfiles = []
        self.vthProfiles = []
        self.synapseFmts = [None]  # keep index 0 unused
        self.stdpPreCfgs = []

        self.synapse_fmt_idxs = {}  # one synfmt per CxSynapses, for now
        self.synapse_axons = collections.OrderedDict()
        self.synapse_entries = collections.OrderedDict()

        self.learning_coreid = None

    @property
    def board(self):
        return self.chip.board

    @property
    def synapses(self):
        return list(self.synapse_axons)

    def validate(self):
        assert len(self.cxProfiles) <= 32  # TODO: check this number
        assert len(self.vthProfiles) <= 16  # TODO: check this number
        assert len(self.synapseFmts) <= 16  # TODO: check this number
        assert len(self.stdpPreCfgs) <= 3

        for cxProfile in self.cxProfiles:
            cxProfile.validate(core=self)
        for vthProfile in self.vthProfiles:
            vthProfile.validate(core=self)
        for synapseFmt in self.synapseFmts:
            if synapseFmt is not None:
                synapseFmt.validate(core=self)
        for traceCfg in self.stdpPreCfgs:
            traceCfg.validate(core=self)

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

    def iterate_inputs(self):
        i0 = 0
        for inp in self.inputs:
            i1 = i0 + inp.n
            cx_idxs = list(range(i0, i1))
            yield inp, cx_idxs
            i0 = i1

    def iterate_synapses(self):
        for group in self.groups:
            for synapses in group.synapses:
                yield synapses

    def add_group(self, group):
        self.groups.append(group)

    def add_input(self, input):
        self.inputs.append(input)

    def add_cx_profile(self, cx_profile):
        self.cxProfiles.append(cx_profile)
        return len(self.cxProfiles) - 1  # index

    def add_vth_profile(self, vth_profile):
        self.vthProfiles.append(vth_profile)
        return len(self.vthProfiles) - 1  # index

    def add_stdp_pre_cfg(self, stdp_pre_cfg):
        self.stdpPreCfgs.append(stdp_pre_cfg)
        return len(self.stdpPreCfgs) - 1  # index

    def n_synapses(self):
        return sum(synapses.size() for group in self.groups
                   for synapses in group.synapses)

    def add_synapses(self, synapses):
        synapse_fmt_idx = self.get_synapse_fmt_idx(synapses.synapse_fmt)
        self.synapse_fmt_idxs[synapses] = synapse_fmt_idx

        a0 = 0
        if len(self.synapse_axons) > 0:
            last = next(reversed(self.synapse_axons))
            a0 = self.synapse_axons[last][-1] + 1
        idxs_per_synapse = synapses.idxs_per_synapse()
        idxs = [a0 + idxs_per_synapse*i for i in range(synapses.n_axons)]
        self.synapse_axons[synapses] = idxs
        self.board.index_synapses(synapses, self.chip, self, idxs)

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

    def get_synapse_fmt_idx(self, synapse_fmt):
        try:
            return self.synapseFmts.index(synapse_fmt)
        except ValueError:
            self.synapseFmts.append(synapse_fmt)
            return len(self.synapseFmts) - 1  # index


class SpikeInput(object):
    class LoihiAxon(object):
        __slots__ = ['chip_id', 'core_id', 'axon_id', 'atom']

        def __init__(self, chip_id, core_id, axon_id, atom=0):
            self.chip_id = chip_id
            self.core_id = core_id
            self.axon_id = axon_id
            self.atom = atom

        def _slots_str(self):
            return ", ".join("%s=%s" % (s, getattr(self, s))
                             for s in self.__slots__)

        def __repr__(self):
            return "%s(%s)" % (type(self).__name__, self._slots_str())

    class LoihiSpike(object):
        __slots__ = ['time', 'axon']

        def __init__(self, time, axon):
            self.time = time
            self.axon = axon

        def __repr__(self):
            return "%s(time=%s, %s)" % (
                type(self).__name__, self.time, self.axon._slots_str())

    def __init__(self):
        self.axon_map = {}  # maps cx_spike_input idx to axon in self.axons
        self.sent_count = 0

    def set_axons(self, board, n2board, cx_spike_input):
        assert len(self.axon_map) == 0
        cx_idxs = np.arange(cx_spike_input.n)
        for axons in cx_spike_input.axons:
            assert (axons.cx_atoms is None or np.all(axons.cx_atoms == 0)), (
                "Cannot send pop spikes to board")
            tchip_idx, tcore_idx, tsyn_ids = board.find_synapses(axons.target)
            tchip = n2board.n2Chips[tchip_idx]
            tcore = tchip.n2Cores[tcore_idx]
            spikes = axons.map_cx_spikes(cx_idxs)
            for cx_idx, spike in zip(cx_idxs, spikes):
                if spike is not None:
                    taxon_idx = int(spike.axon_id)
                    taxon_id = int(tsyn_ids[taxon_idx])
                    self.axon_map.setdefault(cx_idx, []).append(self.LoihiAxon(
                        chip_id=tchip.id, core_id=tcore.id, axon_id=taxon_id))

    def spikes_to_loihi(self, t, cx_idxs):
        for cx_idx in cx_idxs:
            for axon in self.axon_map[cx_idx]:
                yield self.LoihiSpike(time=t, axon=axon)


class Profile(object):
    def __eq__(self, obj):
        return isinstance(obj, type(self)) and all(
            self.__dict__[key] == obj.__dict__[key] for key in self.params)

    def __hash__(self):
        return hash(tuple(self.__dict__[key] for key in self.params))


class CxProfile(Profile):
    DECAY_U_MAX = 2**12 - 1
    DECAY_V_MAX = 2**12 - 1
    REFRACT_DELAY_MAX = 2**6 - 1

    params = ('decayU', 'decayV', 'refractDelay', 'enableNoise')

    def __init__(self, decayV, decayU, refractDelay, enableNoise):
        super(CxProfile, self).__init__()
        self.decayV = decayV
        self.decayU = decayU
        self.refractDelay = refractDelay
        self.enableNoise = enableNoise

    def validate(self, core=None):
        assert self.decayU >= 0
        assert self.decayU <= self.DECAY_U_MAX
        assert self.decayV >= 0
        assert self.decayV <= self.DECAY_V_MAX
        assert self.refractDelay >= 1
        assert self.refractDelay <= self.REFRACT_DELAY_MAX
        assert self.enableNoise in (0, 1)


class VthProfile(Profile):
    """Represents the VthProfile of a compartment (Cx).

    Attributes
    ----------
    vth : int
        The mantissa of the voltage threshold for a compartment. To get the
        actual voltage threshold, this is multiplied by VTH_EXP (64).
    """
    params = ('vth',)

    def __init__(self, vth):
        super(VthProfile, self).__init__()
        self.vth = vth

    def validate(self, core=None):
        assert self.vth > 0
        assert self.vth <= VTH_MAN_MAX
        # if core is not None:
        #     assert self.realVth < core.dendrite_shared_cfg.v_max


class SynapseFmt(Profile):
    INDEX_BITS_MAP = [0, 6, 7, 8, 9, 10, 11, 12]
    WEIGHT_BITS_MAP = [0, 1, 2, 3, 4, 5, 6, 8]

    params = ('wgtLimitMant', 'wgtLimitExp', 'wgtExp', 'discMaxWgt',
              'learningCfg', 'tagBits', 'dlyBits', 'wgtBits',
              'reuseSynData', 'numSynapses', 'cIdxOffset', 'cIdxMult',
              'skipBits', 'idxBits', 'synType', 'fanoutType',
              'compression', 'stdpProfile', 'ignoreDly')

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

    @property
    def realWgtBits(self):
        return self.WEIGHT_BITS_MAP[self.wgtBits]

    @property
    def realIdxBits(self):
        return self.INDEX_BITS_MAP[self.idxBits]

    @property
    def isMixed(self):
        return self.fanoutType == 1

    def bits_per_axon(self, n_weights):
        """For an axon with n weights, compute the weight memory bits used"""
        bits_per_weight = self.realWgtBits + self.dlyBits + self.tagBits
        if self.compression == 0:
            bits_per_weight += self.realIdxBits
        elif self.compression == 3:
            pass
        else:
            raise NotImplementedError("Compression %s" % (self.compression,))

        SYNAPSE_FMT_IDX_BITS = 4
        N_SYNAPSES_BITS = 6
        bits = 0
        synapses_per_group = self.numSynapses + 1
        for i in range(0, n_weights, synapses_per_group):
            n = min(n_weights - i, synapses_per_group)
            bits_i = n*bits_per_weight + SYNAPSE_FMT_IDX_BITS + N_SYNAPSES_BITS
            bits_i = -64 * (-bits_i // 64)
            # ^ round up to nearest 64 (size of one int64 memory unit)
            bits += bits_i

        return bits

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

    def discretize_weights(self, w, dtype=np.int32):
        s = 8 - self.realWgtBits + self.isMixed
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


class StdpProfile(Profile):
    params = (
        'uCodePtr', 'decimateExp', 'numProducts', 'requireY', 'usesXepoch')

    def __init__(self, uCodePtr=0, decimateExp=0, numProducts=0, requireY=0,
                 usesXepoch=0):
        super(StdpProfile, self).__init__()
        self.uCodePtr = uCodePtr
        self.decimateExp = decimateExp
        self.numProducts = numProducts
        self.requireY = requireY
        self.usesXepoch = usesXepoch

    def validate(self, core=None):
        pass


class StdpPreProfile(Profile):
    params = ('updateAlways', 'numTraces', 'numTraceHist', 'stdpProfile')

    def __init__(
            self, updateAlways=1, numTraces=0, numTraceHist=0, stdpProfile=0):
        super(StdpPreProfile, self).__init__()
        self.updateAlways = updateAlways
        self.numTraces = numTraces
        self.numTraceHist = numTraceHist
        self.stdpProfile = stdpProfile

    def validate(self, core=None):
        pass


class TraceCfg(Profile):
    params = ('tau', 'spikeLevelInt', 'spikeLevelFrac')

    def __init__(self, tau=0, spikeLevelInt=0, spikeLevelFrac=0):
        super(TraceCfg, self).__init__()
        self.tau = tau
        self.spikeLevelInt = spikeLevelInt
        self.spikeLevelFrac = spikeLevelFrac

    def validate(self, core=None):
        pass
