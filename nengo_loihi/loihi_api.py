CX_PROFILES_MAX = 32
VTH_PROFILES_MAX = 8
SYNAPSE_FMTS_MAX = 16

VTH_MAN_MAX = 2**17 - 1
VTH_EXP = 2**6
VTH_MAX = VTH_MAN_MAX * VTH_EXP

BIAS_MAN_MAX = 2**12 - 1
BIAS_EXP_MAX = 2**3 - 1
BIAS_MAX = BIAS_MAN_MAX * BIAS_EXP_MAX


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
        self.chips = collections.OrderedDict()

        self.synapses_index = {}

    def _add_chip(self, chip):
        assert chip not in self.chips
        index = len(self.chips)
        self.chips[chip] = index

    def new_chip(self):
        chip = Chip(board=self)
        self._add_chip(chip)
        return chip

    def chip_index(self, chip):
        return self.chips.index(chip)

    def index_synapses(self, synapses, chip, core, a0, a1):
        chip_idx = self.chip_index(chip)
        core_idx = chip.core_index(core)
        self.synapses_index[synapses] = (chip_idx, core_idx, a0, a1)

    def find_synapses(self, synapses):
        group = synapses.parent
        if group.location == 'core':
            return self.synapses_index[synapses]
        elif group.location == 'cpu':
            raise NotImplementedError()
        else:
            raise NotImplementedError()

    # def find_group(self, group):
    #     if group.location == 'core':
    #         for chip_idx, chip in enumerate(self.chips):
    #             for core_idx, core in enumerate(chip.cores):
    #                 result = core.find_group(group)
    #                 if result:
    #                     group_idx, (i0, i1) = result
    #                     return CxSlice(
    #                         self.board_id, chip_idx, core_idx, i0, i1)

    #         raise KeyError("Could not find group %r" % group)
    #     elif group.location == 'cpu':
    #         raise NotImplementedError()
    #     else:
    #         raise NotImplementedError()

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
        self.cores = collections.OrderedDict()

    def _add_core(self, core):
        assert core not in self.cores
        index = len(self.cores)
        self.cores[core] = index

    def new_core(self):
        core = Core(chip=self)
        self._add_core(core)
        return core

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

    def iterate_groups(self):
        i0 = 0
        for group in self.groups:
            i1 = i0 + group.n
            yield group, (i0, i1)
            i0 = i1

    # def iterate_synapses(self):
    #     for group in self.groups:
    #         for synapses in self.synapses:

    # def find_group(self, target_group):
    #     for i, (group, inds) in enumerate(self.groups):
    #         if group is target_group:
    #             return i, inds

    #     return None

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
        return sum(synapses.weights.size for group in self.groups
                   for synapses in group.synapses)

    def add_synapses(self, synapses):
        synapse_fmt = synapses.synapse_fmt
        synapse_fmt_idx = self.add_synapse_fmt(synapse_fmt)
        self.synapse_fmt_idxs[synapses] = synapse_fmt_idx

        last = next(reversed(self.synapse_axons))
        a0 = self.synapse_axons[last][1]
        a1 = a0 + synapses.n_axons
        self.synapse_axons[synapses] = (a0, a1)
        self.board.index_synapses(synapses, self.core, self, a0, a1)

        last = next(reversed(self.synapse_entries))
        s0 = self.synapse_entries[last][1]
        s1 = s0 + synapses.size()
        self.synapse_entries[synapses] = (s0, s1)


class CxProfile(object):
    def __init__(self, decayV, decayU, refDelay):
        self.decayV = decayV
        self.decayU = decayU
        self.refDelay = refDelay

    def __eq__(self, cxProfile):
        return all(self.__dict__[key] == cxProfile.__dict__[key]
                   for key in self.__dict__)


class VthProfile(object):
    def __init__(self, vth):
        self.vth = vth

    def __eq__(self, cxProfile):
        return all(self.__dict__[key] == cxProfile.__dict__[key]
                   for key in self.__dict__)


class SynapseFmt(object):
    INDEX_BITS_MAP = [0, 6, 7, 8, 9, 10, 11, 12]

    def __init__(self, WgtLimitMant=0, WgtLimitExp=0, WgtExp=0, DiscMaxWgt=0,
                 LearningCfg=3, TagBits=0, DlyBits=0, WgtBits=7,
                 ReuseSynData=0, NumSynapses=63, CIdxOffset=0, CIdxMult=0,
                 SkipBits=0, IdxBits=5, SynType=0, FanoutType=0,
                 Compression=0, StdpProfile=0, IgnoreDly=0):
        self.WgtLimitMant = WgtLimitMant
        self.WgtLimitExp = WgtLimitExp
        self.WgtExp = WgtExp
        self.DiscMaxWgt = DiscMaxWgt
        self.LearningCfg = LearningCfg
        self.TagBits = TagBits
        self.DlyBits = DlyBits
        self.WgtBits = WgtBits
        self.ReuseSynData = ReuseSynData
        self.NumSynapses = NumSynapses
        self.CIdxOffset = CIdxOffset
        self.CIdxMult = CIdxMult
        self.SkipBits = SkipBits
        self.IdxBits = IdxBits
        self.SynType = SynType
        self.FanoutType = FanoutType
        self.Compression = Compression
        self.StdpProfile = StdpProfile
        self.IgnoreDly = IgnoreDly

    def set(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def validate(self, core=None):
        assert -7 <= self.WgtExp <= 7
        assert 0 <= self.TagBits < 4
        assert 0 <= self.DlyBits < 8
        assert 0 <= self.WgtBits < 8
        assert 0 <= self.CIdxOffset < 16
        assert 0 <= self.CIdxMult < 16
        assert 0 <= self.IdxBits < 8

    @property
    def width(self):
        return 1 + self.WgtBits

    @property
    def isMixed(self):
        # return 0
        return self.FanoutType == 1

    @property
    def Wscale(self):
        return 14 - self.width + self.WgtExp + self.isMixed
        # ^ = 8 - width(SYN_l) + 6 + wgtExp(SYN_l) + isMixed(SYN_l)

    @property
    def realIdxBits(self):
        return self.INDEX_BITS_MAP[self.IdxBits]
