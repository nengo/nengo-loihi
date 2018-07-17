from __future__ import division

import collections

import numpy as np

CX_PROFILES_MAX = 32
VTH_PROFILES_MAX = 8
SYNAPSE_FMTS_MAX = 16


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

    def add_synapse_fmt(self, synapse_fmt):
        self.synapseFmts.append(synapse_fmt)
        return len(self.synapseFmts) - 1  # index

    def add_stdp_pre_cfg(self, stdp_pre_cfg):
        self.stdpPreCfgs.append(stdp_pre_cfg)
        return len(self.stdpPreCfgs) - 1  # index

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
            a0 = self.synapse_axons[last][-1]
        idx_mult = 2 if synapses.tracing else 1
        idxs = [a0 + idx_mult*i for i in range(synapses.n_axons)]
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
        assert 0 <= self.decayU <= self.DECAY_U_MAX
        assert 0 <= self.decayV <= self.DECAY_V_MAX
        assert 1 <= self.refractDelay <= self.REFRACT_DELAY_MAX
        assert 0 <= self.enableNoise <= 1


class VthProfile(Profile):
    VTH_MAX = 2**17 - 1  # TODO: is this or the one in cx.py right?

    params = ('vth',)

    def __init__(self, vth):
        super(VthProfile, self).__init__()
        self.vth = vth

    def validate(self, core=None):
        assert 0 < self.vth <= self.VTH_MAX
        # if core is not None:
        #     assert self.realVth < core.dendrite_shared_cfg.v_max


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
