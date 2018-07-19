from __future__ import division

import collections
import logging

import numpy as np

from nengo_loihi.cx import bias_to_manexp
from nengo_loihi.hardware.nxsdk_shim import (
    BasicSpikeGenerator, N2Board, TraceCfgGen)

logger = logging.getLogger(__name__)

CX_PROFILES_MAX = 32
VTH_PROFILES_MAX = 8
SYNAPSE_FMTS_MAX = 16


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


def build_board(board):
    n_chips = board.n_chips()
    n_cores_per_chip = board.n_cores_per_chip()
    n_synapses_per_core = board.n_synapses_per_core()

    n2board = N2Board(
        board.board_id, n_chips, n_cores_per_chip, n_synapses_per_core)

    assert len(board.chips) == len(n2board.n2Chips)
    for chip, n2chip in zip(board.chips, n2board.n2Chips):
        logger.debug("Building chip %s", chip)
        build_chip(n2chip, chip)

    return n2board


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


def build_chip(n2chip, chip):
    assert len(chip.cores) == len(n2chip.n2Cores)
    for core, n2core in zip(chip.cores, n2chip.n2Cores):
        logger.debug("Building core %s", core)
        build_core(n2core, core)


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


def build_core(n2core, core):  # noqa: C901
    assert len(core.cxProfiles) < CX_PROFILES_MAX
    assert len(core.vthProfiles) < VTH_PROFILES_MAX

    logger.debug("- Configuring cxProfiles")
    for i, cxProfile in enumerate(core.cxProfiles):
        n2core.cxProfileCfg[i].configure(
            decayV=cxProfile.decayV,
            decayU=cxProfile.decayU,
            refractDelay=cxProfile.refractDelay,
            enableNoise=cxProfile.enableNoise,
            bapAction=1,
        )

    logger.debug("- Configuring vthProfiles")
    for i, vthProfile in enumerate(core.vthProfiles):
        n2core.vthProfileCfg[i].staticCfg.configure(
            vth=vthProfile.vth,
        )

    logger.debug("- Configuring synapseFmts")
    for i, synapseFmt in enumerate(core.synapseFmts):
        if synapseFmt is None:
            continue

        n2core.synapseFmt[i].wgtLimitMant = synapseFmt.wgtLimitMant
        n2core.synapseFmt[i].wgtLimitExp = synapseFmt.wgtLimitExp
        n2core.synapseFmt[i].wgtExp = synapseFmt.wgtExp
        n2core.synapseFmt[i].discMaxWgt = synapseFmt.discMaxWgt
        n2core.synapseFmt[i].learningCfg = synapseFmt.learningCfg
        n2core.synapseFmt[i].tagBits = synapseFmt.tagBits
        n2core.synapseFmt[i].dlyBits = synapseFmt.dlyBits
        n2core.synapseFmt[i].wgtBits = synapseFmt.wgtBits
        n2core.synapseFmt[i].reuseSynData = synapseFmt.reuseSynData
        n2core.synapseFmt[i].numSynapses = synapseFmt.numSynapses
        n2core.synapseFmt[i].cIdxOffset = synapseFmt.cIdxOffset
        n2core.synapseFmt[i].cIdxMult = synapseFmt.cIdxMult
        n2core.synapseFmt[i].skipBits = synapseFmt.skipBits
        n2core.synapseFmt[i].idxBits = synapseFmt.idxBits
        n2core.synapseFmt[i].synType = synapseFmt.synType
        n2core.synapseFmt[i].fanoutType = synapseFmt.fanoutType
        n2core.synapseFmt[i].compression = synapseFmt.compression
        n2core.synapseFmt[i].stdpProfile = synapseFmt.stdpProfile
        n2core.synapseFmt[i].ignoreDly = synapseFmt.ignoreDly

    logger.debug("- Configuring stdpPreCfgs")
    for i, traceCfg in enumerate(core.stdpPreCfgs):
        tcg = TraceCfgGen()
        tc = tcg.genTraceCfg(
            tau=traceCfg.tau,
            spikeLevelInt=traceCfg.spikeLevelInt,
            spikeLevelFrac=traceCfg.spikeLevelFrac,
        )
        tc.writeToRegister(n2core.stdpPreCfg[i])

    # --- learning
    firstLearningIndex = None
    for synapse in core.iterate_synapses():
        if synapse.tracing and firstLearningIndex is None:
            firstLearningIndex = core.synapse_axons[synapse][0]
            core.learning_coreid = n2core.id
            break

    numStdp = 0
    if firstLearningIndex is not None:
        for synapse in core.iterate_synapses():
            axons = np.array(core.synapse_axons[synapse])
            if synapse.tracing:
                numStdp += len(axons)
                assert np.all(len(axons) >= firstLearningIndex)
            else:
                assert np.all(len(axons) < firstLearningIndex)

    if numStdp > 0:
        logger.debug("- Configuring PES learning")
        # add configurations tailored to PES learning
        n2core.stdpCfg.configure(
            firstLearningIndex=firstLearningIndex,
            numRewardAxons=0,
        )

        assert core.stdp_pre_profile_idx is None
        assert core.stdp_profile_idx is None
        core.stdp_pre_profile_idx = 0  # hard-code for now
        core.stdp_profile_idx = 0  # hard-code for now (also in synapse_fmt)
        n2core.stdpPreProfileCfg[0].configure(
            updateAlways=1,
            numTraces=0,
            numTraceHist=0,
            stdpProfile=0,
        )

        # stdpProfileCfg positive error
        n2core.stdpProfileCfg[0].configure(
            uCodePtr=0,
            decimateExp=0,
            numProducts=1,
            requireY=1,
            usesXepoch=1,
        )
        n2core.stdpUcodeMem[0].word = 0x00102108  # 2^-7 learn rate

        # stdpProfileCfg negative error
        n2core.stdpProfileCfg[1].configure(
            uCodePtr=1,
            decimateExp=0,
            numProducts=1,
            requireY=1,
            usesXepoch=1,
        )
        n2core.stdpUcodeMem[1].word = 0x00f02108  # 2^-7 learn rate

        tcg = TraceCfgGen()
        tc = tcg.genTraceCfg(
            tau=0,
            spikeLevelInt=0,
            spikeLevelFrac=0,
        )
        tc.writeToRegister(n2core.stdpPostCfg[0])

    # TODO: allocator should be checking that vmin, vmax are the same
    #   for all groups on a core
    n_cx = 0
    if len(core.groups) > 0:
        group0 = core.groups[0]
        vmin, vmax = group0.vmin, group0.vmax
        assert all(group.vmin == vmin for group in core.groups)
        assert all(group.vmax == vmax for group in core.groups)
        negVmLimit = np.log2(-vmin + 1)
        posVmLimit = (np.log2(vmax + 1) - 9) * 0.5
        assert int(negVmLimit) == negVmLimit
        assert int(posVmLimit) == posVmLimit

        noiseExp0 = group0.noiseExp0
        noiseMantOffset0 = group0.noiseMantOffset0
        noiseAtDendOrVm = group0.noiseAtDendOrVm
        assert all(group.noiseExp0 == noiseExp0 for group in core.groups)
        assert all(group.noiseMantOffset0 == noiseMantOffset0
                   for group in core.groups)
        assert all(group.noiseAtDendOrVm == noiseAtDendOrVm
                   for group in core.groups)

        n2core.dendriteSharedCfg.configure(
            posVmLimit=int(posVmLimit),
            negVmLimit=int(negVmLimit),
            noiseExp0=noiseExp0,
            noiseMantOffset0=noiseMantOffset0,
            noiseAtDendOrVm=noiseAtDendOrVm,
        )

        n2core.dendriteAccumCfg.configure(
            delayBits=3)
        # ^ DelayBits=3 allows 1024 Cxs per core

        for group, cx_idxs, ax_range in core.iterate_groups():
            build_group(n2core, core, group, cx_idxs, ax_range)
            n_cx = max(max(cx_idxs), n_cx)

    for inp, cx_idxs in core.iterate_inputs():
        build_input(n2core, core, inp, cx_idxs)

    logger.debug("- Configuring numUpdates=%d", n_cx // 4 + 1)
    n2core.numUpdates.configure(
        numUpdates=n_cx // 4 + 1,
        numStdp=numStdp,
    )

    n2core.dendriteTimeState[0].tepoch = 2
    n2core.timeState[0].tepoch = 2


def build_group(n2core, core, group, cx_idxs, ax_range):
    assert group.scaleU is False
    assert group.scaleV is False

    logger.debug("Building %s on core.id=%d", group, n2core.id)

    for i, bias in enumerate(group.bias):
        bman, bexp = bias_to_manexp(bias)
        icx = core.cx_profile_idxs[group][i]
        ivth = core.vth_profile_idxs[group][i]

        ii = cx_idxs[i]
        n2core.cxCfg[ii].configure(
            bias=bman, biasExp=bexp, vthProfile=ivth, cxProfile=icx)

        phasex = 'phase%d' % (ii % 4,)
        n2core.cxMetaState[ii // 4].configure(**{phasex: 2})

    logger.debug("- Building %d synapses", len(group.synapses))
    for synapses in group.synapses:
        build_synapses(n2core, core, group, synapses, cx_idxs)

    logger.debug("- Building %d synapses", len(group.axons))
    for axons in group.axons:
        build_axons(n2core, core, group, axons, cx_idxs)

    logger.debug("- Building %d synapses", len(group.probes))
    for probe in group.probes:
        build_probe(n2core, core, group, probe, cx_idxs)


def build_input(n2core, core, spike_input, cx_idxs):
    assert len(spike_input.axons) > 0

    for axon in spike_input.axons:
        build_axons(n2core, core, spike_input, axon, cx_idxs)

    for probe in spike_input.probes:
        build_probe(n2core, core, spike_input, probe, cx_idxs)

    n2board = n2core.parent.parent

    if not hasattr(n2core, 'master_spike_gen'):
        # TODO: this is only needed if precompute=True
        n2core.master_spike_gen = BasicSpikeGenerator(n2board)

    # get core/axon ids
    axon_ids = []
    for axon in spike_input.axons:
        tchip_idx, tcore_idx, tsyn_idxs = core.board.find_synapses(axon.target)
        tchip = n2board.n2Chips[tchip_idx]
        tcore = tchip.n2Cores[tcore_idx]
        axon_ids.append([(tchip.id, tcore.id, tsyn_idx)
                         for tsyn_idx in tsyn_idxs])

    spike_input.spike_gen = n2core.master_spike_gen
    spike_input.axon_ids = axon_ids

    for i, spiked in enumerate(spike_input.spikes):
        for j, s in enumerate(spiked):
            if s:
                for output_axon in axon_ids:
                    n2core.master_spike_gen.addSpike(i, *output_axon[j])

    spike_input.sent_count = len(spike_input.spikes)


def build_synapses(n2core, core, group, synapses, cx_idxs):
    syn_idxs = core.synapse_axons[synapses]
    assert len(syn_idxs) == len(synapses.weights)

    synapse_fmt_idx = core.synapse_fmt_idxs[synapses]
    stdp_pre_cfg_idx = core.stdp_pre_cfg_idxs[synapses]

    target_cxs = set()
    s0 = core.synapse_entries[synapses][0]
    for a, syn_idx in enumerate(syn_idxs):
        wa = synapses.weights[a] // synapses.synapse_fmt.scale
        ia = synapses.indices[a]
        assert len(wa) == len(ia)

        assert np.all(wa <= 255) and np.all(wa >= -256), str(wa)
        for k, (w, i) in enumerate(zip(wa, ia)):
            n2core.synapses[s0 + k].configure(
                CIdx=cx_idxs[i],
                Wgt=w,
                synFmtId=synapse_fmt_idx,
                LrnEn=int(synapses.tracing),
            )
            target_cxs.add(cx_idxs[i])

        n2core.synapseMap[syn_idx].synapsePtr = s0
        n2core.synapseMap[syn_idx].synapseLen = len(wa)
        n2core.synapseMap[syn_idx].discreteMapEntry.configure()

        if synapses.tracing:
            assert core.stdp_pre_profile_idx is not None
            assert stdp_pre_cfg_idx is not None
            n2core.synapseMap[syn_idx+1].singleTraceEntry.configure(
                preProfile=core.stdp_pre_profile_idx, tcs=stdp_pre_cfg_idx)

        s0 += len(wa)

    if synapses.tracing:
        assert core.stdp_profile_idx is not None
        for target_cx in target_cxs:
            # TODO: check that no cx gets configured by multiple synapses
            n2core.stdpPostState[target_cx].configure(
                stdpProfile=core.stdp_profile_idx,
                traceProfile=3,  # TODO: why this value
            )


def build_axons(n2core, core, group, axons, cx_idxs):
    tchip_idx, tcore_idx, tsyn_idxs = core.board.find_synapses(axons.target)
    taxon_idxs = np.asarray(tsyn_idxs)[axons.target_inds]
    n2board = n2core.parent.parent
    tchip_id = n2board.n2Chips[tchip_idx].id
    tcore_id = n2board.n2Chips[tchip_idx].n2Cores[tcore_idx].id
    assert axons.n_axons == len(cx_idxs) == len(taxon_idxs)
    for i in range(axons.n_axons):
        n2core.createDiscreteAxon(
            cx_idxs[i], tchip_id, tcore_id, int(taxon_idxs[i]))


def build_probe(n2core, core, group, probe, cx_idxs):
    assert probe.key in ('u', 'v', 's')
    key_map = {'s': 'spike'}
    key = key_map.get(probe.key, probe.key)

    n2board = n2core.parent.parent
    r = cx_idxs[probe.slice]

    if probe.use_snip:
        probe.snip_info = dict(coreid=n2core.id, cxs=r)
    else:
        p = n2board.monitor.probe(n2core.cxState, r, key)
        core.board.map_probe(probe, p)


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
