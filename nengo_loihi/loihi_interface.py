from __future__ import division

import numpy as np

from nxsdk.arch.n2a.graph.graph import N2Board
# from nxsdk.arch.n2a.graph.inputgen import BasicSpikeGenerator

from nengo_loihi.allocators import one_to_one_allocator
from nengo_loihi.loihi_api import (
    CX_PROFILES_MAX, VTH_PROFILES_MAX, bias_to_manexp)


def build_board(board):
    # from nxsdk.arch.n2a.graph.graph import N2Board

    n_chips = board.n_chips()
    n_cores_per_chip = board.n_cores_per_chip()
    n_synapses_per_core = board.n_synapses_per_core()

    n2board = N2Board(
        board.board_id, n_chips, n_cores_per_chip, n_synapses_per_core)

    assert len(board.chips) == len(n2board.n2Chips)
    for chip, n2chip in zip(board.chips, n2board.n2Chips):
        build_chip(n2chip, chip)

    return n2board


def build_chip(n2chip, chip):
    assert len(chip.cores) == len(n2chip.n2Cores)
    for core, n2core in zip(chip.cores, n2chip.n2Cores):
        build_core(n2core, core)


def build_core(n2core, core):
    assert len(core.cxProfiles) < CX_PROFILES_MAX
    assert len(core.vthProfiles) < VTH_PROFILES_MAX

    for i, cxProfile in enumerate(core.cxProfiles):
        n2core.cxProfileCfg[i].configure(
            decayV=cxProfile.decayV,
            decayU=cxProfile.decayU,
            bapAction=1,
            refractDelay=cxProfile.refDelay,
        )

    for i, vthProfile in enumerate(core.vthProfiles):
        n2core.vthProfileCfg[i].staticCfg.configure(
            vth=vthProfile.vth,
        )

    for i, synapseFmt in enumerate(core.synapseFmts):
        if synapseFmt is None:
            continue

        n2core.synapseFmt[i].wgtExp = synapseFmt.WgtExp
        n2core.synapseFmt[i].wgtBits = synapseFmt.WgtBits
        n2core.synapseFmt[i].numSynapses = synapseFmt.NumSynapses
        n2core.synapseFmt[i].idxBits = synapseFmt.IdxBits
        n2core.synapseFmt[i].compression = synapseFmt.Compression
        n2core.synapseFmt[i].fanoutType = synapseFmt.FanoutType
        # ^ TODO: other parameters

    # TODO: allocator should be checking that vmin, vmax are the same
    #   for all groups on a core
    vmin, vmax = core.groups[0].vmin, core.groups[0].vmax
    assert all(group.vmin == vmin for group in core.groups)
    assert all(group.vmax == vmax for group in core.groups)
    negVmLimit = np.log2(-vmin + 1)
    posVmLimit = (np.log2(vmax + 1) - 9) * 0.5
    assert int(negVmLimit) == negVmLimit
    assert int(posVmLimit) == posVmLimit

    n2core.dendriteSharedCfg.configure(
        posVmLimit=int(posVmLimit),
        negVmLimit=int(negVmLimit))

    n2core.dendriteAccumCfg.configure(
        delayBits=3)
    # ^ DelayBits=3 allows 1024 Cxs per core

    for group, (i0, i1) in core.iterate_groups():
        build_group(n2core, core, group, i0, i1)


def build_group(n2core, core, group, i0, i1):
    assert group.scaleU is False
    assert group.scaleV is False

    # for i in range(256):
    #     n2core.cxMetaState[i].configure(
    #         phase0=2, phase1=2, phase2=2, phase3=2)

    for i, bias in enumerate(group.bias):
        bman, bexp = bias_to_manexp(bias)
        icx = core.cxProfileIdxs[group][i]
        ivth = core.vthProfileIdxs[group][i]

        ii = i0 + i
        n2core.cxCfg[ii].configure(
            bias=bman, biasExp=bexp, vthProfile=ivth, cxProfile=icx)

        phasex = 'phase%d' % (ii % 4,)
        n2core.cxMetaState[ii//4].configure(**{phasex: 2})
        # if ii % 4 == 0:
        #     n2core.cxMetaState[ii//4].configure(phase0=2)
        # elif ii % 4 == 2:
        #     n2core.cxMetaState[ii//4].configure(phase1=2)
        # elif ii % 4 == 3:
        #     n2core.cxMetaState[ii//4].configure(phase2=2)
        # else:
        #     n2core.cxMetaState[ii//4].configure(phase3=2)

    for synapses in group.synapses:
        build_synapses(n2core, core, group, i0, i1, synapses)

    for axons in group.axons:
        build_axons(n2core, core, group, i0, i1, axons)

    for probe in group.probes:
        build_probe(n2core, core, group, i0, i1, probe)

    n2core.numUpdates.configure(numUpdates=1)


def build_synapses(n2core, core, group, i0, i1, synapses):
    a0, a1 = core.synapse_axons[synapses]
    assert (a1 - a0) == len(synapses.weights)

    synapse_fmt_idx = core.synapse_fmt_idxs[synapses]

    s0 = core.synapse_entries[synapses][0]
    for a in range(a1 - a0):
        wa = synapses.weights[a]
        ia = synapses.indices[a]

        for k, (w, i) in enumerate(zip(wa, ia)):
            n2core.synapses[s0+k].CIdx = i0 + i
            assert n2core.synapses[s0+k].CIdx < i1
            n2core.synapses[s0+k].Wgt = w
            n2core.synapses[s0+k].synFmtId = synapse_fmt_idx

        n2core.synapseMap[a0+a].synapsePtr = s0
        n2core.synapseMap[a0+a].synapseLen = len(wa)
        n2core.synapseMap[a0+a].discreteMapEntry.configure()

        s0 += len(wa)


def build_axons(n2core, core, group, i0, i1, axons):
    a0, a1 = core.axon_axons[axons]

    # for now, all axons are one compartment to one axon
    for i in range(group.n):
        n2core.axonMap[i0+i].configure(ptr=a0+i, len=1)

    tchip_idx, tcore_idx, t0, t1 = core.board.find_synapses(axons.target)
    n2board = n2core.parent.parent
    tcore_id = n2board.n2Chips[tchip_idx].n2Cores[tcore_idx].id
    for i in range(axons.n_axons):
        n2core.axonCfg[a0+i].discrete.configure(coreId=tcore_id, axonId=t0+i)


def build_probe(n2core, core, group, i0, i1, probe):
    assert probe.key in ('u', 'v', 's', 'x')
    key_map = {'s': 'spike', 'x': 'u'}
    key = key_map.get(probe.key, probe.key)

    n2board = n2core.parent.parent
    r = list(np.arange(i0, i1)[probe.slice])
    p = n2board.monitor.probe(n2core.cxState, r, key)
    core.board.map_probe(probe, p)


class LoihiSimulator(object):
    def __init__(self, cx_model):
        self.build(cx_model)

    def build(self, cx_model):
        self.model = cx_model

        # --- allocate
        allocator = one_to_one_allocator
        self.board = allocator(self.model)

        # --- build
        self.n2board = build_board(self.board)

    def run_steps(self, steps):
        self.n2board.run(steps)

    def get_probe_output(self, probe):
        cx_probe = self.model.objs[probe]['out']
        n2probe = self.board.probe_map[cx_probe]
        return np.column_stack([p.timeSeries.data for p in n2probe])

        # target = self.model.objs[probe]['in']
        # if isinstance(target, CxGroup):
        #     raise NotImplementedError("Need some way to get this off chip")
        # elif isinstance(target, CxProbe):
        #     n2probe = self.board.probe_map[target]
        #     return n2probe.timeSeries.data
        # else:
        #     raise NotImplementedError()
