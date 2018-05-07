import numpy as np

from nxsdk.arch.n2a.graph.graph import N2Board
# from nxsdk.arch.n2a.graph.inputgen import BasicSpikeGenerator

from .loihi_api import CX_PROFILES_MAX, VTH_PROFILES_MAX, BIAS_MAN_MAX


def build_board(board):
    n_chips = board.n_chips()
    n_cores_per_chip = board.n_cores_per_chip()
    n_synapses_per_core = board.n_synapses_per_core()

    n2board = N2Board(
        board.board_id, n_chips, n_cores_per_chip, n_synapses_per_core)

    assert len(board.chips) == len(n2board.n2Chips)
    for chip, n2chip in zip(board.chips, n2board.n2Chips):
        build_chip(n2board, n2chip, board, chip)


def build_chip(n2chip, chip):
    assert len(chip.cores) == len(n2chip.n2Cores)
    for core, n2core in zip(chip.cores, n2chip.n2Cores):
        build_core(n2board, n2chip, n2core, board, chip, core)


def build_core(n2core, core):
    assert len(core.cxProfiles) < CX_PROFILES_MAX
    assert len(core.vthProfiles) < VTH_PROFILES_MAX

    for i, cxProfile in enumerate(core.cxProfiles):
        n2core.cxProfileCfg[i].configure(
            decayV=cxProfile.decayV,
            decayU=cxProfile.decayU,
            refDelay=cxProfile.refDelay,
        )

    for i, vthProfile in enumerate(core.vthProfiles):
        n2core.vthProfileCfg[i].configure(
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

    for i, bias in enumerate(group.bias):
        bexp = int(np.ceil(np.log2(bias / float(BIAS_MAN_MAX))))
        bman = int(bias / 2**bexp)
        icx = core.cxProfileInds[group][i]
        ivth = core.vthProfileInds[group][i]

        ii = i0 + i
        n2core.cxCfg[ii].configure(
            bias=bman, biasExp=bexp, vthProfile=ivth, cxProfile=icx)

        phasex = 'phase%d' % (ii % 4,)
        n2core.cxMetaState[ii].configure(**{phasex: 2})

    for synapses in group.synapses:
        build_synapses(n2core, core, group, i0, i1, synapses)

    n2core.numUpdates.configure(numUpdates=1)




def build_synapses(n2core, core, group, i0, i1, synapses):
    a0, a1 = core.synapse_axons[synapses]
    assert (a1 - a0) == len(synapses.weights)

    synapse_fmt_idx = core.synapse_fmt_idxs[synapses]

    s0 = core.synapse_entries[synapses]
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



def setupNetwork():

    # --- board
    boardId = 1
    numChips = 1
    # Number of cores per chip
    numCoresPerChip = [2]
    # Number of synapses per core
    numSynapsesPerCore = [[n0, n0*n1]]
    # Initialize the board
    board = N2Board(boardId, numChips, numCoresPerChip, numSynapsesPerCore)

    chip = board.n2Chips[0]
    core0 = chip.n2Cores[0]
    core1 = chip.n2Cores[1]

    # --- core0
    # decayU = 512
    # decayU = 2**12 - 1
    # decayU = 0
    # decayV = 0

    decayU = 273
    decayV = 197
    vth = 1001

    core0.cxProfileCfg[0].configure(decayV=decayV, decayU=decayU)
    core0.vthProfileCfg[0].staticCfg.configure(vth=vth)
    core0.dendriteSharedCfg.configure(posVmLimit=7, negVmLimit=0)

    core0.synapseFmt[1].wgtExp = 0
    core0.synapseFmt[1].wgtBits = 7
    core0.synapseFmt[1].numSynapses = 63
    core0.synapseFmt[1].idxBits = 1
    core0.synapseFmt[1].compression = 3
    core0.synapseFmt[1].fanoutType = 1

    for i, (w, idx) in enumerate(zip(w_in, i_in)):
        core0.synapses[i].CIdx = idx
        core0.synapses[i].Wgt = w
        core0.synapses[i].synFmtId = 1

    core0.synapseMap[0].synapsePtr = 0
    core0.synapseMap[0].synapseLen = len(w_in)
    core0.synapseMap[0].discreteMapEntry.configure()

    for i in range(n0):
        core0.cxCfg[i].configure(bias=0, biasExp=0, vthProfile=0, cxProfile=0)
        core0.axonMap[i].configure(ptr=i, len=1)
        core0.axonCfg[i].discrete.configure(coreId=core1.id, axonId=i)

        # core0.cxMetaState[i].configure(phase0=2)

    core0.numUpdates.configure(numUpdates=1)

    # --- core1
    # decayV = 0
    # decayU = 2**12 - 1
    decayU = 832
    decayV = 173
    vth = 1001

    core1.cxProfileCfg[0].configure(decayV=decayV, decayU=decayU)
    core1.vthProfileCfg[0].staticCfg.configure(vth=vth)
    core1.dendriteSharedCfg.configure(posVmLimit=7, negVmLimit=0)

    core1.synapseFmt[1].wgtExp = 0
    core1.synapseFmt[1].wgtBits = 7
    core1.synapseFmt[1].numSynapses = 63
    core1.synapseFmt[1].idxBits = 1
    core1.synapseFmt[1].compression = 3
    core1.synapseFmt[1].fanoutType = 1

    k0 = 0
    for i in range(n0):
        ws = w_conn[i]
        idxs = i_conn[i]
        k1 = k0 + len(ws)

        for k, (w, idx) in enumerate(zip(ws, idxs)):
            core1.synapses[k0+k].CIdx = idx
            core1.synapses[k0+k].Wgt = w
            core1.synapses[k0+k].synFmtId = 1

        core1.synapseMap[i].synapsePtr = k0
        core1.synapseMap[i].synapseLen = len(ws)
        core1.synapseMap[i].discreteMapEntry.configure()

        k0 = k1

    for i in range(n1):
        # core1.cxMetaState[i].configure(phase0=2)
        core1.cxCfg[i].configure(bias=0, biasExp=0, vthProfile=0, cxProfile=0)

    core1.numUpdates.configure(numUpdates=1)


    # Return the configured board
    return board
