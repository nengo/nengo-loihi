import os

import numpy as np

from nxsdk.arch.n2a.graph.graph import N2Board
from nxsdk.arch.n2a.graph.inputgen import BasicSpikeGenerator


def setupNetwork():

    # n = 200
    # w_in = [np.arange(n) for _ in range(n)]
    # i_in = [np.arange(n) for _ in range(n)]

    w_in = [[-257, 255]]
    i_in = [[0, 1]]

    # na = len(w_in)
    ns = sum(len(ww) for ww in w_in)
    n0 = max(i for ii in i_in for i in ii)
    assert len(w_in) == len(i_in)
    for ww, ii in zip(w_in, i_in):
        assert len(ww) == len(ii)

    # --- board
    boardId = 1
    numChips = 1
    # Number of cores per chip
    numCoresPerChip = [1]
    # Number of synapses per core
    numSynapsesPerCore = [[ns]]
    # Initialize the board
    board = N2Board(boardId, numChips, numCoresPerChip, numSynapsesPerCore)

    chip = board.n2Chips[0]
    core0 = chip.n2Cores[0]

    # --- core0
    decayU = 0
    decayV = 0

    vth = 1001

    core0.cxProfileCfg[0].configure(decayV=decayV, decayU=decayU)
    core0.vthProfileCfg[0].staticCfg.configure(vth=vth)
    core0.dendriteSharedCfg.configure(posVmLimit=7, negVmLimit=0)

    core0.synapseFmt[1].wgtExp = -2
    core0.synapseFmt[1].wgtBits = 7
    core0.synapseFmt[1].numSynapses = 63
    core0.synapseFmt[1].idxBits = 7
    core0.synapseFmt[1].compression = 0
    core0.synapseFmt[1].fanoutType = 1

    s0 = 0
    for a, (wa, ia) in enumerate(zip(w_in, i_in)):
        for k, (w, i) in enumerate(zip(wa, ia)):
            core0.synapses[s0+k].CIdx = i
            core0.synapses[s0+k].Wgt = w
            core0.synapses[s0+k].synFmtId = 1

        core0.synapseMap[a].synapsePtr = s0
        core0.synapseMap[a].synapseLen = len(wa)
        core0.synapseMap[a].discreteMapEntry.configure()

        s0 += len(wa)

    for i in range(n0):
        core0.cxCfg[i].configure(bias=0, biasExp=0, vthProfile=0, cxProfile=0)

    core0.numUpdates.configure(numUpdates=n0//4 + 1)

    return board


if __name__ == '__main__':
    board = setupNetwork()
    chip = board.n2Chips[0]
    core0 = chip.n2Cores[0]

    mon = board.monitor
    u0p = mon.probe(core0.cxState, [0, 1, 2, 3], 'u')
    v0p = mon.probe(core0.cxState, [0, 1, 2, 3], 'v')

    tsteps = 5

    sgen = BasicSpikeGenerator(board)
    for t in range(1, tsteps, 3):
        sgen.addSpike(t, chip.id, core0.id, axonId=0)

    board.run(tsteps)

    for i in range(len(u0p)):
        print("Cx[%d] U: %s" % (i, u0p[i].timeSeries.data))
        print("Cx[%d] V: %s" % (i, v0p[i].timeSeries.data))
