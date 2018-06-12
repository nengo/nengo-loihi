import os

import numpy as np

from nxsdk.arch.n2a.graph.graph import N2Board
from nxsdk.arch.n2a.graph.inputgen import BasicSpikeGenerator

n = 2


def setupNetwork():
    w_conn = 100 * np.eye(n, n).astype(int)
    i_conn = [list(range(len(w))) for w in w_conn]

    n0 = len(w_conn)
    n1 = len(w_conn[0])

    # --- board
    boardId = 1
    numChips = 1
    numCoresPerChip = [2]
    numSynapsesPerCore = [[n0, n0*n1]]
    board = N2Board(boardId, numChips, numCoresPerChip, numSynapsesPerCore)

    chip = board.n2Chips[0]
    core0 = chip.n2Cores[0]
    core1 = chip.n2Cores[1]

    # --- core0
    decayU = 273
    decayV = 197
    vth = 800

    core0.cxProfileCfg[0].configure(decayV=decayV, decayU=decayU)
    core0.vthProfileCfg[0].staticCfg.configure(vth=vth)
    core0.dendriteSharedCfg.configure(posVmLimit=7, negVmLimit=0)

    for i in range(n0):
        core0.cxCfg[i].configure(bias=100, biasExp=6, vthProfile=0, cxProfile=0)
        core0.cxMetaState[i//4].configure(**{'phase%d' % (i % 4): 2})

        core0.axonMap[i].configure(ptr=i, len=1)
        core0.axonCfg[i].discrete.configure(coreId=core1.id, axonId=i)

    core0.numUpdates.configure(numUpdates=n0//4 + 1)

    # --- core1
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
        core1.cxCfg[i].configure(bias=0, biasExp=0, vthProfile=0, cxProfile=0)
        core1.cxMetaState[i//4].configure(**{'phase%d' % (i % 4): 2})

    core1.numUpdates.configure(numUpdates=n1//4 + 1)

    # Return the configured board
    return board


if __name__ == '__main__':
    board = setupNetwork()
    chip = board.n2Chips[0]
    core0 = chip.n2Cores[0]
    core1 = chip.n2Cores[1]

    mon = board.monitor
    u0p = mon.probe(core0.cxState, range(n), 'u')
    v0p = mon.probe(core0.cxState, range(n), 'v')
    s0p = mon.probe(core0.cxState, range(n), 'spike')
    u1p = mon.probe(core1.cxState, range(n), 'u')
    v1p = mon.probe(core1.cxState, range(n), 'v')

    tsteps = 15

    board.run(tsteps)

    print(u0p[0].timeSeries.data)
    print(v0p[0].timeSeries.data)
    print(u1p[0].timeSeries.data)
    print(v1p[0].timeSeries.data)
