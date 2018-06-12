import os

import numpy as np

from nxsdk.arch.n2a.graph.graph import N2Board
from nxsdk.arch.n2a.graph.inputgen import BasicSpikeGenerator

n = 10


def setupNetwork():
    boardId = 1
    numChips = 1
    numCoresPerChip = [1]
    numSynapsesPerCore = [[0]]
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
    core0.dendriteAccumCfg.configure(delayBits=3)

    for i in range(n):
        core0.cxCfg[i].configure(bias=1000, biasExp=0)

        phasex = 'phase%d' % (i % 4,)
        core0.cxMetaState[i//4].configure(**{phasex: 2})

    core0.numUpdates.configure(numUpdates=n//4 + 1)

    return board


if __name__ == '__main__':
    board = setupNetwork()
    chip = board.n2Chips[0]
    core0 = chip.n2Cores[0]

    mon = board.monitor
    v0p = mon.probe(core0.cxState, range(n), 'v')

    tsteps = 5

    board.run(tsteps)

    v = np.column_stack([vp.timeSeries.data for vp in v0p])
    print(v)
