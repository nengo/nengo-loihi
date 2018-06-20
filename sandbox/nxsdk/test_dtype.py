import os

import numpy as np

from nxsdk.arch.n2a.graph.graph import N2Board
from nxsdk.arch.n2a.graph.inputgen import BasicSpikeGenerator


def setupNetwork():
    # --- board
    boardId = 1
    numChips = 1
    numCoresPerChip = [1]
    numSynapsesPerCore = [[0]]
    board = N2Board(boardId, numChips, numCoresPerChip, numSynapsesPerCore)

    chip = board.n2Chips[0]
    core0 = chip.n2Cores[0]

    # for i in range(5):
    for i in np.arange(5):
        core0.cxCfg[i].configure(bias=0, biasExp=0, vthProfile=0, cxProfile=0)

    return board


if __name__ == '__main__':
    board = setupNetwork()
    chip = board.n2Chips[0]
    core0 = chip.n2Cores[0]

    mon = board.monitor
    u0p = mon.probe(core0.cxState, [0, 1], 'u')
    v0p = mon.probe(core0.cxState, [0, 1], 'v')

    tsteps = 15

    board.run(tsteps)

    print(u0p[0].timeSeries.data)
    print(v0p[0].timeSeries.data)
