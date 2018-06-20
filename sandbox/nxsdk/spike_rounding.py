import os

from nxsdk.arch.n2a.graph.graph import N2Board
from nxsdk.arch.n2a.graph.inputgen import BasicSpikeGenerator


def setupNetwork():

    w_in = [-2, 2, 3, 4]
    i_in = list(range(len(w_in)))
    n0 = len(w_in)

    # --- board
    boardId = 1
    numChips = 1
    # Number of cores per chip
    numCoresPerChip = [1]
    # Number of synapses per core
    numSynapsesPerCore = [[n0]]
    # Initialize the board
    board = N2Board(boardId, numChips, numCoresPerChip, numSynapsesPerCore)

    chip = board.n2Chips[0]
    core0 = chip.n2Cores[0]

    # --- core0
    decayU = 0
    decayV = 0

    # decayU = 273
    # decayV = 197
    vth = 1001

    core0.cxProfileCfg[0].configure(decayV=decayV, decayU=decayU)
    core0.vthProfileCfg[0].staticCfg.configure(vth=vth)
    core0.dendriteSharedCfg.configure(posVmLimit=7, negVmLimit=0)

    core0.synapseFmt[1].wgtExp = 0
    core0.synapseFmt[1].wgtBits = 7
    core0.synapseFmt[1].numSynapses = 63
    core0.synapseFmt[1].idxBits = 7
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

    core0.numUpdates.configure(numUpdates=1)

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
