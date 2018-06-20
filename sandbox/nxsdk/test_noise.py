import numpy as np

from nxsdk.arch.n2a.graph.graph import N2Board

n = 256


def setupNetwork():
    boardId = 1
    numChips = 1
    numCoresPerChip = [1]
    numSynapsesPerCore = [[0]]
    board = N2Board(boardId, numChips, numCoresPerChip, numSynapsesPerCore)
    core0 = board.n2Chips[0].n2Cores[0]

    core0.cxProfileCfg[0].configure(
        enableNoise=1, decayV=int(2**12-1), decayU=int(2**12-1))
    core0.vthProfileCfg[0].staticCfg.configure(vth=100000)
    core0.dendriteSharedCfg.configure(noiseExp0=6,
                                        noiseMantOffset0=0,
                                        noiseAtDendOrVm=1)

    for i in range(n):
        core0.cxCfg[i].configure(bias=0, biasExp=0, vthProfile=0, cxProfile=0)

        phasex = 'phase%d' % (i % 4,)
        core0.cxMetaState[i//4].configure(**{phasex: 2})

    core0.numUpdates.configure(numUpdates=(n // 4) + 1)

    return board


if __name__ == '__main__':
    board = setupNetwork()
    core0 = board.n2Chips[0].n2Cores[0]

    mon = board.monitor

    u0Probe = mon.probe(core0.cxState, range(n), 'u')
    v0Probe = mon.probe(core0.cxState, range(n), 'v')

    board.run(100)

    u = np.column_stack([p.timeSeries.data for p in u0Probe])
    v = np.column_stack([p.timeSeries.data for p in v0Probe])

    print("u: [%d, %d] %0.3f %0.3f" % (
        u.min(), u.max(), u.mean(), u.std()))
    print("v: [%d, %d] %0.3f %0.3f" % (
        v.min(), v.max(), v.mean(), v.std()))
