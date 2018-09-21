"""
INTEL CONFIDENTIAL

Copyright © 2018 Intel Corporation.

This software and the related documents are Intel copyrighted
materials, and your use of them is governed by the express
license under which they were provided to you (License). Unless
the License provides otherwise, you may not use, modify, copy,
publish, distribute, disclose or transmit  this software or the
related documents without Intel's prior written permission.

This software and the related documents are provided as is, with
no express or implied warranties, other than those that are
expressly stated in the License.
"""

# -----------------------------------------------------------------------------
# Tutorial: tutorial_07_population_connectivity.py
# -----------------------------------------------------------------------------
# This tutorial illustrates the use of the population-mode connectivity
# feature which allows to share connectivity resources among multiple neurons
# and thus achieves to compress networks with repeated connectivity structure.
# In particular, this is achieved by considering compartments as members of
# populations, assigning compartment ids relative to the population and
# specifying connectivity between populations. The assignment of relative
# compartments ids then allows to share the same axon from source to
# destination population among all compartments of a source population as
# well as to share the fanout synapses for a specific relative compartment id.
# In this minimal example, we construct a neural network of two source and
# two destination populations, all of them containing just 2 compartments to
# illustrate multiplicity of all involved entities (source/destination
# populations, compartments, axons and synapses).
# The source population 0 connects via a shared axon to a shared set of
# synapses which connect in turn to destination population 0. Similarly
# source population 1 connects via another shared axon to the same shared set
# of synapses but to destination population 1.
# srcPop0 -> Shared popAxon0 --                 -- dstPop0
#                              \               /
#                               Shared synapses
#                              /               \
# srcPop1 -> Shared popAxon1 --                 -- dstPop1

# -----------------------------------------------------------------------------
# Import modules
# -----------------------------------------------------------------------------
import os
import matplotlib as mpl
haveDisplay = "DISPLAY" in os.environ
if not haveDisplay:
    mpl.use('Agg')
import matplotlib.pyplot as plt
from nxsdk.arch.n2a.graph.graph import N2Board
from nxsdk.arch.n2a.graph.nodesets.output_axon import OutputAxon


COMPILE = True


def setupNetwork(populationMsgType=32):

    # --------------------------------------------------------------------------
    # Import modules
    # --------------------------------------------------------------------------
    # N2Board ID
    boardId = 1
    # Number of chips
    numChips = 1
    # Number of cores per chip
    numCoresPerChip = [1]
    # Number of synapses per core
    numSynapsesPerCore = [[4]]
    # Initialize the board
    board = N2Board(boardId, numChips, numCoresPerChip, numSynapsesPerCore)
    # Obtain the relevant core (only one in this example)
    n2Core = board.n2Chips[0].n2Cores[0]

    # --------------------------------------------------------------------------
    # Configure shared compartment profiles
    # --------------------------------------------------------------------------
    # Configure decay constant for compartment's input current u
    tauU = 10
    n2Core.cxProfileCfg[0].configure(decayU = int(1/tauU*2**12))
    # Configure membrane potential threshold
    n2Core.vthProfileCfg[0].staticCfg.configure(vth = 40)
    # We use compartments 0, 1, 2, 3, 4, 5, 8, 9 thus, we need to update 3
    # compartments groups
    n2Core.numUpdates.configure(numUpdates = 3)

    # --------------------------------------------------------------------------
    # Configure connectivity using either short 16 bit or long 32 bit
    # population messages. 16 bit messages are more efficient but impose
    # trade-offs between the maximum size of a source population and the
    # maximum number of destination populations per core as 16 bit messages
    # contain 4 bits that can be either allocated to configuring one or the
    # other.
    # --------------------------------------------------------------------------
    if populationMsgType == 16:
        # Configure output population axons:

        # In contrast to non-population-mode connectivity, the axonMap entry
        # corresponding to each source compartment must specify an atom number
        # which is the relative compartment id with respect to the source
        # population. This information will be transmitted with a population
        # spike message and is used to identify the set of fanout synapses
        # connecting to a specific relative source compartment. All axonMap
        # entries of a source population will therefore point to the same
        # axonCfg entry as the output axon is shared among all source
        # compartments. 16 bit population messages require only a single
        # axonCfg entry thus len is 1

        if COMPILE:
            n2Core.axons.append(OutputAxon.pop16Axon(
                srcCxId=0, srcRelCxId=0, dstChipId=0, dstCoreId=4, dstSynMapId=0))
            n2Core.axons.append(OutputAxon.pop16Axon(
                srcCxId=1, srcRelCxId=1, dstChipId=0, dstCoreId=4, dstSynMapId=0))
            n2Core.axons.append(OutputAxon.pop16Axon(
                srcCxId=2, srcRelCxId=0, dstChipId=0, dstCoreId=4, dstSynMapId=1))
            n2Core.axons.append(OutputAxon.pop16Axon(
                srcCxId=3, srcRelCxId=1, dstChipId=0, dstCoreId=4, dstSynMapId=1))
        else:
            # Configure source population 0
            n2Core.axonMap[0].configure(ptr=0, len=1, atom=0)
            n2Core.axonMap[1].configure(ptr=0, len=1, atom=1)
            # Configure source population 1
            n2Core.axonMap[2].configure(ptr=1, len=1, atom=0)
            n2Core.axonMap[3].configure(ptr=1, len=1, atom=1)
            # Configure shared output axons
            n2Core.axonCfg[0].pop16.configure(coreId=4, axonId=0)
            n2Core.axonCfg[1].pop16.configure(coreId=4, axonId=1)

        # Configure input population axons:
        # Configure shared input axon connecting to destination population 0
        n2Core.synapseMap[0].synapsePtr = 0
        n2Core.synapseMap[0].synapseLen = 2
        # popSize specifies the size of the source population
        n2Core.synapseMap[0].popSize = 2
        # pop16: As a compression feature, the user-defined cxBase value is
        # mapped to -> 4*cxBase, therefore we use cxBase = 1 which
        # implies that the destination compartments of dstPop0 will be at 4..5
        n2Core.synapseMap[0].population16MapEntry.configure(cxBase = 1, atomBits=0)
        # Configure shared input axon connecting to destination population 1
        n2Core.synapseMap[1].synapsePtr = 0
        n2Core.synapseMap[1].synapseLen = 2
        n2Core.synapseMap[1].popSize = 2
        # pop16: cxBase -> 4*cxBase: dstPop1 will be at 8..9
        n2Core.synapseMap[1].population16MapEntry.configure(cxBase = 2, atomBits=0)
    elif populationMsgType == 32:
        # Configure output population axons:

        # 32 bit population messages are more flexible and require 2 axonCfg
        # entries per output axon. Therefore axonMap.len must be 2 and ptr
        # must be 0 and 2

        if COMPILE:
            n2Core.axons.append(OutputAxon.pop32Axon(
                srcCxId=0, srcRelCxId=0, dstChipId=0, dstCoreId=4, dstSynMapId=0))
            n2Core.axons.append(OutputAxon.pop32Axon(
                srcCxId=1, srcRelCxId=1, dstChipId=0, dstCoreId=4, dstSynMapId=0))
            n2Core.axons.append(OutputAxon.pop32Axon(
                srcCxId=2, srcRelCxId=0, dstChipId=0, dstCoreId=4, dstSynMapId=1))
            n2Core.axons.append(OutputAxon.pop32Axon(
                srcCxId=3, srcRelCxId=1, dstChipId=0, dstCoreId=4, dstSynMapId=1))
        else:
            # Configure source population 0
            n2Core.axonMap[0].configure(ptr=0, len=2, atom=0)
            n2Core.axonMap[1].configure(ptr=0, len=2, atom=1)
            # Configure source population 1
            n2Core.axonMap[2].configure(ptr=2, len=2, atom=0)
            n2Core.axonMap[3].configure(ptr=2, len=2, atom=1)
            # Configure shared output axons. Only the 0-th pop32 entry requires
            # explicit user input. the 1-st pop32 entry must use default values
            n2Core.axonCfg[0].pop32_0.configure(coreId=4, axonId=0)
            n2Core.axonCfg[1].pop32_1.configure()
            n2Core.axonCfg[2].pop32_0.configure(coreId=4, axonId=1)
            n2Core.axonCfg[3].pop32_1.configure()

        # Configure input population axon connecting to destination population 0
        n2Core.synapseMap[0].synapsePtr = 0
        n2Core.synapseMap[0].synapseLen = 2
        n2Core.synapseMap[0].popSize = 2
        n2Core.synapseMap[0].population32MapEntry.configure(cxBase=4)
        # Configure input population axon connecting to destination population 1
        n2Core.synapseMap[1].synapsePtr = 0
        n2Core.synapseMap[1].synapseLen = 2
        n2Core.synapseMap[1].popSize = 2
        n2Core.synapseMap[1].population32MapEntry.configure(cxBase=8)
    else:
        raise AssertionError('Illegal populationMsgType.')

    # --------------------------------------------------------------------------
    # Configure synapses that are shared by the two pairs of
    # source/destination populations.
    # In order to connect both pairs of source/destination populations with
    # the same set of synapses, we assume the following weight matrix:
    #     [ 4,  8
    #      -4, -8]
    # The first input axon has weights 4, -4 and the second input axon has
    # weights 8, -8. Hence we expect excitation/inhibition in the two
    # destination compartments the two axons connect to by different amounts.
    #  These weights correspond to the following positional synapse ids:
    #     [0, 2
    #      1, 3]
    # Note: Whereas we use absolute compartment ids for non-population-mode
    # connectivity, we use relative destination compartment ids in
    # population-mode with respect to the destination population because
    # synapses are shared and thus connect to multiple compartments.
    # --------------------------------------------------------------------------
    # Configure four synapses with ids 0..3 and weights 4, -4, 8, -8 all
    # using same synapseFmt
    n2Core.synapses[0].CIdx = 0
    n2Core.synapses[0].Wgt = 2
    n2Core.synapses[0].synFmtId = 1
    n2Core.synapses[1].CIdx = 1
    n2Core.synapses[1].Wgt = -4
    n2Core.synapses[1].synFmtId = 1
    n2Core.synapses[2].CIdx = 0
    n2Core.synapses[2].Wgt = 8
    n2Core.synapses[2].synFmtId = 1
    n2Core.synapses[3].CIdx = 1
    n2Core.synapses[3].Wgt = -6
    n2Core.synapses[3].synFmtId = 1
    # Configure a synapseFormat
    n2Core.synapseFmt[1].wgtBits = 7
    n2Core.synapseFmt[1].numSynapses = 63
    n2Core.synapseFmt[1].idxBits = 1
    n2Core.synapseFmt[1].compression = 3
    n2Core.synapseFmt[1].fanoutType = 1

    return board


def runNetwork(board, doPlot):

    # Retrieve the only core we need for experiment
    n2Core = board.n2Chips[0].n2Cores[0]

    # --------------------------------------------------------------------------
    # Configure probes
    # --------------------------------------------------------------------------
    mon = board.monitor
    # Configure u and v probes for both source populations
    uProbesSrcPop0 = mon.probe(n2Core.cxState, range(0, 2), 'u')
    vProbesSrcPop0 = mon.probe(n2Core.cxState, range(0, 2), 'v')
    uProbesSrcPop1 = mon.probe(n2Core.cxState, range(2, 4), 'u')
    vProbesSrcPop1 = mon.probe(n2Core.cxState, range(2, 4), 'v')
    # Configure u and v probes for both destination populations
    uProbesDstPop0 = mon.probe(n2Core.cxState, range(4, 6), 'u')
    vProbesDstPop0 = mon.probe(n2Core.cxState, range(4, 6), 'v')
    uProbesDstPop1 = mon.probe(n2Core.cxState, range(8, 10), 'u')
    vProbesDstPop1 = mon.probe(n2Core.cxState, range(8, 10), 'v')

    # --------------------------------------------------------------------------
    # Run simulation in 4 subsequent phases. In each phase only one of the
    # two source compartments in the two source populations spike due to a
    # driving bias current. This allows to validate that the spikes of 4
    # distinct source compartments passes through the same set of shared
    # synapses and arrives at the corresponding destination compartments.
    # --------------------------------------------------------------------------
    # Execute network for some number of steps in each phase
    numStepsPerRun = 11
    # Activate automatic sync so that network reconfigurations are
    # immediately synced with hardware between runs
    board.sync = True

    # Drive compartment 0 in srcPop0 to spike and run network
    n2Core.cxMetaState[0].configure(phase0=2)
    #     Source population 0
    n2Core.cxCfg[0].configure(bias=4, biasExp=6)
    n2Core.cxCfg[1].configure(bias=0, biasExp=0)
    #     Source population 1
    n2Core.cxCfg[2].configure(bias=0, biasExp=0)
    n2Core.cxCfg[3].configure(bias=0, biasExp=0)
    # Run
    board.run(numStepsPerRun)

    # Drive compartment 1 in srcPop0 to spike and run network
    n2Core.cxState[0].v = 0
    n2Core.cxMetaState[0].configure(phase1=2)
    #     Source population 0
    n2Core.cxCfg[0].configure(bias=0, biasExp=0)
    n2Core.cxCfg[1].configure(bias=4, biasExp=6)
    #     Source population 1
    n2Core.cxCfg[2].configure(bias=0, biasExp=0)
    n2Core.cxCfg[3].configure(bias=0, biasExp=0)
    # Run
    board.run(numStepsPerRun)

    # Drive compartment 2 in srcPop1 to spike and run network
    n2Core.cxState[1].v = 0
    n2Core.cxMetaState[0].configure(phase2=2)
    #     Source population 0
    n2Core.cxCfg[0].configure(bias=0, biasExp=0)
    n2Core.cxCfg[1].configure(bias=0, biasExp=0)
    #     Source population 1
    n2Core.cxCfg[2].configure(bias=4, biasExp=6)
    n2Core.cxCfg[3].configure(bias=0, biasExp=0)
    # Run
    board.run(numStepsPerRun)

    # Drive compartment 3 in srcPop1 to spike and run network
    n2Core.cxState[2].v = 0
    n2Core.cxMetaState[0].configure(phase3=2)
    #     Source population 0
    n2Core.cxCfg[0].configure(bias=0, biasExp=0)
    n2Core.cxCfg[1].configure(bias=0, biasExp=0)
    #     Source population 1
    n2Core.cxCfg[2].configure(bias=0, biasExp=0)
    n2Core.cxCfg[3].configure(bias=4, biasExp=6)
    # Run
    board.run(numStepsPerRun)
    board.run(numStepsPerRun)

    # --------------------------------------------------------------------------
    # Plot
    # The array of graphs illustrates the subsequent generation of source
    # spikes for all source compartments for all source populations. These
    # spikes are then routed to their destinations. srcPop0 only stimulates
    # dstPop0 while srcPop1 only excites dstPop1 but through the same same
    # set of weights as the same excitation/inhibition signature of
    # compartments 4, 5 and 8, 9 illustrate.
    # --------------------------------------------------------------------------
    if doPlot:
        fig = plt.figure(1007)
        # Source population 0
        plt.subplot(4, 4, 1)
        uProbesSrcPop0[0].plot()
        plt.title('u0[srcPop0]')
        plt.subplot(4, 4, 2)
        vProbesSrcPop0[0].plot()
        plt.title('v0[srcPop0]')
        plt.subplot(4, 4, 5)
        uProbesSrcPop0[1].plot()
        plt.title('u1[srcPop0]')
        plt.subplot(4, 4, 6)
        vProbesSrcPop0[1].plot()
        plt.title('v1[srcPop0]')
        # Destination population 0
        plt.subplot(4, 4, 3)
        uProbesDstPop0[0].plot()
        plt.title('u0[dstPop0]')
        plt.subplot(4, 4, 4)
        vProbesDstPop0[0].plot()
        plt.title('v0[dstPop0]')
        plt.subplot(4, 4, 7)
        uProbesDstPop0[1].plot()
        plt.title('u1[dstPop0]')
        plt.subplot(4, 4, 8)
        vProbesDstPop0[1].plot()
        plt.title('v1[dstPop0]')

        # Source population 1
        plt.subplot(4, 4, 9)
        uProbesSrcPop1[0].plot()
        plt.title('u0[srcPop1]')
        plt.subplot(4, 4, 10)
        vProbesSrcPop1[0].plot()
        plt.title('v0[srcPop1]')
        plt.subplot(4, 4, 13)
        uProbesSrcPop1[1].plot()
        plt.title('u1[srcPop1]')
        plt.subplot(4, 4, 14)
        vProbesSrcPop1[1].plot()
        plt.title('v1[srcPop1]')
        # Destination population 1
        plt.subplot(4, 4, 11)
        uProbesDstPop1[0].plot()
        plt.title('u0[dstPop1]')
        plt.subplot(4, 4, 12)
        vProbesDstPop1[0].plot()
        plt.title('v0[dstPop1]')
        plt.subplot(4, 4, 15)
        uProbesDstPop1[1].plot()
        plt.title('u1[dstPop1]')
        plt.subplot(4, 4, 16)
        vProbesDstPop1[1].plot()
        plt.title('v1[dstPop1]')

        if haveDisplay:
            plt.show()
        else:
            fileName = "tutorial_07_fig1007.png"
            print("No display available, saving to file " + fileName + ".")
            fig.savefig(fileName)


if __name__ == '__main__':
    # --------------------------------------------------------------------------
    # Setup the network
    # --------------------------------------------------------------------------
    board = setupNetwork(populationMsgType=16)

    # --------------------------------------------------------------------------
    # Run the network and plot results
    # --------------------------------------------------------------------------
    runNetwork(board, True)
