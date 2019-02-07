from __future__ import division

import logging

from nengo.utils.stdlib import groupby
import numpy as np

from nengo_loihi.discretize import bias_to_manexp
from nengo_loihi.hardware.nxsdk_objects import (
    CX_PROFILES_MAX,
    LoihiSpikeInput,
    VTH_PROFILES_MAX,
)
from nengo_loihi.hardware.nxsdk_shim import (
    BasicSpikeGenerator,
    microcodegen_uci,
    N2Board,
    TraceCfgGen,
)
from nengo_loihi.inputs import SpikeInput

logger = logging.getLogger(__name__)


def build_board(board):
    n_chips = board.n_chips()
    n_cores_per_chip = board.n_cores_per_chip()
    n_synapses_per_core = board.n_synapses_per_core()
    n2board = N2Board(
        board.board_id, n_chips, n_cores_per_chip, n_synapses_per_core)

    # add our own attribute for storing our spike generator
    assert not hasattr(n2board, 'global_spike_generator')
    n2board.global_spike_generator = BasicSpikeGenerator(n2board)

    # custom attr for storing SpikeInputs (filled in build_input)
    assert not hasattr(n2board, 'spike_inputs')
    n2board.spike_inputs = {}

    # build all chips
    assert len(board.chips) == len(n2board.n2Chips)
    for chip, n2chip in zip(board.chips, n2board.n2Chips):
        logger.debug("Building chip %s", chip)
        build_chip(n2chip, chip)

    return n2board


def build_chip(n2chip, chip):
    assert len(chip.cores) == len(n2chip.n2Cores)
    for core, n2core in zip(chip.cores, n2chip.n2Cores):
        logger.debug("Building core %s", core)
        build_core(n2core, core)


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
        if synapse.learning and firstLearningIndex is None:
            firstLearningIndex = core.synapse_axons[synapse][0]
            core.learning_coreid = n2core.id
            break

    numStdp = 0
    if firstLearningIndex is not None:
        for synapse in core.iterate_synapses():
            axons = np.array(core.synapse_axons[synapse])
            if synapse.learning:
                numStdp += len(axons)
                assert np.all(axons >= firstLearningIndex)
            else:
                assert np.all(axons < firstLearningIndex)

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

        # Microcode for the learning rule. `u1` evaluates the learning rule
        # every 2**1 timesteps, `x1` is the pre-trace, `y1` is the post-trace,
        # and 2^-7 is the learning rate. See `help(ruleToUCode)` for more info.
        ucode = microcodegen_uci.ruleToUCode(
            ['dw = u1*x1*y1*(2^-7)'], doOptimize=False)
        assert ucode.numUCodes == 1
        n2core.stdpUcodeMem[0].word = ucode.uCodes[0]

        # stdpProfileCfg negative error
        n2core.stdpProfileCfg[1].configure(
            uCodePtr=1,
            decimateExp=0,
            numProducts=1,
            requireY=1,
            usesXepoch=1,
        )
        # use negative version of above microcode rule
        ucode = microcodegen_uci.ruleToUCode(
            ['dw = -u1*x1*y1*(2^-7)'], doOptimize=False)
        assert ucode.numUCodes == 1
        n2core.stdpUcodeMem[1].word = ucode.uCodes[0]

        tcg = TraceCfgGen()
        tc = tcg.genTraceCfg(
            tau=0,
            spikeLevelInt=0,
            spikeLevelFrac=0,
        )
        tc.writeToRegister(n2core.stdpPostCfg[0])

    # TODO: allocator should be checking that vmin, vmax are the same
    #   for all blocks on a core
    n_cx = 0
    if len(core.blocks) > 0:
        block0 = core.blocks[0]
        vmin, vmax = block0.compartment.vmin, block0.compartment.vmax
        assert all(block.compartment.vmin == vmin
                   for block in core.blocks)
        assert all(block.compartment.vmax == vmax
                   for block in core.blocks)
        negVmLimit = np.log2(-vmin + 1)
        posVmLimit = (np.log2(vmax + 1) - 9) * 0.5
        assert int(negVmLimit) == negVmLimit
        assert int(posVmLimit) == posVmLimit

        noiseExp0 = block0.compartment.noiseExp0
        noiseMantOffset0 = block0.compartment.noiseMantOffset0
        noiseAtDendOrVm = block0.compartment.noiseAtDendOrVm
        assert all(block.compartment.noiseExp0 == noiseExp0
                   for block in core.blocks)
        assert all(block.compartment.noiseMantOffset0 == noiseMantOffset0
                   for block in core.blocks)
        assert all(block.compartment.noiseAtDendOrVm == noiseAtDendOrVm
                   for block in core.blocks)

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

        for block, cx_idxs, ax_range in core.iterate_blocks():
            build_block(n2core, core, block, cx_idxs, ax_range)
            n_cx = max(max(cx_idxs) + 1, n_cx)

    for inp, cx_idxs in core.iterate_inputs():
        build_input(n2core, core, inp, cx_idxs)

    logger.debug("- Configuring numUpdates=%d", n_cx // 4 + 1)
    n2core.numUpdates.configure(
        numUpdates=n_cx // 4 + 1,
        numStdp=numStdp,
    )

    n2core.dendriteTimeState[0].tepoch = 2
    n2core.timeState[0].tepoch = 2


def build_block(n2core, core, block, cx_idxs, ax_range):
    assert block.compartment.scaleU is False
    assert block.compartment.scaleV is False

    logger.debug("Building %s on core.id=%d", block, n2core.id)

    for i, bias in enumerate(block.compartment.bias):
        bman, bexp = bias_to_manexp(bias)
        icx = core.cx_profile_idxs[block][i]
        ivth = core.vth_profile_idxs[block][i]

        ii = cx_idxs[i]
        n2core.cxCfg[ii].configure(
            bias=bman, biasExp=bexp, vthProfile=ivth, cxProfile=icx)

        phasex = 'phase%d' % (ii % 4,)
        n2core.cxMetaState[ii // 4].configure(**{phasex: 2})

    logger.debug("- Building %d synapses", len(block.synapses))
    for synapse in block.synapses:
        build_synapse(n2core, core, block, synapse, cx_idxs)

    logger.debug("- Building %d axons", len(block.axons))
    all_axons = []  # (cx, atom, type, tchip_id, tcore_id, taxon_id)
    for axon in block.axons:
        all_axons.extend(collect_axons(n2core, core, block, axon, cx_idxs))

    build_axons(n2core, core, block, all_axons)

    logger.debug("- Building %d probes", len(block.probes))
    for probe in block.probes:
        build_probe(n2core, core, block, probe, cx_idxs)


def build_input(n2core, core, spike_input, cx_idxs):
    assert len(spike_input.axons) > 0

    for probe in spike_input.probes:
        build_probe(n2core, core, spike_input, probe, cx_idxs)

    n2board = n2core.parent.parent

    assert isinstance(spike_input, SpikeInput)
    loihi_input = LoihiSpikeInput()
    loihi_input.set_axons(core.board, n2board, spike_input)
    assert spike_input not in n2board.spike_inputs
    n2board.spike_inputs[spike_input] = loihi_input

    # add any pre-existing spikes to spikegen
    for t in spike_input.spike_times():
        spikes = spike_input.spike_idxs(t)
        for spike in loihi_input.spikes_to_loihi(t, spikes):
            assert spike.axon.atom == 0, (
                "Cannot send population spikes through spike generator")
            n2board.global_spike_generator.addSpike(
                time=spike.time, chipId=spike.axon.chip_id,
                coreId=spike.axon.core_id, axonId=spike.axon.axon_id)


def build_synapse(n2core, core, block, synapse, cx_idxs):  # noqa C901
    axon_ids = core.synapse_axons[synapse]

    synapse_fmt_idx = core.synapse_fmt_idxs[synapse]
    stdp_pre_cfg_idx = core.stdp_pre_cfg_idxs[synapse]

    atom_bits = synapse.atom_bits()
    axon_bits = synapse.axon_bits()
    atom_bits_extra = synapse.atom_bits_extra()

    target_cxs = set()
    synapse_map = {}  # map weight_idx to (ptr, pop_size, len)
    total_synapse_ptr = int(core.synapse_entries[synapse][0])
    for axon_idx, axon_id in enumerate(axon_ids):
        assert axon_id <= 2**axon_bits

        weight_idx = int(synapse.axon_weight_idx(axon_idx))
        cx_base = synapse.axon_cx_base(axon_idx)

        if weight_idx not in synapse_map:
            weights = synapse.weights[weight_idx]
            indices = synapse.indices[weight_idx]
            weights = weights // synapse.synapse_fmt.scale
            assert weights.ndim == 2
            assert weights.shape == indices.shape
            assert np.all(weights <= 255) and np.all(weights >= -256), str(
                weights)
            n_populations, n_cxs = weights.shape

            synapse_map[weight_idx] = (
                total_synapse_ptr, n_populations, n_cxs)

            for p in range(n_populations):
                for q in range(n_cxs):
                    cx_idx = cx_idxs[indices[p, q]]
                    n2core.synapses[total_synapse_ptr].configure(
                        CIdx=cx_idx,
                        Wgt=weights[p, q],
                        synFmtId=synapse_fmt_idx,
                        LrnEn=int(synapse.learning),
                    )
                    target_cxs.add(cx_idx)
                    total_synapse_ptr += 1

        synapse_ptr, n_populations, n_cxs = synapse_map[weight_idx]
        assert n_populations <= 2**atom_bits

        if cx_base is None:
            # this is a dummy axon with no weights, so set n_cxs to 0
            synapse_ptr = 0
            n_cxs = 0
            cx_base = 0
        else:
            cx_base = int(cx_base)

        assert cx_base <= 256, "Currently limited by hardware"
        n2core.synapseMap[axon_id].synapsePtr = synapse_ptr
        n2core.synapseMap[axon_id].synapseLen = n_cxs
        if synapse.pop_type == 0:  # discrete
            assert n_populations == 1
            n2core.synapseMap[axon_id].discreteMapEntry.configure(
                cxBase=cx_base)
        elif synapse.pop_type == 16:  # pop16
            n2core.synapseMap[axon_id].popSize = n_populations
            assert cx_base % 4 == 0
            n2core.synapseMap[axon_id].population16MapEntry.configure(
                cxBase=cx_base//4, atomBits=atom_bits_extra)
        elif synapse.pop_type == 32:  # pop32
            n2core.synapseMap[axon_id].popSize = n_populations
            n2core.synapseMap[axon_id].population32MapEntry.configure(
                cxBase=cx_base)
        else:
            raise ValueError("Synapse: unrecognized pop_type: %s" % (
                synapse.pop_type,))

        if synapse.learning:
            assert core.stdp_pre_profile_idx is not None
            assert stdp_pre_cfg_idx is not None
            n2core.synapseMap[axon_id+1].singleTraceEntry.configure(
                preProfile=core.stdp_pre_profile_idx, tcs=stdp_pre_cfg_idx)

    assert total_synapse_ptr == core.synapse_entries[synapse][1], (
        "Synapse pointer did not align with precomputed synapse length")

    if synapse.learning:
        assert core.stdp_profile_idx is not None
        for target_cx in target_cxs:
            # TODO: check that no cx gets configured by multiple synapses
            n2core.stdpPostState[target_cx].configure(
                stdpProfile=core.stdp_profile_idx,
                traceProfile=3,  # TODO: why this value
            )


def collect_axons(n2core, core, block, axon, cx_ids):
    synapse = axon.target
    tchip_idx, tcore_idx, tsyn_idxs = core.board.find_synapse(synapse)
    n2board = n2core.parent.parent
    tchip_id = n2board.n2Chips[tchip_idx].id
    tcore_id = n2board.n2Chips[tchip_idx].n2Cores[tcore_idx].id

    cx_idxs = np.arange(len(cx_ids))
    spikes = axon.map_cx_spikes(cx_idxs)

    all_axons = []  # (cx, atom, type, tchip_id, tcore_id, taxon_id)
    for cx_id, spike in zip(cx_ids, spikes):
        taxon_idx = int(spike.axon_id)
        taxon_id = int(tsyn_idxs[taxon_idx])
        atom = int(spike.atom)
        n_populations = synapse.axon_populations(taxon_idx)
        all_axons.append((cx_id, atom, synapse.pop_type,
                          tchip_id, tcore_id, taxon_id))
        if synapse.pop_type == 0:  # discrete
            assert atom == 0
            assert n_populations == 1
        elif synapse.pop_type == 16 or synapse.pop_type == 32:
            n_blocks = len(core.blocks)
            assert (n_blocks == 0
                    or (n_blocks == 1 and block is core.blocks[0]))
            assert len(block.probes) == 0
        else:
            raise ValueError("Axon: unrecognized pop_type: %s" % (
                synapse.pop_type,))

    return all_axons


def build_axons(n2core, core, block, all_axons):  # noqa C901
    if len(all_axons) == 0:
        return

    pop_type0 = all_axons[0][2]
    if pop_type0 == 0:
        for cx_id, atom, pop_type, tchip_id, tcore_id, taxon_id in all_axons:
            assert pop_type == 0, "All axons must be discrete, or none"
            assert atom == 0
            n2core.createDiscreteAxon(
                srcCxId=cx_id,
                dstChipId=tchip_id, dstCoreId=tcore_id, dstSynMapId=taxon_id)

        return
    else:
        assert all(axon[2] != 0 for axon in all_axons), (
            "All axons must be discrete, or none")

    axons_by_cx = groupby(all_axons, key=lambda x: x[0])  # group by cx_id

    axon_id = 0
    axon_map = {}
    for cx_id, axons in axons_by_cx:
        if len(axons) == 0:
            continue

        # axon -> (cx, atom, type, tchip_id, tcore_id, taxon_id)
        assert all(axon[0] == cx_id for axon in axons)
        atom = axons[0][1]
        assert all(axon[1] == atom for axon in axons), (
            "cx atom must be the same for all axons")

        axons = sorted(axons, key=lambda a: a[2:])
        key = tuple(axon[2:] for axon in axons)
        if key not in axon_map:
            axon_id0 = axon_id
            axon_len = 0

            for axon in axons:
                pop_type, tchip_id, tcore_id, taxon_id = axon[2:]
                # note: pop_type==0 should have been handled in code above
                assert pop_type in (16, 32)
                if pop_type == 16:  # pop16
                    n2core.axonCfg[axon_id].pop16.configure(
                        coreId=tcore_id, axonId=taxon_id)
                    axon_id += 1
                    axon_len += 1
                elif pop_type == 32:  # pop32
                    n2core.axonCfg[axon_id].pop32_0.configure(
                        coreId=tcore_id, axonId=taxon_id)
                    n2core.axonCfg[axon_id+1].pop32_1.configure()
                    axon_id += 2
                    axon_len += 2

            axon_map[key] = (axon_id0, axon_len)

        axon_ptr, axon_len = axon_map[key]
        n2core.axonMap[cx_id].configure(ptr=axon_ptr, len=axon_len, atom=atom)


def build_probe(n2core, core, block, probe, cx_idxs):
    key_map = {'current': 'u', 'voltage': 'v', 'spiked': 'spike'}
    assert probe.key in key_map, "probe key not found"
    key = key_map[probe.key]

    n2board = n2core.parent.parent
    r = cx_idxs[probe.slice]

    if probe.use_snip:
        probe.snip_info = dict(coreid=n2core.id, cxs=r, key=key)
    else:
        p = n2board.monitor.probe(n2core.cxState, r, key)
        core.board.map_probe(probe, p)
