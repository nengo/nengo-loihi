from __future__ import division

import logging

import nengo.utils.numpy as npext
from nengo.exceptions import BuildError
from nengo.utils.stdlib import groupby
import numpy as np

from nengo_loihi.discretize import bias_to_manexp
from nengo_loihi.hardware.nxsdk_objects import (
    LoihiSpikeInput,
    MAX_COMPARTMENT_CFGS,
    MAX_VTH_CFGS,
)
from nengo_loihi.hardware.nxsdk_shim import (
    micro_gen,
    NxsdkBoard,
    SpikeGen,
    TraceConfigGenerator,
)
from nengo_loihi.inputs import SpikeInput

logger = logging.getLogger(__name__)


def build_board(board, seed=None):
    n_chips = board.n_chips
    n_cores_per_chip = board.n_cores_per_chip
    n_synapses_per_core = board.n_synapses_per_core
    nxsdk_board = NxsdkBoard(
        board.board_id, n_chips, n_cores_per_chip, n_synapses_per_core)

    # add our own attribute for storing our spike generator
    assert not hasattr(nxsdk_board, 'global_spike_generator')
    nxsdk_board.global_spike_generator = SpikeGen(nxsdk_board)

    # custom attr for storing SpikeInputs (filled in build_input)
    assert not hasattr(nxsdk_board, 'spike_inputs')
    nxsdk_board.spike_inputs = {}

    # build all chips
    assert len(board.chips) == len(nxsdk_board.n2Chips)
    rng = np.random.RandomState(seed)
    for chip, nxsdk_chip in zip(board.chips, nxsdk_board.n2Chips):
        logger.debug("Building chip %s", chip)
        seed = rng.randint(npext.maxint)
        build_chip(nxsdk_chip, chip, seed=seed)

    return nxsdk_board


def build_chip(nxsdk_chip, chip, seed=None):
    assert len(chip.cores) == len(nxsdk_chip.n2Cores)
    rng = np.random.RandomState(seed)
    for core, nxsdk_core in zip(chip.cores, nxsdk_chip.n2Cores):
        logger.debug("Building core %s", core)
        seed = rng.randint(npext.maxint)
        build_core(nxsdk_core, core, seed=seed)


def build_core(nxsdk_core, core, seed=None):  # noqa: C901
    assert len(core.compartment_cfgs) < MAX_COMPARTMENT_CFGS
    assert len(core.vth_cfgs) < MAX_VTH_CFGS

    logger.debug("- Configuring compartments")
    for i, cfg in enumerate(core.compartment_cfgs):
        nxsdk_core.cxProfileCfg[i].configure(
            decayV=cfg.decay_v,
            decayU=cfg.decay_u,
            refractDelay=cfg.refract_delay,
            enableNoise=cfg.enable_noise,
            bapAction=1,
        )

    logger.debug("- Configuring vth_cfgs")
    for i, cfg in enumerate(core.vth_cfgs):
        nxsdk_core.vthProfileCfg[i].staticCfg.configure(
            vth=cfg.vth,
        )

    logger.debug("- Configuring synapse_cfgs")
    for i, cfg in enumerate(core.synapse_cfgs):
        if cfg is None:
            continue

        nxsdk_core.synapseFmt[i].wgtLimitMant = cfg.weight_limit_mant
        nxsdk_core.synapseFmt[i].wgtLimitExp = cfg.weight_limit_exp
        nxsdk_core.synapseFmt[i].wgtExp = cfg.weight_exp
        nxsdk_core.synapseFmt[i].discMaxWgt = cfg.disc_max_weight
        nxsdk_core.synapseFmt[i].learningCfg = cfg.learning_cfg
        nxsdk_core.synapseFmt[i].tagBits = cfg.tag_bits
        nxsdk_core.synapseFmt[i].dlyBits = cfg.delay_bits
        nxsdk_core.synapseFmt[i].wgtBits = cfg.weight_bits
        nxsdk_core.synapseFmt[i].reuseSynData = cfg.reuse_synapse_data
        nxsdk_core.synapseFmt[i].numSynapses = cfg.n_synapses
        nxsdk_core.synapseFmt[i].cIdxOffset = cfg.idx_offset
        nxsdk_core.synapseFmt[i].cIdxMult = cfg.idx_mult
        nxsdk_core.synapseFmt[i].skipBits = cfg.skip_bits
        nxsdk_core.synapseFmt[i].idxBits = cfg.idx_bits
        nxsdk_core.synapseFmt[i].synType = cfg.synapse_type
        nxsdk_core.synapseFmt[i].fanoutType = cfg.fanout_type
        nxsdk_core.synapseFmt[i].compression = cfg.compression
        nxsdk_core.synapseFmt[i].stdpProfile = cfg.stdp_cfg
        nxsdk_core.synapseFmt[i].ignoreDly = cfg.ignore_delay

    logger.debug("- Configuring stdp_pre_cfgs")
    for i, trace_cfg in enumerate(core.stdp_pre_cfgs):
        tcg = TraceConfigGenerator()
        tc = tcg.genTraceCfg(
            tau=trace_cfg.tau,
            spikeLevelInt=trace_cfg.spike_int,
            spikeLevelFrac=trace_cfg.spike_frac,
        )
        tc.writeToRegister(nxsdk_core.stdpPreCfg[i])

    # --- seed randomness
    def seed_trace(trace_random, rng):
        trace_random.random0 = rng.randint(2**32)
        trace_random.random1 = rng.randint(2**32)
        trace_random.random2 = rng.randint(2**32)

    rng = np.random.RandomState(seed)
    # neuron noise
    # TODO: how to set neuron noise?
    # nxsdk_core.dendriteRandom.word = rng.randint(2**32)
    # pre trace rounding
    seed_trace(nxsdk_core.stdpPreRandom, rng)
    # post trace rounding
    seed_trace(nxsdk_core.stdpPostRandom, rng)
    # soma activity trace rounding
    seed_trace(nxsdk_core.somaRandom, rng)
    # synaptic rounding
    nxsdk_core.synapseRepackRandom.word = rng.randint(2**32)

    # --- learning
    first_learning_index = None
    for synapse in core.iterate_synapses():
        if synapse.learning and first_learning_index is None:
            first_learning_index = core.synapse_axons[synapse][0]
            core.learning_coreid = nxsdk_core.id
            break

    num_stdp = 0
    if first_learning_index is not None:
        for synapse in core.iterate_synapses():
            assert synapse.learning, (
                "Currently, all synapses on core are learning or none are")

            axons = np.array(core.synapse_axons[synapse])
            if synapse.learning:
                num_stdp += len(axons)
                assert np.all(axons >= first_learning_index)

    if num_stdp > 0:
        logger.debug("- Configuring PES learning")
        # add configurations tailored to PES learning
        nxsdk_core.stdpCfg.configure(
            firstLearningIndex=first_learning_index,
            numRewardAxons=0,
        )

        assert core.stdp_pre_cfg_idx is None
        assert core.stdp_cfg_idx is None
        core.stdp_pre_cfg_idx = 0  # hard-code for now
        core.stdp_cfg_idx = 0  # hard-code for now (also in synapse_cfg)
        nxsdk_core.stdpPreProfileCfg[0].configure(
            updateAlways=1,
            numTraces=0,
            numTraceHist=0,
            stdpProfile=0,
        )

        # stdpProfileCfg positive error
        nxsdk_core.stdpProfileCfg[0].configure(
            uCodePtr=0,
            decimateExp=0,
            numProducts=1,
            requireY=1,
            usesXepoch=1,
        )

        # Microcode for the learning rule. `u1` evaluates the learning rule
        # every 2**1 timesteps, `x1` is the pre-trace, `y1` is the post-trace,
        # and 2^-7 is the learning rate. See `help(ruleToUCode)` for more info.
        ucode = micro_gen.ruleToUCode(
            ['dw = u1*x1*y1*(2^-7)'], doOptimize=False)
        assert ucode.numUCodes == 1
        nxsdk_core.stdpUcodeMem[0].word = ucode.uCodes[0]

        # stdpProfileCfg negative error
        nxsdk_core.stdpProfileCfg[1].configure(
            uCodePtr=1,
            decimateExp=0,
            numProducts=1,
            requireY=1,
            usesXepoch=1,
        )
        # use negative version of above microcode rule
        ucode = micro_gen.ruleToUCode(
            ['dw = -u1*x1*y1*(2^-7)'], doOptimize=False)
        assert ucode.numUCodes == 1
        nxsdk_core.stdpUcodeMem[1].word = ucode.uCodes[0]

        tcg = TraceConfigGenerator()
        tc = tcg.genTraceCfg(
            tau=0,
            spikeLevelInt=0,
            spikeLevelFrac=0,
        )
        tc.writeToRegister(nxsdk_core.stdpPostCfg[0])

    # TODO: allocator should be checking that vmin, vmax are the same
    #   for all blocks on a core
    n_compartments = 0
    if len(core.blocks) > 0:
        block0 = core.blocks[0]
        vmin, vmax = block0.compartment.vmin, block0.compartment.vmax
        assert all(block.compartment.vmin == vmin
                   for block in core.blocks)
        assert all(block.compartment.vmax == vmax
                   for block in core.blocks)
        neg_limit = np.log2(-vmin + 1)
        pos_limit = (np.log2(vmax + 1) - 9) * 0.5
        assert int(neg_limit) == neg_limit
        assert int(pos_limit) == pos_limit

        noise_exp = block0.compartment.noise_exp
        noise_offset = block0.compartment.noise_offset
        noise_at_membrane = block0.compartment.noise_at_membrane
        assert all(block.compartment.noise_exp == noise_exp
                   for block in core.blocks)
        assert all(block.compartment.noise_offset == noise_offset
                   for block in core.blocks)
        assert all(block.compartment.noise_at_membrane == noise_at_membrane
                   for block in core.blocks)

        if noise_exp < 7:
            # unexpected shifting: exp < 7 acts as exp + 1
            noise_exp = noise_exp - 1

        nxsdk_core.dendriteSharedCfg.configure(
            posVmLimit=int(pos_limit),
            negVmLimit=int(neg_limit),
            noiseExp0=noise_exp,
            noiseMantOffset0=noise_offset,
            noiseAtDendOrVm=noise_at_membrane,
        )

        nxsdk_core.dendriteAccumCfg.configure(
            delayBits=3)
        # ^ DelayBits=3 allows 1024 Cxs per core

        for block, compartment_idxs, ax_range in core.iterate_blocks():
            build_block(nxsdk_core, core, block, compartment_idxs, ax_range)
            n_compartments = max(max(compartment_idxs) + 1, n_compartments)

    for inp, compartment_idxs in core.iterate_inputs():
        build_input(nxsdk_core, core, inp, compartment_idxs)

    logger.debug("- Configuring n_updates=%d", n_compartments // 4 + 1)
    nxsdk_core.numUpdates.configure(
        numUpdates=n_compartments // 4 + 1,
        numStdp=num_stdp,
    )

    nxsdk_core.dendriteTimeState[0].tepoch = 2
    nxsdk_core.timeState[0].tepoch = 2


def build_block(nxsdk_core, core, block, compartment_idxs, ax_range):
    assert block.compartment.scale_u is False
    assert block.compartment.scale_v is False

    logger.debug("Building %s on core.id=%d", block, nxsdk_core.id)

    for i, bias in enumerate(block.compartment.bias):
        bman, bexp = bias_to_manexp(bias)
        icx = core.compartment_cfg_idxs[block][i]
        ivth = core.vth_cfg_idxs[block][i]

        ii = compartment_idxs[i]
        nxsdk_core.cxCfg[ii].configure(
            bias=bman, biasExp=bexp, vthProfile=ivth, cxProfile=icx)

        phasex = 'phase%d' % (ii % 4,)
        nxsdk_core.cxMetaState[ii // 4].configure(**{phasex: 2})

    logger.debug("- Building %d synapses", len(block.synapses))
    for synapse in block.synapses:
        build_synapse(nxsdk_core, core, block, synapse, compartment_idxs)

    logger.debug("- Building %d axons", len(block.axons))
    all_axons = []  # (compartment, atom, type, tchip_id, tcore_id, taxon_id)
    for axon in block.axons:
        all_axons.extend(collect_axons(nxsdk_core, core, block, axon,
                                       compartment_idxs))

    build_axons(nxsdk_core, core, block, all_axons)

    logger.debug("- Building %d probes", len(block.probes))
    for probe in block.probes:
        build_probe(nxsdk_core, core, block, probe, compartment_idxs)


def build_input(nxsdk_core, core, spike_input, compartment_idxs):
    assert len(spike_input.axons) > 0
    nxsdk_board = nxsdk_core.parent.parent

    assert isinstance(spike_input, SpikeInput)
    loihi_input = LoihiSpikeInput()
    loihi_input.set_axons(core.board, nxsdk_board, spike_input)
    assert spike_input not in nxsdk_board.spike_inputs
    nxsdk_board.spike_inputs[spike_input] = loihi_input

    # add any pre-existing spikes to spikegen
    for t in spike_input.spike_times():
        spikes = spike_input.spike_idxs(t)
        for spike in loihi_input.spikes_to_loihi(t, spikes):
            assert spike.axon.atom == 0, (
                "Cannot send population spikes through spike generator")
            nxsdk_board.global_spike_generator.addSpike(
                time=spike.time, chipId=spike.axon.chip_id,
                coreId=spike.axon.core_id, axonId=spike.axon.axon_id)


def build_synapse(  # noqa C901
        nxsdk_core, core, block, synapse, compartment_idxs):
    axon_ids = core.synapse_axons[synapse]

    synapse_cfg_idx = core.synapse_cfg_idxs[synapse]
    stdp_pre_cfg_idx = core.stdp_pre_cfg_idxs[synapse]

    atom_bits = synapse.atom_bits()
    axon_bits = synapse.axon_bits()
    atom_bits_extra = synapse.atom_bits_extra()

    target_compartments = set()
    synapse_map = {}  # map weight_idx to (ptr, pop_size, len)
    total_synapse_ptr = int(core.synapse_entries[synapse][0])
    for axon_idx, axon_id in enumerate(axon_ids):
        assert axon_id <= 2**axon_bits

        weight_idx = int(synapse.axon_weight_idx(axon_idx))
        base = synapse.axon_compartment_base(axon_idx)

        if weight_idx not in synapse_map:
            weights = synapse.weights[weight_idx]
            indices = synapse.indices[weight_idx]
            weights = weights // synapse.synapse_cfg.scale
            assert weights.ndim == 2
            assert weights.shape == indices.shape
            assert np.all(weights <= 255) and np.all(weights >= -256), str(
                weights)
            n_populations, n_compartments = weights.shape

            synapse_map[weight_idx] = (
                total_synapse_ptr, n_populations, n_compartments)

            for p in range(n_populations):
                for q in range(n_compartments):
                    compartment_idx = compartment_idxs[indices[p, q]]
                    nxsdk_core.synapses[total_synapse_ptr].configure(
                        CIdx=compartment_idx,
                        Wgt=weights[p, q],
                        synFmtId=synapse_cfg_idx,
                        LrnEn=int(synapse.learning),
                    )
                    target_compartments.add(compartment_idx)
                    total_synapse_ptr += 1

        synapse_ptr, n_populations, n_compartments = synapse_map[weight_idx]
        assert n_populations <= 2**atom_bits

        if base is None:
            # this is a dummy axon with no weights, so set n_compartments to 0
            synapse_ptr = 0
            n_compartments = 0
            base = 0
        else:
            base = int(base)

        assert base <= 256, "Currently limited by hardware"
        nxsdk_core.synapseMap[axon_id].synapsePtr = synapse_ptr
        nxsdk_core.synapseMap[axon_id].synapseLen = n_compartments
        if synapse.pop_type == 0:  # discrete
            assert n_populations == 1
            nxsdk_core.synapseMap[axon_id].discreteMapEntry.configure(
                cxBase=base)
        elif synapse.pop_type == 16:  # pop16
            nxsdk_core.synapseMap[axon_id].popSize = n_populations
            assert base % 4 == 0
            nxsdk_core.synapseMap[axon_id].population16MapEntry.configure(
                cxBase=base//4, atomBits=atom_bits_extra)
        elif synapse.pop_type == 32:  # pop32
            nxsdk_core.synapseMap[axon_id].popSize = n_populations
            nxsdk_core.synapseMap[axon_id].population32MapEntry.configure(
                cxBase=base)
        else:
            raise BuildError("Synapse: unrecognized pop_type: %s" % (
                synapse.pop_type,))

        if synapse.learning:
            assert core.stdp_pre_cfg_idx is not None
            assert stdp_pre_cfg_idx is not None
            nxsdk_core.synapseMap[axon_id+1].singleTraceEntry.configure(
                preProfile=core.stdp_pre_cfg_idx, tcs=stdp_pre_cfg_idx)

    assert total_synapse_ptr == core.synapse_entries[synapse][1], (
        "Synapse pointer did not align with precomputed synapse length")

    if synapse.learning:
        assert core.stdp_cfg_idx is not None
        for compartment in target_compartments:
            # TODO: check that no compartment configured by multiple synapses
            nxsdk_core.stdpPostState[compartment].configure(
                stdpProfile=core.stdp_cfg_idx,
                traceProfile=3,  # TODO: why this value
            )


def collect_axons(nxsdk_core, core, block, axon, compartment_ids):
    synapse = axon.target
    tchip_idx, tcore_idx, tsyn_idxs = core.board.find_synapse(synapse)
    nxsdk_board = nxsdk_core.parent.parent
    tchip_id = nxsdk_board.n2Chips[tchip_idx].id
    tcore_id = nxsdk_board.n2Chips[tchip_idx].n2Cores[tcore_idx].id

    compartment_idxs = np.arange(len(compartment_ids))
    spikes = axon.map_spikes(compartment_idxs)

    all_axons = []  # (compartment, atom, type, tchip_id, tcore_id, taxon_id)
    for compartment_id, spike in zip(compartment_ids, spikes):
        taxon_idx = int(spike.axon_id)
        taxon_id = int(tsyn_idxs[taxon_idx])
        atom = int(spike.atom)
        n_populations = synapse.axon_populations(taxon_idx)
        all_axons.append((compartment_id, atom, synapse.pop_type,
                          tchip_id, tcore_id, taxon_id))
        if synapse.pop_type == 0:  # discrete
            assert atom == 0
            assert n_populations == 1
        elif synapse.pop_type == 16 or synapse.pop_type == 32:
            n_blocks = len(core.blocks)
            assert (n_blocks == 0
                    or (n_blocks == 1 and block is core.blocks[0]))
            assert len(block.probes) == 0
            tchip_id_source = nxsdk_board.n2Chips[core.chip.index].id
            if tchip_id != tchip_id_source:
                raise BuildError("pop16 and pop32 weights are not "
                                 "supported across multiple chips "
                                 "(%d -> %d); this is likely raised due "
                                 "to convolutional weights being used "
                                 "with a multi-chip allocator" % (
                                     tchip_id_source, tchip_id))
        else:
            raise BuildError("Axon: unrecognized pop_type: %s" % (
                synapse.pop_type,))

    return all_axons


def build_axons(nxsdk_core, core, block, all_axons):  # noqa C901
    if len(all_axons) == 0:
        return

    pop_type0 = all_axons[0][2]
    if pop_type0 == 0:
        for (compartment_id, atom, pop_type, tchip_id, tcore_id,
             taxon_id) in all_axons:
            assert pop_type == 0, "All axons must be discrete, or none"
            assert atom == 0
            nxsdk_core.createDiscreteAxon(
                srcCxId=compartment_id,
                dstChipId=tchip_id,
                dstCoreId=tcore_id,
                dstSynMapId=taxon_id,
            )

        return
    else:
        assert all(axon[2] != 0 for axon in all_axons), (
            "All axons must be discrete, or none")

    axons_by_compartment = groupby(all_axons, key=lambda x: x[0])

    axon_id = 0
    axon_map = {}
    for compartment_id, axons in axons_by_compartment:
        assert len(axons) > 0

        # axon -> (compartment, atom, type, tchip_id, tcore_id, taxon_id)
        assert all(axon[0] == compartment_id for axon in axons)
        atom = axons[0][1]
        assert all(axon[1] == atom for axon in axons), (
            "compartment atom must be the same for all axons")

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
                    nxsdk_core.axonCfg[axon_id].pop16.configure(
                        coreId=tcore_id, axonId=taxon_id)
                    axon_id += 1
                    axon_len += 1
                elif pop_type == 32:  # pop32
                    nxsdk_core.axonCfg[axon_id].pop32_0.configure(
                        coreId=tcore_id, axonId=taxon_id)
                    nxsdk_core.axonCfg[axon_id+1].pop32_1.configure()
                    axon_id += 2
                    axon_len += 2

            axon_map[key] = (axon_id0, axon_len)

        axon_ptr, axon_len = axon_map[key]
        nxsdk_core.axonMap[compartment_id].configure(
            ptr=axon_ptr, len=axon_len, atom=atom)


def build_probe(nxsdk_core, core, block, probe, compartment_idxs):
    key_map = {'current': 'u', 'voltage': 'v', 'spiked': 'spike'}
    assert probe.key in key_map, "probe key not found"
    key = key_map[probe.key]

    nxsdk_board = nxsdk_core.parent.parent
    r = compartment_idxs[probe.slice]

    if probe.use_snip:
        probe.snip_info = dict(
            coreid=nxsdk_core.id, compartment_idxs=r, key=key)
    else:
        p = nxsdk_board.monitor.probe(nxsdk_core.cxState, r, key)
        core.board.map_probe(probe, p)
