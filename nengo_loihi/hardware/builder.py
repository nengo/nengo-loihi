import logging

import nengo.utils.numpy as npext
import numpy as np
from nengo.exceptions import BuildError

from nengo_loihi.builder.discretize import bias_to_manexp
from nengo_loihi.hardware.nxsdk_objects import (
    MAX_COMPARTMENT_CFGS,
    MAX_VTH_CFGS,
    LoihiSpikeInput,
)
from nengo_loihi.hardware.nxsdk_shim import (
    NxsdkBoard,
    SpikeGen,
    TraceConfigGenerator,
    micro_gen,
)
from nengo_loihi.inputs import SpikeInput

logger = logging.getLogger(__name__)


def ceil_div(a, b):
    return -((-a) // b)


def build_board(board, use_snips=False, seed=None):
    n_chips = board.n_chips
    n_cores_per_chip = board.n_cores_per_chip
    n_synapses_per_core = board.n_synapses_per_core
    nxsdk_board = NxsdkBoard(
        board.board_id, n_chips, n_cores_per_chip, n_synapses_per_core
    )

    # add our own attribute for storing our spike generator
    assert not hasattr(nxsdk_board, "global_spike_generator")
    nxsdk_board.global_spike_generator = None if use_snips else SpikeGen(nxsdk_board)

    # custom attr for storing SpikeInputs (filled in build_input)
    assert not hasattr(nxsdk_board, "spike_inputs")
    nxsdk_board.spike_inputs = {}

    # build inputs
    for input in board.inputs:
        build_input(nxsdk_board, board, input)

    # build all chips
    assert len(board.chips) == len(nxsdk_board.n2Chips)
    rng = np.random.RandomState(seed)
    for chip, nxsdk_chip in zip(board.chips, nxsdk_board.n2Chips):
        logger.debug("Building chip %s", chip)
        seed = rng.randint(npext.maxint)
        build_chip(nxsdk_chip, chip, seed=seed)

    # build probes
    logger.debug("Building %d probes", len(board.probes))
    for probe in board.probes:
        build_probe(nxsdk_board, board, probe, use_snips=use_snips)

    return nxsdk_board


def build_chip(nxsdk_chip, chip, seed=None):
    assert len(chip.cores) == len(nxsdk_chip.n2CoresAsList)

    # build cores
    rng = np.random.RandomState(seed)
    for core, nxsdk_core in zip(chip.cores, nxsdk_chip.n2CoresAsList):
        logger.debug("Building core %s", core)
        seed = rng.randint(npext.maxint)
        build_core(nxsdk_core, core, seed=seed)


def build_input(nxsdk_board, board, input):
    if isinstance(input, SpikeInput):
        build_spike_input(nxsdk_board, board, input)
    else:
        raise NotImplementedError(
            "Input type %s not implemented" % type(input).__name__
        )


def build_spike_input(nxsdk_board, board, spike_input):
    assert isinstance(spike_input, SpikeInput)
    assert len(spike_input.axons) > 0

    loihi_input = LoihiSpikeInput()
    loihi_input.set_axons(board, nxsdk_board, spike_input)
    assert spike_input not in nxsdk_board.spike_inputs
    nxsdk_board.spike_inputs[spike_input] = loihi_input

    # add any pre-existing spikes to spikegen
    nxsdk_spike_generator = nxsdk_board.global_spike_generator
    for t in spike_input.spike_times():
        assert (
            nxsdk_spike_generator is not None
        ), "Cannot add pre-existing spikes when using Snips (no spike generator)"

        spikes = spike_input.spike_idxs(t)
        loihi_spikes = loihi_input.spikes_to_loihi(spikes)
        loihi_input.add_spikes_to_generator(t, loihi_spikes, nxsdk_spike_generator)


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
        nxsdk_core.vthProfileCfg[i].staticCfg.configure(vth=cfg.vth)

    logger.debug("- Configuring synapse_cfgs")
    for i, cfg in enumerate(core.synapse_cfgs):
        if cfg is None:
            continue

        obj = nxsdk_core.synapseFmt[i]
        obj.wgtLimitMant = cfg.weight_limit_mant
        obj.wgtLimitExp = cfg.weight_limit_exp
        obj.wgtExp = cfg.weight_exp
        obj.discMaxWgt = cfg.disc_max_weight
        obj.learningCfg = cfg.learning_cfg
        obj.tagBits = cfg.tag_bits
        obj.dlyBits = cfg.delay_bits
        obj.wgtBits = cfg.weight_bits
        obj.reuseSynData = cfg.reuse_synapse_data
        obj.numSynapses = cfg.n_synapses
        obj.cIdxOffset = cfg.idx_offset
        obj.cIdxMult = cfg.idx_mult
        obj.skipBits = cfg.skip_bits
        obj.idxBits = cfg.idx_bits
        obj.synType = cfg.synapse_type
        obj.fanoutType = cfg.fanout_type
        obj.compression = cfg.compression
        obj.stdpProfile = cfg.stdp_cfg
        obj.ignoreDly = cfg.ignore_delay

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
    # d_set (nxsdk_core, b'ZGVuZHJpdGVSYW5kb20=', b'd29yZA==',
    #        val=rng.randint(2 ** 32))
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

    n_stdp = 0
    if first_learning_index is not None:
        for synapse in core.iterate_synapses():
            assert (
                synapse.learning
            ), "Currently, all synapses on core are learning or none are"

            axons = np.array(core.synapse_axons[synapse])
            if synapse.learning:
                n_stdp += len(axons)
                assert np.all(axons >= first_learning_index)

    if n_stdp > 0:
        logger.debug("- Configuring PES learning")
        # add configurations tailored to PES learning
        nxsdk_core.stdpCfg.configure(
            firstLearningIndex=first_learning_index, numRewardAxons=0
        )

        assert core.stdp_pre_cfg_idx is None
        assert core.stdp_cfg_idx is None
        core.stdp_pre_cfg_idx = 0  # hard-code for now
        core.stdp_cfg_idx = 0  # hard-code for now (also in synapse_cfg)
        nxsdk_core.stdpPreProfileCfg[0].configure(
            updateAlways=1, numTraces=0, numTraceHist=0, stdpProfile=0
        )

        # stdp config for positive error
        nxsdk_core.stdpProfileCfg[0].configure(
            uCodePtr=0, decimateExp=0, numProducts=1, requireY=1, usesXepoch=1
        )

        # Microcode for the learning rule. `u1` evaluates the learning rule
        # every 2**1 timesteps, `x1` is the pre-trace, `y1` is the post-trace,
        # and 2^-7 is the learning rate.
        ucode = micro_gen.ruleToUCode(["dw = u1*x1*y1*(2^-7)"], doOptimize=False)
        assert ucode.numUCodes == 1
        nxsdk_core.stdpUcodeMem[0].word = ucode.uCodes[0]

        # stdp config for negative error
        nxsdk_core.stdpProfileCfg[1].configure(
            uCodePtr=1, decimateExp=0, numProducts=1, requireY=1, usesXepoch=1
        )
        # use negative version of above microcode rule
        ucode = micro_gen.ruleToUCode(["dw = -u1*x1*y1*(2^-7)"], doOptimize=False)
        assert ucode.numUCodes == 1
        nxsdk_core.stdpUcodeMem[1].word = ucode.uCodes[0]

        tcg = TraceConfigGenerator()
        tc = tcg.genTraceCfg(tau=0, spikeLevelInt=0, spikeLevelFrac=0)
        tc.writeToRegister(nxsdk_core.stdpPostCfg[0])

    # TODO: allocator should be checking that vmin, vmax are the same
    #   for all blocks on a core
    n_compartments = 0
    if len(core.blocks) > 0:
        block0 = core.blocks[0]
        vmin, vmax = block0.compartment.vmin, block0.compartment.vmax
        assert all(block.compartment.vmin == vmin for block in core.blocks)
        assert all(block.compartment.vmax == vmax for block in core.blocks)
        neg_limit = np.log2(-vmin + 1)
        pos_limit = (np.log2(vmax + 1) - 9) * 0.5
        assert int(neg_limit) == neg_limit
        assert int(pos_limit) == pos_limit

        noise_exp = block0.compartment.noise_exp
        noise_offset = block0.compartment.noise_offset
        noise_at_membrane = block0.compartment.noise_at_membrane
        assert all(block.compartment.noise_exp == noise_exp for block in core.blocks)
        assert all(
            block.compartment.noise_offset == noise_offset for block in core.blocks
        )
        assert all(
            block.compartment.noise_at_membrane == noise_at_membrane
            for block in core.blocks
        )

        if noise_exp < 7:
            # unexpected shifting: exp less than threshold acts as exp + 1
            noise_exp = noise_exp - 1

        nxsdk_core.dendriteSharedCfg.configure(
            posVmLimit=int(pos_limit),
            negVmLimit=int(neg_limit),
            noiseExp0=noise_exp,
            noiseMantOffset0=noise_offset,
            noiseAtDendOrVm=noise_at_membrane,
        )

        nxsdk_core.dendriteAccumCfg.configure(delayBits=3)

        for block, compartment_idxs, ax_range in core.iterate_blocks():
            build_block(nxsdk_core, core, block, compartment_idxs, ax_range)
            n_compartments = max(max(compartment_idxs) + 1, n_compartments)

    n_updates = ceil_div(n_compartments, 4)
    logger.debug("- Configuring n_updates=%d, n_stdp=%d", n_updates, n_stdp)
    nxsdk_core.numUpdates.configure(
        numUpdates=n_updates,
        numStdp=n_stdp,
    )

    nxsdk_core.dendriteTimeState[0].tepoch = 2
    nxsdk_core.timeState[0].tepoch = 2


def build_block(nxsdk_core, core, block, compartment_idxs, ax_range):
    assert block.compartment.scale_u is False
    assert block.compartment.scale_v is False

    logger.debug("Building %s on core.id=%d", block, nxsdk_core.id)

    bmans, bexps = bias_to_manexp(block.compartment.bias)
    for i, (bman, bexp) in enumerate(zip(bmans, bexps)):
        icomp = core.compartment_cfg_idxs[block][i]
        ivth = core.vth_cfg_idxs[block][i]

        ii = compartment_idxs[i]
        nxsdk_core.cxCfg[ii].configure(
            bias=bman, biasExp=bexp, vthProfile=ivth, cxProfile=icomp
        )

        phasex = "phase%d" % (ii % 4,)
        nxsdk_core.cxMetaState[ii // 4].configure(**{phasex: 2})

    logger.debug("- Building %d synapses", len(block.synapses))
    for synapse in block.synapses:
        build_synapse(nxsdk_core, core, block, synapse, compartment_idxs)

    logger.debug("- Building %d axons", len(block.axons))
    pop_id_map = {}
    for axon in block.axons:
        build_axon(nxsdk_core, core, block, axon, compartment_idxs, pop_id_map)


def build_synapse(nxsdk_core, core, block, synapse, compartment_idxs):  # noqa C901
    max_compartment_offset = 256
    axon_ids = core.synapse_axons[synapse]

    synapse_cfg_idx = core.synapse_cfg_idxs[synapse]
    stdp_pre_cfg_idx = core.stdp_pre_cfg_idxs[synapse]

    atom_bits = synapse.atom_bits()
    axon_bits = synapse.axon_bits()
    atom_bits_extra = synapse.atom_bits_extra()
    if atom_bits_extra > 0:
        raise NotImplementedError(
            "Using more than 32 'populations' (e.g. convolutional filters) with "
            "`pop_type=16` axons has not yet been implemented in NxSDK"
        )

    target_compartments = set()
    synapse_map = {}  # map weight_idx to (ptr, pop_size, len)
    total_synapse_ptr = int(core.synapse_entries[synapse][0])
    for axon_idx, axon_id in enumerate(axon_ids):
        assert axon_id is None or axon_id <= 2**axon_bits

        weight_idx = synapse.axon_weight_idx(axon_idx)
        base = synapse.axon_compartment_base(axon_idx)

        if weight_idx not in synapse_map:
            weights = synapse.weights[weight_idx]
            indices = synapse.indices[weight_idx]
            weights = weights // synapse.synapse_cfg.scale
            assert weights.ndim == 2
            assert weights.shape == indices.shape
            assert np.all(weights <= 255) and np.all(weights >= -256), str(weights)

            n_atoms, n_compartments = weights.shape

            synapse_map[weight_idx] = (total_synapse_ptr, n_atoms, n_compartments)

            for p in range(n_atoms):
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

        synapse_ptr, n_atoms, n_compartments = synapse_map[weight_idx]
        assert n_atoms <= 2**atom_bits

        if axon_id is None:  # pragma: no cover
            # This is a dummy axon with no base or no weights, so skip it
            assert base is None or n_compartments == 0
            continue

        # base = int(base)
        assert base <= max_compartment_offset, "Currently limited by hardware"
        nxsdk_core.synapseMap[axon_id].synapsePtr = synapse_ptr
        nxsdk_core.synapseMap[axon_id].synapseLen = n_compartments
        if synapse.pop_type == 0:  # discrete
            assert n_atoms == 1
            nxsdk_core.synapseMap[axon_id].discreteMapEntry.configure(cxBase=base)
        elif synapse.pop_type == 16:  # pop16
            nxsdk_core.synapseMap[axon_id].popSize = n_atoms
            assert base % 4 == 0
            nxsdk_core.synapseMap[axon_id].population16MapEntry.configure(
                cxBase=base // 4,
                atomBits=atom_bits_extra,
            )
        elif synapse.pop_type == 32:  # pop32
            nxsdk_core.synapseMap[axon_id].popSize = n_atoms
            nxsdk_core.synapseMap[axon_id].population32MapEntry.configure(cxBase=base)
        else:
            raise BuildError(f"{synapse}: unrecognized pop_type: {synapse.pop_type}")

        if synapse.learning:
            assert core.stdp_pre_cfg_idx is not None
            assert stdp_pre_cfg_idx is not None
            nxsdk_core.synapseMap[axon_id + 1].singleTraceEntry.configure(
                preProfile=core.stdp_pre_cfg_idx, tcs=stdp_pre_cfg_idx
            )

    assert (
        total_synapse_ptr == core.synapse_entries[synapse][1]
    ), "Synapse pointer did not align with precomputed synapse length"

    if synapse.learning:
        assert core.stdp_cfg_idx is not None
        for compartment in target_compartments:
            # TODO: check that no compartment configured by multiple synapses
            nxsdk_core.stdpPostState[compartment].configure(
                stdpProfile=core.stdp_cfg_idx, traceProfile=3
            )


def build_axon(nxsdk_core, core, block, axon, compartment_ids, pop_id_map):
    synapse = axon.target
    tchip_idx, tcore_idx, taxon_ids = core.board.find_synapse(synapse)
    nxsdk_board = nxsdk_core.parent.parent
    tchip_id = nxsdk_board.n2Chips[tchip_idx].id
    tcore_id = nxsdk_board.n2Chips[tchip_idx].n2CoresAsList[tcore_idx].id

    compartment_idxs = np.arange(len(compartment_ids))
    spikes = axon.map_spikes(compartment_idxs)

    for compartment_id, spike in zip(compartment_ids, spikes):
        if spike is None:
            continue  # this compartment does not route through these axons

        taxon_idx = spike.axon_idx
        taxon_id = taxon_ids[taxon_idx]
        atom = int(spike.atom)
        n_atoms = synapse.axon_populations(taxon_idx)

        if taxon_id is None:
            continue  # this connects to a dummy axon, so do not build

        if synapse.pop_type == 0:  # discrete
            assert atom == 0
            assert n_atoms == 1
            nxsdk_core.createDiscreteAxon(
                srcCxId=compartment_id,
                dstChipId=tchip_id,
                dstCoreId=tcore_id,
                dstSynMapId=taxon_id,
            )

        elif synapse.pop_type in (16, 32):
            n_blocks = len(core.blocks)
            assert n_blocks == 0 or (n_blocks == 1 and block is core.blocks[0])

            # pop_id is a unique index for the population. Must be the same for
            # all axons going to the same target synmap (with different atoms
            # of course), but otherwise different for all axons. Also, each
            # compartment can only have axons belonging to one population.
            if compartment_id not in pop_id_map:
                # If there's already an axon going to this synmap, use that
                # pop_id. Otherwise, make a new pop_id
                pop_key = (tchip_id, tcore_id, taxon_id)
                if pop_key not in pop_id_map:
                    pop_id = max(pop_id_map.values()) + 1 if len(pop_id_map) > 0 else 0
                    pop_id_map[pop_key] = pop_id
                pop_id_map[compartment_id] = pop_id_map[pop_key]
            pop_id = pop_id_map[compartment_id]

            create_axon_kwargs = dict(
                popId=pop_id,
                srcCxId=compartment_id,
                srcRelCxId=atom,
                dstChipId=tchip_id,
                dstCoreId=tcore_id,
                dstSynMapId=taxon_id,
            )
            if synapse.pop_type == 16:
                nxsdk_core.createPop16Axon(**create_axon_kwargs)
            else:
                nxsdk_core.createPop32Axon(**create_axon_kwargs)
        else:
            raise BuildError(f"{synapse}: unrecognized pop_type: {synapse.pop_type}")


def build_probe(nxsdk_board, board, probe, use_snips):
    key_map = {"current": "u", "voltage": "v", "spiked": "spike"}
    assert probe.key in key_map, "probe key not found"
    key = key_map[probe.key]

    assert probe not in board.probe_map
    if use_snips:
        probe_snip = ProbeSnip(key)
        board.probe_map[probe] = probe_snip
    else:
        board.probe_map[probe] = []

    assert len(probe.target) == len(probe.slice) == len(probe.weights)
    for k, target in enumerate(probe.target):
        chip_idx, core_idx, block_idx, compartment_idxs, _ = board.find_block(target)
        assert chip_idx is not None, "Could not find probe target on board"

        nxsdk_chip = nxsdk_board.n2Chips[chip_idx]
        nxsdk_core = nxsdk_chip.n2Cores[core_idx]

        r = compartment_idxs[probe.slice[k]]
        if use_snips:
            probe_snip.chip_idx.append(chip_idx)
            probe_snip.core_id.append(nxsdk_core.id)
            probe_snip.compartment_idxs.append(r)
        else:
            nxsdk_probe = nxsdk_board.monitor.probe(nxsdk_core.cxState, r, key)
            board.probe_map[probe].append(nxsdk_probe)


class ProbeSnip:
    def __init__(self, key):
        assert key in ("u", "v", "spike")
        self.key = key

        self.chip_idx = []
        self.core_id = []
        self.compartment_idxs = []

        self.snip_range = []
