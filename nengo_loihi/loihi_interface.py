from __future__ import division

from distutils.version import LooseVersion
import logging
import os
import sys
import time
import warnings

import jinja2
import numpy as np

from nengo.exceptions import SimulationError

try:
    import nxsdk
    from nxsdk.arch.n2a.compiler.tracecfggen.tracecfggen import TraceCfgGen
    from nxsdk.arch.n2a.graph.graph import N2Board
    from nxsdk.arch.n2a.graph.inputgen import BasicSpikeGenerator
    from nxsdk.arch.n2a.graph.probes import N2SpikeProbe

except ImportError:
    exc_info = sys.exc_info()

    def no_nxsdk(*args, **kwargs):
        raise exc_info[1]
    nxsdk = N2Board = BasicSpikeGenerator = TraceCfgGen = N2SpikeProbe = (
        no_nxsdk)

from nengo_loihi.allocators import one_to_one_allocator
from nengo_loihi.loihi_api import (
    CX_PROFILES_MAX, VTH_PROFILES_MAX, bias_to_manexp)

logger = logging.getLogger(__name__)


def build_board(board):
    n_chips = board.n_chips()
    n_cores_per_chip = board.n_cores_per_chip()
    n_synapses_per_core = board.n_synapses_per_core()

    n2board = N2Board(
        board.board_id, n_chips, n_cores_per_chip, n_synapses_per_core)

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
        if synapse.tracing and firstLearningIndex is None:
            firstLearningIndex = core.synapse_axons[synapse][0]
            core.learning_coreid = n2core.id
            break

    numStdp = 0
    if firstLearningIndex is not None:
        for synapse in core.iterate_synapses():
            axons = np.array(core.synapse_axons[synapse])
            if synapse.tracing:
                numStdp += len(axons)
                assert np.all(len(axons) >= firstLearningIndex)
            else:
                assert np.all(len(axons) < firstLearningIndex)

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
        n2core.stdpUcodeMem[0].word = 0x00102108  # 2^-7 learn rate

        # stdpProfileCfg negative error
        n2core.stdpProfileCfg[1].configure(
            uCodePtr=1,
            decimateExp=0,
            numProducts=1,
            requireY=1,
            usesXepoch=1,
        )
        n2core.stdpUcodeMem[1].word = 0x00f02108  # 2^-7 learn rate

        tcg = TraceCfgGen()
        tc = tcg.genTraceCfg(
            tau=0,
            spikeLevelInt=0,
            spikeLevelFrac=0,
        )
        tc.writeToRegister(n2core.stdpPostCfg[0])

    # TODO: allocator should be checking that vmin, vmax are the same
    #   for all groups on a core
    n_cx = 0
    if len(core.groups) > 0:
        group0 = core.groups[0]
        vmin, vmax = group0.vmin, group0.vmax
        assert all(group.vmin == vmin for group in core.groups)
        assert all(group.vmax == vmax for group in core.groups)
        negVmLimit = np.log2(-vmin + 1)
        posVmLimit = (np.log2(vmax + 1) - 9) * 0.5
        assert int(negVmLimit) == negVmLimit
        assert int(posVmLimit) == posVmLimit

        noiseExp0 = group0.noiseExp0
        noiseMantOffset0 = group0.noiseMantOffset0
        noiseAtDendOrVm = group0.noiseAtDendOrVm
        assert all(group.noiseExp0 == noiseExp0 for group in core.groups)
        assert all(group.noiseMantOffset0 == noiseMantOffset0
                   for group in core.groups)
        assert all(group.noiseAtDendOrVm == noiseAtDendOrVm
                   for group in core.groups)

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

        for group, cx_idxs, ax_range in core.iterate_groups():
            build_group(n2core, core, group, cx_idxs, ax_range)
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


def build_group(n2core, core, group, cx_idxs, ax_range):
    assert group.scaleU is False
    assert group.scaleV is False

    logger.debug("Building %s on core.id=%d", group, n2core.id)

    for i, bias in enumerate(group.bias):
        bman, bexp = bias_to_manexp(bias)
        icx = core.cx_profile_idxs[group][i]
        ivth = core.vth_profile_idxs[group][i]

        ii = cx_idxs[i]
        n2core.cxCfg[ii].configure(
            bias=bman, biasExp=bexp, vthProfile=ivth, cxProfile=icx)

        phasex = 'phase%d' % (ii % 4,)
        n2core.cxMetaState[ii // 4].configure(**{phasex: 2})

    logger.debug("- Building %d synapses", len(group.synapses))
    for synapses in group.synapses:
        build_synapses(n2core, core, group, synapses, cx_idxs)

    logger.debug("- Building %d synapses", len(group.axons))
    for axons in group.axons:
        build_axons(n2core, core, group, axons, cx_idxs)

    logger.debug("- Building %d synapses", len(group.probes))
    for probe in group.probes:
        build_probe(n2core, core, group, probe, cx_idxs)


def build_input(n2core, core, spike_input, cx_idxs):
    assert len(spike_input.axons) > 0

    for axon in spike_input.axons:
        build_axons(n2core, core, spike_input, axon, cx_idxs)

    for probe in spike_input.probes:
        build_probe(n2core, core, spike_input, probe, cx_idxs)

    n2board = n2core.parent.parent

    if not hasattr(n2core, 'master_spike_gen'):
        # TODO: this is only needed if precompute=True
        n2core.master_spike_gen = BasicSpikeGenerator(n2board)

    # get core/axon ids
    axon_ids = []
    for axon in spike_input.axons:
        tchip_idx, tcore_idx, tsyn_idxs = core.board.find_synapses(axon.target)
        tchip = n2board.n2Chips[tchip_idx]
        tcore = tchip.n2Cores[tcore_idx]
        axon_ids.append([(tchip.id, tcore.id, tsyn_idx)
                         for tsyn_idx in tsyn_idxs])

    spike_input.spike_gen = n2core.master_spike_gen
    spike_input.axon_ids = axon_ids

    for i, spiked in enumerate(spike_input.spikes):
        for j, s in enumerate(spiked):
            if s:
                for output_axon in axon_ids:
                    n2core.master_spike_gen.addSpike(i, *output_axon[j])

    spike_input.sent_count = len(spike_input.spikes)


def build_synapses(n2core, core, group, synapses, cx_idxs):
    syn_idxs = core.synapse_axons[synapses]
    assert len(syn_idxs) == len(synapses.weights)

    synapse_fmt_idx = core.synapse_fmt_idxs[synapses]
    stdp_pre_cfg_idx = core.stdp_pre_cfg_idxs[synapses]

    target_cxs = set()
    s0 = core.synapse_entries[synapses][0]
    for a, syn_idx in enumerate(syn_idxs):
        wa = synapses.weights[a] // synapses.synapse_fmt.scale
        ia = synapses.indices[a]
        assert len(wa) == len(ia)

        assert np.all(wa <= 255) and np.all(wa >= -256), str(wa)
        for k, (w, i) in enumerate(zip(wa, ia)):
            n2core.synapses[s0 + k].configure(
                CIdx=cx_idxs[i],
                Wgt=w,
                synFmtId=synapse_fmt_idx,
                LrnEn=int(synapses.tracing),
            )
            target_cxs.add(cx_idxs[i])

        n2core.synapseMap[syn_idx].synapsePtr = s0
        n2core.synapseMap[syn_idx].synapseLen = len(wa)
        n2core.synapseMap[syn_idx].discreteMapEntry.configure()

        if synapses.tracing:
            assert core.stdp_pre_profile_idx is not None
            assert stdp_pre_cfg_idx is not None
            n2core.synapseMap[syn_idx+1].singleTraceEntry.configure(
                preProfile=core.stdp_pre_profile_idx, tcs=stdp_pre_cfg_idx)

        s0 += len(wa)

    if synapses.tracing:
        assert core.stdp_profile_idx is not None
        for target_cx in target_cxs:
            # TODO: check that no cx gets configured by multiple synapses
            n2core.stdpPostState[target_cx].configure(
                stdpProfile=core.stdp_profile_idx,
                traceProfile=3,  # TODO: why this value
            )


def build_axons(n2core, core, group, axons, cx_idxs):
    tchip_idx, tcore_idx, tsyn_idxs = core.board.find_synapses(axons.target)
    taxon_idxs = np.asarray(tsyn_idxs)[axons.target_inds]
    n2board = n2core.parent.parent
    tchip_id = n2board.n2Chips[tchip_idx].id
    tcore_id = n2board.n2Chips[tchip_idx].n2Cores[tcore_idx].id
    assert axons.n_axons == len(cx_idxs) == len(taxon_idxs)
    for i in range(axons.n_axons):
        n2core.createDiscreteAxon(
            cx_idxs[i], tchip_id, tcore_id, int(taxon_idxs[i]))


def build_probe(n2core, core, group, probe, cx_idxs):
    assert probe.key in ('u', 'v', 's')
    key_map = {'s': 'spike'}
    key = key_map.get(probe.key, probe.key)

    n2board = n2core.parent.parent
    r = cx_idxs[probe.slice]

    if probe.use_snip:
        probe.snip_info = dict(coreid=n2core.id, cxs=r)
    else:
        p = n2board.monitor.probe(n2core.cxState, r, key)
        core.board.map_probe(probe, p)


class LoihiSimulator(object):
    """Simulator to place a Model onto a Loihi board and run it.

    Parameters
    ----------
    cx_model : CxModel
        Model specification that will be placed on the Loihi board.
    seed : int, optional (Default: None)
        A seed for stochastic operations.

        .. warning :: Setting the seed has no effect on stochastic
                      operations run on the Loihi board.
    snip_max_spikes_per_step : int, optional (Default: 50)
        Maximum number of spikes that can be sent through
        the nengo_io_h2c channel on one timestep.
    """

    def __init__(self, cx_model, seed=None, snip_max_spikes_per_step=50):
        self.closed = False

        self.check_nxsdk_version()

        self.n2board = None
        self._probe_filters = {}
        self._probe_filter_pos = {}
        self.snip_max_spikes_per_step = snip_max_spikes_per_step

        nxsdk_dir = os.path.realpath(
            os.path.join(os.path.dirname(nxsdk.__file__), "..")
        )
        self.cwd = os.getcwd()
        logger.debug("cd to %s", nxsdk_dir)
        os.chdir(nxsdk_dir)

        if seed is not None:
            warnings.warn("Seed will be ignored when running on Loihi")

        # probeDict is a class attribute, so might contain things left over
        # from previous simulators
        N2SpikeProbe.probeDict.clear()

        self.build(cx_model, seed=seed)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    @staticmethod
    def check_nxsdk_version():
        # raise exception if nxsdk not installed
        if callable(nxsdk):
            nxsdk()

        # if installed, check version
        version = LooseVersion(getattr(nxsdk, "__version__", "0.0.0"))
        minimum = LooseVersion("0.7.0")
        max_tested = LooseVersion("0.7.0")
        if version < minimum:
            raise ImportError("nengo-loihi requires nxsdk>=%s, found %s"
                              % (minimum, version))
        elif version > max_tested:
            warnings.warn("nengo-loihi has not been tested with your nxsdk "
                          "version (%s); latest fully supported version is "
                          "%s" % (version, max_tested))

    def build(self, cx_model, seed=None):
        cx_model.validate()
        self.model = cx_model

        # --- allocate --
        # maps CxModel to cores and chips
        allocator = one_to_one_allocator  # one core per ensemble
        self.board = allocator(self.model)

        # --- build
        self.n2board = build_board(self.board)

    def print_cores(self):
        for j, n2chip in enumerate(self.n2board.n2Chips):
            print("Chip %d, id=%d" % (j, n2chip.id))
            for k, n2core in enumerate(n2chip.n2Cores):
                print("  Core %d, id=%d" % (k, n2core.id))

    def run_steps(self, steps, blocking=True):
        # NOTE: we need to call connect() after snips are created
        self.connect()
        self.n2board.run(steps, aSync=not blocking)

    def wait_for_completion(self):
        self.n2board.finishRun()

    def is_connected(self):
        return self.n2board is not None and self.n2board.nxDriver.hasStarted()

    def connect(self, attempts=10):
        if self.n2board is None:
            raise SimulationError("Must build model before running")

        if self.is_connected():
            return

        logger.info("Connecting to Loihi, max attempts: %d", attempts)
        for i in range(attempts):
            try:
                self.n2board.startDriver()
                if self.is_connected():
                    break
            except Exception as e:
                logger.info("Connection error: %s", e)
                time.sleep(1)
                logger.info("Retrying, attempt %d", i + 1)
        else:
            raise SimulationError("Could not connect to the board")

    def close(self):
        self.n2board.disconnect()
        # TODO: can we chdir back earlier?
        if self.cwd is not None:
            logger.debug("cd to %s", self.cwd)
            os.chdir(self.cwd)
            self.cwd = None

        self.closed = True

    def _filter_probe(self, cx_probe, data):
        dt = self.model.dt
        i = self._probe_filter_pos.get(cx_probe, 0)
        if i == 0:
            shape = data[0].shape
            synapse = cx_probe.synapse
            rng = None
            step = (synapse.make_step(shape, shape, dt, rng, dtype=data.dtype)
                    if synapse is not None else None)
            self._probe_filters[cx_probe] = step
        else:
            step = self._probe_filters[cx_probe]

        if step is None:
            self._probe_filter_pos[cx_probe] = i + len(data)
            return data
        else:
            filt_data = np.zeros_like(data)
            for k, x in enumerate(data):
                filt_data[k] = step((i + k) * dt, x)

            self._probe_filter_pos[cx_probe] = i + k
            return filt_data

    def get_probe_output(self, probe):
        cx_probe = self.model.objs[probe]['out']
        n2probe = self.board.probe_map[cx_probe]
        x = np.column_stack([p.timeSeries.data for p in n2probe])
        x = x if cx_probe.weights is None else np.dot(x, cx_probe.weights)
        return self._filter_probe(cx_probe, x)

    def create_io_snip(self):
        # snips must be created before connecting
        assert not self.is_connected()

        snips_dir = os.path.join(os.path.dirname(__file__), "snips")
        env = jinja2.Environment(
            trim_blocks=True,
            loader=jinja2.FileSystemLoader(snips_dir),
            keep_trailing_newline=True
        )
        template = env.get_template("nengo_io.c.template")

        # --- generate custom code
        # Determine which cores have learning
        n_errors = 0
        total_error_len = 0
        max_error_len = 0
        for core in self.board.chips[0].cores:  # TODO: don't assume 1 chip
            if core.learning_coreid:
                error_len = core.groups[0].n // 2
                max_error_len = max(error_len, max_error_len)
                n_errors += 1
                total_error_len += 2 + error_len

        n_outputs = 1
        probes = []
        cores = set()
        # TODO: should snip_range be stored on the probe?
        snip_range = {}
        for group in self.model.cx_groups.keys():
            for probe in group.probes:
                if probe.use_snip:
                    info = probe.snip_info
                    cores.add(info["coreid"])
                    snip_range[probe] = slice(n_outputs - 1,
                                              n_outputs + len(info["cxs"]) - 1)
                    for cx in info["cxs"]:
                        probes.append((n_outputs, info["coreid"], cx))
                        n_outputs += 1

        # --- write c file using template
        c_path = os.path.join(snips_dir, "nengo_io.c")
        logger.debug(
            "Creating %s with %d outputs, %d error, %d cores, %d probes",
            c_path, n_outputs, n_errors, len(cores), len(probes))
        code = template.render(
            n_outputs=n_outputs,
            n_errors=n_errors,
            max_error_len=max_error_len,
            cores=cores,
            probes=probes,
        )
        with open(c_path, 'w') as f:
            f.write(code)

        # --- create SNIP process and channels
        logger.debug("Creating nengo_io snip process")
        nengo_io = self.n2board.createProcess(
            name="nengo_io",
            cFilePath=c_path,
            includeDir=snips_dir,
            funcName="nengo_io",
            guardName="guard_io",
            phase="mgmt",
        )
        logger.debug("Creating nengo_learn snip process")
        self.n2board.createProcess(
            name="nengo_learn",
            cFilePath=os.path.join(snips_dir, "nengo_learn.c"),
            includeDir=snips_dir,
            funcName="nengo_learn",
            guardName="guard_learn",
            phase="preLearnMgmt",
        )

        size = self.snip_max_spikes_per_step * 2 + 1 + total_error_len
        logger.debug("Creating nengo_io_h2c channel")
        self.nengo_io_h2c = self.n2board.createChannel(b'nengo_io_h2c',
                                                       "int", size)
        logger.debug("Creating nengo_io_c2h channel")
        self.nengo_io_c2h = self.n2board.createChannel(b'nengo_io_c2h',
                                                       "int", n_outputs)
        self.nengo_io_h2c.connect(None, nengo_io)
        self.nengo_io_c2h.connect(nengo_io, None)
        self.nengo_io_c2h_count = n_outputs
        self.nengo_io_snip_range = snip_range
