from __future__ import division

from distutils.version import LooseVersion
import logging
import os
import time
import warnings

import jinja2
from nengo.exceptions import SimulationError
import numpy as np

from nengo_loihi.block import LoihiBlock, Probe
from nengo_loihi.discretize import scale_pes_errors
from nengo_loihi.hardware.allocators import one_to_one_allocator
from nengo_loihi.hardware.builder import build_board
from nengo_loihi.hardware.nxsdk_shim import (
    assert_nxsdk,
    nxsdk,
    N2SpikeProbe,
)
from nengo_loihi.validate import validate_model

logger = logging.getLogger(__name__)


class HardwareInterface(object):
    """Simulator to place a Model onto a Loihi board and run it.

    Parameters
    ----------
    model : Model
        Model specification that will be placed on the Loihi board.
    seed : int, optional (Default: None)
        A seed for stochastic operations.

        .. warning :: Setting the seed has no effect on stochastic
                      operations run on the Loihi board.
    """

    def __init__(self, model, use_snips=True, seed=None):
        self.closed = False
        self.use_snips = use_snips
        self.check_nxsdk_version()

        self.n2board = None
        self.nengo_io_h2c = None  # IO snip host-to-chip channel
        self.nengo_io_c2h = None  # IO snip chip-to-host channel
        self._probe_filters = {}
        self._probe_filter_pos = {}
        self._snip_probe_data = {}
        self._chip2host_sent_steps = 0

        # Maximum number of spikes that can be sent through
        # the nengo_io_h2c channel on one timestep.
        self.snip_max_spikes_per_step = 50

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

        self.build(model, seed=seed)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    @staticmethod
    def check_nxsdk_version():
        # raise exception if nxsdk not installed
        assert_nxsdk()

        # if installed, check version
        version = LooseVersion(getattr(nxsdk, "__version__", "0.0.0"))
        minimum = LooseVersion("0.7.0")
        max_tested = LooseVersion("0.7.5")
        if version < minimum:
            raise ImportError("nengo-loihi requires nxsdk>=%s, found %s"
                              % (minimum, version))
        elif version > max_tested:
            warnings.warn("nengo-loihi has not been tested with your nxsdk "
                          "version (%s); latest fully supported version is "
                          "%s" % (version, max_tested))

    def _iter_blocks(self):
        return iter(self.model.blocks)

    def _iter_probes(self):
        for block in self._iter_blocks():
            for probe in block.probes:
                yield probe

    def build(self, model, seed=None):
        validate_model(model)
        self.model = model
        self.pes_error_scale = getattr(model, 'pes_error_scale', 1.)

        if self.use_snips:
            # tag all probes as being snip-based,
            # having normal probes at the same time as snips causes problems
            for probe in self._iter_probes():
                probe.use_snip = True
                self._snip_probe_data[probe] = []

        # --- allocate --
        # maps Model to cores and chips
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
        if self.use_snips and self.nengo_io_h2c is None:
            self.create_io_snip()

        # NOTE: we need to call connect() after snips are created
        self.connect()
        self.n2board.run(steps, aSync=not blocking)

    def _chip2host_monitor(self, probes_receivers):
        increment = None
        for probe, receiver in probes_receivers.items():
            assert not probe.use_snip
            n2probe = self.board.probe_map[probe]
            x = np.column_stack([
                p.timeSeries.data[self._chip2host_sent_steps:]
                for p in n2probe])
            assert x.ndim == 2

            if len(x) > 0:
                if increment is None:
                    increment = len(x)
                else:
                    assert increment == len(x)

                if probe.weights is not None:
                    x = np.dot(x, probe.weights)

                for j in range(len(x)):
                    receiver.receive(
                        self.model.dt * (self._chip2host_sent_steps + j + 2),
                        x[j])

        if increment is not None:
            self._chip2host_sent_steps += increment

    def _chip2host_snips(self, probes_receivers):
        count = self.nengo_io_c2h_count
        data = self.nengo_io_c2h.read(count)
        time_step, data = data[0], np.array(data[1:], dtype=np.int32)
        snip_range = self.nengo_io_snip_range

        for probe in self._snip_probe_data:
            assert probe.use_snip
            x = data[snip_range[probe]]
            assert x.ndim == 1
            if probe.key == 'spiked':
                if isinstance(probe.target, LoihiBlock):
                    refract_delays = probe.target.compartment.refractDelay
                else:
                    refract_delays = 1

                # Loihi uses the voltage value to indicate where we
                # are in the refractory period. We want to find neurons
                # starting their refractory period.
                x = (x == refract_delays * 128)

            if probe.weights is not None:
                x = np.dot(x, probe.weights)

            receiver = probes_receivers.get(probe, None)
            if receiver is not None:
                # chip->host
                receiver.receive(self.model.dt * time_step, x)
            else:
                # onchip probes
                self._snip_probe_data[probe].append(x)

        self._chip2host_sent_steps += 1

    def chip2host(self, probes_receivers=None):
        if probes_receivers is None:
            probes_receivers = {}

        return (self._chip2host_snips(probes_receivers) if self.use_snips else
                self._chip2host_monitor(probes_receivers))

    def _host2chip_spikegen(self, loihi_spikes):
        # sort all spikes because spikegen needs them in temporal order
        loihi_spikes = sorted(loihi_spikes, key=lambda s: s.time)
        for spike in loihi_spikes:
            assert spike.axon.atom == 0, "Spikegen does not support atom"
            self.n2board.global_spike_generator.addSpike(
                time=spike.time, chipId=spike.axon.chip_id,
                coreId=spike.axon.core_id, axonId=spike.axon.axon_id)

    def _host2chip_snips(self, loihi_spikes, loihi_errors):
        max_spikes = self.snip_max_spikes_per_step
        if len(loihi_spikes) > max_spikes:
            warnings.warn(
                "Too many spikes (%d) sent in one timestep. Increase the "
                "value of `snip_max_spikes_per_step` (currently set to %d). "
                "See\n  https://www.nengo.ai/nengo-loihi/troubleshooting.html"
                "for details." % (len(loihi_spikes), max_spikes))
            del loihi_spikes[max_spikes:]

        msg = [len(loihi_spikes)]
        assert len(loihi_spikes) <= self.snip_max_spikes_per_step
        for spike in loihi_spikes:
            assert spike.axon.chip_id == 0
            msg.extend((spike.axon.core_id, spike.axon.axon_id))
        assert len(loihi_errors) == self.nengo_io_h2c_errors
        for error in loihi_errors:
            msg.extend(error)
        assert len(msg) <= self.nengo_io_h2c.numElements
        self.nengo_io_h2c.write(len(msg), msg)

    def host2chip(self, spikes, errors):
        loihi_spikes = []
        for spike_input, t, s in spikes:
            spike_input = self.n2board.spike_inputs[spike_input]
            loihi_spikes.extend(spike_input.spikes_to_loihi(t, s))

        loihi_errors = []
        for synapse, t, e in errors:
            coreid = None
            for core in self.board.chips[0].cores:
                for block in core.blocks:
                    if synapse in block.synapses:
                        # TODO: assumes one block per core
                        coreid = core.learning_coreid
                        break

                if coreid is not None:
                    break

            assert coreid is not None
            e = scale_pes_errors(e, scale=self.pes_error_scale)
            loihi_errors.append([coreid, len(e)] + e.tolist())

        if self.use_snips:
            return self._host2chip_snips(loihi_spikes, loihi_errors)
        else:
            assert len(loihi_errors) == 0
            return self._host2chip_spikegen(loihi_spikes)

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

    def _filter_probe(self, probe, data):
        dt = self.model.dt
        i = self._probe_filter_pos.get(probe, 0)
        if i == 0:
            shape = data[0].shape
            synapse = probe.synapse
            rng = None
            step = (synapse.make_step(shape, shape, dt, rng, dtype=data.dtype)
                    if synapse is not None else None)
            self._probe_filters[probe] = step
        else:
            step = self._probe_filters[probe]

        if step is None:
            self._probe_filter_pos[probe] = i + len(data)
            return data
        else:
            filt_data = np.zeros_like(data)
            for k, x in enumerate(data):
                filt_data[k] = step((i + k) * dt, x)

            self._probe_filter_pos[probe] = i + k
            return filt_data

    def get_probe_output(self, probe):
        assert isinstance(probe, Probe)
        if probe.use_snip:
            data = self._snip_probe_data[probe]
            if probe.synapse is not None:
                return probe.synapse.filt(data, dt=self.model.dt, y0=0)
            else:
                return data
        n2probe = self.board.probe_map[probe]
        x = np.column_stack([p.timeSeries.data for p in n2probe])
        x = x if probe.weights is None else np.dot(x, probe.weights)
        return self._filter_probe(probe, x)

    def create_io_snip(self):
        # snips must be created before connecting
        assert not self.is_connected(), "still connected"

        snips_dir = os.path.join(
            os.path.dirname(__file__), "snips")
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
                error_len = core.blocks[0].n_neurons // 2
                max_error_len = max(error_len, max_error_len)
                n_errors += 1
                total_error_len += 2 + error_len

        n_outputs = 1
        probes = []
        cores = set()
        # TODO: should snip_range be stored on the probe?
        snip_range = {}
        for block in self.model.blocks:
            for probe in block.probes:
                if probe.use_snip:
                    info = probe.snip_info
                    assert info['key'] in ('u', 'v', 'spike')
                    # For spike probes, we record V and determine if the neuron
                    # spiked in Simulator.
                    cores.add(info["coreid"])
                    snip_range[probe] = slice(n_outputs - 1,
                                              n_outputs + len(info["cxs"]) - 1)
                    for cx in info["cxs"]:
                        probes.append(
                            (n_outputs, info["coreid"], cx, info['key']))
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
        self.nengo_io_h2c_errors = n_errors
        self.nengo_io_c2h_count = n_outputs
        self.nengo_io_snip_range = snip_range
