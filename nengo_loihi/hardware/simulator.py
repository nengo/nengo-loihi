from __future__ import division

import logging
import os
import time
import warnings

import jinja2
import numpy as np

from nengo_loihi.hardware.builder import build_board, one_to_one_allocator
from nengo_loihi.hardware.nxsdk_shim import nxsdk_dir

logger = logging.getLogger(__name__)


class LoihiSimulator(object):
    """
    Simulator to place CxModel onto board and run it.
    """
    def __init__(self, cx_model, seed=None, snip_max_spikes_per_step=50):
        self.n2board = None
        self._probe_filters = {}
        self._probe_filter_pos = {}
        self.snip_max_spikes_per_step = snip_max_spikes_per_step

        self.cwd = os.getcwd()
        logger.debug("cd to %s", nxsdk_dir)
        os.chdir(nxsdk_dir)

        if seed is not None:
            warnings.warn("Seed will be ignored when running on Loihi")

        self.build(cx_model, seed=seed)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def build(self, cx_model, seed=None):
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

    def run_steps(self, steps, async=False):
        # NOTE: we need to call connect() after snips are created
        self.connect()
        self.n2board.run(steps, async=async)

    def wait_for_completion(self):
        self.n2board.finishRun()

    def is_connected(self):
        return self.n2board is not None and self.n2board.nxDriver.hasStarted()

    def connect(self, attempts=10):
        if self.n2board is None:
            raise RuntimeError("Must build model before running")

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
            raise RuntimeError("Could not connect to the board")

    def close(self):
        self.n2board.disconnect()
        # TODO: can we chdir back earlier?
        if self.cwd is not None:
            logger.debug("cd to %s", self.cwd)
            os.chdir(self.cwd)
            self.cwd = None

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
        for core in self.board.chips[0].cores:  # TODO: don't assume 1 chip
            if core.learning_coreid:
                n_errors += 1

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

        size = self.snip_max_spikes_per_step * 2 + 1 + n_errors*2
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
