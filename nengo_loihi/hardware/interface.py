import collections
from distutils.version import LooseVersion
import logging
import os
import shutil
import socket
import struct
import tempfile
import time
import warnings

import jinja2
from nengo.exceptions import SimulationError
import numpy as np

from nengo_loihi.block import LoihiBlock, Probe
from nengo_loihi.compat import make_process_step
from nengo_loihi.discretize import scale_pes_errors
from nengo_loihi.hardware.allocators import OneToOne, RoundRobin
from nengo_loihi.hardware.builder import build_board
from nengo_loihi.nxsdk_obfuscation import d, d_func, d_get
from nengo_loihi.hardware.nxsdk_shim import assert_nxsdk, nxsdk, SnipPhase, SpikeProbe
from nengo_loihi.hardware.validate import validate_board
from nengo_loihi.validate import validate_model

logger = logging.getLogger(__name__)


def ceil_div(a, b):
    return -((-a) // b)


def roundup(a, b):
    return b * ceil_div(a, b)


class HardwareInterface:
    """Place a Model onto a Loihi board and run it.

    Parameters
    ----------
    model : Model
        Model specification that will be placed on the Loihi board.
    use_snips : boolean, optional (Default: True)
        Whether to use snips (e.g., for ``precompute=False``).
    seed : int, optional (Default: None)
        A seed for stochastic operations.
    snip_max_spikes_per_step : int
        The maximum number of spikes that can be sent to the chip in one
        timestep if ``.use_snips`` is True.
    allocator : Allocator, optional (Default: ``OneToOne()``)
        Callable object that allocates the board's devices to given models.
        Defaults to one block and one input per core on a single chip.
    """

    min_nxsdk_version = LooseVersion("0.8.7")
    max_nxsdk_version = LooseVersion("0.9.0")
    channel_packet_size = 64  # size of channel packets in int32s

    def __init__(
        self,
        model,
        use_snips=True,
        seed=None,
        snip_max_spikes_per_step=50,
        allocator=OneToOne(),
    ):
        if isinstance(allocator, RoundRobin) and use_snips:
            raise SimulationError(
                "snips are not supported for the " "RoundRobin allocator"
            )

        self.closed = False
        self.nxsdk_board = None
        self.nengo_io_h2c = None  # IO snip host-to-chip channel
        self.nengo_io_c2h = None  # IO snip chip-to-host channel
        self.host_socket = None  # IO snip superhost (this) <-> host socket
        self.host_socket_connected = False
        self.host_socket_port = None
        self._probe_filters = {}
        self._probe_filter_pos = {}
        self._snip_probe_data = collections.OrderedDict()
        self._chip2host_sent_steps = 0

        self.model = model
        self.use_snips = use_snips
        self.seed = seed
        # Maximum number of spikes that can be sent through
        # the nengo_io_h2c channel on one timestep.
        self.snip_max_spikes_per_step = snip_max_spikes_per_step
        self.allocator = allocator

        self.check_nxsdk_version()

        validate_model(self.model)

        # clear cached content from SpikeProbe class attribute
        d_func(SpikeProbe, b"cHJvYmVEaWN0", b"Y2xlYXI=")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    @classmethod
    def check_nxsdk_version(cls):
        # raise exception if nxsdk not installed
        assert_nxsdk()

        # if installed, check version
        version = LooseVersion(getattr(nxsdk, "__version__", "0.0.0"))
        if version < cls.min_nxsdk_version:
            raise ImportError(
                "nengo-loihi requires nxsdk>=%s, found %s"
                % (cls.min_nxsdk_version, version)
            )
        elif version > cls.max_nxsdk_version:
            warnings.warn(
                "nengo-loihi has not been tested with your nxsdk "
                "version (%s); latest fully supported version is "
                "%s" % (version, cls.max_nxsdk_version)
            )

    def _iter_probes(self):
        for block in self.model.blocks:
            for probe in block.probes:
                yield probe

    @property
    def is_built(self):
        return self.nxsdk_board is not None

    def build(self):
        assert self.nxsdk_board is None, "Cannot rebuild model"

        self.pes_error_scale = getattr(self.model, "pes_error_scale", 1.0)

        if self.use_snips:
            # tag all probes as being snip-based,
            # having normal probes at the same time as snips causes problems
            for probe in self._iter_probes():
                probe.use_snip = True
                self._snip_probe_data[probe] = []

        # --- allocate
        self.board = self.allocator(self.model)

        # --- validate
        validate_board(self.board)

        # --- build
        self.nxsdk_board = build_board(
            self.board, use_snips=self.use_snips, seed=self.seed
        )

        # --- create snips
        if self.use_snips:
            self.create_io_snip()

    def run_steps(self, steps, blocking=True):
        if not self.is_built:
            self.build()

        self.connect()  # returns immediately if already connected

        # start the board running the desired number of steps
        d_get(self.nxsdk_board, b"cnVu")(steps, **{d(b"YVN5bmM="): not blocking})

        # connect to host socket
        if self.host_socket is not None and not self.host_socket_connected:
            # pause to allow host snip to start and listen for connection
            time.sleep(0.1)

            host_address = self.nxsdk_board.executor._host_coordinator.hostAddr
            print(
                "Connecting to host socket at (%s, %s)"
                % (host_address, self.host_socket_port)
            )
            self.host_socket.connect((host_address, self.host_socket_port))
            self.host_socket_connected = True

    def _chip2host_monitor(self, probes_receivers):
        increment = None
        for probe, receiver in probes_receivers.items():
            assert not probe.use_snip
            nxsdk_probe = self.board.probe_map[probe]
            x = np.column_stack(
                [
                    d_get(p, b"dGltZVNlcmllcw==", b"ZGF0YQ==")[
                        self._chip2host_sent_steps :
                    ]
                    for p in nxsdk_probe
                ]
            )
            assert x.ndim == 2

            if len(x) > 0:
                if increment is None:
                    increment = len(x)

                assert increment == len(x), "All x need same number of steps"

                if probe.weights is not None:
                    x = np.dot(x, probe.weights)

                for j in range(len(x)):
                    receiver.receive(
                        self.model.dt * (self._chip2host_sent_steps + j + 2), x[j]
                    )

        if increment is not None:
            self._chip2host_sent_steps += increment

    def _chip2host_snips(self, probes_receivers):
        assert self.host_socket_connected

        recv_size = 4096  # python docs recommend small power of 2, e.g. 4096
        received = self.host_socket.recv(recv_size)  # blocking recv call
        data = received
        while len(received) == recv_size:
            received = self.host_socket.recv(recv_size, flags=socket.MSG_DONTWAIT)
            data += received
        assert len(data) == 4 * self.nengo_io_c2h_count

        data = np.frombuffer(data, dtype=np.int32)
        time_step, data = data[0], data[1:]
        snip_range = self.nengo_io_snip_range

        for probe in self._snip_probe_data:
            assert probe.use_snip
            x = data[snip_range[probe]]
            assert x.ndim == 1
            if probe.key == "spiked":
                assert isinstance(probe.target, LoihiBlock)
                refract_delays = probe.target.compartment.refract_delay

                # Loihi uses the voltage value to indicate where we
                # are in the refractory period. We want to find neurons
                # starting their refractory period.
                x = x == refract_delays * d(b"MTI4", int)

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

    def chip2host(self, probes_receivers):
        return (
            self._chip2host_snips(probes_receivers)
            if self.use_snips
            else self._chip2host_monitor(probes_receivers)
        )

    def _host2chip_spikegen(self, loihi_spikes):
        # sort all spikes because spikegen needs them in temporal order
        loihi_spikes = sorted(loihi_spikes, key=lambda s: s.time)
        for spike in loihi_spikes:
            assert spike.axon.axon_type == 0, "Spikegen cannot send pop spikes"
            assert spike.axon.atom == 0, "Spikegen does not support atom"
            d_func(
                self.nxsdk_board.global_spike_generator,
                b"YWRkU3Bpa2U=",
                kwargs={
                    b"dGltZQ==": spike.time,
                    b"Y2hpcElk": spike.axon.chip_id,
                    b"Y29yZUlk": spike.axon.core_id,
                    b"YXhvbklk": spike.axon.axon_id,
                },
            )

    def _host2chip_snips(self, loihi_spikes, loihi_errors):
        assert self.host_socket_connected

        max_spikes = self.snip_max_spikes_per_step
        if len(loihi_spikes) > max_spikes:
            warnings.warn(
                "Too many spikes (%d) sent in one timestep. Increase the "
                "value of `snip_max_spikes_per_step` (currently set to %d). "
                "See\n  https://www.nengo.ai/nengo-loihi/configuration.html\n"
                "for details." % (len(loihi_spikes), max_spikes)
            )
            del loihi_spikes[max_spikes:]

        msg = [len(loihi_spikes)]
        assert len(loihi_spikes) <= self.snip_max_spikes_per_step
        for spike in loihi_spikes:
            assert spike.axon.chip_id == 0
            msg.extend(SpikePacker.pack(spike))
        assert len(loihi_errors) == self.nengo_io_h2c_errors
        for error in loihi_errors:
            msg.extend(error)

        msg_bytes = struct.pack("%di" % len(msg), *msg)
        i_sent = 0
        while i_sent < len(msg_bytes):
            n_sent = self.host_socket.send(msg_bytes[i_sent:])
            i_sent += n_sent

    def host2chip(self, spikes, errors):
        loihi_spikes = []
        for spike_input, t, s in spikes:
            spike_input = self.nxsdk_board.spike_inputs[spike_input]
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
        d_func(self.nxsdk_board, b"ZmluaXNoUnVu")

    def is_connected(self):
        return self.nxsdk_board is not None and d_func(
            self.nxsdk_board, b"ZXhlY3V0b3I=", b"aGFzU3RhcnRlZA=="
        )

    def connect(self, attempts=10):
        if self.nxsdk_board is None:
            raise SimulationError("Must build model before running")

        if self.is_connected():
            return

        logger.info("Connecting to Loihi, max attempts: %d", attempts)
        for i in range(attempts):
            try:
                d_func(self.nxsdk_board, b"c3RhcnQ=")
                if self.is_connected():
                    break
            except Exception as e:
                logger.warning("Connection error: %s", e)
                time.sleep(1)
                logger.info("Retrying, attempt %d", i + 1)
        else:
            raise SimulationError("Could not connect to the board")

    def close(self):
        if self.host_socket is not None and self.host_socket_connected:
            # send -1 to signal host/chip that we're done
            self.host_socket.send(struct.pack("i", -1))

            # pause to allow chip to receive -1 signal via host
            time.sleep(0.1)

            self.host_socket.close()
            self.host_socket_connected = False

        if self.nxsdk_board is not None:
            d_func(self.nxsdk_board, b"ZGlzY29ubmVjdA==")

        self.closed = True

    def _filter_probe(self, probe, data):
        dt = self.model.dt
        shape = data[0].shape
        i = self._probe_filter_pos.get(probe, 0)
        if i == 0:
            synapse = probe.synapse
            rng = None
            step = (
                make_process_step(synapse, shape, shape, dt, rng, dtype=np.float32)
                if synapse is not None
                else None
            )
            self._probe_filters[probe] = step
        else:
            step = self._probe_filters[probe]

        if step is None:
            self._probe_filter_pos[probe] = i + len(data)
            return data
        else:
            filt_data = np.zeros((len(data),) + shape, dtype=np.float32)
            for k, x in enumerate(data):
                filt_data[k] = step((i + k) * dt, x)

            self._probe_filter_pos[probe] = i + k
            return filt_data

    def get_probe_output(self, probe):
        assert isinstance(probe, Probe)
        if probe.use_snip:
            data = self._snip_probe_data[probe]
        else:
            nxsdk_probe = self.board.probe_map[probe]
            data = np.column_stack(
                [d_get(p, b"dGltZVNlcmllcw==", b"ZGF0YQ==") for p in nxsdk_probe]
            )
            data = data if probe.weights is None else np.dot(data, probe.weights)
        return self._filter_probe(probe, data)

    def create_io_snip(self):
        # snips must be created before connecting
        assert not self.is_connected(), "still connected"

        snips_dir = os.path.join(os.path.dirname(__file__), "snips")
        env = jinja2.Environment(
            trim_blocks=True,
            loader=jinja2.FileSystemLoader(snips_dir),
            keep_trailing_newline=True,
        )

        # --- generate custom code
        # Determine which cores have learning
        n_errors = 0
        total_error_len = 0
        max_error_len = 0
        assert len(self.board.chips) == 1, "Learning not implemented for multiple chips"
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
                    assert info["key"] in ("u", "v", "spike")
                    # For spike probes, we record V and determine if the neuron
                    # spiked in Simulator.
                    cores.add(info["core_id"])
                    snip_range[probe] = slice(
                        n_outputs - 1, n_outputs + len(info["compartment_idxs"]) - 1
                    )
                    for compartment in info["compartment_idxs"]:
                        probes.append(
                            (n_outputs, info["core_id"], compartment, info["key"])
                        )
                        n_outputs += 1

        # obfuscated strings used in templates
        obfs = dict(
            core_class=d(b"TmV1cm9uQ29yZQ=="),
            id_class=d(b"Q29yZUlk"),
            get_channel=d(b"Z2V0Q2hhbm5lbElE"),
            int_type=d(b"aW50MzJfdA=="),
            spike_size=d(b"Mg=="),
            error_info_size=d(b"Mg==", int),
            step=d(b"dGltZV9zdGVw"),
            read=d(b"cmVhZENoYW5uZWw="),
            write=d(b"d3JpdGVDaGFubmVs"),
            spike_shift=d(b"MTY="),
            spike_mask=d(b"MHgwMDAwRkZGRg=="),
            axon_type_0=d(b"MA=="),
            do_axon_type_0=d(b"bnhfc2VuZF9kaXNjcmV0ZV9zcGlrZQ=="),
            axon_type_1=d(b"MzI="),
            do_axon_type_1=d(b"bnhfc2VuZF9wb3AzMl9zcGlrZQ=="),
            data=d(b"dXNlckRhdGE="),
            state=d(b"Y3hfc3RhdGU="),
            neuron=d(b"TkVVUk9OX1BUUg=="),
            # pylint: disable=line-too-long
            pos_pes_cfg=d(
                b"bmV1cm9uLT5zdGRwX3Bvc3Rfc3RhdGVbY29tcGFydG1lbnRfaWR4XSA9ICAgICAgICAgICAgICAgICAgICAgKFBvc3RUcmFjZUVudHJ5KSB7CiAgICAgICAgICAgICAgICAgICAgICAgIC5Zc3Bpa2UwICAgICAgPSAwLAogICAgICAgICAgICAgICAgICAgICAgICAuWXNwaWtlMSAgICAgID0gMCwKICAgICAgICAgICAgICAgICAgICAgICAgLllzcGlrZTIgICAgICA9IDAsCiAgICAgICAgICAgICAgICAgICAgICAgIC5ZZXBvY2gwICAgICAgPSBhYnMoZXJyb3IpLAogICAgICAgICAgICAgICAgICAgICAgICAuWWVwb2NoMSAgICAgID0gMCwKICAgICAgICAgICAgICAgICAgICAgICAgLlllcG9jaDIgICAgICA9IDAsCiAgICAgICAgICAgICAgICAgICAgICAgIC5Uc3Bpa2UgICAgICAgPSAwLAogICAgICAgICAgICAgICAgICAgICAgICAuVHJhY2VQcm9maWxlID0gMywKICAgICAgICAgICAgICAgICAgICAgICAgLlN0ZHBQcm9maWxlICA9IDEKICAgICAgICAgICAgICAgICAgICB9OwogICAgICAgICAgICAgICAgbmV1cm9uLT5zdGRwX3Bvc3Rfc3RhdGVbY29tcGFydG1lbnRfaWR4K25fdmFsc10gPSAgICAgICAgICAgICAgICAgICAgIChQb3N0VHJhY2VFbnRyeSkgewogICAgICAgICAgICAgICAgICAgICAgICAuWXNwaWtlMCAgICAgID0gMCwKICAgICAgICAgICAgICAgICAgICAgICAgLllzcGlrZTEgICAgICA9IDAsCiAgICAgICAgICAgICAgICAgICAgICAgIC5Zc3Bpa2UyICAgICAgPSAwLAogICAgICAgICAgICAgICAgICAgICAgICAuWWVwb2NoMCAgICAgID0gYWJzKGVycm9yKSwKICAgICAgICAgICAgICAgICAgICAgICAgLlllcG9jaDEgICAgICA9IDAsCiAgICAgICAgICAgICAgICAgICAgICAgIC5ZZXBvY2gyICAgICAgPSAwLAogICAgICAgICAgICAgICAgICAgICAgICAuVHNwaWtlICAgICAgID0gMCwKICAgICAgICAgICAgICAgICAgICAgICAgLlRyYWNlUHJvZmlsZSA9IDMsCiAgICAgICAgICAgICAgICAgICAgICAgIC5TdGRwUHJvZmlsZSAgPSAwCiAgICAgICAgICAgICAgICAgICAgfTs="
            ),
            # pylint: disable=line-too-long
            neg_pes_cfg=d(
                b"bmV1cm9uLT5zdGRwX3Bvc3Rfc3RhdGVbY29tcGFydG1lbnRfaWR4XSA9ICAgICAgICAgICAgICAgICAgICAgKFBvc3RUcmFjZUVudHJ5KSB7CiAgICAgICAgICAgICAgICAgICAgICAgIC5Zc3Bpa2UwICAgICAgPSAwLAogICAgICAgICAgICAgICAgICAgICAgICAuWXNwaWtlMSAgICAgID0gMCwKICAgICAgICAgICAgICAgICAgICAgICAgLllzcGlrZTIgICAgICA9IDAsCiAgICAgICAgICAgICAgICAgICAgICAgIC5ZZXBvY2gwICAgICAgPSBhYnMoZXJyb3IpLAogICAgICAgICAgICAgICAgICAgICAgICAuWWVwb2NoMSAgICAgID0gMCwKICAgICAgICAgICAgICAgICAgICAgICAgLlllcG9jaDIgICAgICA9IDAsCiAgICAgICAgICAgICAgICAgICAgICAgIC5Uc3Bpa2UgICAgICAgPSAwLAogICAgICAgICAgICAgICAgICAgICAgICAuVHJhY2VQcm9maWxlID0gMywKICAgICAgICAgICAgICAgICAgICAgICAgLlN0ZHBQcm9maWxlICA9IDAKICAgICAgICAgICAgICAgICAgICB9OwogICAgICAgICAgICAgICAgbmV1cm9uLT5zdGRwX3Bvc3Rfc3RhdGVbY29tcGFydG1lbnRfaWR4K25fdmFsc10gPSAgICAgICAgICAgICAgICAgICAgIChQb3N0VHJhY2VFbnRyeSkgewogICAgICAgICAgICAgICAgICAgICAgICAuWXNwaWtlMCAgICAgID0gMCwKICAgICAgICAgICAgICAgICAgICAgICAgLllzcGlrZTEgICAgICA9IDAsCiAgICAgICAgICAgICAgICAgICAgICAgIC5Zc3Bpa2UyICAgICAgPSAwLAogICAgICAgICAgICAgICAgICAgICAgICAuWWVwb2NoMCAgICAgID0gYWJzKGVycm9yKSwKICAgICAgICAgICAgICAgICAgICAgICAgLlllcG9jaDEgICAgICA9IDAsCiAgICAgICAgICAgICAgICAgICAgICAgIC5ZZXBvY2gyICAgICAgPSAwLAogICAgICAgICAgICAgICAgICAgICAgICAuVHNwaWtlICAgICAgID0gMCwKICAgICAgICAgICAgICAgICAgICAgICAgLlRyYWNlUHJvZmlsZSA9IDMsCiAgICAgICAgICAgICAgICAgICAgICAgIC5TdGRwUHJvZmlsZSAgPSAxCiAgICAgICAgICAgICAgICAgICAgfTs="
            ),
        )

        n_output_packets = ceil_div(n_outputs, self.channel_packet_size)

        # --- write c file using template
        template = env.get_template("nengo_io.c.template")
        self.tmp_snip_dir = tempfile.TemporaryDirectory()
        c_path = os.path.join(self.tmp_snip_dir.name, "nengo_io.c")
        logger.debug(
            "Creating %s with %d outputs, %d error, %d cores, %d probes",
            c_path,
            n_outputs,
            n_errors,
            len(cores),
            len(probes),
        )
        chip_buffer_size = roundup(
            max(
                n_outputs,  # currently, buffer needs to hold all outputs at once
                self.channel_packet_size
                + max(SpikePacker.size(), obfs["error_info_size"]),
            ),
            self.channel_packet_size,
        )
        code = template.render(
            n_outputs=n_outputs,
            n_output_packets=n_output_packets,
            n_errors=n_errors,
            max_error_len=max_error_len,
            buffer_size=chip_buffer_size,
            packet_size=self.channel_packet_size,
            cores=cores,
            probes=probes,
            obfs=obfs,
        )
        with open(c_path, "w") as f:
            f.write(code)

        template = env.get_template("nengo_learn.c.template")
        code = template.render(obfs=obfs)
        with open(os.path.join(snips_dir, "nengo_learn.c"), "w") as f:
            f.write(code)

        # --- create chip processes
        logger.debug("Creating nengo_io chip process")
        nengo_io = d_func(
            self.nxsdk_board,
            b"Y3JlYXRlU25pcA==",
            kwargs={
                b"bmFtZQ==": "nengo_io",
                b"Y0ZpbGVQYXRo": c_path,
                b"aW5jbHVkZURpcg==": snips_dir,
                b"ZnVuY05hbWU=": "nengo_io",
                b"Z3VhcmROYW1l": "guard_io",
                b"cGhhc2U=": d_get(SnipPhase, b"RU1CRURERURfTUdNVA=="),
            },
        )
        logger.debug("Creating nengo_learn chip process")
        c_path = os.path.join(self.tmp_snip_dir.name, "nengo_learn.c")
        shutil.copyfile(os.path.join(snips_dir, "nengo_learn.c"), c_path)
        d_func(
            self.nxsdk_board,
            b"Y3JlYXRlU25pcA==",
            kwargs={
                b"bmFtZQ==": "nengo_learn",
                b"Y0ZpbGVQYXRo": os.path.join(snips_dir, "nengo_learn.c"),
                b"aW5jbHVkZURpcg==": snips_dir,
                b"ZnVuY05hbWU=": "nengo_learn",
                b"Z3VhcmROYW1l": "guard_learn",
                b"cGhhc2U=": d_get(SnipPhase, b"RU1CRURERURfUFJFTEVBUk5fTUdNVA=="),
            },
        )

        # --- create host process (for faster communication via sockets)
        host_socket_port = np.random.randint(50000, 60000)

        max_inputs = n_errors + self.snip_max_spikes_per_step * SpikePacker.size()
        host_buffer_size = roundup(max(max_inputs, n_outputs), self.channel_packet_size)

        host_template = env.get_template("nengo_host.cc.template")
        host_path = os.path.join(self.tmp_snip_dir.name, "nengo_host.cc")
        host_code = host_template.render(
            host_buffer_size=host_buffer_size,
            n_outputs=n_outputs,
            n_output_packets=n_output_packets,
            server_port=host_socket_port,
            input_channel="nengo_io_h2c",
            output_channel="nengo_io_c2h",
            packet_size=self.channel_packet_size,
            obfs=obfs,
        )
        with open(host_path, "w") as f:
            f.write(host_code)

        # make process
        host_process = d_func(
            self.nxsdk_board,
            b"Y3JlYXRlU25pcA==",
            kwargs={
                b"cGhhc2U=": SnipPhase.HOST_CONCURRENT_EXECUTION,
                b"Y3BwRmlsZQ==": host_path,
            },
        )

        # connect to host socket
        self.host_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.host_socket_port = host_socket_port

        # --- create channels
        input_channel_size = (
            1  # first int stores number of spikes
            + self.snip_max_spikes_per_step * SpikePacker.size()
            + total_error_len
        )
        logger.debug("Creating nengo_io_h2c channel (%d)" % input_channel_size)
        self.nengo_io_h2c = d_get(self.nxsdk_board, b"Y3JlYXRlQ2hhbm5lbA==")(
            b"nengo_io_h2c",  # channel name
            **{
                # channel size (in elements)
                d(b"bnVtRWxlbWVudHM="): input_channel_size,
                # size of one packet (in bytes)
                d(b"bWVzc2FnZVNpemU="): 4 * self.channel_packet_size,
                # size of send/receive buffer on chip/host (in packets)
                d(b"c2xhY2s="): 16,
            },
        )
        logger.debug("Creating nengo_io_c2h channel (%d)" % n_outputs)
        self.nengo_io_c2h = d_get(self.nxsdk_board, b"Y3JlYXRlQ2hhbm5lbA==")(
            b"nengo_io_c2h",  # channel name
            **{
                # channel size (in elements)
                d(b"bnVtRWxlbWVudHM="): n_outputs,
                # size of one packet (in bytes)
                d(b"bWVzc2FnZVNpemU="): 4 * self.channel_packet_size,
                # size of send/receive buffer on chip/host (in packets)
                d(b"c2xhY2s="): 16,
            },
        )
        d_get(self.nengo_io_h2c, b"Y29ubmVjdA==")(host_process, nengo_io)
        d_get(self.nengo_io_c2h, b"Y29ubmVjdA==")(nengo_io, host_process)
        self.nengo_io_h2c_errors = n_errors
        self.nengo_io_c2h_count = n_outputs
        self.nengo_io_snip_range = snip_range


class SpikePacker:
    """Packs spikes for sending to chip

    Currently represents a spike as two int32s.
    """

    @classmethod
    def size(cls):
        """The number of int32s used to represent one spike."""
        size = len(cls.pack(None))
        assert size == 2  # must match nengo_io.c.template
        return size

    @classmethod
    def pack(cls, spike):
        """Pack the spike into a tuple of 32-bit integers.

        Parameters
        ----------
        spike : LoihiSpikeInput.LoihiSpike
            The spike to pack.

        Returns
        -------
        packed_spike : tuple of int
            A tuple of length ``size`` to represent this spike.
        """
        chip_id = int(spike.axon.chip_id if spike is not None else 0)
        core_id = int(spike.axon.core_id if spike is not None else 0)
        axon_id = int(spike.axon.axon_id if spike is not None else 0)
        axon_type = int(spike.axon.axon_type if spike is not None else 0)
        atom = int(spike.axon.atom if spike is not None else 0)
        assert chip_id == 0, "Multiple chips not supported"
        assert 0 <= core_id < 1024
        assert 0 <= axon_id < 4096
        assert 0 <= axon_type <= 32
        assert 0 <= atom < 1024

        return (
            int(np.left_shift(core_id, 16)) + axon_id,
            int(np.left_shift(axon_type, 16) + atom),
        )
