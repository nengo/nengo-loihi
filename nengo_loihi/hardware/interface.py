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

from nengo_loihi.block import Probe
from nengo_loihi.compat import make_process_step
from nengo_loihi.discretize import scale_pes_errors
from nengo_loihi.hardware.allocators import OneToOne, RoundRobin
from nengo_loihi.hardware.builder import build_board
from nengo_loihi.nxsdk_obfuscation import d, d_func, d_get
from nengo_loihi.hardware.nxsdk_objects import LoihiSpikeInput
from nengo_loihi.hardware.nxsdk_shim import assert_nxsdk, nxsdk, SnipPhase, SpikeProbe
from nengo_loihi.hardware.validate import validate_board

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
        The maximum number of spikes that can be sent to each chip in one timestep
        if ``.use_snips`` is True.
    allocator : Allocator, optional (Default: ``OneToOne()``)
        Callable object that allocates the board's devices to given models.
        Defaults to one block and one input per core on a single chip.
    """

    min_nxsdk_version = LooseVersion("0.8.7")
    max_nxsdk_version = LooseVersion("0.9.0")
    channel_packet_size = 64  # size of channel packets in int32s
    snip_output_header_len = 1

    def __init__(
        self,
        model,
        use_snips=True,
        seed=None,
        snip_max_spikes_per_step=50,
        allocator=OneToOne(),
    ):
        self.closed = False
        self.nxsdk_board = None
        self.host_socket = None  # IO snip superhost (this) <-> host socket
        self.host_socket_connected = False
        self.host_socket_port = None
        self.error_chip_map = {}  # maps synapses to chip locations for errors
        self._probe_filters = {}
        self._probe_filter_pos = {}
        self._snip_probe_data = collections.OrderedDict()
        self._chip2host_sent_steps = 0

        self.model = model
        self.use_snips = use_snips
        self.seed = seed
        self.snip_max_spikes_per_step = snip_max_spikes_per_step
        self.allocator = allocator

        self.check_nxsdk_version()

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
        return iter(self.model.probes)

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
            self.create_snips()

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

    def _get_weighted_probe(self, probe, outputs):
        # `outputs` shape is (blocks in probe, timesteps, outputs in block)
        assert len(outputs) == len(probe.targets)

        weighted_outputs = []
        weighted_probe = np.shape(probe.weights[0]) is not ()
        assert all(
            weighted_probe == (np.shape(probe.weights[k]) is not ())
            for k in range(len(outputs))
        )
        for k, output in enumerate(outputs):
            output = np.asarray(output)  # , dtype=np.float32)
            if probe.weights[k] is not None:
                output = output.dot(probe.weights[k])
            weighted_outputs.append(output)

        if weighted_probe:
            result = np.sum(weighted_outputs, axis=0)
        else:
            nt = weighted_outputs[0].shape[0] if weighted_outputs[0].ndim == 2 else None
            nc = sum(x.shape[-1] for x in weighted_outputs)
            assert all(
                x.shape[0] == nt if x.ndim == 2 else x.ndim == 1
                for x in weighted_outputs
            )

            result = (
                np.hstack(weighted_outputs)
                if nt is None
                else np.column_stack(weighted_outputs)
            )
            if probe.reindexing is not None:
                result = result[..., probe.reindexing]

            if nt is None and result.ndim == 2:
                assert result.shape[0] == 1, "nt: %s, nc: %s, result.shape: %s" % (
                    nt,
                    nc,
                    result.shape,
                )
                result.shape = (-1,)

            assert (
                nt is None and result.shape == (nc,) or result.shape == (nt, nc)
            ), "nt: %s, nc: %s, result.shape: %s" % (nt, nc, result.shape)

        return result

    def _chip2host_monitor(self, probes_receivers):
        increment = None
        for probe, receiver in probes_receivers.items():
            assert not probe.use_snip
            nxsdk_probes = self.board.probe_map[probe]
            outputs = [
                np.column_stack(
                    [
                        d_get(p, b"dGltZVNlcmllcw==", b"ZGF0YQ==")[
                            self._chip2host_sent_steps :
                        ]
                        for p in nxsdk_probe
                    ]
                )
                for nxsdk_probe in nxsdk_probes
            ]

            if len(outputs) > 0:
                x = self._get_weighted_probe(probe, outputs)

                if increment is None:
                    increment = len(x)

                assert increment == len(x), "All x need same number of steps"

                for j in range(len(x)):
                    receiver.receive(
                        self.model.dt * (self._chip2host_sent_steps + j + 2), x[j]
                    )

        if increment is not None:
            self._chip2host_sent_steps += increment

    def _chip2host_snips(self, probes_receivers):
        assert self.host_socket_connected

        expected_bytes = 4 * self.io_snip_c2h_count
        n_waits = 0  # number of times we've had to wait for more data

        recv_size = 4096  # python docs recommend small power of 2, e.g. 4096
        received = self.host_socket.recv(recv_size)  # blocking recv call
        data = received
        while len(data) < expected_bytes and n_waits < 10:
            if len(received) != recv_size:
                # We did not get all the data we expected. Wait before trying again.
                time.sleep(0.001)
                n_waits += 1

            try:
                received = self.host_socket.recv(recv_size, socket.MSG_DONTWAIT)
                if len(received) > 0:
                    data += received
            except BlockingIOError:  # pragma: no cover
                # No data was available. Hopefully it will be there after we wait.
                received = []

        assert len(data) == expected_bytes, "Received (%d) less than expected (%d)" % (
            len(data),
            expected_bytes,
        )

        raw_data = np.frombuffer(data, dtype=np.int32)
        del data

        # create views into data for different chips
        time_steps = []
        chip_data = []
        i = 0
        for info in self.chip_snip_info:
            data = raw_data[i : i + info["n_outputs"]]
            assert len(data) == info["n_outputs"]
            time_step, data = data[0], data[1:]
            time_steps.append(time_step)
            chip_data.append(data)
            i += self.channel_packet_size * info["n_output_packets"]

        assert all(time_steps == time_steps[0]), "Chips are out of sync!"

        for probe in self._snip_probe_data:
            assert probe.use_snip

            outputs = []
            for chip_idx, data_slice, n_packed_spikes in self.snip_range[probe]:
                data = chip_data[chip_idx][data_slice]
                if n_packed_spikes > 0:
                    packed8 = data.view("uint8")
                    unpacked = np.unpackbits(packed8)
                    unpacked = unpacked.reshape((-1, 8))[:, ::-1].ravel()
                    unpacked = unpacked[:n_packed_spikes]
                    outputs.append(unpacked)
                else:
                    outputs.append(data)

            assert all(x.ndim == 1 for x in outputs)
            x = self._get_weighted_probe(probe, outputs)

            receiver = probes_receivers.get(probe, None)
            if receiver is not None:
                # chip->host
                receiver.receive(self.model.dt * time_steps[chip_idx], x)
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
        nxsdk_spike_generator = self.nxsdk_board.global_spike_generator
        tmax = -1
        for t, spikes in loihi_spikes.items():
            assert t >= tmax, "Spikes must be in order"
            tmax = t
            LoihiSpikeInput.add_spikes_to_generator(t, spikes, nxsdk_spike_generator)

    def _host2chip_snips(self, loihi_spikes, loihi_errors):
        assert self.host_socket_connected
        assert len(loihi_errors) == 0, "Not yet implemented"

        chip_idxs = range(self.board.n_chips)
        # first `n_chips` elements of `msg` record number of elements going to each chip
        msg = [0 for _ in chip_idxs]
        for chip_idx in chip_idxs:
            chip_id = d_get(d_get(self.nxsdk_board, b"bjJDaGlwcw==")[chip_idx], b"aWQ=")
            chip_spikes = (
                loihi_spikes[loihi_spikes["chip_id"] == chip_id]
                if len(loihi_spikes) > 0
                else []
            )

            max_spikes = self.snip_max_spikes_per_step
            if len(chip_spikes) > max_spikes:
                warnings.warn(
                    "Too many spikes (%d) sent in one timestep. Increase the "
                    "value of `snip_max_spikes_per_step` (currently set to %d). "
                    "See\n  https://www.nengo.ai/nengo-loihi/configuration.html\n"
                    "for details." % (len(chip_spikes), max_spikes)
                )
                chip_spikes = chip_spikes[:max_spikes]

            msg_len0 = len(msg)
            msg.append(len(chip_spikes))
            msg.extend(SpikePacker.pack(chip_spikes))

            # assert len(chip_errors) == self.io_snip_h2c_errors[chip_idx]
            # for error in chip_errors:
            #     chip_msg.extend(error)

            msg[chip_idx] = len(msg) - msg_len0

        # encode message to bytes and send to host
        msg_bytes = struct.pack("%di" % len(msg), *msg)
        i_sent = 0
        while i_sent < len(msg_bytes):
            n_sent = self.host_socket.send(msg_bytes[i_sent:])
            i_sent += n_sent

    def host2chip(self, spikes, errors):
        loihi_spikes = collections.OrderedDict()
        for spike_input, t, s in spikes:
            loihi_spike_input = self.nxsdk_board.spike_inputs[spike_input]
            loihi_spikes.setdefault(t, []).extend(loihi_spike_input.spikes_to_loihi(s))

        assert (
            self.use_snips or len(errors) == 0
        ), "Learning only supported with snips (`precompute=False`)"
        error_info = []
        error_vecs = []
        for synapse, t, e in errors:
            core_id = self.error_chip_map[synapse]
            error_info.append([core_id, len(e)])
            error_vecs.append(e)

        loihi_errors = []
        if len(error_vecs) > 0:
            error_vecs = np.concatenate(error_vecs)
            error_vecs = scale_pes_errors(error_vecs, scale=self.pes_error_scale)

            i = 0
            for core_id, e_len in error_info:
                loihi_errors.append(
                    [core_id, e_len] + error_vecs[i : i + e_len].tolist()
                )
                i += e_len

        if self.use_snips:
            if len(loihi_spikes) > 0:
                assert len(loihi_spikes) == 1, "SNIPs process one timestep at a time"
                loihi_spikes = next(iter(loihi_spikes.values()))
                loihi_spikes = np.hstack(loihi_spikes) if len(loihi_spikes) > 0 else []
            else:
                loihi_spikes = []
            return self._host2chip_snips(loihi_spikes, loihi_errors)
        else:
            return self._host2chip_spikegen(loihi_spikes)

    def wait_for_completion(self):
        d_func(self.nxsdk_board, b"ZmluaXNoUnVu")

    def is_connected(self):
        return self.nxsdk_board is not None and d_func(
            self.nxsdk_board, b"ZXhlY3V0b3I=", b"aGFzU3RhcnRlZA=="
        )

    def connect(self, attempts=3):
        if self.nxsdk_board is None:
            raise SimulationError("Must build model before running")

        if self.is_connected():
            return

        logger.info("Connecting to Loihi, max attempts: %d", attempts)
        last_exception = None
        for i in range(attempts):
            try:
                d_func(self.nxsdk_board, b"c3RhcnQ=")
                if self.is_connected():
                    break
            except Exception as e:
                last_exception = e
                logger.warning("Connection error: %s", e)
                time.sleep(1)
                logger.info("Retrying, attempt %d", i + 1)
        else:
            raise SimulationError("Board connection error%s" % (
                ": %s" % last_exception if last_exception is not None else ""
            ))

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
            nxsdk_probes = self.board.probe_map[probe]
            outputs = [
                np.column_stack(
                    [d_get(p, b"dGltZVNlcmllcw==", b"ZGF0YQ==") for p in nxsdk_probe]
                )
                for nxsdk_probe in nxsdk_probes
            ]
            data = self._get_weighted_probe(probe, outputs)

        return self._filter_probe(probe, data)

    def create_snips(self):
        assert not self.is_connected(), "Board must be disconnected to create snips"

        n_chips = self.board.n_chips
        chip_idxs = list(range(n_chips))
        chip_snip_info = [{} for _ in range(n_chips)]

        self.host_snip = None  # host snip process
        self.io_snip = {}  # chip snip processes for IO snips
        self.io_snip_h2c_errors = {}
        self.io_snip_snip_range = {}
        self.learn_snip = {}  # chip snip processes for learn snips

        snips_dir = os.path.join(os.path.dirname(__file__), "snips")
        env = jinja2.Environment(
            trim_blocks=True,
            loader=jinja2.FileSystemLoader(snips_dir),
            keep_trailing_newline=True,
        )
        self.tmp_snip_dir = tempfile.TemporaryDirectory()

        # --- determine required information for learning
        assert len(self.error_chip_map) == 0
        self.io_snip_h2c_errors = [None] * n_chips

        for chip_idx, chip in enumerate(self.board.chips):
            n_errors = 0
            total_error_len = 0
            # max_error_len = 0
            for core in chip.cores:
                if core.learning_coreid is None:
                    continue

                assert (
                    len(core.blocks) == 1
                ), "Learning not implemented with multiple blocks per core"
                block = core.blocks[0]
                error_len = block.n_neurons // 2
                # max_error_len = max(error_len, max_error_len)
                n_errors += 1
                total_error_len += 2 + error_len

                for synapse in block.synapses:
                    self.error_chip_map[synapse] = core.learning_coreid

            cinfo = chip_snip_info[chip_idx]
            cinfo["n_errors"] = n_errors
            cinfo["total_error_len"] = total_error_len
            # cinfo["max_error_len"] = max_error_len
            self.io_snip_h2c_errors[chip_idx] = n_errors

        # --- determine required information for receiving outputs
        output_offset = self.snip_output_header_len  # first output is timestamp
        # TODO: should snip_range be stored on the probe?
        snip_range = {}

        for info in chip_snip_info:
            info["cores"] = set()
            info["probes"] = []
            info["i_output"] = 0

        for probe in self._iter_probes():
            if not probe.use_snip:
                continue

            pinfo = probe.snip_info
            assert pinfo["key"] in ("u", "v", "spike")

            snip_range[probe] = []
            for block, chip_idx, core_id, compartment_idxs in zip(
                probe.targets,
                pinfo["chip_idx"],
                pinfo["core_id"],
                pinfo["compartment_idxs"],
            ):
                cinfo = chip_snip_info[chip_idx]
                cinfo["cores"].add(core_id)
                i_output = cinfo["i_output"]

                key = pinfo["key"]
                if pinfo["key"] == "spike":
                    refract_delay = block.compartment.refract_delay[0]
                    assert np.all(block.compartment.refract_delay == refract_delay)
                    key = refract_delay * d(b"MTI4", int)

                n_comps = len(compartment_idxs)
                comp0 = compartment_idxs[0]
                comp_diff = np.diff(compartment_idxs)
                comp_step = comp_diff[0]
                is_ranged_comps = np.all(comp_diff == comp_step)
                is_packed_spikes = is_ranged_comps and (pinfo["key"] == "spike")
                n_packed_spikes = n_comps if is_packed_spikes else 0

                output_len = ceil_div(n_comps, 32) if is_packed_spikes else n_comps
                output_slice = slice(i_output, i_output + output_len)
                snip_range[probe].append((chip_idx, output_slice, n_packed_spikes))

                offset = output_offset + i_output
                if is_ranged_comps:
                    cinfo["probes"].append(
                        (offset, key, core_id, comp0, comp_step, n_comps)
                    )
                else:
                    for i, comp in enumerate(compartment_idxs):
                        chip_snip_info[chip_idx]["probes"].append(
                            (offset + i, key, core_id, comp, 0, 1)
                        )
                cinfo["i_output"] += output_len

        # number of outputs (in ints and packets) for each chip
        for info in chip_snip_info:
            info["n_outputs"] = output_offset + info["i_output"]
            info["n_output_packets"] = ceil_div(
                info["n_outputs"], self.channel_packet_size
            )

        # total number of outputs expected back from the host (including packet padding)
        self.io_snip_c2h_count = self.channel_packet_size * sum(
            info["n_output_packets"] for info in chip_snip_info
        )

        self.chip_snip_info = chip_snip_info
        self.snip_range = snip_range

        # --- create host process (for faster communication via sockets)
        input_channels = ["nengo_io_h2c_chip_%d" % chip_idx for chip_idx in chip_idxs]
        output_channels = ["nengo_io_c2h_chip_%d" % chip_idx for chip_idx in chip_idxs]

        socket_port = np.random.randint(50000, 60000)

        read_size = roundup(1024, self.channel_packet_size)
        write_packets = ceil_div(read_size, self.channel_packet_size)
        write_size = write_packets * self.channel_packet_size
        # double buffer size, just so we can do a full extra read/write if we need to
        buffer_size = 2 * roundup(max(read_size, write_size), self.channel_packet_size)

        template = env.get_template("nengo_host.cc.template")
        c_path = os.path.join(self.tmp_snip_dir.name, "nengo_host.cc")
        code = template.render(
            n_chips=n_chips,
            buffer_size=buffer_size,
            packet_size=self.channel_packet_size,
            read_size=read_size,
            write_packets=write_packets,
            output_packets=", ".join(
                "%d" % info["n_output_packets"] for info in chip_snip_info
            ),
            server_port=socket_port,
            input_channels=input_channels,
            output_channels=output_channels,
            obfs=snip_obfs,
        )
        with open(c_path, "w") as f:
            f.write(code)

        # make process
        self.host_snip = d_func(
            self.nxsdk_board,
            b"Y3JlYXRlU25pcA==",
            kwargs={
                b"cGhhc2U=": SnipPhase.HOST_CONCURRENT_EXECUTION,
                b"Y3BwRmlsZQ==": c_path,
            },
        )

        # connect to host socket
        self.host_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.host_socket.setsockopt(socket.SOL_TCP, socket.TCP_NODELAY, 1)
        self.host_socket_port = socket_port

        # --- create chip processes
        for chip_idx in chip_idxs:
            self.create_chip_snip(
                chip_idx,
                env,
                input_channels[chip_idx],
                output_channels[chip_idx],
                chip_snip_info[chip_idx],
            )

    def create_chip_snip(
        self, chip_idx, env, input_channel_name, output_channel_name, info
    ):
        chip_id = d_get(d_get(self.nxsdk_board, b"bjJDaGlwcw==")[chip_idx], b"aWQ=")
        include_dir = self.tmp_snip_dir.name
        src_dir = self.tmp_snip_dir.name

        # --- create IO snip
        c_filename = "nengo_io_chip_%d.c" % chip_idx
        h_filename = "nengo_io_chip_%d.h" % chip_idx
        c_path = os.path.join(src_dir, c_filename)
        h_path = os.path.join(include_dir, h_filename)

        template = env.get_template("nengo_io.c.template")
        logger.debug(
            "Creating %s with %d outputs, %d error, %d cores, %d probes",
            c_path,
            info["n_outputs"],
            info["n_errors"],
            len(info["cores"]),
            len(info["probes"]),
        )
        chip_buffer_size = roundup(
            max(
                info["n_outputs"],  # currently, buffer needs to hold all outputs
                self.channel_packet_size
                + max(SpikePacker.size, snip_obfs["error_info_size"]),
            ),
            self.channel_packet_size,
        )
        code = template.render(
            header_file=h_filename,
            n_outputs=info["n_outputs"],
            n_output_packets=info["n_output_packets"],
            n_errors=info["n_errors"],
            buffer_size=chip_buffer_size,
            packet_size=self.channel_packet_size,
            input_channel=input_channel_name,
            output_channel=output_channel_name,
            cores=info["cores"],
            probes=info["probes"],
            obfs=snip_obfs,
        )
        with open(c_path, "w") as f:
            f.write(code)

        # write header file using template
        template = env.get_template("nengo_io.h.template")
        code = template.render()
        with open(h_path, "w") as f:
            f.write(code)

        # create SNIP process
        logger.debug("Creating nengo_io chip %d process" % chip_idx)
        self.io_snip[chip_idx] = d_func(
            self.nxsdk_board,
            b"Y3JlYXRlU25pcA==",
            kwargs={
                b"bmFtZQ==": "nengo_io_chip" + str(chip_id),
                b"Y0ZpbGVQYXRo": c_path,
                b"aW5jbHVkZURpcg==": include_dir,
                b"ZnVuY05hbWU=": "nengo_io",
                b"Z3VhcmROYW1l": "guard_io",
                b"cGhhc2U=": d_get(SnipPhase, b"RU1CRURERURfTUdNVA=="),
                b"Y2hpcElk": chip_id,
            },
        )

        # --- create learning snip
        h_filename = "nengo_learn_chip_%d.h" % chip_idx
        c_filename = "nengo_learn_chip_%d.c" % chip_idx
        c_path = os.path.join(src_dir, c_filename)
        h_path = os.path.join(include_dir, h_filename)

        # write c file using template
        template = env.get_template("nengo_learn.c.template")
        code = template.render(header_file=h_filename, obfs=snip_obfs)
        with open(c_path, "w") as f:
            f.write(code)

        # write header file using template
        template = env.get_template("nengo_learn.h.template")
        code = template.render()
        with open(h_path, "w") as f:
            f.write(code)

        # create SNIP process
        logger.debug("Creating nengo_learn chip %d process" % chip_idx)
        self.learn_snip[chip_idx] = d_func(
            self.nxsdk_board,
            b"Y3JlYXRlU25pcA==",
            kwargs={
                b"bmFtZQ==": "nengo_learn",
                b"Y0ZpbGVQYXRo": c_path,
                b"aW5jbHVkZURpcg==": include_dir,
                b"ZnVuY05hbWU=": "nengo_learn",
                b"Z3VhcmROYW1l": "guard_learn",
                b"cGhhc2U=": d_get(SnipPhase, b"RU1CRURERURfUFJFTEVBUk5fTUdNVA=="),
                b"Y2hpcElk": chip_id,
            },
        )

        # --- create channels
        input_channel_size = (
            self.snip_output_header_len  # first int stores number of spikes
            + self.snip_max_spikes_per_step * SpikePacker.size
            + info["total_error_len"]
        )
        logger.debug(
            "Creating %s channel (%d)" % (input_channel_name, input_channel_size)
        )
        input_channel = d_get(self.nxsdk_board, b"Y3JlYXRlQ2hhbm5lbA==")(
            str.encode(input_channel_name),
            **{
                # channel size (in elements)
                d(b"bnVtRWxlbWVudHM="): input_channel_size,
                # size of one packet (in bytes)
                d(b"bWVzc2FnZVNpemU="): 4 * self.channel_packet_size,
                # size of send/receive buffer on chip/host (in packets)
                d(b"c2xhY2s="): 16,
            },
        )
        logger.debug(
            "Creating %s channel (%d)" % (output_channel_name, info["n_outputs"])
        )
        output_channel = d_get(self.nxsdk_board, b"Y3JlYXRlQ2hhbm5lbA==")(
            str.encode(output_channel_name),
            **{
                # channel size (in elements)
                d(b"bnVtRWxlbWVudHM="): info["n_outputs"],
                # size of one packet (in bytes)
                d(b"bWVzc2FnZVNpemU="): 4 * self.channel_packet_size,
                # size of send/receive buffer on chip/host (in packets)
                d(b"c2xhY2s="): 16,
            },
        )
        d_get(input_channel, b"Y29ubmVjdA==")(self.host_snip, self.io_snip[chip_idx])
        d_get(output_channel, b"Y29ubmVjdA==")(self.io_snip[chip_idx], self.host_snip)


class SpikePacker:
    """Packs spikes for sending to chip

    Currently represents a spike as two int32s.
    """

    size = 2  # must match nengo_io.c.template

    @classmethod
    def pack(cls, spikes):
        """Pack the spike into a tuple of 32-bit integers.

        Parameters
        ----------
        spike : structured ndarray of spikes
            The spikes to pack.

        Returns
        -------
        packed_spike : tuple of int
            A tuple of length ``size * n_spikes`` to represent this spike.
        """
        if len(spikes) == 0:
            return []

        assert np.all(
            spikes["chip_id"] == spikes["chip_id"][0]
        ), "All spikes must go to the same chip"
        assert np.all(spikes["core_id"] < 1024)
        assert np.all(spikes["axon_id"] < 4096)
        assert np.all(spikes["axon_type"] <= 32)
        assert np.all(spikes["atom"] < 1024)

        axon_type = spikes["axon_type"]
        axon_type[axon_type == 16] += spikes["atom_bits_extra"][axon_type == 16]
        return np.array(
            [
                np.left_shift(spikes["core_id"], 16) + spikes["axon_id"],
                np.left_shift(axon_type, 16) + spikes["atom"],
            ]
        ).T.ravel()


# obfuscated strings used in SNIP templates
snip_obfs = dict(
    core_class=d(b"TmV1cm9uQ29yZQ=="),
    id_class=d(b"Q29yZUlk"),
    get_channel=d(b"Z2V0Q2hhbm5lbElE"),
    int_type=d(b"aW50MzJfdA=="),
    spike_size=d(b"Mg=="),
    error_info_size=d(b"Mg==", int),
    s_data=d(b"dXNlckRhdGE="),
    s_step=d(b"dGltZV9zdGVw"),
    read=d(b"cmVhZENoYW5uZWw="),
    write=d(b"d3JpdGVDaGFubmVs"),
    spike_shift=d(b"MTY="),
    spike_mask=d(b"MHgwMDAwRkZGRg=="),
    do_axon_type_0=d(b"bnhfc2VuZF9kaXNjcmV0ZV9zcGlrZQ=="),
    do_axon_type_16=d(b"bnhfc2VuZF9wb3AxNl9zcGlrZQ=="),
    do_axon_type_32=d(b"bnhfc2VuZF9wb3AzMl9zcGlrZQ=="),
    comp_state=d(b"Y3hfc3RhdGU="),
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
