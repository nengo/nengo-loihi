import collections
from distutils.version import LooseVersion
import logging
import os
from select import select
import socket
import struct
import tempfile
import time
import warnings

import jinja2
from nengo.exceptions import SimulationError
import numpy as np

from nengo_loihi.compat import make_process_step
from nengo_loihi.discretize import scale_pes_errors
from nengo_loihi.hardware.allocators import OneToOne, RoundRobin
from nengo_loihi.hardware.builder import build_board
from nengo_loihi.nxsdk_obfuscation import d, d_func, d_get
from nengo_loihi.hardware.nxsdk_objects import LoihiSpikeInput
from nengo_loihi.hardware.nxsdk_shim import assert_nxsdk, nxsdk, SnipPhase, SpikeProbe
from nengo_loihi.hardware.validate import validate_board
from nengo_loihi.probe import LoihiProbe
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

    connection_retries = 3
    min_nxsdk_version = LooseVersion("0.8.7")
    max_nxsdk_version = LooseVersion("0.9.0")

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
                "snips are not supported for the RoundRobin allocator"
            )

        self.closed = False
        self.error_chip_map = {}  # maps synapses to chip locations for errors

        self._probe_filters = {}
        self._probe_filter_pos = {}

        self.model = model
        self.use_snips = use_snips
        self.seed = seed

        self.check_nxsdk_version()

        validate_model(self.model)

        # clear cached content from SpikeProbe class attribute
        d_func(SpikeProbe, b"cHJvYmVEaWN0", b"Y2xlYXI=")

        # If we're using snips, tag all probes as being snip-based,
        # as having normal probes at the same time as snips causes problems.
        # This must be done before the build process to ensure information
        # is stored properly.
        if self.use_snips:
            for probe in self.model.probes:
                probe.use_snip = True

        # --- allocate
        self.board = allocator(self.model)

        # --- validate
        validate_board(self.board)

        # --- build
        self.nxsdk_board = build_board(
            self.board, use_snips=self.use_snips, seed=self.seed
        )

        # --- create snips or non-snip infrastructure
        self.snips, self.no_snips = None, None
        if self.use_snips:
            self.snips = Snips(
                self.model, self.board, self.nxsdk_board, snip_max_spikes_per_step
            )
            self.chip2host = self.snips.chip2host

        else:
            self.no_snips = NoSnips(
                self.model.dt,
                self.board.probe_map,
                self.nxsdk_board.global_spike_generator,
            )
            self.chip2host = self.no_snips.chip2host

    def __enter__(self):
        if self.closed:
            raise SimulationError(
                "Loihi interface has been closed and cannot be reopened."
            )

        if not self.connected:
            self.connect()
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

    @property
    def connected(self):
        return self.nxsdk_board is not None and d_func(
            self.nxsdk_board, b"ZXhlY3V0b3I=", b"aGFzU3RhcnRlZA=="
        )

    def close(self):
        if self.snips is not None and self.snips.connected:
            self.snips.close()

        if self.nxsdk_board is not None:
            d_func(self.nxsdk_board, b"ZGlzY29ubmVjdA==")
            self.nxsdk_board = None

        self.closed = True

    def connect(self):
        """Connects to the board."""

        logger.info("Connecting to Loihi, max attempts: %d", self.connection_retries)
        last_exception = None
        for i in range(self.connection_retries):
            try:
                d_func(self.nxsdk_board, b"c3RhcnQ=")
                if self.connected:
                    break
            except Exception as e:
                last_exception = e
                logger.warning("Connection error: %s", e)
                time.sleep(1)
                logger.info("Retrying, attempt %d", i + 1)
        else:
            raise SimulationError(
                "Board connection error%s"
                % (": %s" % last_exception if last_exception is not None else "")
            )

    def get_probe_output(self, probe):
        assert isinstance(probe, LoihiProbe)
        if probe.use_snip:
            data = self.snips.probe_data[probe]
        else:
            nxsdk_probes = self.board.probe_map[probe]
            outputs = [
                np.column_stack(
                    [d_get(p, b"dGltZVNlcmllcw==", b"ZGF0YQ==") for p in nxsdk_probe]
                )
                for nxsdk_probe in nxsdk_probes
            ]
            data = probe.weight_outputs(outputs)

        # --- Filter probed data
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

    def _find_learning_core_id(self, synapse):
        # TODO: make multi-chip when we add multi-chip snips
        for core in self.board.chips[0].cores:
            for block in core.blocks:
                if synapse in block.synapses:
                    assert (
                        len(core.blocks) == 1
                    ), "Learning not implemented with multiple blocks per core"
                    return core.learning_coreid

        raise ValueError("Could not find core ID for synapse %r" % synapse)

    def host2chip(self, spikes, errors):
        loihi_spikes = collections.OrderedDict()
        for spike_input, t, s in spikes:
            loihi_spike_input = self.nxsdk_board.spike_inputs[spike_input]
            loihi_spikes.setdefault(t, []).extend(loihi_spike_input.spikes_to_loihi(s))

        error_info = []
        error_vecs = []
        for synapse, t, e in errors:
            core_id = self.error_chip_map.get(synapse, None)
            if core_id is None:
                core_id = self._find_learning_core_id(synapse)
                self.error_chip_map[synapse] = core_id

            error_info.append([core_id, len(e)])
            error_vecs.append(e)

        loihi_errors = []
        if len(error_vecs) > 0:
            error_vecs = np.concatenate(error_vecs)
            error_vecs = scale_pes_errors(error_vecs, scale=self.model.pes_error_scale)

            i = 0
            for core_id, e_len in error_info:
                loihi_errors.append(
                    [core_id, e_len] + error_vecs[i : i + e_len].tolist()
                )
                i += e_len

        if self.use_snips:
            if len(loihi_spikes) > 0:
                assert len(loihi_spikes) == 1, "Snips process one timestep at a time"
                loihi_spikes = next(iter(loihi_spikes.values()))
                loihi_spikes = np.hstack(loihi_spikes) if len(loihi_spikes) > 0 else []
            else:
                loihi_spikes = []
            return self.snips.host2chip(loihi_spikes, loihi_errors)
        else:
            assert len(loihi_errors) == 0
            return self.no_snips.host2chip(loihi_spikes)

    def run_steps(self, steps, blocking=True):
        assert self.connected, "Interface is not built"

        # start the board running the desired number of steps
        d_get(self.nxsdk_board, b"cnVu")(steps, **{d(b"YVN5bmM="): not blocking})

        # connect snips
        if self.use_snips and not self.snips.connected:
            self.snips.connect(self.nxsdk_board)

    def wait_for_completion(self):
        d_func(self.nxsdk_board, b"ZmluaXNoUnVu")


class NoSnips:
    def __init__(self, dt, probe_map, spike_generator):
        self.sent_steps = 0
        self.dt = dt
        self.probe_map = probe_map
        self.spike_generator = spike_generator

    def chip2host(self, probes_receivers):
        increment = None
        for probe, receiver in probes_receivers.items():
            assert not probe.use_snip
            nxsdk_probes = self.probe_map[probe]
            outputs = [
                np.column_stack(
                    [
                        d_get(p, b"dGltZVNlcmllcw==", b"ZGF0YQ==")[self.sent_steps :]
                        for p in nxsdk_probe
                    ]
                )
                for nxsdk_probe in nxsdk_probes
            ]

            if len(outputs) > 0:
                x = probe.weight_outputs(outputs)

                if increment is None:
                    increment = len(x)

                assert increment == len(x), "All x need same number of steps"

                for j in range(len(x)):
                    receiver.receive(self.dt * (self.sent_steps + j + 2), x[j])

        if increment is not None:
            self.sent_steps += increment

    def host2chip(self, loihi_spikes):
        nxsdk_spike_generator = self.spike_generator
        tmax = -1
        for t, spikes in loihi_spikes.items():
            assert t >= tmax, "Spikes must be in order"
            tmax = t
            LoihiSpikeInput.add_spikes_to_generator(t, spikes, nxsdk_spike_generator)


class Snips:

    channel_packet_elements = 64  # size of channel packets in int32s
    channel_bytes_per_element = 4  # bytes per int32 (channel packets element size)
    packet_bytes = channel_packet_elements * channel_bytes_per_element

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
        do_axon_type_0=d(b"bnhfc2VuZF9kaXNjcmV0ZV9zcGlrZQ=="),
        do_axon_type_16=d(b"bnhfc2VuZF9wb3AxNl9zcGlrZQ=="),
        do_axon_type_32=d(b"bnhfc2VuZF9wb3AzMl9zcGlrZQ=="),
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

    def __init__(self, model, board, nxsdk_board, max_spikes_per_step):
        self.model = model
        self.max_spikes_per_step = max_spikes_per_step

        self.probe_data = collections.OrderedDict()
        self.snips_dir = os.path.join(os.path.dirname(__file__), "snips")
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.env = jinja2.Environment(
            trim_blocks=True,
            loader=jinja2.FileSystemLoader(self.snips_dir),
            keep_trailing_newline=True,
        )

        self.sent_steps = 0
        self.processes = {}
        self.channels = {}

        for probe in self.model.probes:
            self.probe_data[probe] = []

        (
            self.n_errors,
            self.n_outputs,
            self.n_output_packets,
            self.snip_range,
        ) = self._create_io_snip(board, nxsdk_board)
        self.host_socket = HostSocket()
        self._create_host_snip(nxsdk_board)
        self._create_learn_snip(nxsdk_board)

        d_get(self.channels["h2c"], b"Y29ubmVjdA==")(
            self.processes["host"], self.processes["nengo_io"]
        )
        d_get(self.channels["c2h"], b"Y29ubmVjdA==")(
            self.processes["nengo_io"], self.processes["host"]
        )

    def _render_template(self, filename, **template_data):
        template = self.env.get_template("{}.template".format(filename))
        path = os.path.join(self.tmp_dir.name, filename)
        code = template.render(obfs=self.obfs, **template_data)
        with open(path, "w") as f:
            f.write(code)
        return path

    def _create_io_snip(self, board, nxsdk_board):

        # Determine which cores have learning
        n_errors = 0
        total_error_len = 0
        max_error_len = 0
        assert len(board.chips) == 1, "Learning not implemented for multiple chips"
        for core in board.chips[0].cores:  # TODO: don't assume 1 chip
            if core.learning_coreid:
                error_len = core.blocks[0].n_neurons // 2
                max_error_len = max(error_len, max_error_len)
                n_errors += 1
                total_error_len += 2 + error_len

        output_offset = 1  # first output is timestamp
        i_output = 0
        probes = []
        cores = set()
        # TODO: should snip_range be stored on the probe?
        snip_range = {}
        for probe in self.model.probes:
            if probe.use_snip:
                info = probe.snip_info
                assert info["key"] in ("u", "v", "spike")

                snip_range[probe] = []
                for block, core_id, compartment_idxs in zip(
                    probe.target, info["core_id"], info["compartment_idxs"]
                ):
                    key = info["key"]
                    if info["key"] == "spike":
                        refract_delay = block.compartment.refract_delay[0]
                        assert np.all(block.compartment.refract_delay == refract_delay)
                        key = refract_delay * d(b"MTI4", int)

                    cores.add(core_id)
                    n_comps = len(compartment_idxs)
                    comp0 = compartment_idxs[0]
                    comp_diff = np.diff(compartment_idxs)
                    comp_step = comp_diff[0] if n_comps > 1 else 0
                    is_ranged_comps = n_comps > 1 and np.all(comp_diff == comp_step)
                    is_packed_spikes = is_ranged_comps and (info["key"] == "spike")
                    n_packed_spikes = n_comps if is_packed_spikes else 0

                    output_len = ceil_div(n_comps, 32) if is_packed_spikes else n_comps
                    output_slice = slice(i_output, i_output + output_len)
                    snip_range[probe].append((output_slice, n_packed_spikes))

                    if is_ranged_comps:
                        probes.append(
                            (
                                output_offset + i_output,
                                key,
                                core_id,
                                comp0,
                                comp_step,
                                n_comps,
                            )
                        )
                        i_output += output_len
                    else:
                        for comp in compartment_idxs:
                            probes.append(
                                (output_offset + i_output, key, core_id, comp, 0, 1)
                            )
                            i_output += 1

        n_outputs = output_offset + i_output

        n_output_packets = ceil_div(n_outputs, self.channel_packet_elements)
        chip_buffer_size = roundup(
            max(
                n_outputs,  # currently, buffer needs to hold all outputs at once
                self.channel_packet_elements
                + max(SpikePacker.size, self.obfs["error_info_size"]),
            ),
            self.channel_packet_elements,
        )

        # --- write c file using template
        logger.debug(
            "Creating nengo_io.c with %d outputs, %d error, %d cores, %d probes",
            n_outputs,
            n_errors,
            len(cores),
            len(probes),
        )
        path = self._render_template(
            "nengo_io.c",
            n_outputs=n_outputs,
            n_output_packets=n_output_packets,
            n_errors=n_errors,
            max_error_len=max_error_len,
            buffer_size=chip_buffer_size,
            packet_elements=self.channel_packet_elements,
            cores=cores,
            probes=probes,
        )

        logger.debug("Creating nengo_io chip process")
        self.processes["nengo_io"] = d_func(
            nxsdk_board,
            b"Y3JlYXRlU25pcA==",
            kwargs={
                b"bmFtZQ==": "nengo_io",
                b"Y0ZpbGVQYXRo": path,
                b"aW5jbHVkZURpcg==": self.snips_dir,
                b"ZnVuY05hbWU=": "nengo_io",
                b"Z3VhcmROYW1l": "guard_io",
                b"cGhhc2U=": d_get(SnipPhase, b"RU1CRURERURfTUdNVA=="),
            },
        )

        # --- create channels
        input_channel_size = (
            1  # first int stores number of spikes
            + self.max_spikes_per_step * SpikePacker.size
            + total_error_len
        )
        logger.debug("Creating nengo_io_h2c channel (%d)" % input_channel_size)
        self.channels["h2c"] = d_get(nxsdk_board, b"Y3JlYXRlQ2hhbm5lbA==")(
            b"nengo_io_h2c",  # channel name
            **{
                # channel size (in elements)
                d(b"bnVtRWxlbWVudHM="): input_channel_size,
                # size of one packet (in bytes)
                d(b"bWVzc2FnZVNpemU="): self.packet_bytes,
                # size of send/receive buffer on chip/host (in packets)
                d(b"c2xhY2s="): 16,
            },
        )
        logger.debug("Creating nengo_io_c2h channel (%d)" % n_outputs)
        self.channels["c2h"] = d_get(nxsdk_board, b"Y3JlYXRlQ2hhbm5lbA==")(
            b"nengo_io_c2h",  # channel name
            **{
                # channel size (in elements)
                d(b"bnVtRWxlbWVudHM="): n_outputs,
                # size of one packet (in bytes)
                d(b"bWVzc2FnZVNpemU="): self.packet_bytes,
                # size of send/receive buffer on chip/host (in packets)
                d(b"c2xhY2s="): 16,
            },
        )

        return n_errors, n_outputs, n_output_packets, snip_range

    def _create_host_snip(self, nxsdk_board):
        # --- create host process (for faster communication via sockets)
        max_inputs = self.n_errors + self.max_spikes_per_step * SpikePacker.size
        host_buffer_size = roundup(
            max(max_inputs, self.n_outputs), self.channel_packet_elements
        )

        path = self._render_template(
            "nengo_host.cpp",
            host_buffer_size=host_buffer_size,
            n_outputs=self.n_outputs,
            n_output_packets=self.n_output_packets,
            server_port=self.host_socket.port,
            input_channel="nengo_io_h2c",
            output_channel="nengo_io_c2h",
            packet_bytes=self.packet_bytes,
        )

        # make process
        self.processes["host"] = d_func(
            nxsdk_board,
            b"Y3JlYXRlU25pcA==",
            kwargs={
                b"cGhhc2U=": SnipPhase.HOST_CONCURRENT_EXECUTION,
                b"Y3BwRmlsZQ==": path,
            },
        )

    def _create_learn_snip(self, nxsdk_board):
        path = self._render_template("nengo_learn.c")

        logger.debug("Creating nengo_learn chip process")
        self.processes["learn"] = d_func(
            nxsdk_board,
            b"Y3JlYXRlU25pcA==",
            kwargs={
                b"bmFtZQ==": "nengo_learn",
                b"Y0ZpbGVQYXRo": path,
                b"aW5jbHVkZURpcg==": self.snips_dir,
                b"ZnVuY05hbWU=": "nengo_learn",
                b"Z3VhcmROYW1l": "guard_learn",
                b"cGhhc2U=": d_get(SnipPhase, b"RU1CRURERURfUFJFTEVBUk5fTUdNVA=="),
            },
        )

    @property
    def connected(self):
        return self.host_socket is not None and self.host_socket.connected

    def close(self):
        if self.host_socket is not None:
            self.host_socket.close()

    def connect(self, nxsdk_board):
        if self.host_socket is not None:
            self.host_socket.connect(nxsdk_board)

    def chip2host(self, probes_receivers):
        assert self.host_socket.connected

        data = self.host_socket.recv_bytes(
            self.channel_bytes_per_element * self.n_outputs
        )
        time_step, data = data[0], data[1:]
        snip_range = self.snip_range

        for probe in self.probe_data:
            assert probe.use_snip

            outputs = []
            for r, n_packed_spikes in snip_range[probe]:
                if n_packed_spikes > 0:
                    packed32 = data[r]
                    packed8 = packed32.view("uint8")
                    unpacked = np.unpackbits(packed8)
                    unpacked = unpacked.reshape((-1, 8))[:, ::-1].ravel()
                    unpacked = unpacked[:n_packed_spikes]
                    outputs.append(unpacked)
                else:
                    outputs.append(data[r])

            assert all(x.ndim == 1 for x in outputs)

            weighted_outputs = probe.weight_outputs(outputs)[0]

            receiver = probes_receivers.get(probe, None)
            if receiver is not None:
                # chip->host
                receiver.receive(self.model.dt * time_step, weighted_outputs)
            else:
                # onchip probes
                self.probe_data[probe].append(weighted_outputs)

        self.sent_steps += 1

    def host2chip(self, loihi_spikes, loihi_errors):
        assert self.host_socket.connected

        max_spikes = self.max_spikes_per_step
        if len(loihi_spikes) > max_spikes:
            warnings.warn(
                "Too many spikes (%d) sent in one timestep. Increase the "
                "value of `snip_max_spikes_per_step` (currently set to %d). "
                "See\n  https://www.nengo.ai/nengo-loihi/configuration.html\n"
                "for details." % (len(loihi_spikes), max_spikes)
            )
            loihi_spikes = loihi_spikes[:max_spikes]

        data = [len(loihi_spikes)]
        data.extend(SpikePacker.pack(loihi_spikes))

        assert len(loihi_errors) == self.n_errors
        for error in loihi_errors:
            data.extend(error)

        self.host_socket.send_all(data)


class HostSocket:
    recv_retries = 10
    recv_size = 4096  # python docs recommend small power of 2, e.g. 4096
    recv_timeout = 0.01

    def __init__(self):
        self.connected = False

        self.port = np.random.randint(50000, 60000)
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setsockopt(socket.SOL_TCP, socket.TCP_NODELAY, 1)

    def connect(self, nxsdk_board):
        # pause to allow host snip to start and listen for connection
        time.sleep(0.1)

        host_address = d_get(
            nxsdk_board,
            b"ZXhlY3V0b3I==",
            b"X2hvc3RfY29vcmRpbmF0b3I==",
            b"aG9zdEFkZHI=",
        )
        logger.info("Connecting to host socket at (%s, %s)", host_address, self.port)
        self.socket.connect((host_address, self.port))
        self.connected = True

    def close(self):
        # send -1 to signal host/chip that we're done
        self.send_all([-1])

        # pause to allow chip to receive -1 signal via host
        time.sleep(0.1)

        self.socket.close()
        self.connected = False

    def recv_bytes(self, bytes_expected):
        data = bytearray([])
        n_retries = 0

        while len(data) < bytes_expected and n_retries < self.recv_retries:
            ready, _, _ = select([self.socket], [], [], self.recv_timeout)
            if self.socket in ready:
                data += bytearray(self.socket.recv(self.recv_size))
            else:  # pragma: no cover
                n_retries += 1

        assert len(data) == bytes_expected, "Received (%d) less than expected (%d)" % (
            len(data),
            bytes_expected,
        )

        return np.frombuffer(data, dtype=np.int32)

    def send_all(self, data):
        msg_bytes = struct.pack("%di" % len(data), *data)
        self.socket.sendall(msg_bytes)


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

        assert np.all(spikes["chip_id"] == 0), "Multiple chips not supported"
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
