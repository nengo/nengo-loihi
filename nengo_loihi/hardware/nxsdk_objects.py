import collections

import numpy as np

from nengo_loihi.block import Config
from nengo_loihi.nxsdk_obfuscation import d, d_get

MAX_COMPARTMENT_CFGS = d(b"MzI=", int)
MAX_VTH_CFGS = d(b"OA==", int)
MAX_SYNAPSE_CFGS = d(b"OA==", int)


class Board:
    """An entire Loihi Board, with multiple Chips"""

    def __init__(self, board_id=1):
        self.board_id = board_id

        self.chips = []
        self.chip_idxs = {}

        self.synapse_index = {}

        self.inputs = []

        self.probes = []
        # When using snips, this maps to a ProbeSnip instance
        # When not using snips, this maps to an NxSDK probe
        self.probe_map = {}

    @property
    def n_chips(self):
        return len(self.chips)

    @property
    def n_cores_per_chip(self):
        return [chip.n_cores for chip in self.chips]

    @property
    def n_synapses_per_core(self):
        return [[core.n_synapses for core in chip.cores] for chip in self.chips]

    def _add_chip(self, chip):
        assert chip not in self.chips
        self.chip_idxs[chip] = len(self.chips)
        self.chips.append(chip)

    def add_input(self, input):
        self.inputs.append(input)

    def new_chip(self):
        chip = Chip(board=self)
        self._add_chip(chip)
        return chip

    def chip_index(self, chip):
        return self.chip_idxs[chip]

    def index_synapse(self, synapse, chip, core, idxs):
        chip_idx = self.chip_index(chip)
        core_idx = chip.core_index(core)
        self.synapse_index[synapse] = (chip_idx, core_idx, idxs)

    def find_block(self, target_block):
        for chip in self.chips:
            for core in chip.cores:
                if target_block not in core.blocks:  # early skipping for speed
                    continue

                for block, compartment_idxs, ax_range in core.iterate_blocks():
                    if target_block is block:
                        return (
                            self.chip_index(chip),
                            chip.core_index(core),
                            core.blocks.index(block),
                            compartment_idxs,
                            ax_range,
                        )

                raise RuntimeError(
                    "Block is in core, but not found?!"
                )  # pragma: no cover

        return (None, None, None, None, None)  # pragma: no cover

    def find_synapse(self, synapse):
        return self.synapse_index[synapse]


class Chip:
    """A Loihi Chip on a Board, with multiple Cores."""

    def __init__(self, board):
        self.board = board

        self.cores = []
        self.core_idxs = {}

    @property
    def n_cores(self):
        return len(self.cores)

    def _add_core(self, core):
        assert core not in self.cores
        self.core_idxs[core] = len(self.cores)
        self.cores.append(core)

    def new_core(self):
        core = Core(chip=self)
        self._add_core(core)
        return core

    def core_index(self, core):
        return self.core_idxs[core]


class Core:
    """A Loihi Core, implementing one or more Blocks."""

    def __init__(self, chip):
        self.chip = chip
        self.blocks = []

        self.compartment_cfgs = []
        self.vth_cfgs = []
        self.synapse_cfgs = [None]  # keep index 0 unused
        self.stdp_pre_cfgs = []

        self.synapse_cfg_idxs = {}  # one synfmt per Synapse, for now

        # for each Synapse, provides a map from axon index to axon id
        self.synapse_axons = collections.OrderedDict()

        # for each Synapse, provides the indices occupied in the synapse weight table
        self.synapse_entries = collections.OrderedDict()

        self.learning_coreid = None

    @property
    def board(self):
        return self.chip.board

    @property
    def synapses(self):
        return list(self.synapse_axons)

    @property
    def n_synapses(self):
        return sum(
            synapse.size() for block in self.blocks for synapse in block.synapses
        )

    def iterate_blocks(self):
        i0 = 0
        a0 = 0
        for block in self.blocks:
            i1 = i0 + block.compartment.n_compartments
            a1 = a0 + sum(ax.n_axons for ax in block.axons)
            compartment_idxs = np.arange(i0, i1)
            ax_range = (a0, a1)
            yield block, compartment_idxs, ax_range
            i0 = i1
            a0 = a1

    def iterate_synapses(self):
        for block in self.blocks:
            for synapse in block.synapses:
                yield synapse

    def add_block(self, block):
        self.blocks.append(block)

    def add_compartment_cfg(self, compartment_cfg):
        self.compartment_cfgs.append(compartment_cfg)
        return len(self.compartment_cfgs) - 1  # index

    def add_vth_cfg(self, vth_cfg):
        self.vth_cfgs.append(vth_cfg)
        return len(self.vth_cfgs) - 1  # index

    def add_stdp_pre_cfg(self, stdp_pre_cfg):
        self.stdp_pre_cfgs.append(stdp_pre_cfg)
        return len(self.stdp_pre_cfgs) - 1  # index

    def add_synapse(self, synapse):
        synapse_cfg_idx = self.get_synapse_cfg_idx(synapse.synapse_cfg)
        self.synapse_cfg_idxs[synapse] = synapse_cfg_idx

        # determine starting ID for this synapse's axons
        id0 = 0
        if len(self.synapse_axons) > 0:
            last = next(reversed(self.synapse_axons))
            id0 = self.synapse_axons[last][-1] + 1

        # determine the ID for each synapse axon index
        idxs_per_synapse = synapse.idxs_per_synapse()
        i = id0
        ids = []
        for idx in range(synapse.n_axons):
            base = synapse.axon_compartment_base(idx)
            w, _ = synapse.axon_weights_indices(idx)
            if base is None or w.size == 0:
                # dummy axon, which we will not build
                ids.append(None)
            else:
                ids.append(i)
                i += idxs_per_synapse

        self.synapse_axons[synapse] = ids
        self.board.index_synapse(synapse, self.chip, self, ids)

        # determine the indices in the synapse weight table that this synapse occupies
        s0 = 0
        if len(self.synapse_entries) > 0:
            last = next(reversed(self.synapse_entries))
            s0 = self.synapse_entries[last][1]
        s1 = s0 + synapse.size()
        self.synapse_entries[synapse] = (s0, s1)

    def get_synapse_cfg(self, synapse):
        return self.synapse_cfgs[self.synapse_cfg_idxs[synapse]]

    def get_synapse_cfg_idx(self, synapse_cfg):
        try:
            return self.synapse_cfgs.index(synapse_cfg)
        except ValueError:
            self.synapse_cfgs.append(synapse_cfg)
            return len(self.synapse_cfgs) - 1  # index


class LoihiSpikeInput:
    """Stores information needed to send spikes to the actual chip.

    This acts as a bridge between a SpikeInput and the actual chip.
    It maps positions in the spike input to actual locations on the chip.

    Attributes
    ----------
    axon_map : {int: ndarray(dtype=spike_dtype)}
        Map from axon indices in the SpikeInput to LoihiAxons targeting
        particular locations on the chip.

    Notes
    -----
    spike_dtype is a numpy datatype to represent a Loihi axon or spike.
    It represents the following information.

    t : np.int32
        The timestep at which to send the spike to the chip (unused for axons)
    axon_type : np.int32
        The population type of axon. discrete = 0, pop16 = 16, pop32 = 32.
    chip_id : np.int32
        The actual ID of the target chip on the board.
    core_id : np.int32
        The actual ID of the target core on the board.
    axon_id : np.int32
        The actual ID of the target axon on the board.
    atom : np.int32
        The population index (atom), used if this axon sends population spikes
        (i.e. axon_type != 0).
    atom_bits_extra : np.int32
        The number of extra bits used for the atom (pop16 axons only).
    """

    spike_dtype = np.dtype(
        [
            ("t", np.int32),
            ("axon_type", np.int32),
            ("chip_id", np.int32),
            ("core_id", np.int32),
            ("axon_id", np.int32),
            ("atom", np.int32),
            ("atom_bits_extra", np.int32),
        ]
    )

    @classmethod
    def add_spikes_to_generator(cls, t, spikes, basic_spike_generator):
        methods = {
            0: getattr(basic_spike_generator, d(b"YWRkU3Bpa2U=")),
            16: getattr(basic_spike_generator, d(b"YWRkUG9wMTZTcGlrZQ==")),
            32: getattr(basic_spike_generator, d(b"YWRkUG9wMzJTcGlrZQ==")),
        }
        time = d(b"dGltZQ==")
        chip_id = d(b"Y2hpcElk")
        core_id = d(b"Y29yZUlk")
        axon_id = d(b"YXhvbklk")
        atom = d(b"c3JjQXRvbQ==")
        atom_bits_extra = d(b"YXRvbUJpdHM=")

        for spike in spikes:
            axon_type = int(spike["axon_type"])
            kwargs = {
                time: t,
                chip_id: spike["chip_id"].item(),
                core_id: spike["core_id"].item(),
                axon_id: spike["axon_id"].item(),
            }
            if axon_type == 0:
                assert spike["atom"] == 0, "Atom must be zero for discrete spikes"
            else:
                kwargs[atom] = spike["atom"]
                if axon_type == 16:
                    kwargs[atom_bits_extra] = spike["atom_bits_extra"]

            methods[axon_type](**kwargs)

    def __init__(self):
        self.axon_map = {}  # maps spike_input idx to axon in self.axons

    def set_axons(self, board, nxsdk_board, spike_input):
        """Initialize the axon map for this object.

        Parameters
        ----------
        board : Board
            The nengo_loihi object representing the Loihi board.
        nxsdk_board : NxsdkBoard
            The nxsdk object representing the Loihi board.
        spike_input : SpikeInput
            The SpikeInput containing information about which axons are
            to be targeted.
        """
        assert len(self.axon_map) == 0
        input_idxs = np.arange(spike_input.n_neurons)
        for axon in spike_input.axons:
            synapse = axon.target
            atom_bits_extra = synapse.atom_bits_extra()
            tchip_idx, tcore_idx, taxon_ids = board.find_synapse(synapse)
            tchip = d_get(nxsdk_board, b"bjJDaGlwcw==")[tchip_idx]
            tcore = d_get(tchip, b"bjJDb3Jlcw==")[tcore_idx]
            spikes = axon.map_spikes(input_idxs)
            for input_idx, spike in zip(input_idxs, spikes):
                self.axon_map.setdefault(input_idx, [])

                taxon_id = taxon_ids[spike.axon_idx] if spike is not None else None
                if taxon_id is None:
                    continue  # this is a dummy axon, so do not connect

                self.axon_map[input_idx].append(
                    np.array(
                        (
                            -1,
                            axon.pop_type,
                            tchip.id,
                            tcore.id,
                            taxon_id,
                            spike.atom,
                            atom_bits_extra,
                        ),
                        dtype=self.spike_dtype,
                    )
                )

    def spikes_to_loihi(self, input_idxs):
        """Map spike input indices to axons targeting chip locations.

        Parameters
        ----------
        input_idxs : list of int
            Indices of positions in the SpikeInput that are currently spiking.

        Returns
        -------
        axons : generator of ndarray(dtype=spike_dtype)
            Axons targeting physical locations on the chip.
        """
        return (axon for i in input_idxs for axon in self.axon_map[i])


class CompartmentConfig(Config):
    DECAY_U_MAX = d(b"NDA5NQ==", int)
    DECAY_V_MAX = d(b"NDA5NQ==", int)
    REFRACT_DELAY_MAX = d(b"NjM=", int)

    params = ("decay_u", "decay_v", "refract_delay", "enable_noise")

    def __init__(self, decay_v, decay_u, refract_delay, enable_noise):
        super().__init__()
        self.decay_v = decay_v
        self.decay_u = decay_u
        self.refract_delay = refract_delay
        self.enable_noise = enable_noise


class VthConfig(Config):
    """Represents the Vth config information of a compartment.

    Attributes
    ----------
    vth : int
        The mantissa of the voltage threshold for a compartment. To get the
        actual voltage threshold, this is multiplied by VTH_EXP.
    """

    params = ("vth",)

    def __init__(self, vth):
        super().__init__()
        self.vth = vth


class TraceConfig(Config):
    params = ("tau", "spike_int", "spike_frac")

    def __init__(self, tau=0, spike_int=0, spike_frac=0):
        super().__init__()
        self.tau = tau
        self.spike_int = spike_int
        self.spike_frac = spike_frac
