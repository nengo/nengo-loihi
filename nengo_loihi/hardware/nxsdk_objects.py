from __future__ import division

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

    def new_chip(self):
        chip = Chip(board=self)
        self._add_chip(chip)
        return chip

    def chip_index(self, chip):
        return self.chip_idxs[chip]

    def map_probe(self, probe, nxsdk_probe):
        assert probe not in self.probe_map
        self.probe_map[probe] = nxsdk_probe

    def index_synapse(self, synapse, chip, core, idxs):
        chip_idx = self.chip_index(chip)
        core_idx = chip.core_index(core)
        self.synapse_index[synapse] = (chip_idx, core_idx, idxs)

    def find_synapse(self, synapse):
        return self.synapse_index[synapse]


class Chip:
    """A Loihi Chip on a Board, with multiple Cores."""

    def __init__(self, board):
        self.board = board

        self.cores = []
        self.core_idxs = {}

    @property
    def index(self):
        """Index of this Chip in the parent Board"""
        return self.board.chip_index(self)

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
        self.inputs = []

        self.compartment_cfgs = []
        self.vth_cfgs = []
        self.synapse_cfgs = [None]  # keep index 0 unused
        self.stdp_pre_cfgs = []

        self.synapse_cfg_idxs = {}  # one synfmt per Synapse, for now
        self.synapse_axons = collections.OrderedDict()
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
            compartment_idxs = list(range(i0, i1))
            ax_range = (a0, a1)
            yield block, compartment_idxs, ax_range
            i0 = i1
            a0 = a1

    def iterate_inputs(self):
        i0 = 0
        for inp in self.inputs:
            i1 = i0 + inp.n_neurons
            compartment_idxs = list(range(i0, i1))
            yield inp, compartment_idxs
            i0 = i1

    def iterate_synapses(self):
        for block in self.blocks:
            for synapse in block.synapses:
                yield synapse

    def add_block(self, block):
        self.blocks.append(block)

    def add_input(self, input):
        self.inputs.append(input)

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

        a0 = 0
        if len(self.synapse_axons) > 0:
            last = next(reversed(self.synapse_axons))
            a0 = self.synapse_axons[last][-1] + 1
        idxs_per_synapse = synapse.idxs_per_synapse()
        idxs = [a0 + idxs_per_synapse * i for i in range(synapse.n_axons)]
        self.synapse_axons[synapse] = idxs
        self.board.index_synapse(synapse, self.chip, self, idxs)

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
    axon_map : {int: LoihiAxon}
        Map from axon indices in the SpikeInput to LoihiAxons targeting
        particular locations on the chip.
    """

    class LoihiAxon:
        """Represents an axon going to the chip.

        Parameters
        ----------
        axon_type : int
            The population type of axon. 0 for discrete, 16 for pop16,
            and 32 for pop32.
        chip_id : int
            The actual ID of the target chip on the board.
        core_id : int
            The actual ID of the target core on the board.
        axon_id : int
            The actual ID of the target axon on the board.
        atom : int
            The population index (atom), used if this axon sends population
            spikes (i.e. axon_type != 0).
        """

        __slots__ = ["axon_type", "chip_id", "core_id", "axon_id", "atom"]

        def __init__(self, axon_type, chip_id, core_id, axon_id, atom=0):
            # TODO: obfuscate axon_type, or atom?
            assert axon_type in (0, 16, 32)
            self.axon_type = axon_type
            self.chip_id = chip_id
            self.core_id = core_id
            self.axon_id = axon_id
            self.atom = atom

        def _slots_str(self):
            return ", ".join("%s=%s" % (s, getattr(self, s)) for s in self.__slots__)

        def __repr__(self):
            return "%s(%s)" % (type(self).__name__, self._slots_str())

    class LoihiSpike:
        """Represents a spike going to the chip.

        Parameters
        ----------
        time : int
            The timestep at which the spike should be sent to the chip.
        axon : LoihiSpikeInput.LoihiAxon
            The axon information to target the spike to a particular chip axon.
        """

        __slots__ = ["time", "axon"]

        def __init__(self, time, axon):
            self.time = time
            self.axon = axon

        def __repr__(self):
            return "%s(time=%s, %s)" % (
                type(self).__name__,
                self.time,
                self.axon._slots_str(),
            )

    def __init__(self):
        self.axon_map = {}  # maps spike_input idx to axon in self.axons
        self.sent_count = 0

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
            axon_type = axon.pop_type
            assert axon_type in (0, 32), "Only discrete and pop32 supported"
            tchip_idx, tcore_idx, tsyn_ids = board.find_synapse(axon.target)
            tchip = d_get(nxsdk_board, b"bjJDaGlwcw==")[tchip_idx]
            tcore = d_get(tchip, b"bjJDb3Jlcw==")[tcore_idx]
            spikes = axon.map_spikes(input_idxs)
            for input_idx, spike in zip(input_idxs, spikes):
                if spike is not None:
                    taxon_idx = int(spike.axon_id)
                    taxon_id = int(tsyn_ids[taxon_idx])
                    self.axon_map.setdefault(input_idx, []).append(
                        self.LoihiAxon(
                            axon_type=axon_type,
                            chip_id=tchip.id,
                            core_id=tcore.id,
                            axon_id=taxon_id,
                            atom=spike.atom,
                        )
                    )

    def spikes_to_loihi(self, t, input_idxs):
        """Map spike input indices to spikes for the chip.

        Parameters
        ----------
        t : int
            Current timestep.
        input_idxs : list of int
            Indices of positions in the SpikeInput that are currently spiking.

        Returns
        -------
        spikes : generator of LoihiSpike
            Spikes targeting physical locations on the chip.
        """
        for input_idx in input_idxs:
            for axon in self.axon_map[input_idx]:
                yield self.LoihiSpike(time=t, axon=axon)


class CompartmentConfig(Config):
    DECAY_U_MAX = d(b"NDA5NQ==", int)
    DECAY_V_MAX = d(b"NDA5NQ==", int)
    REFRACT_DELAY_MAX = d(b"NjM=", int)

    params = ("decay_u", "decay_v", "refract_delay", "enable_noise")

    def __init__(self, decay_v, decay_u, refract_delay, enable_noise):
        super(CompartmentConfig, self).__init__()
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
        super(VthConfig, self).__init__()
        self.vth = vth


class TraceConfig(Config):
    params = ("tau", "spike_int", "spike_frac")

    def __init__(self, tau=0, spike_int=0, spike_frac=0):
        super(TraceConfig, self).__init__()
        self.tau = tau
        self.spike_int = spike_int
        self.spike_frac = spike_frac
