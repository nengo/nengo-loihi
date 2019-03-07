from __future__ import division

import collections

import numpy as np

from nengo_loihi.block import Profile


CX_PROFILES_MAX = 32
VTH_PROFILES_MAX = 8
SYNAPSE_FMTS_MAX = 16


class Board:
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
        return [[core.n_synapses for core in chip.cores]
                for chip in self.chips]

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

    def map_probe(self, probe, n2probe):
        assert probe not in self.probe_map
        self.probe_map[probe] = n2probe

    def index_synapse(self, synapse, chip, core, idxs):
        chip_idx = self.chip_index(chip)
        core_idx = chip.core_index(core)
        self.synapse_index[synapse] = (chip_idx, core_idx, idxs)

    def find_synapse(self, synapse):
        return self.synapse_index[synapse]


class Chip:
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
    def __init__(self, chip):
        self.chip = chip
        self.blocks = []
        self.inputs = []

        self.cxProfiles = []
        self.vthProfiles = []
        self.synapseFmts = [None]  # keep index 0 unused
        self.stdpPreCfgs = []

        self.synapse_fmt_idxs = {}  # one synfmt per Synapse, for now
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
        return sum(synapse.size() for block in self.blocks
                   for synapse in block.synapses)

    def iterate_blocks(self):
        i0 = 0
        a0 = 0
        for block in self.blocks:
            i1 = i0 + block.compartment.n_compartments
            a1 = a0 + sum(ax.n_axons for ax in block.axons)
            cx_idxs = list(range(i0, i1))
            ax_range = (a0, a1)
            yield block, cx_idxs, ax_range
            i0 = i1
            a0 = a1

    def iterate_inputs(self):
        i0 = 0
        for inp in self.inputs:
            i1 = i0 + inp.n_neurons
            cx_idxs = list(range(i0, i1))
            yield inp, cx_idxs
            i0 = i1

    def iterate_synapses(self):
        for block in self.blocks:
            for synapse in block.synapses:
                yield synapse

    def add_block(self, block):
        self.blocks.append(block)

    def add_input(self, input):
        self.inputs.append(input)

    def add_cx_profile(self, cx_profile):
        self.cxProfiles.append(cx_profile)
        return len(self.cxProfiles) - 1  # index

    def add_vth_profile(self, vth_profile):
        self.vthProfiles.append(vth_profile)
        return len(self.vthProfiles) - 1  # index

    def add_stdp_pre_cfg(self, stdp_pre_cfg):
        self.stdpPreCfgs.append(stdp_pre_cfg)
        return len(self.stdpPreCfgs) - 1  # index

    def add_synapse(self, synapse):
        synapse_fmt_idx = self.get_synapse_fmt_idx(synapse.synapse_fmt)
        self.synapse_fmt_idxs[synapse] = synapse_fmt_idx

        a0 = 0
        if len(self.synapse_axons) > 0:
            last = next(reversed(self.synapse_axons))
            a0 = self.synapse_axons[last][-1] + 1
        idxs_per_synapse = synapse.idxs_per_synapse()
        idxs = [a0 + idxs_per_synapse*i for i in range(synapse.n_axons)]
        self.synapse_axons[synapse] = idxs
        self.board.index_synapse(synapse, self.chip, self, idxs)

        s0 = 0
        if len(self.synapse_entries) > 0:
            last = next(reversed(self.synapse_entries))
            s0 = self.synapse_entries[last][1]
        s1 = s0 + synapse.size()
        self.synapse_entries[synapse] = (s0, s1)

    def get_synapse_fmt(self, synapse):
        return self.synapseFmts[self.synapse_fmt_idxs[synapse]]

    def get_synapse_fmt_idx(self, synapse_fmt):
        try:
            return self.synapseFmts.index(synapse_fmt)
        except ValueError:
            self.synapseFmts.append(synapse_fmt)
            return len(self.synapseFmts) - 1  # index


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

        __slots__ = ['axon_type', 'chip_id', 'core_id', 'axon_id', 'atom']

        def __init__(self, axon_type, chip_id, core_id, axon_id, atom=0):
            assert axon_type in (0, 16, 32)
            self.axon_type = axon_type
            self.chip_id = chip_id
            self.core_id = core_id
            self.axon_id = axon_id
            self.atom = atom

        def _slots_str(self):
            return ", ".join("%s=%s" % (s, getattr(self, s))
                             for s in self.__slots__)

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

        __slots__ = ['time', 'axon']

        def __init__(self, time, axon):
            self.time = time
            self.axon = axon

        def __repr__(self):
            return "%s(time=%s, %s)" % (
                type(self).__name__, self.time, self.axon._slots_str())

    def __init__(self):
        self.axon_map = {}  # maps cx_spike_input idx to axon in self.axons
        self.sent_count = 0

    def set_axons(self, board, n2board, spike_input):
        """Initialize the axon map for this object.

        Parameters
        ----------
        board : Board
            The nengo_loihi object representing the Loihi board.
        n2board : nxsdk.N2Board
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
            tchip = n2board.n2Chips[tchip_idx]
            tcore = tchip.n2Cores[tcore_idx]
            spikes = axon.map_cx_spikes(input_idxs)
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
                        ))

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


class CxProfile(Profile):
    DECAY_U_MAX = 2**12 - 1
    DECAY_V_MAX = 2**12 - 1
    REFRACT_DELAY_MAX = 2**6 - 1

    params = ('decayU', 'decayV', 'refractDelay', 'enableNoise')

    def __init__(self, decayV, decayU, refractDelay, enableNoise):
        super(CxProfile, self).__init__()
        self.decayV = decayV
        self.decayU = decayU
        self.refractDelay = refractDelay
        self.enableNoise = enableNoise


class VthProfile(Profile):
    """Represents the VthProfile of a compartment (Cx).

    Attributes
    ----------
    vth : int
        The mantissa of the voltage threshold for a compartment. To get the
        actual voltage threshold, this is multiplied by VTH_EXP (64).
    """
    params = ('vth',)

    def __init__(self, vth):
        super(VthProfile, self).__init__()
        self.vth = vth


class TraceCfg(Profile):
    params = ('tau', 'spikeLevelInt', 'spikeLevelFrac')

    def __init__(self, tau=0, spikeLevelInt=0, spikeLevelFrac=0):
        super(TraceCfg, self).__init__()
        self.tau = tau
        self.spikeLevelInt = spikeLevelInt
        self.spikeLevelFrac = spikeLevelFrac
