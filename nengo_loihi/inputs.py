import numpy as np
from nengo.utils.numpy import is_integer


class LoihiInput:
    def __init__(self, label=None):
        self.label = label
        self.axons = []

    def add_axon(self, axon):
        self.axons.append(axon)


class SpikeInput(LoihiInput):
    def __init__(self, n_neurons, label=None):
        super().__init__(label=label)
        self.n_neurons = n_neurons
        self.spikes = {}  # map sim timestep index to list of spike inds

    def add_spikes(self, ti, spike_idxs):
        assert is_integer(ti)
        ti = int(ti)
        assert ti > 0, "Spike times must be >= 1 (got %d)" % ti
        assert ti not in self.spikes
        self.spikes[ti] = spike_idxs

    def spike_times(self):
        return sorted(self.spikes)

    def spike_idxs(self, ti):
        return self.spikes.get(ti, [])


class DVSInput(LoihiInput):
    """Live input from a spiking DVS camera."""

    dvs_height = 180
    dvs_width = 240
    dvs_polarity = 2
    N_CORES = -(-240 * 180 * 2 // 1024)  # ceil of DVS inputs/compartments per core

    def __init__(self, pool=(1, 1), channels_last=True, label=None):
        super(DVSInput, self).__init__(label=label)
        self.dvs_size = self.dvs_height * self.dvs_width * self.dvs_polarity

        self.channels_last = channels_last
        self.pool = pool

        self.height = int(np.ceil(self.dvs_height / self.pool[0]))
        self.width = int(np.ceil(self.dvs_width / self.pool[1]))
        self.polarity = self.dvs_polarity
        self.size = self.height * self.width * self.polarity
        self.n_neurons = self.size  # builder assumes inputs have this

        # file-specific inputs
        self.file_node = None  # DVSFileChipNode handling the input
