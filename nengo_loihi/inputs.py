from nengo import Process
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


class ChipProcess(Process):
    """Abstract base class for Node processes to be placed on the Loihi board.

    Such processes must then have a NengoLoihi builder to put them on the Loihi board.
    """
