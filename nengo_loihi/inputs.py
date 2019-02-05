from __future__ import division

from nengo import Node
from nengo.exceptions import SimulationError
from nengo.params import Default
from nengo.utils.compat import is_integer
import numpy as np


class LoihiInput(object):
    def __init__(self, label=None):
        self.label = label
        self.axons = []

    def add_axon(self, axon):
        self.axons.append(axon)


class SpikeInput(LoihiInput):
    def __init__(self, n_neurons, label=None):
        super(SpikeInput, self).__init__(label=label)
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


class HostSendNode(Node):
    """For sending host->chip messages"""

    def __init__(self, dimensions, label=Default):
        self.queue = []
        super(HostSendNode, self).__init__(
            self.update,
            size_in=dimensions,
            size_out=0,
            label=label,
        )

    def update(self, t, x):
        assert len(self.queue) == 0 or t > self.queue[-1][0]
        self.queue.append((t, x))


class HostReceiveNode(Node):
    """For receiving chip->host messages"""

    def __init__(self, dimensions, label=Default):
        self.queue = [(0, np.zeros(dimensions))]
        self.queue_index = 0
        super(HostReceiveNode, self).__init__(
            self.update,
            size_in=0,
            size_out=dimensions,
            label=label,
        )

    def update(self, t):
        while (len(self.queue) > self.queue_index + 1
               and self.queue[self.queue_index][0] < t):
            self.queue_index += 1
        return self.queue[self.queue_index][1]

    def receive(self, t, x):
        self.queue.append((t, x))


class ChipReceiveNode(Node):
    """For receiving host->chip messages"""

    def __init__(self, dimensions, size_out, label=Default):
        self.raw_dimensions = dimensions
        self.spikes = []
        self.spike_input = None  # set by builder
        super(ChipReceiveNode, self).__init__(
            self.update, size_in=0, size_out=size_out, label=label)

    def clear(self):
        self.spikes.clear()

    def receive(self, t, x):
        assert len(self.spikes) == 0 or t > self.spikes[-1][0]
        assert x.ndim == 1
        self.spikes.append((t, x.nonzero()[0]))

    def update(self, t):
        raise SimulationError("ChipReceiveNodes should not be run")

    def collect_spikes(self):
        assert self.spike_input is not None
        for t, x in self.spikes:
            yield (self.spike_input, t, x)


class ChipReceiveNeurons(ChipReceiveNode):
    """Passes spikes directly (no on-off neuron encoding)"""
    def __init__(self, dimensions, neuron_type=None, label=Default):
        self.neuron_type = neuron_type
        super(ChipReceiveNeurons, self).__init__(
            dimensions, dimensions, label=label)
