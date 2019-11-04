from collections import OrderedDict

from nengo import Node
from nengo.exceptions import SimulationError
from nengo.params import Default
import numpy as np


class HostSendNode(Node):
    """For sending host->chip messages"""

    def __init__(self, dimensions, label=Default):
        self.queue = []
        super(HostSendNode, self).__init__(
            self.update, size_in=dimensions, size_out=0, label=label
        )

    def update(self, t, x):
        assert len(self.queue) == 0 or t > self.queue[-1][0]
        self.queue.append((t, x))


class HostReceiveNode(Node):
    """For receiving chip->host messages"""

    def __init__(self, dimensions, label=Default):
        self.queue = [(0, np.zeros(dimensions))]
        super(HostReceiveNode, self).__init__(
            self.update, size_in=0, size_out=dimensions, label=label
        )

    def update(self, t):
        t1, x = self.queue[-1]
        assert t >= t1
        return x

    def receive(self, t, x):
        self.queue.append((t, x))


class ChipReceiveNode(Node):
    """For receiving host->chip messages"""

    def __init__(self, dimensions, size_out, label=Default):
        self.raw_dimensions = dimensions
        self.spikes = []
        self.spike_input = None  # set by builder
        super(ChipReceiveNode, self).__init__(
            self.update, size_in=0, size_out=size_out, label=label
        )

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
        super(ChipReceiveNeurons, self).__init__(dimensions, dimensions, label=label)


class PESModulatoryTarget:
    def __init__(self, target):
        self.target = target
        self.errors = OrderedDict()

    def clear(self):
        self.errors.clear()

    def receive(self, t, x):
        assert len(self.errors) == 0 or t >= next(reversed(self.errors))
        if t in self.errors:
            self.errors[t] += x
        else:
            self.errors[t] = np.array(x)

    def collect_errors(self):
        for t, x in self.errors.items():
            yield (self.target, t, x)
