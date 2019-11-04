from collections import OrderedDict

from nengo import Node
from nengo.exceptions import SimulationError
from nengo.params import Default
import numpy as np


class HostSendNode(Node):
    """For sending host->chip messages"""

    def __init__(self, dimensions, label=Default, check_output=False):
        self.queue = []
        super(HostSendNode, self).__init__(
            self.update, size_in=dimensions, size_out=0, label=label
        )
        if hasattr(self, "check_output"):
            self.check_output = check_output

    def update(self, t, x):
        assert len(self.queue) == 0 or t > self.queue[-1][0]
        self.queue.append((t, x))


class HostReceiveNode(Node):
    """For receiving chip->host messages"""

    def __init__(self, dimensions, label=Default, check_output=False):
        self.queue = [(0, np.zeros(dimensions))]
        super(HostReceiveNode, self).__init__(
            self.update, size_in=0, size_out=dimensions, label=label
        )
        if hasattr(self, "check_output"):
            self.check_output = check_output

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
        self.error_target = None
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

    def collect_errors(self):
        return ()

    def collect_spikes(self):
        return self.spikes


class ChipReceiveNeurons(ChipReceiveNode):
    """Passes spikes directly (no on-off neuron encoding)"""

    def __init__(self, dimensions, neuron_type=None, label=Default):
        self.neuron_type = neuron_type
        super(ChipReceiveNeurons, self).__init__(dimensions, dimensions, label=label)


class PESModulatoryTarget:
    def __init__(self, target):
        self.errors = OrderedDict()
        self.error_target = target
        self.spike_input = None

    def clear(self):
        self.errors.clear()

    def receive(self, t, x):
        assert len(self.errors) == 0 or t >= next(reversed(self.errors))
        if t in self.errors:
            self.errors[t] += x
        else:
            self.errors[t] = np.array(x)

    def collect_errors(self):
        return self.errors.items()

    def collect_spikes(self):
        return ()
