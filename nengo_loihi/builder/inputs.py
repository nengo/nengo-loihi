import numpy as np
from nengo import Node
from nengo.exceptions import SimulationError
from nengo.params import Default


class HostSendNode(Node):
    """For sending host->chip messages"""

    def __init__(self, dimensions, label=Default):
        self.queue = []
        super().__init__(self.update, size_in=dimensions, size_out=0, label=label)

    def update(self, t, x):
        assert len(self.queue) == 0 or t > self.queue[-1][0]
        self.queue.append((t, x))


class HostReceiveNode(Node):
    """For receiving chip->host messages"""

    def __init__(self, dimensions, label=Default):
        self.queue = [(0, np.zeros(dimensions))]
        self.queue_index = 0
        super().__init__(self.update, size_in=0, size_out=dimensions, label=label)

    def update(self, t):
        while (
            len(self.queue) > self.queue_index + 1
            and self.queue[self.queue_index][0] < t
        ):
            self.queue_index += 1
        return self.queue[self.queue_index][1]

    def receive(self, t, x):
        self.queue.append((t, x))


class ChipReceiveNode(Node):
    """Represents chip end-point for host->chip encoded connections.

    This Node does not do anything, other than act as a placeholder in ``Connection``s.

    Attributes
    ----------
    spike_target : SpikeInput
        Spike input that will inject the spikes from the sender onto the chip.
    """

    def __init__(self, dimensions, size_out, label=Default):
        self.raw_dimensions = dimensions
        self.spike_target = None
        super().__init__(self.update, size_in=0, size_out=size_out, label=label)

    def update(self, t):
        raise SimulationError("{} should not be run".format(type(self).__name__))


class ChipReceiveNeurons(ChipReceiveNode):
    """Passes spikes directly (no on-off neuron encoding).

    This Node does not do anything, other than act as a placeholder in ``Connection``s.

    Attributes
    ----------
    spike_target : SpikeInput
        Spike input that will inject the spikes from the sender onto the chip.
    """

    def __init__(self, dimensions, neuron_type=None, label=Default):
        self.neuron_type = neuron_type
        super().__init__(dimensions, dimensions, label=label)


class PESModulatoryTarget:
    """Represents chip end-point for host->chip messages.

    Attributes
    ----------
    error_target : nengo.Probe
        Probe on the output of the learning connection. This can then be used to look
        up the synapse for weight adjustment.
    """

    def __init__(self, target):
        super().__init__()
        self.error_target = target
