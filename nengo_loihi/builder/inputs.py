import collections

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

    def update(self, t, x):
        assert len(self.queue) == 0 or t > self.queue[-1][0]
        self.queue.append((t, x.copy()))


class HostReceiveNode(Node):
    """For receiving chip->host messages"""

    def __init__(self, dimensions, label=Default, check_output=False):
        self.queue = collections.deque()
        super(HostReceiveNode, self).__init__(
            self.update, size_in=0, size_out=dimensions, label=label
        )

    def update(self, t):
        if t <= 0:
            return np.zeros(self.size_out)

        t1, x = self.queue.popleft()
        assert abs(t - t1) < 1e-8
        return x

    def receive(self, t, x):
        # we assume that x will not be mutated (i.e. we do not need to copy)
        self.queue.append((t, x))


class ChipReceiver:
    """Abstract interface for objects that receive spikes or errors to send to the chip.

    These objects are used in ``Simulator._collect_receiver_info``, where the output
    from the corresponding sender can be routed to the correct target.

    Attributes
    ----------
    error_target : nengo.Probe
        Probe on the output of the learning connection. This can then be used to look
        up the synapse for weight adjustment.
    spike_target : SpikeInput
        Spike input that will inject the spikes from the sender onto the chip.
    """

    def __init__(self):
        self.error_target = None
        self.spike_target = None


class ChipReceiveNode(Node, ChipReceiver):
    """Represents chip end-point for host->chip encoded connections.

    This Node does not do anything, other than act as a placeholder in ``Connection``s.
    """

    def __init__(self, dimensions, size_out, label=Default):
        self.raw_dimensions = dimensions
        super(ChipReceiveNode, self).__init__(
            self.update, size_in=0, size_out=size_out, label=label
        )

    def update(self, t):
        raise SimulationError("ChipReceiveNodes should not be run")


class ChipReceiveNeurons(ChipReceiveNode):
    """Passes spikes directly (no on-off neuron encoding)"""

    def __init__(self, dimensions, neuron_type=None, label=Default):
        self.neuron_type = neuron_type
        super(ChipReceiveNeurons, self).__init__(dimensions, dimensions, label=label)


class PESModulatoryTarget(ChipReceiver):
    """Represents chip end-point for host->chip messages.

    This Node does not do anything, other than act as a placeholder in ``Connection``s.
    All the action happens in
    """

    def __init__(self, target):
        super(PESModulatoryTarget, self).__init__()
        self.error_target = target
