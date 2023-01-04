import numpy as np
from nengo import Node, Process
from nengo.exceptions import SimulationError
from nengo.params import Default


class HostSendProcess(Process):
    class FillQueueStep:
        """Fill a simple queue to be processed externally."""

        def __init__(self):
            self.queue = []

        def __call__(self, t, x):
            assert len(self.queue) == 0 or t > self.queue[-1][0]
            self.queue.append((t, np.copy(x)))

    def make_step(self, shape_in, shape_out, dt, rng, state):
        assert shape_out == (0,)
        return self.FillQueueStep()


class HostSendNode(Node):
    """For sending host->chip messages."""

    def __init__(self, dimensions, label=Default):
        super().__init__(HostSendProcess(), size_in=dimensions, size_out=0, label=label)


class HostReceiveProcess(Process):
    class EmptyQueueStep:
        """Empty a queue that is filled externally."""

        def __init__(self, size_out):
            self.size_out = size_out
            self.queue = [(0, np.zeros(self.size_out))]

        def __call__(self, t):
            while len(self.queue) > 1 and self.queue[0][0] < t:
                self.queue.pop(0)
            return self.queue[0][1]

        def receive(self, t, x):
            assert not self.queue or t >= self.queue[0][0]
            self.queue.append((t, x))

    def make_step(self, shape_in, shape_out, dt, rng, state):
        assert shape_in == (0,)
        return self.EmptyQueueStep(shape_out)


class HostReceiveNode(Node):
    """For receiving chip->host messages."""

    def __init__(self, dimensions, label=Default):
        super().__init__(
            HostReceiveProcess(), size_in=0, size_out=dimensions, label=label
        )


class ChipReceiveProcess(Process):
    class RaiseSimErrorStep:
        def __init__(self, message):
            self.message = message

        def __call__(self, t):
            raise SimulationError(self.message)

    def make_step(self, shape_in, shape_out, dt, rng, state):
        return self.RaiseSimErrorStep("ChipReceiveProcess should not be run")


class ChipReceiveNode(Node):
    """
    Represents chip end-point for host->chip encoded connections.

    This Node does not do anything, other than act as a placeholder in ``Connection``s.

    Attributes
    ----------
    raw_dimensions : int
        Number of dimensions originally passed in.
    """

    def __init__(self, dimensions, size_out, label=Default):
        self.raw_dimensions = dimensions
        super().__init__(
            ChipReceiveProcess(), size_in=0, size_out=size_out, label=label
        )


class ChipReceiveNeurons(ChipReceiveNode):
    """
    Passes spikes directly (no on-off neuron encoding).

    This Node does not do anything, other than act as a placeholder in ``Connection``s.

    Attributes
    ----------
    neuron_type : NeuronType
       NeuronType object that is used to generate spikes.
    """

    def __init__(self, dimensions, neuron_type=None, label=Default):
        self.neuron_type = neuron_type
        super().__init__(dimensions, dimensions, label=label)


class PESModulatoryTarget:
    """
    Represents chip end-point for host->chip messages.

    Attributes
    ----------
    error_target : nengo.Probe
        Probe on the output of the learning connection. This can then be used to look
        up the synapse for weight adjustment.
    """

    def __init__(self, target):
        super().__init__()
        self.error_target = target
