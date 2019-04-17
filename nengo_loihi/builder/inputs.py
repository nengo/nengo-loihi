import os

import numpy as np
from nengo import Node
from nengo.exceptions import SimulationError
from nengo.params import Default
from nengo.processes import Process

from nengo_loihi.dvs import DVSEvents


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

    def __init__(self, dimensions, size_out, label=Default, output=None):
        self.raw_dimensions = dimensions
        self.spike_target = None
        output = self.update if output is None else output
        super().__init__(output, size_in=0, size_out=size_out, label=label)

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

    def __init__(self, dimensions, neuron_type=None, label=Default, output=None):
        self.neuron_type = neuron_type
        super().__init__(dimensions, dimensions, label=label, output=output)


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


class DVSFileChipNode(ChipReceiveNeurons):
    """Node for DVS input to Loihi chip from a pre-recorded file

    Parameters
    ----------
    file_path : string
        The path of the file to read from. Can be a ``.aedat`` or ``.events`` file.
    kind : "aedat" or "events" or None, optional
        The format of the file. If ``None`` (default), this will be detected from the
        file extension.
    t_start : float, optional
        Offset for the time in the file to start at, in seconds.
    rel_time : bool, optional
        Whether to make all times relative to the first event, or not. Defaults
        to True for ``.aedat`` files and False otherwise.
    pool : (int, int), optional
        Number of pixels to pool over in the vertical and horizontal
        directions, respectively.
    channels_last : bool, optional
        Whether to make the channels (i.e. the polarity) the least-significant
        index (True) or the most-significant index (False).
    """

    def __init__(
        self,
        file_path,
        format=None,
        t_start=0,
        rel_time=None,
        pool=(1, 1),
        channels_last=True,
        label=None,
    ):
        self.file_path = file_path
        assert os.path.exists(self.file_path)
        self.format = format
        self.t_start = t_start
        self.rel_time = rel_time

        self.dvs_height = 180
        self.dvs_width = 240
        self.dvs_polarity = 2
        self.channels_last = channels_last
        self.pool = pool

        self.height = int(np.ceil(self.dvs_height / self.pool[0]))
        self.width = int(np.ceil(self.dvs_width / self.pool[1]))
        self.polarity = self.dvs_polarity
        self.size = self.height * self.width * self.polarity

        # for outputting images if the node is run in Nengo
        output = DVSFileImageProcess(self)

        super().__init__(self.size, label=label, output=output)

    def _read_events(self):
        """Helper function to read events from the target file."""

        dvs_events = DVSEvents()
        dvs_events.read_file(self.file_path, kind=self.format, rel_time=self.rel_time)
        events = dvs_events.events

        pool_y, pool_x = self.pool
        if self.channels_last:
            stride_x = self.polarity
            stride_y = self.polarity * self.width
            stride_p = 1
        else:
            stride_x = 1
            stride_y = self.width
            stride_p = self.width * self.height

        events_t = events[:]["t"]
        events_idx = (
            (events[:]["y"] // pool_y) * stride_y
            + (events[:]["x"] // pool_x) * stride_x
            + events[:]["p"] * stride_p
        )
        return events_t, events_idx


class DVSFileImageProcess(Process):
    """Convert DVS events to images."""

    def __init__(self, dvs_file_chip_node, **kwargs):
        super().__init__(default_size_in=0, **kwargs)
        self.dvs_file_chip_node = dvs_file_chip_node

    def make_step(self, shape_in, shape_out, dt, rng, state):
        assert shape_in == (0,)
        assert len(shape_out) == 1

        height = self.dvs_file_chip_node.height
        width = self.dvs_file_chip_node.width
        polarity = self.dvs_file_chip_node.polarity
        t_start = self.dvs_file_chip_node.t_start
        events_t, events_idx = self.dvs_file_chip_node._read_events()

        def step_dvsfileimage(t):
            t = t_start + t
            t0 = (t - dt) * 1e6
            t1 = t * 1e6

            et = events_t
            idxs = events_idx[(et >= t0) & (et < t1)]

            image = np.zeros(height * width * polarity)
            np.add.at(image, idxs, 1 / dt)
            return image

        return step_dvsfileimage
