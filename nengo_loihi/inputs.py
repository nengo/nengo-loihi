import os

from nengo.utils.compat import is_integer
import numpy as np

from nengo_loihi.dvs import get_dvs_reader
from nengo_loihi.builder.inputs import ChipReceiveNeurons


class LoihiInput:
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


class DVSInput(LoihiInput):
    """Live input from a spiking DVS camera."""

    N_CORES = -(-240*180*2 // 1024)  # ceil of DVS inputs/compartments per core

    def __init__(self, pool=(1, 1), channels_last=True, label=None):
        super(DVSInput, self).__init__(label=label)
        self.dvs_height = 180
        self.dvs_width = 240
        self.dvs_polarity = 2

        self.channels_last = channels_last
        self.pool = pool

        self.height = int(np.ceil(self.dvs_height / self.pool[0]))
        self.width = int(np.ceil(self.dvs_width / self.pool[1]))
        self.polarity = self.dvs_polarity
        self.size = self.height * self.width * self.polarity
        self.n_neurons = self.size  # builder assumes inputs have this

        # file-specific inputs
        self.file_node = None  # DVSFileChipNode handling the input


class DVSFileChipNode(ChipReceiveNeurons):
    """Node for DVS input to Loihi chip from a pre-recorded file

    Parameters
    ----------
    filename : string
        The path of the file to read from. Can be a `.aedat` or `.events` file.
    format : string ('aedat' or 'events'), optional
        The format of the file. By default, this will be detected from the
        file extension.
    t_start : float, optional
        Offset for the time in the file to start at, in seconds.
    rel_time : bool, optional
        Whether to make all times relative to the first event, or not. Defaults
        to True for 'aedat' files and False otherwise.
    pool : (int, int), optional
        Number of pixels to pool over in the vertical and horizontal
        directions, respectively.
    channels_last : bool, optional
        Whether to make the channels the least-significant index (True) or the
        most-significant index (False).
    use_cores : bool, optional
        Whether to use Loihi cores to map the input, simulating the live DVS.
    """

    def __init__(self, filename, format=None, t_start=0, rel_time=None,
                 pool=(1, 1), channels_last=True, use_cores=False, label=None):
        self.filename = filename
        assert os.path.exists(self.filename)
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
        d = self.height * self.width * self.polarity
        super(DVSFileChipNode, self).__init__(d, d, label=label)

        self.use_cores = use_cores

        # for nengo node reading (`update` function)
        self._events_t = None
        self._events_idx = None
        self.dt = 0.001

    def read_events(self, pool_xy=None, stride_xyp=None):
        reader = get_dvs_reader(self.filename, format=self.format)
        events = reader.read_events(rel_time=self.rel_time)

        if pool_xy is not None:
            pool_x, pool_y = pool_xy
        else:
            pool_y, pool_x = self.pool

        if stride_xyp is not None:
            stride_x, stride_y, stride_p = stride_xyp
        elif self.channels_last:
            stride_x = self.polarity
            stride_y = self.polarity*self.width
            stride_p = 1
        else:
            stride_x = 1
            stride_y = self.width
            stride_p = self.width*self.height

        self._events_t = events[:]['t']
        self._events_idx = (
            (events[:]['y'] // pool_y)*stride_y
            + (events[:]['x'] // pool_x)*stride_x
            + events[:]['p']*stride_p)
        return self._events_t, self._events_idx

    def update(self, t):
        if self._events_t is None:
            self.read_events()

        t = self.t_start + t
        t0 = (t - self.dt) * 1e6
        t1 = (t) * 1e6

        et = self._events_t
        idxs = self._events_idx[(et >= t0) & (et <= t1)]

        image = np.zeros(self.height * self.width * self.polarity)
        image[idxs] += 1 / self.dt
        return image
