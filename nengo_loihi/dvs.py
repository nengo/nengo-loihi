import ctypes
import os
import warnings

import numpy as np

from nengo_loihi.inputs import ChipProcess


class DVSFileChipProcess(ChipProcess):
    """Process for DVS input to Loihi chip from a pre-recorded file.

    Parameters
    ----------
    file_path : string
        The path of the file to read from. Can be a ``.aedat`` or ``.events`` file.
    file_fmt : "aedat" or "events" or None, optional
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
    dvs_height : int, optional
        The actual height (in pixels) of the DVS sensor. Only change this if your
        sensor has a non-standard height. If you wish to make the output of this node
        smaller, use the ``pool`` argument instead.
    dvs_width : int, optional
        The actual width (in pixels) of the DVS sensor. Only change this if your
        sensor has a non-standard width. If you wish to make the output of this node
        smaller, use the ``pool`` argument instead.
    **kwargs
        Extra arguments to pass to the `nengo.Process` constructor.

    Examples
    --------

    This example shows how to create the process, use it in a `~nengo.Node`, and
    connect it to neurons on the Loihi chip. The DVS events loaded from the file will
    be transferred immediately to the Loihi chip; none of the simulation is on the host.

    .. testcode::

        with nengo.Network() as net:
            dvs_process = nengo_loihi.dvs.DVSFileChipProcess("my-dvs-events.aedat")
            u = nengo.Node(dvs_process)
            ens = nengo.Ensemble(dvs_process.size, dimensions=1)
            nengo.Connection(u, ens.neurons)
    """

    def __init__(
        self,
        file_path,
        file_fmt=None,
        t_start=0,
        rel_time=None,
        pool=(1, 1),
        channels_last=True,
        dvs_height=180,
        dvs_width=240,
        **kwargs
    ):
        self.file_path = file_path
        self.file_fmt = file_fmt
        self.t_start = t_start
        self.rel_time = rel_time

        self.dvs_height = dvs_height
        self.dvs_width = dvs_width
        self.dvs_polarity = 2
        self.channels_last = channels_last
        self.pool = pool

        self.height = int(np.ceil(self.dvs_height / self.pool[0]))
        self.width = int(np.ceil(self.dvs_width / self.pool[1]))
        self.polarity = self.dvs_polarity
        self.size = self.height * self.width * self.polarity

        super().__init__(default_size_in=0, default_size_out=self.size, **kwargs)

    def make_step(self, shape_in, shape_out, dt, rng, state):
        """Make the step function to display the DVS events as image frames.

        This step function is only called when using this process in a
        `nengo.Simulator`. When using it in a `nengo_loihi.Simulator`, the events
        are transferred directly to the Loihi board.
        """
        assert shape_in == (0,)
        assert len(shape_out) == 1

        height = self.height
        width = self.width
        polarity = self.polarity
        t_start = self.t_start
        events_t, events_idx = self._read_events()

        def step_dvsfileimage(t):
            t = t_start + t
            t_lower = (t - dt) * 1e6
            t_upper = t * 1e6

            idxs = events_idx[(events_t >= t_lower) & (events_t < t_upper)]

            image = np.zeros(height * width * polarity)
            np.add.at(image, idxs, 1 / dt)
            return image

        return step_dvsfileimage

    def _read_events(self):
        """Helper function to read events from the target file."""

        dvs_events = DVSEvents()
        dvs_events.read_file(
            self.file_path, file_fmt=self.file_fmt, rel_time=self.rel_time
        )
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
            (events[:]["y"].astype(np.int32) // pool_y) * stride_y
            + (events[:]["x"].astype(np.int32) // pool_x) * stride_x
            + events[:]["p"].astype(np.int32) * stride_p
        )
        return events_t, events_idx


class DVSEvents:
    """A group of events from a Dynamic Vision Sensor (DVS) spiking camera.

    Attributes
    ----------
    events : structured `numpy.ndarray`
        A structured array with the following fields:

          * "y": The vertical coordinate of the event.
          * "x": The horizontal coordinate of the event.
          * "p": The polarity of the event (``0`` for off, ``1`` for on).
          * "v": The event trigger (``0`` for DVS events, ``1`` for external events).
          * "t": The event timestamp in microseconds.

    n_events : int
        The number of events.
    """

    events_dtype = np.dtype(
        [("y", "u2"), ("x", "u2"), ("p", "u1"), ("v", "u1"), ("t", "u4")]
    )

    def __init__(self):
        self.events = None

    @property
    def n_events(self):
        """The number of events (equals ``len(self.events)``)."""
        return len(self.events)

    @staticmethod
    def from_file(file_path, **kwargs):
        """Create a new `.DVSEvents` object with events from a file.

        Parameters
        ----------
        file_path : str
            The path to the events file.
        **kwargs
            Additional keyword arguments to pass to `.DVSEvents.read_file`.
        """
        events = DVSEvents()
        events.read_file(file_path, **kwargs)
        return events

    def init_events(self, event_data=None, n_events=None):
        """Initialize ``events`` array.

        Only required if configuring events manually (i.e. not reading from a file).

        Parameters
        ----------
        event_data : list of tuples, optional
            Each tuple is of the form ``(y, x, p, v, t)``. See `.DVSEvents` for the
            definitions of these fields.
        n_events : int, optional
            The number of events in the new (empty) events array, in the case that
            ``event_data`` is not provided.
        """
        assert (event_data is not None) or (n_events is not None)

        if event_data is not None:
            n_events = len(event_data) if n_events is None else n_events
            assert n_events == len(
                event_data
            ), "Specified number of events (%d) does not match length of data (%d)" % (
                n_events,
                len(event_data),
            )

        if self.events is not None:
            warnings.warn("`events` has already been initialized. Overwriting.")

        if event_data is not None:
            self.events = np.array(event_data, dtype=self.events_dtype)
        else:
            self.events = np.zeros(n_events, dtype=self.events_dtype)

    def read_file(self, file_path, file_fmt=None, rel_time=None):
        """Read events from a file.

        Parameters
        ----------
        file_path : str
            The path to the events file.
        file_fmt : "aedat" or "events" or None, optional
            The file format of the events file. If ``None``, will be detected
            based on the file extension.
        rel_time : bool, optional
            Whether timestamps should be relative to the first event, or absolute.
        """
        assert os.path.exists(file_path), "File does not exist: %r" % (file_path,)

        if file_fmt is None:
            file_fmt = self._get_extension(file_path)
            if file_fmt == "":
                raise ValueError(
                    "Events file %r has no extension. Could not detect file format. "
                    "Please pass a value for `file_fmt` to specify the format."
                    % (file_path,)
                )

        if file_fmt == "aedat":
            io = AEDatFileIO(file_path)
            io.read_events(dvs_events=self, rel_time=rel_time)
        elif file_fmt == "events":
            io = EventsFileIO(file_path)
            io.read_events(dvs_events=self, rel_time=rel_time)
        else:
            raise ValueError(
                "Unrecognized file format %r for file %r" % (file_fmt, file_path)
            )

    def write_file(self, file_path):
        """Write events to a file.

        Currently only supports the ``.events`` file format.

        Parameters
        ----------
        file_path : str
            The path to the events file.
        """

        file_fmt = self._get_extension(file_path)
        if file_fmt == "":
            raise ValueError(
                "The provided path %r has no extension. Please use the '.events' "
                "extension." % (file_path,)
            )

        if file_fmt == "events":
            io = EventsFileIO(file_path)
            io.write_events(self)
        else:
            raise ValueError(
                "Unsupported file format %r for writing events files" % (file_fmt,)
            )

    def _get_extension(self, file_path):
        _, ext = os.path.splitext(file_path)
        ext = ext.lower()
        return ext[1:] if len(ext) > 0 and ext[0] == "." else ext


class DVSFileIO:
    """Abstract base class for reading DVS event files."""

    def __init__(self, file_path, rel_time):
        self.file_path = file_path
        self.rel_time = rel_time

    def read_events(self, rel_time=None):
        raise NotImplementedError()

    def write_events(self, rel_time=None):
        raise NotImplementedError()


class AEDatFileIO(DVSFileIO):
    """Reader for events files using the AEDat file format."""

    class AEDatEvent(ctypes.BigEndianStructure):
        """Class for parsing AEDat event entries.

        Based on the format spec [1]_.

        References
        ----------
        .. [1] "AEDat file formats", inivation-docs,
           https://inivation.github.io/inivation-docs/Software%20user%20guides/AEDAT_file_formats.html
        """

        _fields_ = [
            ("type", ctypes.c_uint64, 1),
            ("y", ctypes.c_uint64, 9),
            ("x", ctypes.c_uint64, 10),
            ("polarity", ctypes.c_uint64, 1),
            ("trigger", ctypes.c_uint64, 1),
            ("adc_sample", ctypes.c_uint64, 10),
            ("t", ctypes.c_uint64, 32),
        ]

    def __init__(self, file_path):
        super().__init__(file_path, rel_time=True)

    def _read_header(self, fh):
        header = []
        buf = fh.read(1024)
        while buf[0] == ord("#"):
            end = buf.find(b"\n")
            if end > 0:
                header.append(buf[1 : end + 1])
                buf = buf[end + 1 :]

            if len(buf) == 0 or end < 0:
                buf = buf + fh.read(1024)

        assert len(header) > 0, "AEDat missing header in file %r" % self.file_path
        header0 = header[0].decode("ascii").strip()
        assert header0.startswith("!AER-DAT")

        version = header0[8:]
        assert version == "2.0", "Only AEDat format version 2.0 is currently supported"

        return buf

    def read_events(self, rel_time=None, dvs_events=None):
        rel_time = self.rel_time if rel_time is None else rel_time

        with open(self.file_path, "rb") as fh:
            buf = self._read_header(fh)
            buf = buf + fh.read(1024)

            raw_events = []
            while len(buf) >= 8:
                b, buf = buf[:8], buf[8:]
                e = self.AEDatEvent.from_buffer_copy(b)
                raw_events.append(e)

                if len(buf) < 8:
                    buf = buf + fh.read(1024)

        if len(buf) > 0:
            warnings.warn("Mangled event at end of %r" % (self.file_path,))

        etuple = lambda e: (e.y, e.x, e.polarity, e.trigger, e.t)
        dvs_events = DVSEvents() if dvs_events is None else dvs_events
        dvs_events.init_events(event_data=[etuple(e) for e in raw_events])

        # should be sorted, but make sure
        dvs_events.events.sort(order="t", kind="stable")

        if rel_time and len(dvs_events.events) > 0:
            dvs_events.events[:]["t"] -= dvs_events.events[0]["t"]

        return dvs_events

    def write_events(self, rel_time=None):
        raise NotImplementedError("Writing AEDat files not yet supported")


class EventsFileIO(DVSFileIO):
    def __init__(self, file_path):
        super().__init__(file_path, rel_time=False)

    def read_events(self, rel_time=None, dvs_events=None):
        if rel_time is None:
            rel_time = self.rel_time

        packet_size = 8  # number of words per packet

        # TODO: could read file in chunks to reduce overall memory
        with open(self.file_path, "rb") as f:
            data = f.read()
        data = np.fromstring(data, np.uint8)

        assert len(data) % packet_size == 0
        n_events = len(data) // packet_size

        dvs_events = DVSEvents() if dvs_events is None else dvs_events
        dvs_events.init_events(n_events=n_events)

        # find x and y values for events
        dvs_events.events[:]["y"] = (
            (data[1::packet_size].astype("uint16") << 8) + data[::packet_size]
        ) >> 2
        dvs_events.events[:]["x"] = (
            (data[3::packet_size].astype("uint16") << 8) + data[2::packet_size]
        ) >> 1

        # get the polarity (1 for on events, 0 for off events)
        dvs_events.events[:]["p"] = (data[::packet_size] & 0x02) == 0x02
        dvs_events.events[:]["v"] = (data[::packet_size] & 0x01) == 0x01

        # find the time stamp for each event
        t = data[7::packet_size].astype(np.uint32)
        t = (t << 8) + data[6::packet_size]
        t = (t << 8) + data[5::packet_size]
        t = (t << 8) + data[4::packet_size]
        dvs_events.events[:]["t"] = t

        # return events sorted by time
        dvs_events.events.sort(order="t", kind="stable")

        if rel_time and len(dvs_events.events) > 0:
            dvs_events.events[:]["t"] -= dvs_events.events[0]["t"]

        return dvs_events

    def write_events(self, dvs_events):
        events = dvs_events.events

        # reformat events (currently ignores "v")
        event_data = np.zeros(
            len(events), dtype=[("y", "<u2"), ("x", "<u2"), ("t", "<u4")]
        )
        event_data[:]["t"] = events[:]["t"]
        event_data[:]["x"] = events[:]["x"] << 1
        event_data[:]["y"] = (events[:]["y"] << 2) + (events[:]["p"] << 1)

        with open(self.file_path, "wb") as fh:
            fh.write(event_data.tobytes())
