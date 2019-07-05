import collections
import ctypes
import itertools
import os

import numpy as np


class AEDatEvent(ctypes.BigEndianStructure):
    _fields_ = [
        ("type", ctypes.c_uint64, 1),
        ("y", ctypes.c_uint64, 9),
        ("x", ctypes.c_uint64, 10),
        ("polarity", ctypes.c_uint64, 1),
        ("trigger", ctypes.c_uint64, 1),
        ("adc_sample", ctypes.c_uint64, 10),
        ("t", ctypes.c_uint64, 32),
    ]

    def __str__(self):
        return "Event(%s, y=%3d, x=%3d, p=%d, t=%10d)" % (
            self.type, self.y, self.x, self.polarity, self.t)


class Reader:
    def __init__(self, filepath, rel_time):
        self.filepath = filepath
        self.rel_time = rel_time

    def read_events(self, rel_time=None):
        raise NotImplementedError()


class AEDatReader(Reader):
    def __init__(self, filepath):
        super().__init__(filepath, rel_time=True)

        self.version = None  # set in parse_header()
        self.parse_header()

    def read_header(self):
        with open(self.filepath, 'r') as fh:
            lines = []
            while True:
                try:
                    line = fh.readline()
                    if line.startswith('#'):
                        lines.append(line[1:].strip())
                    else:
                        break
                except UnicodeDecodeError:
                    break  # This means we're out of the ASCII header

        return lines

    def parse_header(self):
        header = self.read_header()
        assert header[0].startswith('!AER-DAT')
        version = header[0][8:]
        assert version == '2.0'
        self.version = version

    def read_events(self, rel_time=None):
        if rel_time is None:
            rel_time = self.rel_time

        with open(self.filepath, 'rb') as fh:
            header = True
            buf = fh.read(1024)
            while header:
                if buf[0] == ord('#'):
                    end = buf.find(b'\n')
                    if end > 0:
                        buf = buf[end+1:]
                    if len(buf) == 0 or end < 0:
                        buf = buf + fh.read(1024)
                else:
                    header = False

            raw_events = []
            while len(buf) > 8:
                b, buf = buf[:8], buf[8:]
                e = AEDatEvent.from_buffer_copy(b)
                raw_events.append(e)

                if len(buf) < 8:
                    buf = buf + fh.read(1024)

        raw_events.sort(key=lambda e: e.t)  # should be sorted, but make sure

        etuple = lambda e: (e.y, e.x, e.polarity, e.trigger, e.t)
        events = np.array([etuple(e) for e in raw_events], dtype=np.dtype([
            ('y', 'u2'), ('x', 'u2'), ('p', 'u1'), ('v', 'u1'), ('t', 'u4')]))

        if rel_time and len(events) > 0:
            t0 = events[0]['t']
            events[:]['t'] -= t0

        return events

    def get_event_dict(self):
        events = self.read_events(rel_time=True)
        return collections.OrderedDict(
            itertools.groupby(events, key=lambda e: e.t))


class EventsReader(Reader):
    def __init__(self, filepath):
        super().__init__(filepath, rel_time=False)

    def read_events(self, rel_time=None):
        if rel_time is None:
            rel_time = self.rel_time

        packet_size = 8  # number of words per packet

        # TODO: could read file in chunks to reduce overall memory
        with open(self.filepath, 'rb') as f:
            data = f.read()
        data = np.fromstring(data, np.uint8)

        assert len(data) % packet_size == 0
        n = len(data) // packet_size

        events = np.zeros(n, dtype=np.dtype([
            ('y', 'u2'), ('x', 'u2'), ('p', 'u1'), ('v', 'u1'), ('t', 'u4')]))

        # find x and y values for events
        events[:]['y'] = ((data[1::packet_size].astype('uint16') << 8)
                          + data[::packet_size]) >> 2
        events[:]['x'] = ((data[3::packet_size].astype('uint16') << 8)
                          + data[2::packet_size]) >> 1

        # get the polarity (1 for on events, 0 for off events)
        events[:]['p'] = (data[::packet_size] & 0x02) == 0x02
        events[:]['v'] = (data[::packet_size] & 0x01) == 0x01

        # find the time stamp for each event
        t = data[7::packet_size].astype(np.uint32)
        t = (t << 8) + data[6::packet_size]
        t = (t << 8) + data[5::packet_size]
        t = (t << 8) + data[4::packet_size]
        if rel_time:
            t -= t[0]

        events[:]['t'] = t

        # t = t.astype(float) / 1000000   # convert microseconds to seconds
        return events


def get_dvs_reader(filename, format=None):
    assert os.path.exists(filename)
    if format is None:
        _, ext = os.path.splitext(filename)
        if ext == '.events':
            format = 'events'
        elif ext == '.aedat':
            format = 'aedat'
        else:
            raise ValueError("Unrecognized extension %r" % ext)
    format = format.lower()

    if format == 'aedat':
        return AEDatReader(filename)
    if format == 'events':
        return EventsReader(filename)
    else:
        raise ValueError("Unrecognized format %r" % format)


def play_dvs_file(  # noqa: C901, pragma: no cover
        filename, format=None, scale=2, fps=20):
    import time
    import sdl2.ext

    reader = get_dvs_reader(filename, format=format)
    events = reader.read_events()
    t_end = events[-1]['t'] * 1e-6

    width = 240
    height = 180

    backgroundColor = sdl2.ext.RGBA(0x808080FF)
    offColor = sdl2.ext.RGBA(0x000000FF)
    onColor = sdl2.ext.RGBA(0xFFFFFFFF)

    sdl2.ext.init()
    window = sdl2.ext.Window("DVS", size=(width * scale, height * scale))
    window.show()
    renderer = sdl2.ext.Renderer(window,
                                 flags=sdl2.render.SDL_RENDERER_SOFTWARE)
    renderer.logical_size = (width, height)
    renderer.clear(color=backgroundColor)
    renderer.present()

    spf = 1. / fps

    running = True
    playing = True
    t = 0
    while running:
        sdl_events = sdl2.ext.get_events()
        for event in sdl_events:
            if event.type == sdl2.SDL_QUIT:
                running = False
                break
            elif event.type == 768 and event.key.keysym.sym == 32:  # space
                # space bar down: toggle whether we're playing
                playing = not playing
            elif event.type == 768 and event.key.keysym.sym == 114:  # 'r'
                # 'r' key down: restart
                t = 0

        if running and playing:
            t1 = t + spf

            real_time = time.time()

            # --- plot
            pol0 = []
            pol1 = []

            t0_us = t * 1e6
            t1_us = t1 * 1e6
            m = (events[:]['t'] >= t0_us) & (events[:]['t'] < t1_us)
            events_m = events[m]

            for x, y, p in zip(events_m['x'], events_m['y'], events_m['p']):
                if p == 0:
                    pol0.extend((x, y))
                else:
                    pol1.extend((x, y))

            renderer.clear(color=backgroundColor)
            renderer.draw_point(pol0, color=offColor)
            renderer.draw_point(pol1, color=onColor)
            renderer.present()

            real_time = time.time() - real_time

            if real_time < spf:
                time.sleep(spf - real_time)

            t = t1
            if t > t_end:
                t = 0
                time.sleep(spf)
                renderer.clear(color=offColor)
                renderer.present()
                time.sleep(0.5)

    sdl2.ext.quit()


def save_dvs_board(statepath):
    import nxsdk.api.n2a as nx
    from scipy.sparse import identity
    from nxsdk.compiler.nxsdkcompiler.n2_compiler import N2Compiler
    from nxsdk_modules.dvs.src.dvs import DVS

    net = nx.NxNet()

    dvs = DVS(net=net, dimX=240, dimY=180, dimP=2)

    cp = nx.CompartmentPrototype(
        vThMant=100,
        enableHomeostasis=1,
        compartmentCurrentDecay=4095,
        compartmentVoltageDecay=4095,
        activityTimeConstant=0,
        activityImpulse=1,
        minActivity=20,
        maxActivity=80,
        homeostasisGain=0,
        tEpoch=1,
    )

    cg1 = net.createCompartmentGroup(size=dvs.numPixels, prototype=cp)

    connproto = nx.ConnectionPrototype(
        weight=255,
        signMode=nx.SYNAPSE_SIGN_MODE.EXCITATORY)

    cMask = identity(dvs.numPixels)
    dvs.outputs.rawDVS.connect(cg1, prototype=connproto, connectionMask=cMask)

    compiler = N2Compiler()
    board = compiler.compile(net)

    board.start()

    board.dumpNeuroCores(str(statepath))
    print("Saved to %s" % str(statepath))
    board.disconnect()
