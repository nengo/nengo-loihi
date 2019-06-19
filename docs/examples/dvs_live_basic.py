import matplotlib.pyplot as plt
import numpy as np
import nengo
import nengo_loihi
from nengo_loihi.inputs import DVSChipNode
import sdl2.ext


backgroundColor = sdl2.ext.RGBA(0x808080FF)
offColor = sdl2.ext.RGBA(0x000000FF)
onColor = sdl2.ext.RGBA(0xFFFFFFFF)


with nengo.Network() as net:
    u = DVSChipNode(pool=(20, 20), channels_last=True)
    h = u.height
    w = u.width
    p = u.polarity

    e = nengo.Ensemble(
        h * w, 1,
        neuron_type=nengo.SpikingRectifiedLinear(),
        gain=nengo.dists.Choice([0.1]),
        bias=nengo.dists.Choice([0]),
    )

    nengo.Connection(u[::2], e.neurons, synapse=0.01)
    probe = nengo.Probe(e.neurons)

    # probe_v = nengo.Probe(e.neurons, 'voltage')


def plot(sim, renderer):
    """Visualize soma activity"""
    data = sim.data[probe][-1]
    print((data > 0).sum())

    # print(sim.data[probe_v].max())

    # x, y lists for polarity 0
    pol0 = list()

    # x, y lists for polarity 1
    pol1 = list()

    # soma activity shows up 2 timesteps after spike injected
    # showing plots for timestep 1 and 2 here
    for ind, val in enumerate(data):
        if val > 0:
            i = ind // width
            j = ind % width
            assert i < height

            # renderer uses bottom left corner indexing
            x = j
            y = height - 1 - i
            pol1.append(x)
            pol1.append(y)

    if renderer is not None:
        renderer.clear(color=backgroundColor)
        renderer.draw_point(pol0, color=offColor)
        renderer.draw_point(pol1, color=onColor)
        renderer.present()


# t_step = 0.01
t_step = 0.1

with nengo_loihi.Simulator(net) as sim:

    width = u.width
    height = u.height
    # width = 24
    # height = 18

    try:
        sdl2.ext.init()
        scale = 5 * 4
        window = sdl2.ext.Window("DVS", size=(width * scale, height * scale))
        window.show()
        renderer = sdl2.ext.Renderer(window)
        renderer.logical_size = (width, height)
        renderer.clear(color=backgroundColor)
        renderer.present()
    except sdl2.ext.common.SDLError as e:
        print("Running without window due to: %s" % str(e))
        renderer = None

    running = True
    while running:
        events = sdl2.ext.get_events() if renderer else []
        for event in events:
            if event.type == sdl2.SDL_QUIT:
                running = False
                break
        if running:
            sim.run(t_step)
            plot(sim, renderer)

    sdl2.ext.quit()
