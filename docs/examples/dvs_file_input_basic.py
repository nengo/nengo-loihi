import matplotlib.pyplot as plt
import numpy as np
import nengo
import nengo_loihi
from nengo_loihi.inputs import DVSFileChipNode


with nengo.Network() as net:
    u = DVSFileChipNode(t_start=1.0, filename='davis240c-5sec-handmove.aedat')
    h = u.height
    w = u.width
    p = u.polarity

    e = nengo.Ensemble(
        9*12, 1,
        neuron_type=nengo.neurons.SpikingRectifiedLinear(),
        max_rates=nengo.dists.Choice([500]),
        intercepts=nengo.dists.Choice([0]),
    )

    slice0 = [240*2*i + 2*j + k
              for i in range(0, 180, 20)
              for j in range(0, 240, 20)
              for k in range(1)]
    nengo.Connection(u[slice0], e.neurons)

    p = nengo.Probe(e.neurons, synapse=0.03)

with nengo_loihi.Simulator(net) as sim:
    sim.run(0.5)


print(sim.data[p].sum(axis=0).reshape(9, 12))
