import os
import matplotlib as mpl
haveDisplay = "DISPLAY" in os.environ
if not haveDisplay:
    mpl.use('Agg')

import nengo
import nengo_loihi
import numpy as np
import matplotlib.pyplot as plt

import nxsdk
Simulator = nengo_loihi.Simulator


with nengo.Network(seed=1) as model:
    stim = nengo.Node(lambda t: 0.5)#np.sin(2*np.pi*t))

    a = nengo.Ensemble(100, 1, label='b',
                       max_rates=nengo.dists.Uniform(100, 120),
                       intercepts=nengo.dists.Uniform(-0.5, 0.5)
			)
    nengo.Connection(stim, a, synapse=None)

    out = nengo.Node(None, size_in=1)
    nengo.Connection(a, out)

    p = nengo.Probe(a)
    p2 = nengo.Probe(out, synapse=0.02)

with Simulator(model) as sim:
    sim.run(0.1)

print(sim.data[p2])
