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

    a = nengo.Ensemble(10, 1, label='b', seed=1,
                       max_rates=nengo.dists.Uniform(100, 120),
                       intercepts=nengo.dists.Uniform(-0.5, 0.5)
			)
    nengo.Connection(stim, a, synapse=None)

    out = nengo.Node(None, size_in=a.n_neurons)
    nengo.Connection(a.neurons, out, synapse=None)

    #p = nengo.Probe(a)
    p2 = nengo.Probe(out)

with Simulator(model, precompute=True) as sim:
    sim.run(0.03)

print(sim.data[p2])

