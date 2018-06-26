import matplotlib as mpl
mpl.use('Agg')

import numpy as np
import matplotlib.pyplot as plt

import nengo
import nengo_loihi

import nxsdk
Simulator = nengo_loihi.Simulator
print("Running on Loihi")

with nengo.Network(seed=None) as model:
    u = nengo.Node(lambda t: np.sin(2*np.pi*t))
    up = nengo.Probe(u, synapse=None)

    a = nengo.Ensemble(100, 2,
                       max_rates=nengo.dists.Uniform(100, 120),
                       intercepts=nengo.dists.Uniform(-.95,.95))
    nengo.Connection(u, a)

    ap = nengo.Probe(a, synapse = 0.02)

with Simulator(model, precompute=True) as sim:
    sim.run(1.)

plt.figure()
plt.plot(sim.trange(), sim.data[ap])
plt.plot(sim.trange(), sim.data[up])

plt.savefig('v_node_ens.png')