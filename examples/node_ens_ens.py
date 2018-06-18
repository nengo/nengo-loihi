import os
import matplotlib as mpl
haveDisplay = "DISPLAY" in os.environ
if not haveDisplay:
    mpl.use('Agg')

import nengo
import nengo_loihi
import numpy as np
import matplotlib.pyplot as plt

try:
    import nxsdk
    Simulator = nengo_loihi.Simulator
    print("Running on Loihi")
except ImportError:
    Simulator = nengo_loihi.NumpySimulator
    print("Running in simulation")

tend = 2.0

# a_fn = lambda x: x
a_fn = lambda x: x**2
# a_fn = lambda x: np.abs(x)
solver = nengo.solvers.LstsqL2(weights=False)
# solver = nengo.solvers.LstsqL2(weights=True)

bnp = None
with nengo.Network(seed=1) as model:
    # u = nengo.Node(output=0.0, label='u')
    # u = nengo.Node(output=0.5, label='u')
    # u = nengo.Node(output=1.0, label='u')
    u = nengo.Node(output=nengo.processes.WhiteSignal(tend, high=5, seed=2))
    up = nengo.Probe(u, synapse=None)

    a = nengo.Ensemble(100, d, label='a',
                       max_rates=nengo.dists.Uniform(100, 120),
                       intercepts=nengo.dists.Uniform(-0.5, 0.5))
    nengo.Connection(u, a, synapse=None)
    ap = nengo.Probe(a)
    anp = nengo.Probe(a.neurons)
    avp = nengo.Probe(a.neurons[:5], 'voltage')

    b = nengo.Ensemble(101, 1, label='b',
                       max_rates=nengo.dists.Uniform(100, 120),
                       intercepts=nengo.dists.Uniform(-0.5, 0.5))
    ab_conn = nengo.Connection(a, b, function=a_fn, solver=solver)
    bp = nengo.Probe(b)
    bnp = nengo.Probe(b.neurons)
    bup = nengo.Probe(b.neurons[:5], 'input')
    bvp = nengo.Probe(b.neurons[:5], 'voltage')

    c = nengo.Ensemble(1, 1, label='c')
    bc_conn = nengo.Connection(b, c)

with Simulator(model, max_time=tend) as sim:
    sim.run(tend)

print(sim.data[avp][-10:])
print(sim.data[bup][-10:])
print(sim.data[bvp][-10:])

acount = sim.data[anp].sum(axis=0)
print(acount)

if bnp is not None:
    bcount = sim.data[bnp].sum(axis=0)
    b_decoders = sim.data[bc_conn].weights
    print(bcount)
    print("Spike decoded value: %s" % (np.dot(b_decoders, bcount) * sim.dt,))

plt.figure()
output_filter = nengo.synapses.Alpha(0.02)
print(output_filter.filtfilt(sim.data[bp])[::100])
plt.plot(sim.trange(), output_filter.filtfilt(sim.data[up]))
plt.plot(sim.trange(), output_filter.filtfilt(sim.data[ap]))
plt.plot(sim.trange(), output_filter.filtfilt(sim.data[bp]))

plt.savefig('node_ens_ens.png')
plt.show()
