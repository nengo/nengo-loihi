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
    target = 'loihi'
    print("Running on Loihi")
except ImportError:
    target = 'sim'
    print("Running in simulation")


a_fn = lambda x: x + 0.5


with nengo.Network(seed=1) as model:
    a = nengo.Ensemble(100, 1, label='b',
                       max_rates=nengo.dists.Uniform(100, 120),
                       intercepts=nengo.dists.Uniform(-0.5, 0.5))
    ap = nengo.Probe(a)
    # anp = nengo.Probe(a.neurons)
    avp = nengo.Probe(a.neurons[:5], 'voltage')

    b = nengo.Ensemble(101, 1, label='b',
                       max_rates=nengo.dists.Uniform(100, 120),
                       intercepts=nengo.dists.Uniform(-0.5, 0.5))
    # ab_conn = nengo.Connection(a, b, function=a_fn)
    ab_conn = nengo.Connection(a, b, function=a_fn, solver=nengo.solvers.LstsqL2(weights=True))
    bp = nengo.Probe(b)
    bnp = nengo.Probe(b.neurons)
    bup = nengo.Probe(b.neurons[:5], 'input')
    bvp = nengo.Probe(b.neurons[:5], 'voltage')


# with nengo.Simulator(model) as sim:
# with nengo_loihi.Simulator(model, target='sim') as sim:
# with nengo_loihi.Simulator(model, target='loihi') as sim:
with nengo_loihi.Simulator(model, target=target) as sim:
    # sim.run(0.1)
    # sim.run(0.5)
    # sim.run(1.5)
    sim.run(1.0)

print(sim.data[avp][-10:])
print(sim.data[bup][-10:])
print(sim.data[bvp][-10:])

# ac = sim.data[anp].sum(axis=0)
# bc = sim.data[bnp].sum(axis=0)
# a_decoders = sim.data[ab_conn].weights
# print(ac)
# print(np.dot(a_decoders, ac) * sim.dt)

plt.figure()
output_filter = nengo.synapses.Alpha(0.02)
print(output_filter.filtfilt(sim.data[bp])[::100])
# plt.plot(sim.trange(), output_filter.filt(sim.data[ap]))
# plt.plot(sim.trange(), output_filter.filt(sim.data[bp]))
plt.plot(sim.trange(), output_filter.filtfilt(sim.data[ap]))
plt.plot(sim.trange(), output_filter.filtfilt(sim.data[bp]))

plt.figure()
plt.plot(sim.trange(), sim.data[avp][:, :10])

plt.savefig('ens_ens.png')
plt.show()
