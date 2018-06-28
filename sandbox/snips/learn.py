import os
import matplotlib as mpl
haveDisplay = "DISPLAY" in os.environ
if not haveDisplay:
    print('no display')
    mpl.use('Agg')

import nengo
import nengo_loihi
import numpy as np
import matplotlib.pyplot as plt

import nxsdk
Simulator = nengo_loihi.Simulator


with nengo.Network(seed=1) as model:
    stim = nengo.Node(lambda t: np.sin(2*np.pi*t))

    a = nengo.Ensemble(100, 1, label='b',
                       max_rates=nengo.dists.Uniform(100, 120),
                       intercepts=nengo.dists.Uniform(-0.5, 0.5)
			)
    nengo.Connection(stim, a, synapse=None)

    def output(t, x):
        # print(x)
        return x
        
    out = nengo.Node(output, size_in=1, size_out=1)
    c = nengo.Connection(a, out,
                         learning_rule_type=nengo.PES(),
                         function=lambda x: 0,
                         synapse=0.01)

    error = nengo.Node(None, size_in=1)
   
    nengo.Connection(out, error, transform=-1)
    nengo.Connection(stim, error, transform=1)

    nengo.Connection(error, c.learning_rule, transform=1.0)

    p = nengo.Probe(a)
    p2 = nengo.Probe(out)
    p3 = nengo.Probe(error)

with Simulator(model, precompute=False) as sim:
    sim.run(5)

output_filter = nengo.synapses.Lowpass(0.01)

plt.figure()
plt.plot(sim.trange(), output_filter.filt(sim.data[p]))
plt.plot(sim.trange(), output_filter.filt(sim.data[p2]))
plt.plot(sim.trange(), output_filter.filt(sim.data[p3]))
print('saving...')
plt.savefig('pes_learning_snips.png')
