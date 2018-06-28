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
print("Running on Loihi")


import nengo
import nengo_loihi
import numpy as np

try:  # Run on Loihi if available
    import nxsdk
    Simulator = nengo_loihi.Simulator
    print("Running on Loihi")
except ImportError:
    Simulator = nengo_loihi.NumpySimulator
    print("Running in simulation")

with nengo.Network() as model:
    stim = nengo.Node(lambda t: [np.sin(t*2*np.pi)])

    a = nengo.Ensemble(23, 1,
                       max_rates=nengo.dists.Uniform(100, 120),
                       intercepts=nengo.dists.Uniform(-0.5, 0.5))

    b = nengo.Node(None, size_in=1)

    nengo.Connection(stim, a)
    conn = nengo.Connection(a, b,
                            learning_rule_type=nengo.PES(),
                            function=lambda x: 0)

    error = nengo.Node(None, size_in=1)
    nengo.Connection(b, error)
    nengo.Connection(stim, error, transform=-1)
    nengo.Connection(error, conn.learning_rule)

    ap = nengo.Probe(a)
    bp = nengo.Probe(b)

with Simulator(model) as sim:
    sim.run(.1)

plt.figure()
output_filter = nengo.synapses.Lowpass(0.01)
plt.plot(sim.trange(), output_filter.filt(sim.data[ap]))
plt.plot(sim.trange(), output_filter.filt(sim.data[bp]))

plt.savefig('pes_learning_snips.png')

