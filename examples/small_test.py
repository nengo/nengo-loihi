import os
import matplotlib as mpl
haveDisplay = "DISPLAY" in os.environ
if not haveDisplay:
    mpl.use('Agg')


import numpy as np
import matplotlib.pyplot as plt

import nengo
import nengo_loihi

try:
    import nxsdk
except ImportError:
    print("This script is to compare loihi and simulated loihi, but ",
            "the loihi cannot be found")
    raise

sims = {'simulation': nengo_loihi.NumpySimulator,
        'silicon': nengo_loihi.Simulator}
x={}
u={}
v={}
for name, simulator in sims.items():
    n = 4
    with nengo.Network(seed=1) as model:
        inp = nengo.Node(lambda t: np.sin(2*np.pi*t))

        a = nengo.Ensemble(n, 1,
                           max_rates=nengo.dists.Uniform(100, 120),
                           intercepts=nengo.dists.Uniform(-0.5, 0.5))
        ap = nengo.Probe(a, synapse=0.01)
        aup = nengo.Probe(a.neurons, 'input')
        avp = nengo.Probe(a.neurons, 'voltage')

        nengo.Connection(inp, a)

    with simulator(model, precompute=True) as sim:
        print("Running in {}".format(name))
        sim.run(3.)

    synapse = nengo.synapses.Lowpass(0.01)
    x[name] = synapse.filt(sim.data[ap])

    u[name] = sim.data[aup][:25]
    u[name] = np.round(u[name]*1000) if str(u[name].dtype).startswith('float') else u[name]

    v[name] = sim.data[avp][:25]
    v[name] = np.round(v[name]*1000) if str(v[name].dtype).startswith('float') else v[name]


    plt.figure()
    plt.subplot(311)
    plt.plot(sim.trange(), sim.data[ap])
    plt.subplot(312)
    plt.plot(sim.trange(), x[name])

    if haveDisplay:
        plt.show()
    else:
        plt.savefig('compare_{}.png'.format(name))
    
print('simulation',np.column_stack((np.arange(u['simulation'].shape[0]), u['simulation'])))
print('silicon', np.column_stack((np.arange(u['silicon'].shape[0]), u['silicon'])))
print('simulation',np.column_stack((np.arange(v['simulation'].shape[0]), v['simulation'])))
print('silicon', np.column_stack((np.arange(v['silicon'].shape[0]), v['silicon'])))
print('simulation', x['simulation'][0:30])
print('silicon', x['silicon'][0:30])

