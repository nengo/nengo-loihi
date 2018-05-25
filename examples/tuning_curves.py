import os
import matplotlib as mpl
haveDisplay = "DISPLAY" in os.environ
if not haveDisplay:
    mpl.use('Agg')

import matplotlib.pyplot as plt
import numpy as np

import nengo
import nengo_loihi

try:
    import nxsdk
    Simulator = lambda model: nengo_loihi.Simulator(model, target='loihi')
    print("Running on Loihi")
except ImportError:
    Simulator = lambda model: nengo_loihi.Simulator(model, target='sim')
    print("Running in simulation")


tau_ref = 0.002
NeuronType = lambda tau_ref: nengo.LIF(tau_ref=tau_ref)
# NeuronType = lambda tau_ref: nengo.RectifiedLinear()

n = 256
encoders = np.ones((n, 1))
# gain = np.ones(n)
gain = np.zeros(n)
bias = np.linspace(0, 30, n)
# bias = np.linspace(1, 5, n)
# bias = np.linspace(0, 1.01, n)

with nengo.Network() as model:
    a = nengo.Ensemble(n, 1, neuron_type=NeuronType(tau_ref),
                       encoders=encoders, gain=gain, bias=bias)
    ap = nengo.Probe(a.neurons)

t_final = 1.0
with Simulator(model) as sim:
    sim.run(t_final)

scount = sim.data[ap].sum(axis=0)
# print(scount)

ref = NeuronType(tau_ref).rates(0., gain, bias)
refshift = NeuronType(tau_ref + 0.0005).rates(0., gain, bias)
plt.plot(bias, ref, 'k', label='ref')
plt.plot(bias, refshift, 'b', label='ref shift')
plt.plot(bias, scount, 'r', label='loihi')
plt.legend(loc='best')

plt.savefig('tuning_curves.png')
plt.show()
