import os
import matplotlib as mpl
haveDisplay = "DISPLAY" in os.environ
if not haveDisplay:
    mpl.use('Agg')

import matplotlib.pyplot as plt
import numpy as np

import nengo
import nengo_loihi

from loihi_neurons import LoihiLIF

try:
    import nxsdk
    Simulator = lambda model: nengo_loihi.Simulator(model, target='loihi')
    print("Running on Loihi")
except ImportError:
    Simulator = lambda model: nengo_loihi.Simulator(model, target='sim')
    # Simulator = nengo.Simulator
    print("Running in simulation")


dt = 0.001

n = 256
encoders = np.ones((n, 1))
# gain = np.ones(n)
gain = np.zeros(n)
bias = np.linspace(0, 30, n)
# bias = np.linspace(1, 5, n)
# bias = np.linspace(0, 1.01, n)

if 0:
    tau_rc = 0.02
    tau_ref = 0.002
    neuron_type = LoihiLIF(tau_rc=tau_rc, tau_ref=tau_ref)
    # neuron_type = nengo.LIF(tau_rc=tau_rc, tau_ref=tau_ref)
    loihi_neuron_type = LoihiLIF(tau_rc=tau_rc, tau_ref=tau_ref)
    amp = 1.

    bias = np.linspace(0, 30, n)
else:
    tau_rc = np.inf
    tau_ref = 0.000
    neuron_type = nengo.RectifiedLinear()
    neuron_type = LoihiLIF(tau_rc=tau_rc, tau_ref=tau_ref)
    loihi_neuron_type = LoihiLIF(tau_rc=tau_rc, tau_ref=tau_ref)

    bias = np.linspace(0, 1.001, n)
    # bias = np.linspace(0, 1001, n)
    # bias = np.linspace(0, 1001, n)


with nengo.Network() as model:
    a = nengo.Ensemble(n, 1, neuron_type=neuron_type,
                       encoders=encoders, gain=gain, bias=bias)
    ap = nengo.Probe(a.neurons)

t_final = 1.0
with Simulator(model) as sim:
    sim.run(t_final)

# scount = sim.data[ap].sum(axis=0)
# print(scount)
scount = (sim.data[ap] > 0).sum(axis=0)
print(scount)

ref = neuron_type.rates(0., gain, bias)
loihi_ref = loihi_neuron_type.rates(0., gain, bias)
plt.plot(bias, scount, 'r', label='loihi')
plt.plot(bias, ref, 'k--', label='ref')
plt.plot(bias, loihi_ref, 'g:', label='loihi ref')
plt.legend(loc='best')

# plt.savefig('tuning_curves.png')
plt.show()
