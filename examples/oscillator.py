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
    Simulator = nengo_loihi.Simulator
    print("Running on Loihi")
except ImportError:
    Simulator = nengo_loihi.NumpySimulator
    print("Running in simulation")

tau = 0.1
alpha = 1.0


def f(x):
    x0, x1 = x
    r = np.sqrt(x0**2 + x1**2)
    a = np.arctan2(x1, x0)
    dr = -(r - 1)
    da = alpha
    r = r + tau*dr
    a = a + tau*da
    return [r*np.cos(a), r*np.sin(a)]


n = 200

if 1:
    # for some reason this works:
    rng = np.random.RandomState(3)
    max_rates = nengo.dists.Uniform(100, 120).sample(n, rng=rng)
    intercepts = nengo.dists.Uniform(-0.5, 0.5).sample(n, rng=rng)
    intercepts[0] = -1
else:
    # but this doesn't:
    max_rates = nengo.dists.Uniform(100, 120)  # can't have this too high
    intercepts = nengo.dists.Uniform(-0.8, 0.8)


with nengo.Network(seed=1) as model:
    a = nengo.Ensemble(n, 2, label='a',
                       max_rates=max_rates, intercepts=intercepts, seed=4)
    ap = nengo.Probe(a, synapse=0.01)
    # anp = nengo.Probe(a.neurons)
    aup = nengo.Probe(a.neurons, 'input')
    avp = nengo.Probe(a.neurons, 'voltage')

    # nengo.Connection(a, a, function=f, synapse=tau, seed=3)
    c = nengo.Connection(a, a, function=f, synapse=tau, seed=3,
                         solver=nengo.solvers.LstsqL2(weights=True))

    b = nengo.Ensemble(100, 2, label='b')
    ab = nengo.Connection(a, b, synapse=None)


with Simulator(model) as sim:
    sim.run(10.)

synapse = nengo.synapses.Alpha(0.01)
x = synapse.filtfilt(sim.data[ap])

nshow = 8

u = sim.data[aup][:25, :nshow]
u = np.round(u*1000) if str(u.dtype).startswith('float') else u
print(np.column_stack((np.arange(u.shape[0]), u)))

v = sim.data[avp][:25, :nshow]
v = np.round(v*1000) if str(v.dtype).startswith('float') else v
print(np.column_stack((np.arange(v.shape[0]), v)))

print(x[-4000::200])


plt.subplot(311)
plt.plot(sim.trange(), sim.data[ap])

plt.subplot(312)
plt.plot(sim.trange(), x)

# plt.subplot(313)
# synapse = nengo.synapses.Alpha(0.01)
# spikes = sim.data[anp].astype(np.float32)
# decoders = sim.data[ab].weights
# plt.plot(sim.trange(), np.dot(synapse.filtfilt(spikes), decoders.T))

plt.savefig('oscillator.png')
plt.show()
