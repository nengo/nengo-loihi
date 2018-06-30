import matplotlib.pyplot as plt
import nengo
import numpy as np

import nengo_loihi

seed = 4
tau = 0.1
alpha = 3.0
weights = False
n = 200


def f(x):
    x0, x1 = x
    r = np.sqrt(x0**2 + x1**2)
    a = np.arctan2(x1, x0)
    dr = -(r - 1)
    da = alpha
    r = r + tau*dr
    a = a + tau*da
    return [r*np.cos(a), r*np.sin(a)]


nengo_loihi.set_defaults()
with nengo.Network(seed=seed) as model:
    a = nengo.Ensemble(n, 2, label='a')
    ap = nengo.Probe(a, synapse=0.01)
    anp = nengo.Probe(a.neurons)
    aup = nengo.Probe(a.neurons[:8], 'input')
    avp = nengo.Probe(a.neurons[:8], 'voltage')

    c = nengo.Connection(a, a,
                         function=f,
                         synapse=tau,
                         solver=nengo.solvers.LstsqL2(weights=weights))

    b = nengo.Ensemble(100, 2, label='b')
    ab = nengo.Connection(a, b, synapse=None)

with nengo_loihi.Simulator(model) as sim:
    sim.run(10.)

if __name__ == "__main__":
    x = nengo.synapses.Alpha(0.01).filtfilt(sim.data[ap])

    plt.subplot(211)
    plt.plot(sim.trange(), sim.data[ap])

    plt.subplot(212)
    plt.plot(sim.trange(), x)
