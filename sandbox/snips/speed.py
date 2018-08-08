import nengo
import nengo_loihi
import numpy as np
import timeit

with nengo.Network(seed=1) as model:
    stim = nengo.Node(lambda t: 0.5)#np.sin(2*np.pi*t))

    a = nengo.Ensemble(100, 1, label='b',
                       max_rates=nengo.dists.Uniform(100, 120),
                       intercepts=nengo.dists.Uniform(-0.5, 0.5)
			)
    nengo.Connection(stim, a, synapse=None)

    out = nengo.Node(None, size_in=1)
    nengo.Connection(a, out)

    p = nengo.Probe(out, synapse=0.02)

with nengo_loihi.Simulator(model, precompute=False) as sim:

    T = 0.5
    start = timeit.default_timer()
    sim.run(T)
    end = timeit.default_timer()
    print((end-start)/T)
