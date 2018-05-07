import nengo
import nengo_loihi

import matplotlib.pyplot as plt

a_fn = lambda x: x + 0.5


with nengo.Network(seed=1) as model:
    a = nengo.Ensemble(100, 1, label='b',
                       max_rates=nengo.dists.Uniform(100, 120),
                       intercepts=nengo.dists.Uniform(-0.5, 0.5))
    ap = nengo.Probe(a)

    b = nengo.Ensemble(101, 1, label='b',
                       max_rates=nengo.dists.Uniform(100, 120),
                       intercepts=nengo.dists.Uniform(-0.5, 0.5))
    # nengo.Connection(a, b, function=a_fn)
    nengo.Connection(a, b, function=a_fn, solver=nengo.solvers.LstsqL2(weights=True))
    bp = nengo.Probe(b)


# with nengo.Simulator(model) as sim:
with nengo_loihi.Simulator(model, target='sim') as sim:
    sim.run(0.5)

output_filter = nengo.synapses.Alpha(0.02)
plt.plot(sim.trange(), output_filter.filtfilt(sim.data[ap]))
plt.plot(sim.trange(), output_filter.filtfilt(sim.data[bp]))

plt.show()
