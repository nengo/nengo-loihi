import nengo
import nengo_loihi
import numpy as np

D = 3

with nengo.Network(seed=1) as model:
    stim = nengo.Node(lambda t: [0.5]*D)

    a = nengo.Ensemble(500, D, label='a',
                       max_rates=nengo.dists.Uniform(100, 120),
                       intercepts=nengo.dists.Uniform(-0.9, 0.9)
			)
    nengo.Connection(stim, a, synapse=None)

    def output(t, x):
        return x

    out = nengo.Node(output, size_in=1, size_out=1)
    c = nengo.Connection(a, out,
                         learning_rule_type=nengo.PES(learning_rate=1e-3),
                         function=lambda x: 0,
                         synapse=0.01)

    error = nengo.Node(None, size_in=1)

    nengo.Connection(out, error, transform=1)
    nengo.Connection(stim[0], error, transform=-1)

    nengo.Connection(error, c.learning_rule, transform=1.0)

    p = nengo.Probe(out, synapse=0.05)

T = 0.01
with nengo_loihi.Simulator(model, precompute=False) as sim:
    sim.run(T)

print(sim.time_per_step)

print(sim.data[p][-10:])
