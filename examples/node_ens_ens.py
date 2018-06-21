import matplotlib.pyplot as plt
import nengo
import numpy as np

import nengo_loihi

try:
    import nxsdk  # pylint: disable=unused-import
    Simulator = nengo_loihi.Simulator
    print("Running on Loihi")
except ImportError:
    Simulator = nengo_loihi.NumpySimulator
    print("Running in simulation")

seed = 1
tend = 2.0
weights = False
a_fn = lambda x: x**2
d = 2

bnp = None
with nengo.Network(seed=seed) as model:
    u = nengo.Node(
        output=nengo.processes.WhiteSignal(tend, high=5, seed=seed+1),
        size_out=d)
    up = nengo.Probe(u, synapse=None)

    a = nengo.Ensemble(100, d, label='a',
                       max_rates=nengo.dists.Uniform(100, 120),
                       intercepts=nengo.dists.Uniform(-0.5, 0.5))
    nengo.Connection(u, a, synapse=None)
    ap = nengo.Probe(a)
    anp = nengo.Probe(a.neurons)
    avp = nengo.Probe(a.neurons[:5], 'voltage')

    b = nengo.Ensemble(101, d, label='b',
                       max_rates=nengo.dists.Uniform(100, 120),
                       intercepts=nengo.dists.Uniform(-0.5, 0.5))
    ab_conn = nengo.Connection(a, b,
                               function=a_fn,
                               solver=nengo.solvers.LstsqL2(weights=weights))
    bp = nengo.Probe(b)
    bnp = nengo.Probe(b.neurons)
    bup = nengo.Probe(b.neurons[:5], 'input')
    bvp = nengo.Probe(b.neurons[:5], 'voltage')

    c = nengo.Ensemble(1, d, label='c')
    bc_conn = nengo.Connection(b, c)

with Simulator(model, precompute=True) as sim:
    sim.run(tend)

if __name__ == "__main__":
    print(sim.data[avp][-10:])
    print(sim.data[bup][-10:])
    print(sim.data[bvp][-10:])

    acount = sim.data[anp].sum(axis=0)
    print(acount)

    if bnp is not None:
        bcount = sim.data[bnp].sum(axis=0)
        b_decoders = sim.data[bc_conn].weights
        print(bcount)
        print("Spike decoded value: %s" % (
            np.dot(b_decoders, bcount) * sim.dt,))

    plt.figure()
    output_filter = nengo.synapses.Alpha(0.02)
    t = sim.trange()
    print(output_filter.filtfilt(sim.data[bp])[::100])
    plt.plot(t, output_filter.filtfilt(sim.data[up]), c="b", label="u")
    plt.plot(t, output_filter.filtfilt(sim.data[ap]), c="g", label="a")
    plt.plot(t, output_filter.filtfilt(sim.data[bp]), c="r", label="b")
    plt.legend()

    plt.savefig('node_ens_ens.png')
    plt.show()
