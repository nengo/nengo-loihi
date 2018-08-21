import matplotlib.pyplot as plt
import nengo
import numpy as np

import nengo_loihi

seed = 1
weights = True
bnp = None


def a_fn(x):
    return x + 0.5


nengo_loihi.set_defaults()
with nengo.Network(seed=seed) as model:
    a = nengo.Ensemble(100, 1, label='a')
    ap = nengo.Probe(a)
    anp = nengo.Probe(a.neurons)
    avp = nengo.Probe(a.neurons[:5], 'voltage')

    b = nengo.Ensemble(101, 1, label='b')
    ab_conn = nengo.Connection(a, b,
                               function=a_fn,
                               solver=nengo.solvers.LstsqL2(weights=weights))
    bp = nengo.Probe(b)
    bnp = nengo.Probe(b.neurons)
    bup = nengo.Probe(b.neurons[:5], 'input')
    bvp = nengo.Probe(b.neurons[:5], 'voltage')

    c = nengo.Ensemble(1, 1, label='c')
    bc_conn = nengo.Connection(b, c)

with nengo_loihi.Simulator(model) as sim:
    sim.run(1.0)

if __name__ == "__main__":
    print(sim.data[avp][-10:])
    print(sim.data[bup][-10:])
    print(sim.data[bvp][-10:])

    print(sim.data[anp].sum(axis=0))

    if bnp is not None:
        bcount = sim.data[bnp].sum(axis=0)
        b_decoders = sim.data[bc_conn].weights
        print(bcount)
        print("Spike decoded value: %s" % (
            np.dot(b_decoders, bcount) * sim.dt,))

    plt.figure()
    output_filter = nengo.synapses.Alpha(0.02)
    print(output_filter.filtfilt(sim.data[bp])[::100])
    plt.plot(sim.trange(), output_filter.filtfilt(sim.data[ap]))
    plt.plot(sim.trange(), output_filter.filtfilt(sim.data[bp]))

    plt.savefig('ens_ens.png')
    plt.show()
