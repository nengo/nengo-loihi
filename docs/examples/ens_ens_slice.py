import matplotlib.pyplot as plt
import nengo

import nengo_loihi

seed = 1
b_vals = [-0.5, 0.75]
weights = True


def a_fn(x):
    return [xx + bb for xx, bb in zip(x, b_vals)]


nengo_loihi.set_defaults()
with nengo.Network(seed=seed) as model:
    a = nengo.Ensemble(100, 2, label='a')
    b = nengo.Ensemble(101, 2, label='b')
    bp = nengo.Probe(b)
    ab_conn = nengo.Connection(a, b, function=a_fn)

    c = nengo.Ensemble(102, 2, label='c')
    cp = nengo.Probe(c)
    solver = nengo.solvers.LstsqL2(weights=weights)
    bc_conn = nengo.Connection(b[1], c[0], solver=solver)
    bc_conn = nengo.Connection(b[0], c[1], solver=solver)

with nengo_loihi.Simulator(model) as sim:
    sim.run(1.0)

if __name__ == "__main__":
    plt.figure()
    output_filter = nengo.synapses.Alpha(0.02)
    plt.plot(sim.trange(), output_filter.filtfilt(sim.data[bp]))
    plt.plot(sim.trange(), output_filter.filtfilt(sim.data[cp]))
    plt.legend(['b%d' % d for d in range(b.dimensions)]
               + ['c%d' % d for d in range(c.dimensions)])

    plt.savefig('ens_ens_slice.png')
    plt.show()
