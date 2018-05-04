import nengo
import nengo_loihi

import matplotlib.pyplot as plt


with nengo.Network(seed=1) as model:
    # u = nengo.Node(output=-1)
    # u = nengo.Node(output=0)
    # u = nengo.Node(output=0.03)
    u = nengo.Node(output=0.29)
    # u = nengo.Node(output=0.51)
    # u = nengo.Node(output=0.7)
    # u = nengo.Node(output=0.75)
    # u = nengo.Node(output=1.01)
    # u = nengo.Node(output=1.1)
    # u = nengo.Node(output=0.99)
    up = nengo.Probe(u)

    # a = nengo.Ensemble(10, 1, label='a',
    a = nengo.Ensemble(100, 1, label='a',
                       max_rates=nengo.dists.Uniform(100, 120),
                       intercepts=nengo.dists.Uniform(-0.5, 0.5))
    # a = nengo.Ensemble(100, 1, label='a')
    ac = nengo.Connection(u, a, synapse=None)
    ap = nengo.Probe(a)

    b = nengo.Ensemble(101, 1, label='b',
                       max_rates=nengo.dists.Uniform(100, 120),
                       intercepts=nengo.dists.Uniform(-0.5, 0.5))
    nengo.Connection(a, b)
    # nengo.Connection(a, b, solver=nengo.solvers.LstsqL2(weights=True))
    bp = nengo.Probe(b)


# with nengo.Simulator(model) as sim:
with nengo_loihi.Simulator(model, target='sim') as sim:

    # sim.run(0.1)

    # for i in range(10):
    #     # sim.step()
    #     # print(sim.model.objs[ac]['

    #     if isinstance(sim, nengo.Simulator):
    #         print(sim.signals[sim.model.sig[a.neurons]['in']])
    #         print(sim.signals[sim.model.sig[a.neurons]['voltage']])
    #     else:
    #         print("U: %s" % sim.simulator.U[sim.simulator.group_slices[
    #             sim.model.objs[a]['out']]])
    #         print("V: %s" % sim.simulator.V[sim.simulator.group_slices[
    #             sim.model.objs[a]['out']]])

    #     sim.step()

    # sim.run(0.01)
    # sim.run(0.05)
    sim.run(0.5)

    # sim.run(5.0)


# plt.plot(sim.trange(), sim.data[ap])
# plt.plot(sim.trange(), sim.data[bp])

# output_filter = nengo.synapses.Alpha(0.005)
output_filter = nengo.synapses.Alpha(0.02)
plt.plot(sim.trange(), output_filter.filtfilt(sim.data[up]))
plt.plot(sim.trange(), output_filter.filtfilt(sim.data[ap]))
plt.plot(sim.trange(), output_filter.filtfilt(sim.data[bp]))

plt.show()
