import warnings

import matplotlib.pyplot as plt
import nengo
import nengo_loihi
import numpy as np

warnings.filterwarnings("ignore", category=FutureWarning)

n_dim = 5


def input_f(t):
    phases = (1 / n_dim) * np.arange(n_dim)
    return np.sin(6 * (t + phases))


with nengo.Network(seed=0) as net:
    u = nengo.Node(input_f, label="u")
    # a = nengo.Ensemble(1000, n_dim, radius=1.3, label="a")
    a = nengo.Ensemble(2000, n_dim, radius=1.3, label="a")
    b = nengo.Ensemble(1000, n_dim, radius=1.3, label="b")

    nengo.Connection(u, a, label="u-a")
    nengo.Connection(a, b, label="a-b")

    up = nengo.Probe(u)
    ap = nengo.Probe(a, synapse=nengo.Alpha(0.01))
    bp = nengo.Probe(b, synapse=nengo.Alpha(0.01))

    anp = nengo.Probe(a.neurons)


def ind_string(inds):
    d = np.diff(inds)
    if np.all(d == d[0]):
        return "slice(%d, %d, %d)" % (inds[0], inds[-1], d[0])
    else:
        return str(inds)


# with nengo_loihi.Simulator(net) as sim:
with nengo_loihi.Simulator(net, target="loihi", dismantle=True) as sim:
    for input in sim.model.inputs:
        print("Input %s: %d" % (input.label, input.n_neurons))
        for axon in input.axons:
            print(
                "  Axon %s: target %s: %d"
                % (axon.label, axon.target.label, axon.n_axons)
            )
    for block in sim.model.blocks:
        print("Block %s: %d" % (block.label, block.compartment.n_compartments))
        for synapse in block.synapses:
            print("  Synapse %s: %d" % (synapse.label, synapse.n_axons))
        for axon in block.axons:
            print(
                "  Axon %s: target %s: %d"
                % (axon.label, axon.target.label, axon.n_axons)
            )
            print("    inds: %s" % ind_string(axon.compartment_map))

    import time
    tt = time.time()

    sim.run(1.0)

    total_runtime = time.time() - tt
    print("total runtime: %0.3f" % total_runtime)

    # hardware = sim.sims["loihi"]
    # print("weighted probe time: %0.3f" % (hardware.timer0))
    # print("weighted probe percent: %0.1f%%" % (100 * hardware.timer0 / total_runtime))

spikes = sim.data[anp]
spike_counts = (spikes > 0).sum(axis=0)
# print(spike_counts[0::10])
# print(spike_counts[1::10])

# voltages = sim.data[avp]
# print(voltages.std(axis=0))

plt.figure()
ax = plt.subplot(211)
ax.plot(sim.trange(), sim.data[up], "--")
ax.set_prop_cycle(None)
ax.plot(sim.trange(), sim.data[ap])

ax = plt.subplot(212)
# ax = plt.subplot(111)
ax.plot(sim.trange(), sim.data[up], "--")
ax.set_prop_cycle(None)
ax.plot(sim.trange(), sim.data[bp])

plt.show()
