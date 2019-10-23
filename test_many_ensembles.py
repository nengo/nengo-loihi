import matplotlib.pyplot as plt
import nengo
import nengo_loihi
import numpy as np


def input_f(t, phase_offset):
    return np.sin(6 * t + phase_offset + np.array([0, np.pi / 2]))


n_ensembles = 15
n_neurons = 600
phase_step = 1  # in radians
dimensions = len(input_f(0, 0))


with nengo.Network(seed=34) as net:
    nengo_loihi.set_defaults()

    inputs = []
    ensembles = []
    probes = []
    for i in range(n_ensembles):

        def input_f_i(t, i=i):
            return input_f(t, phase_offset=i * phase_step)

        input = nengo.Node(input_f_i)
        ensemble = nengo.Ensemble(n_neurons, dimensions=dimensions)
        probe = nengo.Probe(ensemble, synapse=nengo.synapses.Alpha(0.01))
        nengo.Connection(input, ensemble, synapse=None)

        inputs.append(input)
        ensembles.append(ensemble)
        probes.append(probe)


with nengo_loihi.Simulator(net, precompute=False) as sim:
    sim.run(1.0)
    #sim.run(10.0)
    print("Time/step: %0.2f ms" % (sim.timers["snips"] / sim.n_steps * 1000))

plt.figure()

rows = 2
cols = 2

for i in range(rows * cols):
    plt.subplot(rows, cols, i + 1)
    plt.plot(sim.trange(), sim.data[probes[i]])

plt.savefig("test_many_ensembles.png")
plt.show()
