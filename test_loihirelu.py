import cProfile as profile

import matplotlib.pyplot as plt
import nengo
import nengo_loihi
import numpy as np

with nengo.Network() as net:
    n_neurons = 1000
    # n_neurons = 100000

    currents = np.linspace(0, 1500, n_neurons)

    a = nengo.Ensemble(
        n_neurons,
        1,
        gain=nengo.dists.Choice([1]),
        bias=currents,
        neuron_type=nengo_loihi.neurons.LoihiRectifiedLinear(),
    )
    ap = nengo.Probe(a.neurons)


with nengo.Simulator(net) as sim:
    # profile.runctx("sim.run_steps(1000)", globals(), locals(), sort=2)
    sim.run_steps(1000)


plt.plot(currents, sim.data[ap][0], "k")
plt.plot(currents, sim.data[ap][-1], "b")
plt.show()
