import matplotlib.pyplot as plt
import nengo
import numpy as np

import nengo_loihi

n_neurons = 250  # number of neurons in on-chip ensemble
input_d = 2  # number of input dimensions
dt = 0.001

encode_neuron_type = nengo_loihi.decode_neurons.OnOffDecodeNeurons(dt=dt, is_input=True)

weights = nengo.dists.UniformHypersphere(surface=True).sample(n_neurons, d=input_d)

with nengo.Network() as net:
    nengo_loihi.set_defaults()
    nengo_loihi.add_params(net)

    # input node
    f = 2 * np.pi
    inp = nengo.Node(lambda t: [np.sin(f * t), np.cos(f * t)])
    assert input_d == inp.size_out

    # off-chip ensemble to turn input into spikes
    encode_ens = encode_neuron_type.get_ensemble(input_d)
    net.config[encode_ens].on_chip = False
    nengo.Connection(inp, encode_ens, synapse=None)

    # on-chip ensemble to connect input to using `weights`
    onchip_ens = nengo.Ensemble(n_neurons, input_d, encoders=weights)
    onchip_weights = dt * encode_neuron_type.get_post_encoders(weights).T
    nengo.Connection(encode_ens.neurons, onchip_ens.neurons, transform=onchip_weights)

    p = nengo.Probe(onchip_ens, synapse=nengo.Alpha(0.01))

with nengo_loihi.Simulator(net) as sim:
    sim.run(2.0)

plt.plot(sim.trange(), sim.data[p])
plt.show()
