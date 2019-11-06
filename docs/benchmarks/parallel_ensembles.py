import cProfile
import timeit

import matplotlib.pyplot as plt
import nengo
import nengo_loihi
from nengo_loihi import decode_neurons
import numpy as np

rng = np.random.RandomState(1)
seed = rng.randint(0, 2**31)

LoihiEmulator = lambda net: nengo_loihi.Simulator(net, target="sim")
LoihiSimulator = lambda net: nengo_loihi.Simulator(
    net, target="loihi", hardware_options=dict(snip_max_spikes_per_step=300),
)
NengoSimulator = lambda net: nengo.Simulator(net)

Simulator = LoihiSimulator

funnel_input = True
funnel_output = True
learning = True
profile = False
#profile = True

simtime = 10

#n_input = 6
#n_output = 3
n_input = 10
n_output = 5

#n_neurons = 10
n_neurons = 400
#n_neurons = 1000
#n_ensembles = 1
n_ensembles = 20
#pes_learning_rate = 5e-9
pes_learning_rate = 5e-6
#pes_learning_rate = 5e-4

# synapse time constants
tau_input = 0.01
tau_error = 0.01
tau_output = 0.01

encoders_dist = nengo.dists.UniformHypersphere(surface=True)
encoders = encoders_dist.sample(n_neurons * n_ensembles, n_input, rng=rng)
encoders = encoders.reshape(n_ensembles, n_neurons, n_input)

input_freq = np.pi
input_phases = (1 / n_input) * np.arange(n_input)

# desired transform between input and output
assert n_input % n_output == 0
inout_transform = np.repeat(np.eye(n_output), n_input / n_output, axis=1)
inout_transform /= inout_transform.sum(axis=1, keepdims=True)

net = nengo.Network(seed=seed)
net.config[nengo.Ensemble].neuron_type = nengo.LIF()
net.config[nengo.Ensemble].intercepts = nengo.dists.Uniform(-1, 0.5)
net.config[nengo.Ensemble].max_rates = nengo.dists.Uniform(100, 200)
# net.config[nengo.Connection].synapse = None

with net:
    error_rng = np.random.RandomState(5)

    def ctrl_func(t, u_adapt):
        inputs = np.sin(input_freq*t + 2*np.pi*input_phases)

        target_output = np.dot(inout_transform, inputs)
        errors = u_adapt - target_output

        return inputs.tolist() + errors.tolist()

    ctrl = nengo.Node(ctrl_func, size_in=n_output, size_out=n_input+n_output, label="ctrl")
    ctrl_probe = nengo.Probe(ctrl)

    input = ctrl[:n_input]
    error = ctrl[n_input:]
    output = ctrl

    inp2ens_transform = None
    if funnel_input:
        input_decodeneurons = decode_neurons.Preset10DecodeNeurons()
        onchip_input = input_decodeneurons.get_ensemble(dim=n_input)
        nengo.Connection(input, onchip_input, synapse=None)
        inp2ens_transform = np.hstack(
            [np.eye(n_input), -np.eye(n_input)] * input_decodeneurons.pairs_per_dim
        )
        input = onchip_input

    if funnel_output:
        output_decodeneurons = decode_neurons.Preset10DecodeNeurons()
        onchip_output = output_decodeneurons.get_ensemble(dim=n_output)
        out2ctrl_transform = np.hstack(
            [np.eye(n_output), -np.eye(n_output)] * output_decodeneurons.pairs_per_dim
        ) / 2000.
        nengo.Connection(
            onchip_output.neurons,
            output,
            transform=out2ctrl_transform,
            synapse=tau_output,
        )
        output = onchip_output

    for ii in range(n_ensembles):
        ens = nengo.Ensemble(
            n_neurons=n_neurons,
            dimensions=n_input,
            radius=np.sqrt(n_input),
            encoders=encoders[ii],
            label="ens%02d" % ii,
        )

        if inp2ens_transform is not None:
            inp2ens_transform_ii = np.dot(encoders[ii], inp2ens_transform)
            nengo.Connection(
                input.neurons,
                ens.neurons,
                transform=inp2ens_transform_ii,
                synapse=tau_input
            )
        else:
            nengo.Connection(input, ens, synapse=tau_input)

        conn_kwargs = dict()
        if learning:
            conn_kwargs["transform"] = rng.uniform(-0.01, 0.01, size=(n_output, n_input))
            conn_kwargs["learning_rule_type"] = nengo.PES(
                pes_learning_rate, pre_synapse=tau_error,
            )
        else:
            conn_kwargs["transform"] = inout_transform / n_ensembles

        conn = nengo.Connection(ens, output, **conn_kwargs)

        if learning:
            nengo.Connection(error, conn.learning_rule, synapse=None)


with Simulator(net) as sim:
    sim.run(0.001)  # eliminate any extra startup from timing

    steps = sim.n_steps
    timer = timeit.default_timer()
    if profile:
        cProfile.runctx("sim.run(%s)" % simtime, globals(), locals(), sort="cumtime")
    else:
        sim.run(simtime)
    timer = timeit.default_timer() - timer
    steps = sim.n_steps - steps
    print('Run time/step: %0.2f ms' % (1000 * timer / steps))

    inputs = sim.data[ctrl_probe][:, :n_input]
    targets = np.dot(inputs, inout_transform.T)
    errors = sim.data[ctrl_probe][:, n_input:]
    outputs = errors + targets

    error_steps = min(sim.n_steps, 2000)
    start_error = np.abs(errors[:error_steps]).mean()
    end_error = np.abs(errors[-error_steps:]).mean()
    print("Error: start %0.3f, end %0.3f" % (start_error, end_error))

    plt.plot(sim.trange(), targets, ":")
    plt.plot(sim.trange(), outputs)
    plt.savefig("parallel_ensembles.pdf")
