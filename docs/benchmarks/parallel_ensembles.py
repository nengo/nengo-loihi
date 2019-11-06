import nengo
import nengo_loihi
from nengo_loihi import decode_neurons
import numpy as np
import timeit
import traceback

from utils import AreaIntercepts, Triangular

rng = np.random.RandomState(1)

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
seed = 0

# synapse time constants
tau_input = 0.01  # on input connection
tau_training = 0.01  # on the training signal
#tau_output = 0.2  # on the output from the adaptive ensemble
tau_output = 0.01  # on the output from the adaptive ensemble
# NOTE: the time constant on the neural activity used in the learning
# connection is the default 0.005, and can be set by specifying the
# pre_synapse parameter inside the PES rule instantiation

# set up neuron intercepts
intercepts_bounds = [-0.3, 0.1]
intercepts_mode = 0.1

intercepts_dist = AreaIntercepts(
    dimensions=n_input,
    base=Triangular(intercepts_bounds[0], intercepts_mode, intercepts_bounds[1]),
)
intercepts = intercepts_dist.sample(n=n_neurons * n_ensembles, rng=rng)
intercepts = intercepts.reshape(n_ensembles, n_neurons)

np.random.seed = seed
encoders_dist = nengo.dists.UniformHypersphere(surface=True)
encoders = encoders_dist.sample(n_neurons * n_ensembles, n_input, rng=rng)
encoders = encoders.reshape(n_ensembles, n_neurons, n_input)

nengo_model = nengo.Network(seed=seed)
nengo_model.config[nengo.Ensemble].neuron_type = nengo.LIF()
# nengo_model.config[nengo.Connection].synapse = None

with nengo_model:
    error_rng = np.random.RandomState(5)

    assert n_input % n_output == 0
    inout_transform = np.repeat(np.eye(n_output), n_input / n_output, axis=1)
    inout_transform /= inout_transform.sum(axis=1, keepdims=True)

    #inout_transform = error_rng.uniform(-1, 1, size=(n_output, n_input))
    #inout_transform /= np.sqrt((inout_transform**2).sum(axis=1, keepdims=True))

    def arm_func(t, u_adapt):
        freq = np.pi
        phases = (1 / n_input) * np.arange(n_input)
        inputs = np.sin(freq*t + 2*np.pi*phases)

        target_output = np.dot(inout_transform, inputs)
        errors = u_adapt - target_output

        if int(1000 * t) % 1000 == 0:
            print(("i", inputs))
            print(("u", u_adapt))
            print(("o", target_output))
            print(("e", errors))

        return inputs.tolist() + errors.tolist()

    arm = nengo.Node(arm_func, size_in=n_output, size_out=n_input+n_output, label="arm")
    arm_probe = nengo.Probe(arm)

    input_decodeneurons = decode_neurons.Preset10DecodeNeurons()
    onchip_input = input_decodeneurons.get_ensemble(dim=n_input)
    nengo.Connection(arm[:n_input], onchip_input, synapse=None)
    inp2ens_transform = np.hstack(
        [np.eye(n_input), -np.eye(n_input)] * input_decodeneurons.pairs_per_dim
    )

    output_decodeneurons = decode_neurons.Preset10DecodeNeurons()
    onchip_output = output_decodeneurons.get_ensemble(dim=n_output)
    out2arm_transform = np.hstack(
        [np.eye(n_output), -np.eye(n_output)] * output_decodeneurons.pairs_per_dim
    ) / 2000.
    nengo.Connection(onchip_output.neurons, arm, transform=out2arm_transform, synapse=tau_output)

    adapt_ens = []
    conn_learn = []
    for ii in range(n_ensembles):
        adapt_ens.append(
            nengo.Ensemble(
                n_neurons=n_neurons,
                dimensions=n_input,
                intercepts=intercepts[ii],
                radius=np.sqrt(n_input),
                encoders=encoders[ii],
                label="ens%02d" % ii,
            )
        )

        # hook up input signal to adaptive population to provide context
        inp2ens_transform_ii = np.dot(encoders[ii], inp2ens_transform)
        nengo.Connection(onchip_input.neurons, adapt_ens[ii].neurons,
                         transform=inp2ens_transform_ii, synapse=tau_input)

        conn_learn.append(
            nengo.Connection(
                adapt_ens[ii],
                onchip_output,
                learning_rule_type=nengo.PES(
                    pes_learning_rate, pre_synapse=tau_training,
                ),
                #transform=inout_transform / n_ensembles,
                #transform=inout_transform / n_ensembles * 0.1,
                transform=rng.uniform(-0.01, 0.01, size=(n_output, n_input)),
                #transform=np.zeros((n_output, n_input)),
                #transform=np.zeros((n_output, n_neurons)),
                #transform=np.zeros((onchip_output.n_neurons, n_neurons)),
            )
        )

        # hook up the training signal to the learning rule
        nengo.Connection(
            arm[n_input:], conn_learn[ii].learning_rule,
            synapse=None,
            #synapse=tau_training
        )

import cProfile

with nengo_loihi.Simulator(
        nengo_model,
        target='loihi',
        hardware_options=dict(snip_max_spikes_per_step=300)) as sim:
# with nengo.Simulator(nengo_model) as sim:

    hw = sim.sims["loihi"]
    print("n_errors: %s" % hw.nengo_io_h2c_errors)
    print("n_outputs: %s" % hw.nengo_io_c2h_count)

    sim.run(0.001)
    start = timeit.default_timer()
    #sim.run(0.005)
    #sim.run(1)
    #sim.run(4)
    sim.run(10)
    #sim.run(30)
    #cProfile.runctx("sim.run(1)", globals(), locals(), sort="cumtime")
    #cProfile.runctx("sim.run(0.001)", globals(), locals(), sort="cumtime")
    print('Run time: %0.5f' % (timeit.default_timer() - start))
    print('Run time/step: %0.2f ms' % (1000 * sim.timers["snips"] / sim.n_steps))


    inputs = sim.data[arm_probe][:, :n_input]
    targets = np.dot(inputs, inout_transform.T)
    errors = sim.data[arm_probe][:, n_input:]
    outputs = errors + targets

    error_steps = min(sim.n_steps, 2000)
    start_error = np.abs(errors[:error_steps]).mean()
    end_error = np.abs(errors[-error_steps:]).mean()
    print("Error: start %0.3f, end %0.3f" % (start_error, end_error))

import matplotlib.pyplot as plt
plt.plot(sim.trange(), targets, ":")
plt.plot(sim.trange(), outputs)
plt.savefig("dynamics_adaptation_test.pdf")
