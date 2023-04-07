import collections

import matplotlib.pyplot as plt
import nengo
import numpy as np
from nengo.utils.numpy import rms
from nxsdk.api.enums.api_enums import ProbeParameter
from nxsdk.graph.monitor.probes import PerformanceProbeCondition

import nengo_loihi
from nengo_loihi.hardware.allocators import Greedy, GreedyInterchip, RoundRobin


def add_energy_probe(sim, steps):
    assert sim.sims["loihi"] is not None, "Must be on Loihi"
    if sim.sims["loihi"].nxsdk_board is None:
        sim.sims["loihi"].build()

    energy_probe_cond = PerformanceProbeCondition(
        tStart=1, tEnd=steps, bufferSize=1024, binSize=4
    )
    energy_probe = sim.sims["loihi"].nxsdk_board.probe(
        ProbeParameter.ENERGY, energy_probe_cond
    )
    return energy_probe


def compute_power_metrics(probe):
    # Get the total raw power numbers from loihi (measured in mW, converted to W)
    power_data = np.asarray(probe.rawPowerTotal, dtype=float) / 1000.0
    # compute the mean power
    power_measured = np.mean(power_data)
    # Get the total recorded energy from loihi (measured in uJ, converted to J)
    energy = probe.totalEnergy / 1000000.0
    # Get the total execution time from loihi (measured in us, converted to s)
    execution_time = probe.totalExecutionTime / 1000000.0
    # Calculate wattage from recorded energy and execution times
    # power_calc = energy / execution_time

    # raw power measurements should be more accurate for fluctuating power
    power = power_measured

    return dict(power_data=power_data, energy=energy, power=power, time=execution_time)


def measure_idle_power_n2n(run_time=10.0, n_chips=1):
    dt = 0.001
    multichip = n_chips > 1

    # empirically estimated steps/second
    steps_per_second = 65e3 if multichip else 125e3
    steps = int(round(run_time * steps_per_second))

    with nengo.Network(seed=0) as net:
        # Create an ensemble that fires one spike per step
        neuron_type = nengo_loihi.neurons.LoihiSpikingRectifiedLinear()
        gain = [1.0]
        bias = [1.002 / dt]
        ens_a = nengo.Ensemble(1, 1, neuron_type=neuron_type, gain=gain, bias=bias)
        ens_b = nengo.Ensemble(1, 1, neuron_type=neuron_type)

        nengo.Connection(ens_a.neurons, ens_b.neurons, transform=[[0.001]])

    # Create the nengo_loihi simulator to run the network to get the energy data
    args = dict(hardware_options=dict(n_chips=n_chips, allocator=RoundRobin()))

    sim = nengo_loihi.Simulator(net, precompute=True, dt=dt, seed=0, **args)
    probe = add_energy_probe(sim, steps)

    with sim:
        sim.run_steps(steps)

    return compute_power_metrics(probe)


def get_network(num_ensemble=3, factor=20, seed=0):
    n_neurons = 100
    dim = 1

    transform = np.eye(n_neurons)
    gain = np.ones((n_neurons))
    neuron_type = nengo_loihi.LoihiSpikingRectifiedLinear(amplitude=1)

    with nengo.Network(seed=seed) as net:
        ensembles = collections.OrderedDict()
        probes = collections.OrderedDict()

        for n in range(num_ensemble):
            bias = np.random.uniform(0.1, 1.0, size=(n_neurons)) * 10 * factor

            x = nengo.Ensemble(
                n_neurons,
                dim,
                gain=gain,
                bias=bias,
                neuron_type=neuron_type,
                label="a%d" % n,
            )
            y = nengo.Ensemble(
                n_neurons,
                dim,
                neuron_type=neuron_type,
                gain=gain,
                bias=np.zeros((n_neurons)),
                label="b%d" % n,
            )
            nengo.Connection(
                x.neurons,
                y.neurons,
                transform=transform,
                synapse=0.001,
            )

            # TODO: we could add probes here to see how they affect power consumption
            if False:  # n == 0 or n == (num_ensemble-1):
                probes["%d" % n] = nengo.Probe(ensembles["%d_1" % n].neurons)

    return net, probes


seed = 0
np.random.seed(seed)

dt = 0.001
sim_t = 25
steps = int(sim_t / dt)

num_ensemble = 50
factor = 10
net, probes = get_network(num_ensemble=num_ensemble, factor=factor, seed=seed)

n_chips = 2
multichip_allocator = GreedyInterchip(cores_per_chip=64)
allocator = multichip_allocator if n_chips > 1 else Greedy()

sim_loihi = nengo_loihi.Simulator(
    net,
    precompute=True,
    hardware_options={"n_chips": n_chips, "allocator": allocator},
)

# add energy probe and measure idle power BEFORE we
# connect to the board with our main simulator
energy_probe = add_energy_probe(sim_loihi, steps)
idle_power = measure_idle_power_n2n(run_time=10.0, n_chips=n_chips)

# entering this `with` block connects to the Loihi board
with sim_loihi:
    sim_loihi.run_steps(steps)


if 0:
    # Make an NxSDK plot of time of different execution phases
    plt.figure()
    energy_probe.plotExecutionTime()
    plt.savefig("phase_time_plot_%d_%d.png" % (n_chips, factor))
    plt.close()


sim_power = compute_power_metrics(energy_probe)
dyn_power = sim_power["power"] - idle_power["power"]
dyn_energy = dyn_power * sim_power["time"]
print(f"Idle power ({idle_power['time']:0.3f} s): {idle_power['power']:0.3f} W")
print(f"Sim power ({sim_power['time']:0.3f} s): {sim_power['power']:0.3f} W")
print(f"Dynamic Power: {dyn_power * 1e3:0.2f} mW")
print(f"Dynamic Energy: {dyn_energy * 1e3:0.2f} mJ")
