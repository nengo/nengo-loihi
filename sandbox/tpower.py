import nengo
import numpy as np
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
    power_calc = energy / execution_time

    # raw power measurements should be more accurate for fluctuating power
    power = power_measured

    return dict(power_data=power_data, energy=energy, power=power, time=execution_time)


def measure_idle_power_n2n(run_time=10.0, multichip=False):
    dt = 0.001

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
    args = dict(hardware_options=dict())
    if multichip:
        args["hardware_options"]["allocator"] = RoundRobin()
        args["hardware_options"]["n_chips"] = 2

    sim = nengo_loihi.Simulator(net, precompute=True, dt=dt, seed=0, **args)
    probe = add_energy_probe(sim, steps)

    with sim:
        sim.run_steps(steps)

    return compute_power_metrics(probe)


with nengo.Network() as net:
    ens = nengo.Ensemble(dimensions=1, n_neurons=100)
    input = nengo.Node(np.sin)
    output = nengo.Node(size_in=1)

    nengo.Connection(input, ens)
    nengo.Connection(ens, output)

n_chips = 1
multichip_allocator = GreedyInterchip(cores_per_chip=64)
allocator = multichip_allocator if n_chips > 1 else Greedy()

sim = nengo_loihi.Simulator(
    net,
    precompute=True,
    hardware_options=dict(n_chips=n_chips, allocator=allocator),
)

# add energy probe to record power usage of network
runtime = 3
n_steps = int(runtime / 0.001)

energy_probe = add_energy_probe(sim, n_steps)

# measure idle power, right before we actually run
idle_power = measure_idle_power_n2n(run_time=runtime, multichip=True)

with sim:
    sim.run(runtime)

    #### RETURN DICT OF DATA
    data = {}

    sim_power = compute_power_metrics(energy_probe)
    dyn_power = sim_power["power"] - idle_power["power"]
    dyn_energy = dyn_power * (energy_probe.totalExecutionTime * 1e-6)
    n_inferences = n_steps
    energy_inference = dyn_energy / n_inferences

    data["Total execution time (s)"] = energy_probe.totalExecutionTime * 1e-6
    data["N_inferences"] = n_inferences
    data["N_inferences/s"] = n_inferences / (energy_probe.totalExecutionTime * 1e-6)

    data["Idle (W)"] = idle_power["power"]
    data["Running (W)"] = sim_power["power"]
    data["Dynamic (J)"] = dyn_energy
    data["Dynamic J/inf"] = energy_inference

print(data)
