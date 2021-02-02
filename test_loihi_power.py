import numpy as np
import nengo
from nengo.utils.numpy import rms
import nengo_loihi
from nengo_loihi.hardware.allocators import Greedy, GreedyComms, RoundRobin
import logging
# from power_profile import (
#     add_energy_probe,
#     measure_idle_power_n2n,
#     compute_power_metrics,
# )
import time
import collections
import pandas as pd
import matplotlib.pyplot as plt

from nxsdk.api.enums.api_enums import ProbeParameter
from nxsdk.graph.monitor.probes import PerformanceProbeCondition


def add_energy_probe(sim, steps):
    assert sim.sims["loihi"] is not None, "Must be on Loihi"
    if sim.sims["loihi"].nxsdk_board is None:
        sim.sims["loihi"].build()

    energy_probe_cond = PerformanceProbeCondition(
        tStart=1, tEnd=steps, bufferSize=1024 * 4, binSize=4
    )
    energy_probe = sim.sims["loihi"].nxsdk_board.probe(
        ProbeParameter.ENERGY, energy_probe_cond
    )
    return energy_probe


seed = 0
np.random.seed(seed)

n_rept = 2

num_ensemble = 50
n_neurons = 100
dim = 1
nChips = 2

dt = 0.001
sim_t = 2
steps = int(sim_t / dt)

transform = np.eye(n_neurons)
gain = np.ones((n_neurons))
neuron_type = nengo_loihi.LoihiSpikingRectifiedLinear(amplitude=1)


def get_network(num_ensemble=3, factor=20):
    with nengo.Network(seed=seed) as net:
        ensembles = collections.OrderedDict()
        conns = collections.OrderedDict()
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

            if False:  # n == 0 or n == (num_ensemble-1):
                probes["%d" % n] = nengo.Probe(ensembles["%d_1" % n].neurons)

    return net, probes


# n_chips = [1, nChips]
n_chips = [nChips]

# factors = list(range(11, 102, 10))
factors = list(range(10, 101, 30))

# multichip_allocator = RoundRobin()
multichip_allocator = GreedyComms(cores_per_chip=64)

metric_data = collections.defaultdict(list)
for factor in factors:
    net, probes = get_network(
        num_ensemble=num_ensemble, factor=factor
    )
    for n_chip in n_chips:
        # time.sleep(4)
        energys = []
        powers = []
        idle_chip_power = []
        run_power = []
        exec_time = []
        for _ in range(n_rept):
            allocator = multichip_allocator if n_chip > 1 else Greedy()

            sim_loihi = nengo_loihi.Simulator(
                net,
                precompute=True,
                hardware_options={"n_chips": n_chip, "allocator": allocator},
            )

            energy_probe = add_energy_probe(sim_loihi, steps)
            # idle_power = measure_idle_power_n2n(run_time=10.0, n_chips=n_chip)

            with sim_loihi:
                board = sim_loihi.sims["loihi"].board

                # print allocation
                # for chip_idx, chip in enumerate(board.chips):
                #     print("Chip %d:" % (chip_idx,))

                #     block_labels = []
                #     for core_idx, core in enumerate(chip.cores):
                #         for block_idx, block in enumerate(core.blocks):
                #             block_labels.append(block.label)

                #     print(block_labels)

                # run simulation
                sim_loihi.run_steps(steps)
                if len(probes.keys()) > 1:
                    ranges_min = []
                    ranges_max = []
                    for n, probe_key in enumerate(probes.keys()):
                        probe = probes[probe_key]
                        data = sim_loihi.data[probe]
                        nonzero = (data > 0).any(axis=0)
                        output_noscale = data[:, nonzero] if sum(nonzero) > 0 else data
                        output = output_noscale
                        rates = output.mean(axis=0)
                        # print(probe_key, rates.mean(), rates.min(), rates.max())
                        ranges_min.append(rates.min())
                        ranges_max.append(rates.max())
                    print(np.array(ranges_min).min())
                    print(np.array(ranges_max).max())

            # plt.figure()
            # energy_probe.plotExecutionTime()
            # plt.savefig("phase_time_plot_%d_%d.png" % (n_chip, factor))
            # plt.close()

            exec_time.append(energy_probe.totalExecutionTime * 1e-6)

            # sim_power = compute_power_metrics(energy_probe)
            # dyn_power = sim_power["power"] - idle_power["power"]
            # dyn_energy = dyn_power * (energy_probe.totalExecutionTime * 1e-6)
            # print("Dynamic Power", dyn_power)
            # print("Dynamic Energy", dyn_energy)
            # energys.append(dyn_energy)
            # powers.append(dyn_power)
            # idle_chip_power.append(idle_power["power"])
            # run_power.append(sim_power["power"])
            # print(idle_power["power"], sim_power["power"])

            del sim_loihi
            del energy_probe
            # del idle_power
            # del sim_power

        # metric_data["Power_%d" % n_chip].append(np.mean(powers))
        # metric_data["Energy_%d" % n_chip].append(np.mean(energys))
        metric_data["Execution_time_%d" % n_chip].append(np.mean(exec_time))
        # metric_data["Power_std_%d" % n_chip].append(np.std(powers))
        # metric_data["Energy_std_%d" % n_chip].append(np.std(energys))
        metric_data["Execution_time_std_%d" % n_chip].append(np.std(exec_time))

    metric_data["Factor"].append(factor)

data = pd.DataFrame(metric_data)

data.to_csv("interchip_metrics.csv")

plt.figure()
base_rate = 10 * np.asarray(metric_data["Factor"])
for n_chip in n_chips:
    plt.plot(
        base_rate, metric_data["Execution_time_%d" % n_chip], label="%d chips" % n_chip
    )
plt.legend(loc=2)
plt.savefig("execution_time.pdf")
