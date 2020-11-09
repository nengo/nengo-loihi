import numpy as np
import nengo
from nengo.utils.numpy import rms
import nengo_loihi
from nengo_loihi.hardware.allocators import Greedy, RoundRobin
import logging
from profiling.loihi_monitor import (
    add_energy_probe,
    measure_idle_power_n2n,
    compute_power_metrics,
)
import time
import collections
import pandas as pd
import matplotlib.pyplot as plt


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
    with nengo.Network(seed=seed) as model:
        ensembles = collections.OrderedDict()
        conns = collections.OrderedDict()
        probes = collections.OrderedDict()

        for n in range(num_ensemble):
            bias = np.random.uniform(0.1, 1.0, size=(n_neurons)) * 10 * factor

            ensembles["%d" % 0] = nengo.Ensemble(
                n_neurons,
                dim,
                gain=gain,
                bias=bias,
                neuron_type=neuron_type,
                label="ens_%d" % 0,
            )
            for i in range(1, 1 - nChips):
                ensembles["%d_1" % i] = nengo.Ensemble(
                    n_neurons,
                    dim,
                    neuron_type=neuron_type,
                    gain=gain,
                    bias=np.zeros((n_neurons)),
                    label="ens_1_%d" % n,
                )
                conns["%d" % i] = nengo.Connection(
                    ensembles["%d" % i - 1].neurons,
                    ensembles["%d_1" % i].neurons,
                    transform=transform,
                    synapse=0.001,
                )

            if False:  # n == 0 or n == (num_ensemble-1):
                probes["%d" % n] = nengo.Probe(ensembles["%d_1" % n].neurons)
    return model, ensembles, conns, probes


n_chips = [1, nChips]

metric_data = collections.defaultdict(list)
for factor in range(11, 102, 10):
    model, ensembles, conns, probes = get_network(
        num_ensemble=num_ensemble, factor=factor
    )
    for option in [0, 1]:
        # time.sleep(4)
        energys = []
        powers = []
        idle_chip_power = []
        run_power = []
        exec_time = []
        for _ in range(n_rept):
            n_chip = n_chips[option]

            sim_loihi = nengo_loihi.Simulator(
                model,
                precompute=True,
                hardware_options={"n_chips": n_chip, "allocator": RoundRobin()},
            )

            energy_probe = add_energy_probe(sim_loihi, steps)
            idle_power = measure_idle_power_n2n(run_time=10.0, n_chips=n_chip)

            with sim_loihi:
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
            sim_power = compute_power_metrics(energy_probe)
            plt.figure()
            energy_probe.plotExecutionTime()
            plt.savefig("phase_time_plot_%d_%d.png" % (option, factor))
            plt.close()

            dyn_power = sim_power["power"] - idle_power["power"]
            dyn_energy = dyn_power * (energy_probe.totalExecutionTime * 1e-6)
            print("Dynamic Power", dyn_power)
            print("Dynamic Energy", dyn_energy)
            energys.append(dyn_energy)
            powers.append(dyn_power)
            exec_time.append(energy_probe.totalExecutionTime * 1e-6)
            idle_chip_power.append(idle_power["power"])
            run_power.append(sim_power["power"])
            print(idle_power["power"], sim_power["power"])
            del sim_loihi
            del energy_probe
            del idle_power
            del sim_power

        metric_data["Power_%d" % option].append(np.mean(powers))
        metric_data["Energy_%d" % option].append(np.mean(energys))
        metric_data["Execution_time_%d" % option].append(np.mean(exec_time))
        metric_data["Power_std_%d" % option].append(np.std(powers))
        metric_data["Energy_std_%d" % option].append(np.std(energys))
        metric_data["Execution_time_std_%d" % option].append(np.std(exec_time))

    metric_data["Factor"].append(factor)

data = pd.DataFrame(metric_data)

data.to_csv("interchip_metrics.csv")
