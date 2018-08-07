import numpy as np

from nengo_loihi.group import CoreGroup
from nengo_loihi.emulator import Emulator
from nengo_loihi.model import CxModel
from nengo_loihi.probes import Probe


def test_simulator_noise(plt, seed):
    model = CxModel()
    group = CoreGroup(10)
    group.compartments.configure_relu()

    group.compartments.bias[...] = np.linspace(0, 0.01, group.n_compartments)

    group.compartments.enableNoise[:] = 1
    group.compartments.noiseExp0 = -2
    group.compartments.noiseMantOffset0 = 0
    group.compartments.noiseAtDendOrVm = 1

    probe = Probe(target=group, key='v')
    group.probes.add(probe)
    model.add_group(group)

    model.discretize()

    sim = Emulator(model, seed=seed)
    sim.run_steps(1000)
    y = sim.probe_outputs[probe]

    plt.plot(y)
    plt.yticks(())
