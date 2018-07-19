import numpy as np

from nengo_loihi.emulator import Emulator
from nengo_loihi.model import CxGroup, CxProbe, CxModel


def test_simulator_noise(plt, seed):
    model = CxModel()
    group = CxGroup(10)
    group.configure_relu()

    group.bias[:] = np.linspace(0, 0.01, group.n)

    group.enableNoise[:] = 1
    group.noiseExp0 = -2
    group.noiseMantOffset0 = 0
    group.noiseAtDendOrVm = 1

    probe = CxProbe(target=group, key='v')
    group.add_probe(probe)
    model.add_group(group)

    model.discretize()

    sim = Emulator(model, seed=seed)
    sim.run_steps(1000)
    y = sim.probe_outputs[probe]

    plt.plot(y)
    plt.yticks(())
