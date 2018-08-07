import numpy as np
import pytest

from nengo_loihi.builder import CoreGroup, Model
from nengo_loihi.hardware import HAS_NXSDK, LoihiSimulator
from nengo_loihi.probes import Probe


@pytest.mark.skipif(not HAS_NXSDK, reason="Test requires NxSDK")
def test_simulator_noise(plt, seed):
    model = Model()
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

    sim = LoihiSimulator(model, seed=seed)
    sim.run_steps(1000)
    y = np.column_stack([
        p.timeSeries.data for p in sim.board.probe_map[probe]])

    plt.plot(y)
    plt.yticks(())
