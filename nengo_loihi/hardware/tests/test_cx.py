import numpy as np
import pytest

from nengo_loihi.cx import CxGroup, CxProbe, CxModel
from nengo_loihi.hardware import HAS_NXSDK, LoihiSimulator


@pytest.mark.skipif(not HAS_NXSDK, reason="Test requires NxSDK")
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

    sim = LoihiSimulator(model, seed=seed)
    sim.run_steps(1000)
    y = np.column_stack([
        p.timeSeries.data for p in sim.board.probe_map[probe]])

    plt.plot(y)
    plt.yticks(())
