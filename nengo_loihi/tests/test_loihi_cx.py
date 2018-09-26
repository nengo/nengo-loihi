from nengo.exceptions import SimulationError
import numpy as np
import pytest

from nengo_loihi.loihi_cx import CxGroup, CxModel, CxProbe, CxSimulator


def test_simulator_noise(request, plt, seed):
    target = request.config.getoption("--target")

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

    if target == 'loihi':
        sim = model.get_loihi(seed=seed)
        sim.run_steps(1000)
        y = np.column_stack([
            p.timeSeries.data for p in sim.board.probe_map[probe]])
    else:
        sim = model.get_simulator(seed=seed)
        sim.run_steps(1000)
        y = sim.probe_outputs[probe]

    plt.plot(y)
    plt.yticks(())


def test_strict_mode():
    # Tests should be run in strict mode
    assert CxSimulator.strict

    with pytest.raises(SimulationError):
        CxSimulator.error("Error in emulator")
    CxSimulator.strict = False
    with pytest.warns(UserWarning):
        CxSimulator.error("Error in emulator")

    # Strict mode is a global setting so we set it back to True
    # for subsequent test runs.
    CxSimulator.strict = True
