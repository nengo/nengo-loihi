import nengo
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


def test_tau_s_warning(Simulator):
    with nengo.Network() as net:
        stim = nengo.Node(0)
        ens = nengo.Ensemble(10, 1)
        nengo.Connection(stim, ens, synapse=0.1)
        nengo.Connection(ens, ens,
                         synapse=0.001,
                         solver=nengo.solvers.LstsqL2(weights=True))

    with pytest.warns(UserWarning) as record:
        with Simulator(net):
            pass
    # The 0.001 synapse is applied first due to splitting rules putting
    # the stim -> ens connection later than the ens -> ens connection
    assert any(rec.message.args[0] == (
        "tau_s is currently 0.001, which is smaller than 0.005. "
        "Overwriting tau_s with 0.005.") for rec in record)

    with net:
        nengo.Connection(ens, ens,
                         synapse=0.1,
                         solver=nengo.solvers.LstsqL2(weights=True))
    with pytest.warns(UserWarning) as record:
        with Simulator(net):
            pass
    assert any(rec.message.args[0] == (
        "tau_s is already set to 0.1, which is larger than 0.005. Using 0.1."
    ) for rec in record)
