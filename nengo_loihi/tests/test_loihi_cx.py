import nengo
from nengo.exceptions import SimulationError
import numpy as np
import pytest

from nengo_loihi.loihi_api import VTH_MAX
from nengo_loihi.loihi_cx import (
    CxAxons, CxGroup, CxModel, CxProbe, CxSimulator, CxSpikeInput, CxSynapses)
from nengo_loihi.loihi_interface import LoihiSimulator


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
        with LoihiSimulator(model, use_snips=False, seed=seed) as sim:
            sim.run_steps(1000)
            y = sim.get_probe_output(probe)
    else:
        with CxSimulator(model, seed=seed) as sim:
            sim.run_steps(1000)
            y = sim.get_probe_output(probe)

    plt.plot(y)
    plt.yticks(())


def test_strict_mode():
    # Tests should be run in strict mode
    assert CxSimulator.strict

    try:
        with pytest.raises(SimulationError):
            CxSimulator.error("Error in emulator")
        CxSimulator.strict = False
        with pytest.warns(UserWarning):
            CxSimulator.error("Error in emulator")
    finally:
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


@pytest.mark.skipif(pytest.config.getoption("--target") != "loihi",
                    reason="need Loihi as comparison")
@pytest.mark.parametrize('n_axons', [200, 1000])
def test_uv_overflow(n_axons, Simulator, plt, allclose):
    # TODO: Currently this is not testing the V overflow, since it is higher
    #  and I haven't been able to figure out a way to make it overflow.
    nt = 15

    model = CxModel()

    # n_axons controls number of input spikes and thus amount of overflow
    input = CxSpikeInput(n_axons)
    for t in np.arange(1, nt+1):
        input.add_spikes(t, np.arange(n_axons))  # send spikes to all axons
    model.add_input(input)

    group = CxGroup(1)
    group.configure_relu()
    group.configure_filter(0.1)
    group.vmin = -2**22

    synapses = CxSynapses(n_axons)
    synapses.set_full_weights(np.ones((n_axons, 1)))
    group.add_synapses(synapses)

    axons = CxAxons(n_axons)
    axons.target = synapses
    input.add_axons(axons)

    probe_u = CxProbe(target=group, key='u')
    group.add_probe(probe_u)
    probe_v = CxProbe(target=group, key='v')
    group.add_probe(probe_v)
    probe_s = CxProbe(target=group, key='s')
    group.add_probe(probe_s)

    model.add_group(group)
    model.discretize()

    group.vth[:] = VTH_MAX  # must set after `discretize`

    assert CxSimulator.strict  # Tests should be run in strict mode
    CxSimulator.strict = False
    try:
        with CxSimulator(model) as emu:
            with pytest.warns(UserWarning):
                emu.run_steps(nt)
            emu_u = emu.get_probe_output(probe_u)
            emu_v = emu.get_probe_output(probe_v)
            emu_s = emu.get_probe_output(probe_s)
    finally:
        CxSimulator.strict = True  # change back to True for subsequent tests

    with LoihiSimulator(model, use_snips=False) as sim:
        sim.run_steps(nt)
        sim_u = sim.get_probe_output(probe_u)
        sim_v = sim.get_probe_output(probe_v)
        sim_s = sim.get_probe_output(probe_s)
        sim_v[sim_s > 0] = 0  # since Loihi has placeholder voltage after spike

    plt.subplot(311)
    plt.plot(emu_u)
    plt.plot(sim_u)

    plt.subplot(312)
    plt.plot(emu_v)
    plt.plot(sim_v)

    plt.subplot(313)
    plt.plot(emu_s)
    plt.plot(sim_s)

    assert allclose(emu_u, sim_u)
    assert allclose(emu_v, sim_v)
