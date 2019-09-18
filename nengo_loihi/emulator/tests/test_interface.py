from nengo.exceptions import SimulationError
import numpy as np
import pytest

from nengo_loihi.block import Axon, LoihiBlock, Synapse, Probe
from nengo_loihi.builder import Model
from nengo_loihi.discretize import discretize_model, VTH_MAX
from nengo_loihi.emulator import EmulatorInterface
from nengo_loihi.emulator.interface import (
    BlockInfo,
    CompartmentState,
    NoiseState,
    SynapseState,
)
from nengo_loihi.hardware import HardwareInterface
from nengo_loihi.inputs import SpikeInput


@pytest.mark.parametrize("strict", (True, False))
def test_strict_mode(strict, monkeypatch):
    # Tests should be run in strict mode
    assert EmulatorInterface.strict

    model = Model()
    model.add_block(LoihiBlock(1))

    monkeypatch.setattr(EmulatorInterface, "strict", strict)
    emu = EmulatorInterface(model)
    assert emu.strict == strict

    if strict:
        check = pytest.raises(SimulationError, match="Error in emulator")
    else:
        check = pytest.warns(UserWarning)

    with check:
        emu.compartment.error("Error in emulator")


@pytest.mark.target_loihi
@pytest.mark.parametrize("n_axons", [200, 1000])
def test_uv_overflow(n_axons, plt, allclose, monkeypatch):
    # TODO: Currently this is not testing the V overflow, since it is higher
    #  and I haven't been able to figure out a way to make it overflow.
    nt = 15

    model = Model()

    # n_axons controls number of input spikes and thus amount of overflow
    input = SpikeInput(n_axons)
    for t in np.arange(1, nt + 1):
        input.add_spikes(t, np.arange(n_axons))  # send spikes to all axons
    model.add_input(input)

    block = LoihiBlock(1)
    block.compartment.configure_relu()
    block.compartment.configure_filter(0.1)

    synapse = Synapse(n_axons)
    synapse.set_weights(np.ones((n_axons, 1)))
    block.add_synapse(synapse)

    axon = Axon(n_axons)
    axon.target = synapse
    input.add_axon(axon)

    probe_u = Probe(target=block, key="current")
    block.add_probe(probe_u)
    probe_v = Probe(target=block, key="voltage")
    block.add_probe(probe_v)
    probe_s = Probe(target=block, key="spiked")
    block.add_probe(probe_s)

    model.add_block(block)
    discretize_model(model)

    # must set these after `discretize` to specify discretized values
    block.compartment.vmin = -(2 ** 22) + 1
    block.compartment.vth[:] = VTH_MAX

    assert EmulatorInterface.strict  # Tests should be run in strict mode
    monkeypatch.setattr(EmulatorInterface, "strict", False)
    with EmulatorInterface(model) as emu:
        with pytest.warns(UserWarning):
            emu.run_steps(nt)
        emu_u = emu.get_probe_output(probe_u)
        emu_v = emu.get_probe_output(probe_v)
        emu_s = emu.get_probe_output(probe_s)

    with HardwareInterface(model, use_snips=False) as sim:
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


def test_dtype_errors():
    block = LoihiBlock(1)
    block_info = BlockInfo([block])
    block.compartment.vth = block.compartment.vth.astype(np.float64)

    with pytest.raises(ValueError, match="dtype.*not supported"):
        CompartmentState(block_info)
    with pytest.raises(ValueError, match="dtype.*not supported"):
        NoiseState(block_info)
    with pytest.raises(ValueError, match="dtype.*not supported"):
        SynapseState(block_info)
