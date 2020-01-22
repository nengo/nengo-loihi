from nengo.exceptions import BuildError
import numpy as np
import pytest

from nengo_loihi.block import Axon, LoihiBlock, Probe, Synapse
from nengo_loihi.builder import Model
from nengo_loihi.discretize import discretize_model
from nengo_loihi.emulator import EmulatorInterface
from nengo_loihi.hardware import HardwareInterface
from nengo_loihi.inputs import SpikeInput


def test_compartment_errors():
    block = LoihiBlock(90)

    # set filter to a very large value so current scaling can't be applied
    with pytest.raises(BuildError, match="[Cc]urrent.*scaling"):
        block.compartment.configure_filter(1e6)

    # set to a value when previously configured to larger value
    block.compartment.configure_filter(0.006)
    with pytest.warns(UserWarning, match="tau_s.*larger"):
        block.compartment.configure_filter(0.004)

    # set to a value when previously configured to smaller value
    block.compartment.configure_filter(0.003)
    with pytest.warns(UserWarning, match="tau_s.*smaller"):
        block.compartment.configure_filter(0.007)

    # set tau_rc to a very large value so voltage scaling can't be applied
    with pytest.raises(BuildError, match="[Vv]oltage.*scaling"):
        block.compartment.configure_lif(tau_rc=1e6)


def test_strings():
    block = LoihiBlock(3, label="myBlock")
    assert str(block) == "LoihiBlock(myBlock)"
    assert str(block.compartment) == "Compartment()"

    synapse = Synapse(2, label="mySynapse")
    assert str(synapse) == "Synapse(mySynapse)"

    axon = Axon(2, target=None, compartment_map=[], label="myAxon")
    assert str(axon) == "Axon(myAxon)"

    spike = Axon.Spike(axon_idx=7, atom=2)
    assert str(spike) == "Spike(axon_idx=7, atom=2)"


# TODO: Only targeting sim due to bug with negative cx_base on Loihi
@pytest.mark.target_sim
def test_negative_base(request, seed):
    n_axons = 3

    model = Model()

    input = SpikeInput(n_axons)
    input.add_spikes(1, list(range(n_axons)))
    model.add_input(input)

    block = LoihiBlock(3)
    block.compartment.configure_relu()
    model.add_block(block)

    synapse = Synapse(n_axons)
    weights = [0.1, 0.1, 0.1]
    indices = [0, 1, 2]
    axon_to_weight_map = list(range(n_axons))
    bases = [0, 1, -1]
    synapse.set_population_weights(
        weights, indices, axon_to_weight_map, bases, pop_type=32
    )
    block.add_synapse(synapse)

    axon = Axon(n_axons, target=synapse, compartment_map=np.arange(3))
    input.add_axon(axon)

    probe = Probe(target=block, key="voltage")
    block.add_probe(probe)

    discretize_model(model)

    n_steps = 2
    if request.config.getoption("--target") == "loihi":
        with HardwareInterface(model, use_snips=False, seed=seed) as sim:
            sim.run_steps(n_steps)
            y = sim.get_probe_output(probe)
    else:
        with EmulatorInterface(model, seed=seed) as sim:
            sim.run_steps(n_steps)
            y = sim.get_probe_output(probe)

    # Compartments 0 and 2 should change from axons 0 and 1.
    # Axon 2 should have no effect, and not change compartment 1 (the sum of
    # its base and index), or other compartments (e.g. 2 if base ignored)
    assert np.allclose(y[1, 1], 0), "Third axon not ignored"
    assert np.allclose(y[1, 0], y[1, 2]), "Third axon targeting another"
    assert not np.allclose(y[1], y[0]), "Voltage not changing"
