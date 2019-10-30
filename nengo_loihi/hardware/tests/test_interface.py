import nengo
from nengo.exceptions import SimulationError
import numpy as np
import pytest

import nengo_loihi
from nengo_loihi.block import Axon, LoihiBlock, Synapse
from nengo_loihi.compat import signals_allclose
from nengo_loihi.builder import Model
from nengo_loihi.discretize import discretize_model
from nengo_loihi.hardware import interface as hardware_interface
from nengo_loihi.hardware.allocators import OneToOne
from nengo_loihi.hardware.builder import build_board
from nengo_loihi.nxsdk_obfuscation import d_set


class MockNxsdk:
    def __init__(self):
        self.__version__ = None


def test_error_on_old_version(monkeypatch):
    mock = MockNxsdk()
    mock.__version__ = "0.5.5"

    monkeypatch.setattr(hardware_interface, "nxsdk", mock)
    with pytest.raises(ImportError, match="nxsdk"):
        hardware_interface.HardwareInterface.check_nxsdk_version()


def test_no_warn_on_current_version(monkeypatch):
    mock = MockNxsdk()
    mock.__version__ = str(hardware_interface.HardwareInterface.max_nxsdk_version)

    monkeypatch.setattr(hardware_interface, "nxsdk", mock)
    monkeypatch.setattr(hardware_interface, "assert_nxsdk", lambda: True)
    with pytest.warns(None) as record:
        hardware_interface.HardwareInterface.check_nxsdk_version()
    assert len(record) == 0


def test_warn_on_future_version(monkeypatch):
    mock = MockNxsdk()
    mock.__version__ = "100.0.0"

    monkeypatch.setattr(hardware_interface, "nxsdk", mock)
    monkeypatch.setattr(hardware_interface, "assert_nxsdk", lambda: True)
    with pytest.warns(UserWarning):
        hardware_interface.HardwareInterface.check_nxsdk_version()


def test_builder_poptype_errors():
    pytest.importorskip("nxsdk")

    # Test error in build_synapse
    model = Model()
    block = LoihiBlock(1)
    block.compartment.configure_lif()
    model.add_block(block)

    synapse = Synapse(1)
    synapse.set_weights([[1]])
    synapse.pop_type = 8
    block.add_synapse(synapse)

    discretize_model(model)

    allocator = OneToOne()  # one core per ensemble
    board = allocator(model)

    with pytest.raises(ValueError, match="[Ss]ynapse.*[Uu]nrec.*pop.*type"):
        build_board(board)

    # Test error in collect_axons
    model = Model()
    block0 = LoihiBlock(1)
    block0.compartment.configure_lif()
    model.add_block(block0)
    block1 = LoihiBlock(1)
    block1.compartment.configure_lif()
    model.add_block(block1)

    axon = Axon(1)
    block0.add_axon(axon)

    synapse = Synapse(1)
    synapse.set_weights([[1]])
    synapse.pop_type = 8
    axon.target = synapse
    block1.add_synapse(synapse)

    discretize_model(model)

    board = allocator(model)

    with pytest.raises(ValueError, match="[Aa]xon.*[Uu]nrec.*pop.*type"):
        build_board(board)


@pytest.mark.target_loihi
def test_interface_connection_errors(Simulator):
    with nengo.Network() as net:
        nengo.Ensemble(2, 1)

    # test unbuilt model error
    with Simulator(net) as sim:
        sim.sims["loihi"].nxsdk_board = None

        with pytest.raises(SimulationError, match="build.*before running"):
            sim.step()

    # test failed connection error
    def start(*args, **kwargs):
        raise Exception("Mock failure to connect")

    with Simulator(net) as sim:
        interface = sim.sims["loihi"]
        d_set(interface.nxsdk_board, b"c3RhcnQ=", val=start)

        with pytest.raises(SimulationError, match="[Cc]ould not connect"):
            interface.connect(attempts=1)


@pytest.mark.target_loihi
def test_snip_input_count(Simulator, seed, plt):
    with nengo.Network(seed=seed) as model:
        a = nengo.Ensemble(100, 1)
        for i in range(30):
            stim = nengo.Node(0.5)
            nengo.Connection(stim, a, synapse=None)
    with Simulator(model) as sim:
        with pytest.warns(UserWarning, match="Too many spikes"):
            sim.run(0.01)
