import nengo
from nengo.exceptions import SimulationError
import pytest

from nengo_loihi.block import Axon, LoihiBlock, Synapse
from nengo_loihi.builder import Model
from nengo_loihi.discretize import discretize_model
from nengo_loihi.hardware import interface as hardware_interface
from nengo_loihi.hardware.allocators import one_to_one_allocator
from nengo_loihi.hardware.builder import build_board


class MockNxsdk:
    def __init__(self):
        self.__version__ = None


def test_error_on_old_version(monkeypatch):
    mock = MockNxsdk()
    mock.__version__ = "0.5.5"

    monkeypatch.setattr(hardware_interface, 'nxsdk', mock)
    with pytest.raises(ImportError):
        hardware_interface.HardwareInterface.check_nxsdk_version()


def test_no_warn_on_current_version(monkeypatch):
    mock = MockNxsdk()
    mock.__version__ = "0.7.0"

    monkeypatch.setattr(hardware_interface, 'nxsdk', mock)
    monkeypatch.setattr(hardware_interface, 'assert_nxsdk', lambda: True)
    with pytest.warns(None) as record:
        hardware_interface.HardwareInterface.check_nxsdk_version()
    assert len(record) == 0


def test_warn_on_future_version(monkeypatch):
    mock = MockNxsdk()
    mock.__version__ = "100.0.0"

    monkeypatch.setattr(hardware_interface, 'nxsdk', mock)
    monkeypatch.setattr(hardware_interface, 'assert_nxsdk', lambda: True)
    with pytest.warns(UserWarning):
        hardware_interface.HardwareInterface.check_nxsdk_version()


def test_builder_poptype_errors():
    pytest.importorskip('nxsdk')

    # Test error in build_synapse
    model = Model()
    block = LoihiBlock(1)
    block.compartment.configure_lif()
    model.add_block(block)

    synapse = Synapse(1)
    synapse.set_full_weights([1])
    synapse.pop_type = 8
    block.add_synapse(synapse)

    discretize_model(model)

    allocator = one_to_one_allocator  # one core per ensemble
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
    synapse.set_full_weights([1])
    synapse.pop_type = 8
    axon.target = synapse
    block1.add_synapse(synapse)

    discretize_model(model)

    allocator = one_to_one_allocator  # one core per ensemble
    board = allocator(model)

    with pytest.raises(ValueError, match="[Aa]xon.*[Uu]nrec.*pop.*type"):
        build_board(board)


@pytest.mark.skipif(pytest.config.getoption('--target') != 'loihi',
                    reason="Loihi-only test")
def test_interface_connection_errors(Simulator):
    with nengo.Network() as net:
        nengo.Ensemble(2, 1)

    # test unbuilt model error
    with Simulator(net) as sim:
        sim.sims['loihi'].n2board = None

        with pytest.raises(SimulationError, match="build.*before running"):
            sim.step()

    # test failed connection error
    def startDriver(*args, **kwargs):
        raise Exception("Mock failure to connect")

    with Simulator(net) as sim:
        interface = sim.sims['loihi']
        interface.n2board.startDriver = startDriver

        with pytest.raises(SimulationError, match="[Cc]ould not connect"):
            interface.connect(attempts=1)
