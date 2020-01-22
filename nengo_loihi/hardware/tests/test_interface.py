import nengo
from nengo.exceptions import SimulationError
import pytest

from nengo_loihi.block import Axon, LoihiBlock, Synapse
from nengo_loihi.builder import Model
from nengo_loihi.discretize import discretize_model
from nengo_loihi.hardware import interface as hardware_interface
from nengo_loihi.hardware.allocators import OneToOne
from nengo_loihi.hardware.builder import build_board
from nengo_loihi.hardware.nxsdk_shim import NxsdkBoard
from nengo_loihi.nxsdk_obfuscation import d


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

    synapse = Synapse(1)
    synapse.set_weights([[1]])
    synapse.pop_type = 8
    block1.add_synapse(synapse)

    axon = Axon(1, target=synapse, compartment_map=[0])
    block0.add_axon(axon)

    discretize_model(model)

    board = allocator(model)

    with pytest.raises(ValueError, match="[Aa]xon.*[Uu]nrec.*pop.*type"):
        build_board(board)


@pytest.mark.target_loihi
def test_interface_connection_errors(Simulator, monkeypatch):
    with nengo.Network() as net:
        nengo.Ensemble(2, 1)

    # test opening closed interface error
    sim = Simulator(net)
    interface = sim.sims["loihi"]
    interface.close()
    with pytest.raises(SimulationError, match="cannot be reopened"):
        with interface:
            pass
    sim.close()

    # test failed connection error
    def start(*args, **kwargs):
        raise Exception("Mock failure to connect")

    monkeypatch.setattr(NxsdkBoard, d(b"c3RhcnQ="), start)

    with pytest.raises(SimulationError, match="[Cc]ould not connect"):
        with Simulator(net):
            pass


@pytest.mark.filterwarnings("ignore:Model is precomputable.")
@pytest.mark.target_loihi
def test_snip_input_count(Simulator, seed, plt):
    with nengo.Network(seed=seed) as model:
        a = nengo.Ensemble(100, 1)
        for i in range(30):
            stim = nengo.Node(0.5)
            nengo.Connection(stim, a, synapse=None)
    with Simulator(model, precompute=False) as sim:
        with pytest.warns(UserWarning, match="Too many spikes"):
            sim.run(0.01)


@pytest.mark.target_loihi
def test_find_learning_core_id():
    # This is mostly just a test for the ValueError, since it was uncovered
    allocator = OneToOne()

    model = Model()

    for _ in range(9):
        block = LoihiBlock(1)
        block.compartment.configure_lif()
        model.add_block(block)

        synapse = Synapse(1)
        synapse.set_weights([[1]])
        block.add_synapse(synapse)

    good_synapse = Synapse(n_axons=1)
    good_synapse.set_weights([[1]])
    good_core_idx = 3
    good_core_id = 392
    list(model.blocks)[good_core_idx].add_synapse(good_synapse)

    bad_synapse = Synapse(n_axons=1)
    bad_synapse.set_weights([[1]])

    discretize_model(model)

    interface = hardware_interface.HardwareInterface(model, use_snips=False)
    interface.board = allocator(model)

    interface.board.chips[0].cores[good_core_idx].learning_coreid = good_core_id
    assert interface._find_learning_core_id(good_synapse) == good_core_id

    with pytest.raises(ValueError, match="Could not find core ID"):
        interface._find_learning_core_id(bad_synapse)
