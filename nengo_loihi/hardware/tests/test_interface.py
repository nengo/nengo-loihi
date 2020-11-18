import socket

import nengo
import numpy as np
import pytest
from nengo.exceptions import SimulationError

from nengo_loihi.block import Axon, LoihiBlock, Synapse
from nengo_loihi.builder.builder import Model
from nengo_loihi.builder.discretize import discretize_model
from nengo_loihi.hardware import interface as hardware_interface
from nengo_loihi.hardware.allocators import Greedy
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

    allocator = Greedy()  # one core per ensemble
    board = allocator(model, n_chips=1)

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

    board = allocator(model, n_chips=1)

    with pytest.raises(ValueError, match="[Aa]xon.*[Uu]nrec.*pop.*type"):
        build_board(board)


def test_host_snip_recv_bytes():
    host_snip = hardware_interface.HostSnip(None)

    # We bypass the host_snip.connect method and connect manually
    host_address = "127.0.0.1"  # Standard loopback interface address

    # Configure socket to send data to itself
    host_snip.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    host_snip.socket.bind((host_address, host_snip.port))
    host_snip.socket.connect((host_address, host_snip.port))

    # Generate random data to send
    data = np.random.randint(0, 8192, size=1100, dtype=np.int32)

    # Correctly receive data in two chunks
    # Note that chunks are 4096 bytes at the smallest (HostSnip.recv_size)
    host_snip.send_all(data)
    received = host_snip.recv_bytes(1024 * 4)
    assert np.all(received == data[:1024])
    rest = 1100 - 1024
    received = host_snip.recv_bytes(rest * 4)
    assert np.all(received == data[-rest:])

    # Send too little data
    host_snip.send_all(data)
    with pytest.raises(RuntimeError, match="less than expected"):
        host_snip.recv_bytes(1536 * 4)

    # Send shutdown signal at the end
    data[-1] = -1
    host_snip.send_all(data)
    with pytest.raises(RuntimeError, match="shutdown signal from chip"):
        # With proper amount received
        host_snip.recv_bytes(1100 * 4)
    host_snip.send_all(data)
    with pytest.raises(RuntimeError, match="shutdown signal from chip"):
        # With early stop
        host_snip.recv_bytes(2048 * 4)


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

    with pytest.raises(SimulationError, match="Mock failure to connect"):
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
