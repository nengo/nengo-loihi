import nengo
from nengo.exceptions import SimulationError
import pytest


from nengo_loihi.inputs import ChipReceiveNode


def test_chipreceivenode_run_error():
    with nengo.Network() as net:
        ChipReceiveNode(dimensions=1, size_out=1)

    with pytest.raises(SimulationError, match="should not be run"):
        with nengo.Simulator(net) as sim:
            sim.step()
