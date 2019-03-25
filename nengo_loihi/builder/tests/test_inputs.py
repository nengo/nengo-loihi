import nengo
from nengo.exceptions import SimulationError
import numpy as np
import pytest

from nengo_loihi.builder.inputs import ChipReceiveNode, PESModulatoryTarget


def test_chipreceivenode_run_error():
    with nengo.Network() as net:
        ChipReceiveNode(dimensions=1, size_out=1)

    with pytest.raises(SimulationError, match="should not be run"):
        with nengo.Simulator(net) as sim:
            sim.step()


def test_pesmodulatorytarget_interface():
    target = "target"
    p = PESModulatoryTarget(target)

    t0 = 4
    e0 = [1.8, 2.4, 3.3]
    t1 = t0 + 3
    e1 = [7.2, 2.2, 4.1]
    e01 = np.array(e0) + np.array(e1)

    p.receive(t0, e0)
    assert isinstance(p.errors[t0], np.ndarray)
    assert np.allclose(p.errors[t0], e0)

    p.receive(t0, e1)
    assert np.allclose(p.errors[t0], e01)

    with pytest.raises(AssertionError):
        p.receive(t0 - 1, e0)  # time needs to be >= last time

    p.receive(t1, e1)
    assert np.allclose(p.errors[t1], e1)

    errors = list(p.collect_errors())
    assert len(errors) == 2
    assert errors[0][:2] == (target, t0) and np.allclose(errors[0][2], e01)
    assert errors[1][:2] == (target, t1) and np.allclose(errors[1][2], e1)

    p.clear()
    assert len(list(p.collect_errors())) == 0
