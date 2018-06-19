import nengo
import numpy as np
import pytest


@pytest.mark.parametrize("val", (-0.75, -0.5, 0, 0.5, 0.75))
@pytest.mark.parametrize("type", ("array", "func"))
def test_input_node(Simulator, val, type):
    with nengo.Network() as net:
        if type == "array":
            input = [val]
        else:
            input = lambda t: [val]
        a = nengo.Node(input)

        b = nengo.Ensemble(100, 1, max_rates=nengo.dists.Uniform(100, 120),
                           intercepts=nengo.dists.Uniform(-0.5, 0.5))
        nengo.Connection(a, b)

        # create a second path so that we test nodes with multiple outputs
        c = nengo.Ensemble(100, 1, max_rates=nengo.dists.Uniform(100, 120),
                           intercepts=nengo.dists.Uniform(-0.5, 0.5))
        nengo. Connection(a, c)

        p_b = nengo.Probe(b, synapse=0.1)
        p_c = nengo.Probe(c, synapse=0.1)

    with Simulator(net, max_time=1.0) as sim:
        sim.run(1.0)

        # TODO: seems like error margins should be smaller than this?
        assert np.allclose(sim.data[p_b][-100:], val, atol=0.15)
        assert np.allclose(sim.data[p_c][-100:], val, atol=0.15)
