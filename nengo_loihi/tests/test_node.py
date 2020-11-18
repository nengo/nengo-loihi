import nengo
import numpy as np
import pytest


@pytest.mark.filterwarnings("ignore:Model is precomputable.")
@pytest.mark.parametrize("precompute", [True, False])
def test_multiple_stims(allclose, Simulator, seed, precompute):
    with nengo.Network(seed=seed) as model:
        N = 10
        a = nengo.Ensemble(N, 1, seed=seed)
        b = nengo.Ensemble(N, 1, seed=seed)
        nengo.Connection(nengo.Node(0.5), a)
        nengo.Connection(nengo.Node(0.5), b)

        p_a = nengo.Probe(a)
        p_b = nengo.Probe(b)
    with Simulator(model, precompute=precompute) as sim:
        sim.run(0.1)

    # Note: these should ideally be identical,
    #  but noise in the spiking DecodeNeurons will
    #  make them different.  If spiking decoders are
    #  implemented, then these should be identical.
    assert allclose(np.mean(sim.data[p_a]), np.mean(sim.data[p_b]), atol=0.05)
