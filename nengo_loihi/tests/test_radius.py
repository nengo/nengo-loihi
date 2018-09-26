import nengo
import numpy as np
import pytest


@pytest.mark.parametrize('radius', [0.01, 1, 100])
def test_radius_probe(Simulator, seed, radius):
    with nengo.Network(seed=seed) as model:
        stim = nengo.Node(radius / 2.)
        ens = nengo.Ensemble(n_neurons=100, dimensions=1,
                             radius=radius,
                             intercepts=nengo.dists.Uniform(-0.95, 0.95))
        nengo.Connection(stim, ens)
        p = nengo.Probe(ens, synapse=0.1)
    with Simulator(model, precompute=True) as sim:
        sim.run(0.5)

    assert np.allclose(sim.data[p][-1:], radius / 2., rtol=0.1)


@pytest.mark.parametrize('radius1', [0.01, 100])
@pytest.mark.parametrize('radius2', [0.01, 100])
@pytest.mark.parametrize('weights', [True, False])
def test_radius_ens_ens(Simulator, seed, radius1, radius2, weights):
    with nengo.Network(seed=seed) as model:
        stim = nengo.Node(radius1 / 2.)
        a = nengo.Ensemble(n_neurons=100, dimensions=1,
                           radius=radius1,
                           intercepts=nengo.dists.Uniform(-0.95, 0.95))
        b = nengo.Ensemble(n_neurons=100, dimensions=1,
                           radius=radius2,
                           intercepts=nengo.dists.Uniform(-0.95, 0.95))
        nengo.Connection(stim, a)
        nengo.Connection(a, b, synapse=0.01, transform=radius2 / radius1,
                         solver=nengo.solvers.LstsqL2(weights=weights))
        p = nengo.Probe(b, synapse=0.1)
    with Simulator(model, precompute=True) as sim:
        sim.run(0.4)

    assert np.allclose(sim.data[p][-1:], radius2 / 2., rtol=0.2)
