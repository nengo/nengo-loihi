import pytest
import nengo
import numpy as np


@pytest.mark.parametrize('D', [1, 2, 3])
def test_node_ens(Simulator, seed, plt, D):
    with nengo.Network(seed=seed) as model:
        stim = nengo.Node(lambda t: [np.sin(t*2*np.pi)]*D)

        a = nengo.Ensemble(100*D, D,
                           max_rates=nengo.dists.Uniform(100, 120),
                           intercepts=nengo.dists.Uniform(-0.5, 0.5))
        nengo.Connection(stim, a)

        p_stim = nengo.Probe(stim, synapse=0.1)
        p_a = nengo.Probe(a, synapse=0.1)
    with Simulator(model, max_time=1.0) as sim:
        sim.run(1.0)

    plt.subplot(2, 1, 1)
    plt.plot(sim.trange(), sim.data[p_stim])
    plt.title('Input (should be %d sine waves)' % D)
    plt.legend(['%d' % i for i in range(D)], loc='best')
    plt.subplot(2, 1, 2)
    plt.title('Output (should be %d sine waves)' % D)
    plt.plot(sim.trange(), sim.data[p_a])
    plt.legend(['%d' % i for i in range(D)], loc='best')

    assert np.allclose(sim.data[p_stim],
                       sim.data[p_a], atol=0.3)
