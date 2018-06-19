import pytest

import numpy as np

import nengo


@pytest.mark.parametrize('D1', [1, 2, 3])
@pytest.mark.parametrize('D2', [1, 2, 3])
@pytest.mark.parametrize('func', [False, True])
def test_node2ens(Simulator, seed, plt, D1, D2, func):
    with nengo.Network(seed=seed) as model:
        stim = nengo.Node(lambda t: [np.sin(t*2*np.pi)]*D1)

        a = nengo.Ensemble(100, D2,
                           max_rates=nengo.dists.Uniform(100, 120),
                           intercepts=nengo.dists.Uniform(-0.5, 0.5))

        transform = np.identity(max(D1, D2))
        transform = transform[:D2, :D1]
        if func:
            def function(x):
                return -x
        else:
            function = None
        nengo.Connection(stim, a,
                         transform=transform,
                         function=function)

        p_stim = nengo.Probe(stim)
        p_a = nengo.Probe(a, synapse=0.03)

    with Simulator(model) as sim:
        sim.run(1.0)

    if func:
        if D2 >= D1:
            assert np.allclose(
                sim.data[p_stim][:, :D1],
                -sim.data[p_a][:, :D1],
                atol=0.6)
            assert np.allclose(
                np.zeros((len(sim.data[p_stim]), D2-D1)),
                -sim.data[p_a][:, D1:D2],
                atol=0.6)
        else:
            assert(np.allclose(
                sim.data[p_stim][:, :D2],
                -sim.data[p_a],
                atol=0.6))
    else:
        if D2 >= D1:
            assert np.allclose(
                sim.data[p_stim][:, :D1],
                sim.data[p_a][:, :D1],
                atol=0.6)
            assert np.allclose(
                np.zeros((len(sim.data[p_stim]), D2-D1)),
                sim.data[p_a][:, D1:D2],
                atol=0.6)
        else:
            assert np.allclose(
                sim.data[p_stim][:, :D2],
                sim.data[p_a],
                atol=0.6)

    plt.subplot(2, 1, 1)
    plt.plot(sim.trange(), sim.data[p_stim])
    plt.title('Input (should be %d sine waves)' % D1)
    plt.legend(['%d' % i for i in range(D1)], loc='best')
    plt.subplot(2, 1, 2)
    n_sine = min(D1, D2)
    status = ' flipped' if func else ''
    n_flat = D2 - n_sine
    plt.title('Output (should be %d%s sine waves and %d flat lines)' %
              (n_sine, status, n_flat))
    plt.plot(sim.trange(), sim.data[p_a])
    plt.legend(['%d' % i for i in range(D2)], loc='best')
