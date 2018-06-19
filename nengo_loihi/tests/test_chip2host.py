import pytest

import numpy as np

import nengo
import nengo.utils.matplotlib


@pytest.mark.parametrize('pre_d', [1, 3])
@pytest.mark.parametrize('post_d', [1, 3])
@pytest.mark.parametrize('func', [False, True])
def test_ens2node(Simulator, seed, plt, pre_d, post_d, func):
    with nengo.Network(seed=seed) as model:
        stim = nengo.Node(lambda t: [np.sin(t * 2 * np.pi)] * pre_d)

        a = nengo.Ensemble(100, pre_d,
                           max_rates=nengo.dists.Uniform(100, 120),
                           intercepts=nengo.dists.Uniform(-0.5, 0.5))

        nengo.Connection(stim, a)

        data = []
        output = nengo.Node(lambda t, x: data.append(x),
                            size_in=post_d, size_out=0)

        transform = np.identity(max(pre_d, post_d))
        transform = transform[:post_d, :pre_d]
        if func:
            def conn_func(x):
                return -x
        else:
            conn_func = None
        nengo.Connection(a, output, transform=transform, function=conn_func)

        p_stim = nengo.Probe(stim)

    with Simulator(model) as sim:
        sim.run(1.0)

    filt = nengo.synapses.Lowpass(0.03)
    filt_data = filt.filt(np.array(data))

    # TODO: improve the bounds on these tests
    if post_d >= pre_d:
        assert np.allclose(
            filt_data[:, :pre_d] * (-1 if func else 1),
            sim.data[p_stim][:, :pre_d],
            atol=0.6)
        assert np.allclose(filt_data[:, pre_d:], 0, atol=0.6)
    else:
        assert np.allclose(
            filt_data * (-1 if func else 1),
            sim.data[p_stim][:, :post_d],
            atol=0.6)

    plt.subplot(2, 1, 1)
    plt.plot(sim.trange(), sim.data[p_stim])
    plt.title('Input (should be %d sine waves)' % pre_d)
    plt.legend(['%d' % i for i in range(pre_d)], loc='best')
    plt.subplot(2, 1, 2)
    n_sine = min(pre_d, post_d)
    status = ' flipped' if func else ''
    n_flat = post_d - n_sine
    plt.title('Output (should be %d%s sine waves and %d flat lines)' %
              (n_sine, status, n_flat))
    plt.plot(sim.trange(), filt_data)
    plt.legend(['%d' % i for i in range(post_d)], loc='best')


def test_neurons2node(Simulator, seed, plt):
    with nengo.Network(seed=seed) as model:
        stim = nengo.Node(lambda t: [np.sin(t * 2 * np.pi)])
        p_stim = nengo.Probe(stim)

        a = nengo.Ensemble(100, 1,
                           max_rates=nengo.dists.Uniform(100, 120),
                           intercepts=nengo.dists.Choice([0]))

        nengo.Connection(stim, a)

        data = []
        output = nengo.Node(lambda t, x: data.append(x),
                            size_in=a.n_neurons, size_out=0)
        nengo.Connection(a.neurons, output, synapse=None)

    with Simulator(model) as sim:
        sim.run(1.0)

    nengo.utils.matplotlib.rasterplot(sim.trange(), np.array(data),
                                      ax=plt.gca())
    plt.twinx()
    plt.plot(sim.trange(), sim.data[p_stim])
    plt.title('Raster plot for sine input')

    pre = np.asarray(data[:len(data) // 2 - 100])
    post = np.asarray(data[len(data) // 2 + 100:])
    on_neurons = np.squeeze(sim.data[a].encoders == 1)
    assert np.sum(pre[:, on_neurons]) > 0
    assert np.sum(post[:, on_neurons]) == 0
    assert np.sum(pre[:, np.logical_not(on_neurons)]) == 0
    assert np.sum(post[:, np.logical_not(on_neurons)]) > 0
