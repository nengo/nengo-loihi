import nengo
from nengo.utils.matplotlib import rasterplot
import numpy as np
import pytest


# This test sometimes (but not consistently) fails on the chip for various
# combinations of the parameter values. This possibly has to do with
# interneuron noise and not representing the values well.
@pytest.mark.xfail
@pytest.mark.parametrize("val", (-0.75, -0.5, 0, 0.5, 0.75))
@pytest.mark.parametrize("type", ("array", "func"))
def test_input_node(allclose, Simulator, val, type):
    with nengo.Network() as net:
        if type == "array":
            input = [val]
        else:
            input = lambda t: [val]
        a = nengo.Node(input)

        b = nengo.Ensemble(100, 1)
        nengo.Connection(a, b)

        # create a second path so that we test nodes with multiple outputs
        c = nengo.Ensemble(100, 1)
        nengo.Connection(a, c)

        p_b = nengo.Probe(b, synapse=0.1)
        p_c = nengo.Probe(c, synapse=0.1)

    with Simulator(net, precompute=True) as sim:
        sim.run(1.0)

    # TODO: seems like error margins should be smaller than this?
    assert allclose(sim.data[p_b][-100:], val, atol=0.15)
    assert allclose(sim.data[p_c][-100:], val, atol=0.15)


@pytest.mark.parametrize('pre_d', [1, 3])
@pytest.mark.parametrize('post_d', [1, 3])
@pytest.mark.parametrize('func', [False, True])
def test_ens2node(allclose, Simulator, seed, plt, pre_d, post_d, func):
    with nengo.Network(seed=seed) as model:
        stim = nengo.Node(lambda t: [np.sin(t * 2 * np.pi)] * pre_d)

        a = nengo.Ensemble(100, pre_d)

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
        assert allclose(
            filt_data[:, :pre_d] * (-1 if func else 1),
            sim.data[p_stim][:, :pre_d],
            atol=0.6)
        assert allclose(filt_data[:, pre_d:], 0, atol=0.6)
    else:
        assert allclose(
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
                           intercepts=nengo.dists.Choice([0]))

        nengo.Connection(stim, a)

        data = []
        output = nengo.Node(lambda t, x: data.append(x),
                            size_in=a.n_neurons, size_out=0)
        nengo.Connection(a.neurons, output, synapse=None)

    with Simulator(model) as sim:
        sim.run(1.0)

    rasterplot(sim.trange(), np.array(data), ax=plt.gca())
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


@pytest.mark.parametrize('pre_d', [1, 3])
@pytest.mark.parametrize('post_d', [1, 3])
@pytest.mark.parametrize('func', [False, True])
def test_node2ens(allclose, Simulator, seed, plt, pre_d, post_d, func):
    with nengo.Network(seed=seed) as model:
        stim = nengo.Node(lambda t: [np.sin(t * 2 * np.pi)] * pre_d)

        a = nengo.Ensemble(100, post_d)

        transform = np.identity(max(pre_d, post_d))
        transform = transform[:post_d, :pre_d]
        if func:
            def function(x):
                return -x
        else:
            function = None
        nengo.Connection(stim, a, transform=transform, function=function)

        p_stim = nengo.Probe(stim)
        p_a = nengo.Probe(a, synapse=0.03)

    with Simulator(model) as sim:
        sim.run(1.0)

    # TODO: improve the bounds on these tests
    if post_d >= pre_d:
        assert allclose(
            sim.data[p_stim][:, :pre_d],
            sim.data[p_a][:, :pre_d] * (-1 if func else 1),
            atol=0.6)
        assert allclose(
            sim.data[p_a][:, pre_d:] * (-1 if func else 1), 0, atol=0.6)
    else:
        assert allclose(
            sim.data[p_stim][:, :post_d],
            sim.data[p_a] * (-1 if func else 1),
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
    plt.plot(sim.trange(), sim.data[p_a])
    plt.legend(['%d' % i for i in range(post_d)], loc='best')
