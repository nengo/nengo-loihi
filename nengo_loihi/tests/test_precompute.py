import numpy as np

import nengo
import nengo.utils.matplotlib
import pytest


@pytest.mark.skipif(pytest.config.getoption("--target") != "loihi",
                    reason="Loihi only test")
def test_precompute(allclose, Simulator, seed, plt):
    with nengo.Network(seed=seed) as model:
        D = 2
        stim = nengo.Node(lambda t: [np.sin(t * 2 * np.pi)] * D)

        a = nengo.Ensemble(100, D)

        nengo.Connection(stim, a)

        output = nengo.Node(size_in=D)

        nengo.Connection(a, output)

        p_stim = nengo.Probe(stim, synapse=0.03)
        p_a = nengo.Probe(a, synapse=0.03)
        p_out = nengo.Probe(output, synapse=0.03)

    with Simulator(model, precompute=False) as sim1:
        sim1.run(1.0)
    with Simulator(model, precompute=True) as sim2:
        sim2.run(1.0)

    plt.subplot(2, 1, 1)
    plt.plot(sim1.trange(), sim1.data[p_stim])
    plt.plot(sim1.trange(), sim1.data[p_a])
    plt.plot(sim1.trange(), sim1.data[p_out])
    plt.title('precompute=False')
    plt.subplot(2, 1, 2)
    plt.plot(sim2.trange(), sim2.data[p_stim])
    plt.plot(sim2.trange(), sim2.data[p_a])
    plt.plot(sim2.trange(), sim2.data[p_out])
    plt.title('precompute=True')

    assert np.array_equal(sim1.data[p_stim], sim2.data[p_stim])
    assert allclose(sim1.data[p_a], sim2.data[p_a], atol=0.2)
    assert allclose(sim1.data[p_out], sim2.data[p_out], atol=0.2)


@pytest.mark.skipif(pytest.config.getoption("--target") != "loihi",
                    reason="Loihi only test")
@pytest.mark.xfail(pytest.config.getoption("--target") == "loihi",
                   reason="Fails allclose check")
def test_input_node_precompute(allclose, Simulator, plt):
    input_fn = lambda t: np.sin(2 * np.pi * t)
    targets = ["sim", "loihi"]
    x = {}
    u = {}
    v = {}
    for target in targets:
        n = 4
        with nengo.Network(seed=1) as model:
            inp = nengo.Node(input_fn)

            a = nengo.Ensemble(n, 1)
            ap = nengo.Probe(a, synapse=0.01)
            aup = nengo.Probe(a.neurons, 'input')
            avp = nengo.Probe(a.neurons, 'voltage')

            nengo.Connection(inp, a)

        with Simulator(model, precompute=True, target=target) as sim:
            print("Running in {}".format(target))
            sim.run(3.)

        synapse = nengo.synapses.Lowpass(0.03)
        x[target] = synapse.filt(sim.data[ap])

        u[target] = sim.data[aup][:25]
        u[target] = (
            np.round(u[target] * 1000)
            if str(u[target].dtype).startswith('float') else
            u[target])

        v[target] = sim.data[avp][:25]
        v[target] = (
            np.round(v[target] * 1000)
            if str(v[target].dtype).startswith('float') else
            v[target])

        plt.plot(sim.trange(), x[target], label=target)

    t = sim.trange()
    u = input_fn(t)
    plt.plot(t, u, 'k:', label='input')
    plt.legend(loc='best')

    assert allclose(x['sim'], x['loihi'], atol=0.1, rtol=0.01)
