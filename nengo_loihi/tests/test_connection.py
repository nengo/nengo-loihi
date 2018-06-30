import nengo
import numpy as np
import pytest


@pytest.mark.parametrize('weight_solver', [False, True])
@pytest.mark.parametrize('target_value', [-0.75, 0.4, 1.0])
def test_ens_ens_constant(weight_solver, target_value, Simulator, seed, plt):
    a_fn = lambda x: x + target_value
    solver = nengo.solvers.LstsqL2(weights=weight_solver)

    bnp = None
    with nengo.Network(seed=seed) as model:
        a = nengo.Ensemble(100, 1, label='a')

        b = nengo.Ensemble(101, 1, label='b')
        nengo.Connection(a, b, function=a_fn, solver=solver)
        bp = nengo.Probe(b)
        bnp = nengo.Probe(b.neurons)

        c = nengo.Ensemble(1, 1, label='c')
        bc_conn = nengo.Connection(b, c)

    with Simulator(model) as sim:
        sim.run(1.0)

    t = sim.trange()
    bcount = (sim.data[bnp] > 0).mean(axis=0)

    b_decoders = sim.data[bc_conn].weights
    dec_value = np.dot(b_decoders, bcount)

    output_filter = nengo.synapses.Alpha(0.03)
    target_output = target_value * np.ones_like(t)
    sim_output = output_filter.filt(sim.data[bp])
    plt.plot(t, target_output, 'k')
    plt.plot(t, sim_output)

    assert np.allclose(dec_value, target_value, rtol=0.1, atol=0.1)
    t_check = t > 0.5
    assert np.allclose(sim_output[t_check], target_output[t_check],
                       rtol=0.15, atol=0.15)


@pytest.mark.parametrize('precompute', [True, False])
def test_node_to_neurons(precompute, Simulator, plt):
    tfinal = 1.0

    x = np.array([0.7, 0.3])
    A = np.array([[1, 1],
                  [1, -1],
                  [1, -0.5]])
    y = np.dot(A, x)

    gain = [3] * len(y)
    bias = [0] * len(y)

    neuron_type = nengo.LIF()
    z = neuron_type.rates(y, gain, bias)

    with nengo.Network() as model:
        u = nengo.Node(x, label='u')
        a = nengo.Ensemble(len(y), 1, label='a',
                           neuron_type=neuron_type, gain=gain, bias=bias)
        ap = nengo.Probe(a.neurons)
        nengo.Connection(u, a.neurons, synapse=None, transform=A)

    with Simulator(model, precompute=precompute) as sim:
        sim.run(tfinal)

    tsum = 0.5
    t = sim.trange()
    rates = (sim.data[ap][t > t[-1] - tsum] > 0).sum(axis=0) / tsum

    bar_width = 0.35
    plt.bar(np.arange(len(z)), z, bar_width, color='k', label='z')
    plt.bar(np.arange(len(z)) + bar_width, rates, bar_width, label='rates')
    plt.legend(loc='best')

    assert np.allclose(rates, z, atol=3, rtol=0.1)
