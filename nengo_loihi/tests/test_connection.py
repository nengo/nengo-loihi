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
        a = nengo.Ensemble(100, 1, label='b',
                           max_rates=nengo.dists.Uniform(100, 120),
                           intercepts=nengo.dists.Uniform(-0.5, 0.5))

        b = nengo.Ensemble(101, 1, label='b',
                           max_rates=nengo.dists.Uniform(100, 120),
                           intercepts=nengo.dists.Uniform(-0.5, 0.5))
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
