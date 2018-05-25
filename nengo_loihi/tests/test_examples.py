import pytest

import numpy as np

import nengo


@pytest.mark.parametrize('weights', [False, True])
def test_ens_ens(Simulator, seed, plt, weights):
    solver = nengo.solvers.LstsqL2(weights=weights)

    with nengo.Network(seed=seed) as model:
        a = nengo.Ensemble(100, 1, label='a',
                           max_rates=nengo.dists.Uniform(100, 120),
                           intercepts=nengo.dists.Uniform(-0.5, 0.5))
        ap = nengo.Probe(a)

        b = nengo.Ensemble(101, 1, label='b',
                           max_rates=nengo.dists.Uniform(100, 120),
                           intercepts=nengo.dists.Uniform(-0.5, 0.5))
        nengo.Connection(a, b, function=lambda x: x + 0.5, solver=solver)
        bp = nengo.Probe(b)

    with Simulator(model) as sim:
        sim.run(1.0)

    output_filter = nengo.synapses.Alpha(0.02)
    plt.plot(sim.trange(), output_filter.filtfilt(sim.data[ap]))
    plt.plot(sim.trange(), output_filter.filtfilt(sim.data[bp]))


# def test_node_ens_ens(Simulator, plt):
#     with nengo.Network(seed=1) as model:
#         u = nengo.Node(output=0.5)
#         up = nengo.Probe(u)

#         a = nengo.Ensemble(100, 1, label='a',
#                            max_rates=nengo.dists.Uniform(100, 120),
#                            intercepts=nengo.dists.Uniform(-0.5, 0.5))
#         nengo.Connection(u, a, synapse=None)
#         ap = nengo.Probe(a)

#         b = nengo.Ensemble(101, 1, label='b',
#                            max_rates=nengo.dists.Uniform(100, 120),
#                            intercepts=nengo.dists.Uniform(-0.5, 0.5))
#         nengo.Connection(a, b)
#         bp = nengo.Probe(b)

#     with Simulator(model) as sim:
#         sim.run(0.5)

#     output_filter = nengo.synapses.Alpha(0.02)
#     plt.plot(sim.trange(), output_filter.filtfilt(sim.data[up]))
#     plt.plot(sim.trange(), output_filter.filtfilt(sim.data[ap]))
#     plt.plot(sim.trange(), output_filter.filtfilt(sim.data[bp]))


@pytest.mark.parametrize('weights', [False, True])
def test_oscillator(Simulator, seed, plt, weights):
    solver = nengo.solvers.LstsqL2(weights=weights)
    tau = 0.1
    alpha = 1.0

    def f(x):
        x0, x1 = x
        r = np.sqrt(x0**2 + x1**2)
        a = np.arctan2(x1, x0)
        dr = -(r - 1)
        da = alpha
        r = r + tau*dr
        a = a + tau*da
        return [r*np.cos(a), r*np.sin(a)]

    with nengo.Network(seed=seed) as model:
        a = nengo.Ensemble(200, 2, label='a',
                           max_rates=nengo.dists.Uniform(100, 120),
                           intercepts=nengo.dists.Uniform(-0.5, 0.5))
        ap = nengo.Probe(a)

        nengo.Connection(a, a, function=f, synapse=tau, solver=solver)

    with Simulator(model) as sim:
        sim.run(4.0)

    output_filter = nengo.synapses.Alpha(0.02)
    plt.plot(sim.trange(), output_filter.filtfilt(sim.data[ap]))
