import numpy as np

import nengo
import pytest

import nengo_loihi
import nengo_loihi.splitter as splitter


def test_interneuron_structures():
    D = 2
    radius = 2.0
    with nengo.Network() as model:
        nengo_loihi.add_params(model)

        stim = nengo.Node(np.eye(D)[0])
        ens = nengo.Ensemble(n_neurons=10, dimensions=D, radius=radius)

        def conn_func(x):
            return x
        solver = nengo.solvers.NoSolver(None)
        synapse = nengo.synapses.Lowpass(0.1)
        transform = np.random.uniform(-1, 1, (D,D))
        nengo.Connection(stim, ens,
                         function=conn_func,
                         solver=solver,
                         synapse=synapse,
                         transform=transform)

    inter_rate = 1000
    inter_n = 1

    host, chip, _, _, _ = splitter.split(model, inter_rate, inter_n,
                                  spiking_interneurons_on_host=True)

    assert len(host.all_ensembles) == 1
    assert len(host.all_connections) == 2
    conn = host.connections[0]
    assert conn.pre is stim
    assert conn.function is conn_func
    assert conn.solver is solver
    assert conn.synapse is None   # TODO: should this be synapse?
    assert np.allclose(conn.transform, transform / radius)

    host, chip, _, _, _ = splitter.split(model, inter_rate, inter_n,
                                     spiking_interneurons_on_host=False)

    assert len(host.all_ensembles) == 0
    assert len(host.all_connections) == 1
    conn = host.connections[0]
    assert conn.pre is stim
    assert conn.function is conn_func
    assert conn.solver is solver
    assert conn.synapse is synapse   # TODO: should this be None?
    assert np.allclose(conn.transform, transform / radius)


def test_no_interneuron_input():
    synapse = 0.1
    with nengo.Network() as model:
        nengo_loihi.add_params(model)

        stim = nengo.Node(np.sin)
        ens = nengo.Ensemble(n_neurons=1, dimensions=1)
        nengo.Connection(stim, ens, synapse=synapse)
        probe = nengo.Probe(stim, synapse=synapse)

    host, chip, h2c, _, _ = splitter.split(model, inter_rate=1000, inter_n=1,
                                           spiking_interneurons_on_host=False)

    assert len(h2c) == 1
    sender, receiver = list(h2c.items())[0]

    with nengo.Simulator(host) as sim:
        sim.run(1.0)

    assert np.allclose(sim.trange(), [q[0] for q in sender.queue])
    assert np.allclose(sim.data[probe], [q[1] for q in sender.queue])


@pytest.mark.skipif(pytest.config.getoption("--target") != "loihi",
                    reason="Loihi only test")
def test_no_interneurons_running(Simulator, seed, plt):

    synapse = 0.1
    with nengo.Network() as model:
        stim = nengo.Node(lambda t: 0 if t%1.0 < 0.5 else 1)
        ens = nengo.Ensemble(n_neurons=100, dimensions=1)
        nengo.Connection(stim, ens, synapse=synapse)
        p_stim = nengo.Probe(stim)
        p_neurons = nengo.Probe(ens.neurons)
    with Simulator(model, precompute=False) as sim:
        sim.run(2.0)

    nengo.utils.matplotlib.rasterplot(sim.trange(), sim.data[p_neurons])
    plt.plot(sim.trange(), sim.data[p_stim])






