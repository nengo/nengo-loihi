import nengo
import pytest


def test_stopping(Simulator, seed):
    model = nengo.Network(seed=seed)
    with model:
        stim = nengo.Node(0.5)
        ens = nengo.Ensemble(n_neurons=100, dimensions=1)
        nengo.Connection(stim, ens)

        def output_func(t, x):
            if t > 0.05:
                raise Exception('Stopping')
        output = nengo.Node(output_func, size_in=1)
        nengo.Connection(ens, output)

    with Simulator(model, precompute=False) as sim:
        with pytest.raises(Exception):
            sim.run(0.1)


def test_closing(Simulator, seed):
    model = nengo.Network(seed=seed)
    with model:
        stim = nengo.Node(0.5)
        ens = nengo.Ensemble(n_neurons=100, dimensions=1)
        nengo.Connection(stim, ens)

        def output_func(t, x):
            if t > 0.05:
                sim.close()
        output = nengo.Node(output_func, size_in=1)
        nengo.Connection(ens, output)

    sim = Simulator(model, precompute=False)

    sim.run(0.1)
    assert sim.n_steps == 51
