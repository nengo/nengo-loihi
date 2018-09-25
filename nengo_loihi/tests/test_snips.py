import nengo
import pytest
import numpy as np


@pytest.mark.skipif(pytest.config.getoption("--target") != "loihi",
                    reason="snips only exist on loihi")
def test_snip_input_count(Simulator, seed, plt):
    with nengo.Network(seed=seed) as model:
        a = nengo.Ensemble(100, 1)
        for i in range(30):
            stim = nengo.Node(0.5)
            nengo.Connection(stim, a, synapse=None)
    with Simulator(model, precompute=False) as sim:
        with pytest.warns(UserWarning, match="Too many spikes"):
            sim.run(0.01)


@pytest.mark.skipif(pytest.config.getoption("--target") != "loihi",
                    reason="snips only exist on loihi")
@pytest.mark.parametrize("snip_io_steps", [1, 10])
def test_snip_skipping(Simulator, seed, plt, snip_io_steps):
    dt = 0.001
    with nengo.Network(seed=seed) as model:
        a = nengo.Ensemble(200, 1)

        def stim_func(t):
            step = int(t/dt)
            if step % snip_io_steps == 1 % snip_io_steps:
                return 0
            else:
                return 1
        stim = nengo.Node(stim_func)
        nengo.Connection(stim, a, synapse=None)
        output = nengo.Node(None, 1)
        nengo.Connection(a, output, synapse=0.1)
        p_output = nengo.Probe(output)
        p_a = nengo.Probe(a, synapse=0.1)

    with Simulator(model, dt=dt, precompute=False,
                   snip_io_steps=snip_io_steps) as sim:
        sim.run(1.0)

    assert np.allclose(sim.data[p_a], 0, atol=0.1)
    assert np.allclose(sim.data[p_output], 0, atol=0.1)
