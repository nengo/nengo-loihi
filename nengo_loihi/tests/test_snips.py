import nengo
import pytest


@pytest.mark.skipif(pytest.config.getoption("--target") != "loihi",
                    reason="snips only exist on loihi")
def test_snip_input_count(Simulator, seed, plt):
    with nengo.Network(seed=seed) as model:
        a = nengo.Ensemble(100, 1)
        for i in range(30):
            stim = nengo.Node(0.5)
            nengo.Connection(stim, a, synapse=None)
    with Simulator(model) as sim:
        with pytest.warns(UserWarning, match="Too many spikes"):
            sim.run(0.01)
