import nengo
import numpy as np
import pytest

from nengo_loihi.builder import Model
from nengo_loihi.decode_neurons import (
    DecodeNeurons,
    NoisyDecodeNeurons,
    OnOffDecodeNeurons,
    Preset5DecodeNeurons,
    Preset10DecodeNeurons,
)


@pytest.mark.parametrize(
    "decode_neurons, tolerance",
    [
        (OnOffDecodeNeurons(), 0.35),
        (NoisyDecodeNeurons(5), 0.12),
        (NoisyDecodeNeurons(10), 0.11),
        (Preset5DecodeNeurons(), 0.06),
        (Preset10DecodeNeurons(), 0.03),
    ],
)
def test_add_inputs(decode_neurons, tolerance, Simulator, seed, plt):
    sim_time = 2.0
    pres_time = sim_time / 4
    eval_time = sim_time / 8

    stim_values = [[0.5, 0.5], [0.5, -0.9], [-0.7, -0.3], [-0.3, 1.0]]
    stim_times = np.arange(0, sim_time, pres_time)
    stim_fn_a = nengo.processes.Piecewise(
        {t: stim_values[i][0] for i, t in enumerate(stim_times)}
    )
    stim_fn_b = nengo.processes.Piecewise(
        {t: stim_values[i][1] for i, t in enumerate(stim_times)}
    )

    with nengo.Network(seed=seed) as model:
        stim_a = nengo.Node(stim_fn_a)
        stim_b = nengo.Node(stim_fn_b)

        a = nengo.Ensemble(n_neurons=100, dimensions=1)
        b = nengo.Ensemble(n_neurons=100, dimensions=1)

        nengo.Connection(stim_a, a)
        nengo.Connection(stim_b, b)

        c = nengo.Ensemble(n_neurons=100, dimensions=1)
        nengo.Connection(a, c)
        nengo.Connection(b, c)

        out_synapse = nengo.Alpha(0.03)
        stim_synapse = out_synapse.combine(nengo.Alpha(0.005)).combine(
            nengo.Alpha(0.005)
        )
        p_stim_a = nengo.Probe(stim_a, synapse=stim_synapse)
        p_stim_b = nengo.Probe(stim_b, synapse=stim_synapse)
        p_c = nengo.Probe(c, synapse=out_synapse)

    build_model = Model()
    build_model.decode_neurons = decode_neurons

    with Simulator(model, model=build_model) as sim:
        sim.run(sim_time)

    t = sim.trange()
    tmask = np.zeros(t.shape, dtype=bool)
    for pres_t in np.arange(0, sim_time, pres_time):
        t0 = pres_t + pres_time - eval_time
        t1 = pres_t + pres_time
        tmask |= (t >= t0) & (t <= t1)

    target = sim.data[p_stim_a] + sim.data[p_stim_b]
    error = np.abs(sim.data[p_c][tmask] - target[tmask]).mean()

    plt.plot(t, target)
    plt.plot(t, sim.data[p_c])
    plt.ylim([-1.1, 1.1])
    plt.title("error = %0.2e" % error)

    assert error < tolerance


@pytest.mark.parametrize("decode_neurons, tolerance", [(OnOffDecodeNeurons(), 0.01)])
def test_node_neurons(decode_neurons, tolerance, Simulator, seed, plt):
    sim_time = 0.2

    stim_fn = lambda t: 0.9 * np.sin(2 * np.pi * t / sim_time)
    out_synapse = nengo.Alpha(0.03)
    stim_synapse = out_synapse.combine(nengo.Alpha(0.005))

    with nengo.Network(seed=seed) as model:
        stim = nengo.Node(stim_fn)
        a = nengo.Ensemble(n_neurons=100, dimensions=1)
        nengo.Connection(stim, a)

        p_stim = nengo.Probe(stim, synapse=stim_synapse)
        p_a = nengo.Probe(a, synapse=out_synapse)

    build_model = Model()
    build_model.node_neurons = decode_neurons

    with Simulator(model, model=build_model) as sim:
        sim.run(sim_time)

    t = sim.trange()
    target = sim.data[p_stim]
    error = np.abs(sim.data[p_a] - target).mean()

    plt.plot(t, target)
    plt.plot(t, sim.data[p_a])
    plt.ylim([-1.1, 1.1])
    plt.title("error = %0.2e" % error)

    assert error < tolerance


def test_decode_neuron_str():
    assert str(DecodeNeurons(dt=0.005)) == "DecodeNeurons(dt=0.005)"
    assert str(OnOffDecodeNeurons(pairs_per_dim=2, dt=0.002, rate=None)) == (
        "OnOffDecodeNeurons(pairs_per_dim=2, dt=0.002, rate=250)"
    )
    assert str(NoisyDecodeNeurons(1, rate=20)) == (
        "NoisyDecodeNeurons(pairs_per_dim=1, dt=0.001, rate=20, noise_exp=-2)"
    )
    assert str(Preset5DecodeNeurons()) == ("Preset5DecodeNeurons(dt=0.001, rate=200)")
    assert str(Preset10DecodeNeurons(dt=0.0001, rate=0.5)) == (
        "Preset10DecodeNeurons(dt=0.0001, rate=0.5)"
    )
