import nengo
import numpy as np
import pytest

from nengo_loihi.builder import (
    DecodeNeurons,
    get_gain_bias,
    Model,
    NoisyDecodeNeurons,
    OnOffDecodeNeurons,
    Preset5DecodeNeurons,
    Preset10DecodeNeurons,
)


@pytest.mark.parametrize("passed_intercepts", [
    nengo.dists.Uniform(-1, 1), np.linspace(-1, 1, 10000),
])
def test_intercept_limit(passed_intercepts, rng):
    model = Model()
    assert model.intercept_limit == 0.95

    ens = nengo.Ensemble(10000, 1,
                         intercepts=passed_intercepts,
                         add_to_container=False)
    with pytest.warns(UserWarning):
        _, _, _, intercepts = get_gain_bias(ens, rng, model.intercept_limit)
    assert np.all(intercepts <= model.intercept_limit)


def test_decode_neuron_str():
    assert str(DecodeNeurons(dt=0.005)) == "DecodeNeurons(dt=0.005)"
    assert str(OnOffDecodeNeurons(pairs_per_dim=2, dt=0.002, rate=None)) == (
        "OnOffDecodeNeurons(pairs_per_dim=2, dt=0.002, rate=250)")
    assert str(NoisyDecodeNeurons(1, rate=20)) == (
        "NoisyDecodeNeurons(pairs_per_dim=1, dt=0.001, rate=20, noise_exp=-2)")
    assert str(Preset5DecodeNeurons()) == (
        "Preset5DecodeNeurons(dt=0.001, rate=200)")
    assert str(Preset10DecodeNeurons(dt=0.0001, rate=0.5)) == (
        "Preset10DecodeNeurons(dt=0.0001, rate=0.5)")
