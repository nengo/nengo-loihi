import nengo
import numpy as np
import pytest

from nengo_loihi.builder import get_gain_bias, Model


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
