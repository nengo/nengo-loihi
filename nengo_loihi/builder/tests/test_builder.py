import nengo
import numpy as np
import pytest
from nengo.exceptions import BuildError

import nengo_loihi.builder.probe
from nengo_loihi.builder import Model
from nengo_loihi.builder.ensemble import get_gain_bias


@pytest.mark.parametrize(
    "passed_intercepts", [nengo.dists.Uniform(-1, 1), np.linspace(-1, 1, 10000)]
)
def test_intercept_limit(passed_intercepts, rng):
    model = Model()
    assert model.intercept_limit == 0.95

    ens = nengo.Ensemble(10000, 1, intercepts=passed_intercepts, add_to_container=False)
    with pytest.warns(UserWarning):
        _, _, _, intercepts = get_gain_bias(ens, rng, model.intercept_limit)
    assert np.all(intercepts <= model.intercept_limit)


def test_build_callback(Simulator):
    with nengo.Network() as net:
        a = nengo.Ensemble(3, 1)
        b = nengo.Ensemble(3, 1)
        c = nengo.Connection(a, b)

    objs = []

    def build_callback(obj):
        objs.append(obj)

    model = Model()
    model.build_callback = build_callback
    with Simulator(net, model=model):
        pass

    for obj in (a, b, c):
        assert obj in objs, "%s not in objs" % obj


def test_probemap_bad_type_error(Simulator, monkeypatch):
    with nengo.Network() as net:
        a = nengo.Ensemble(2, 1)
        nengo.Probe(a)

    # need to monkeypatch it so Ensemble is not in probemap, since all types
    # not in probemap are caught in validation when creating the probe
    monkeypatch.setattr(nengo_loihi.builder.probe, "probemap", {})
    with pytest.raises(BuildError, match="not probeable"):
        with Simulator(net):
            pass


def test_builder_strings():
    model = Model(label="myModel")
    assert str(model) == "Model(myModel)"
