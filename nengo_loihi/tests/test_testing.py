import pytest
import numpy as np


def test_seed(request, seed):
    if request.config.getoption("seed_offset"):
        pytest.skip()
    assert seed == 2004919645


def test_allclose(allclose, capsys):
    assert not allclose(0, 1)
    assert allclose(0, 1, rtol=1)
    assert allclose(0, 1, atol=1)
    assert allclose(0, 2, rtol=0.5, atol=1)
    assert allclose(np.arange(4), np.roll(np.arange(4), 2), xtol=2)
    assert allclose(np.arange(4), np.roll(np.arange(4), -2), xtol=2)
    assert allclose(np.arange(4), np.roll(np.arange(4), 1), xtol=2)
    assert allclose(np.arange(4), np.roll(np.arange(4), -1), xtol=2)
    assert not allclose(np.arange(4), np.roll(np.arange(4), 2), xtol=1)
    assert not allclose(np.nan, np.nan)
    assert allclose(np.nan, np.nan, equal_nan=True)
    capsys.readouterr()
    assert not allclose(np.arange(4), np.ones(4))
    assert capsys.readouterr().out == ("allclose first 5 failures:\n"
                                       "  (0,): 0 1.0\n"
                                       "  (2,): 2 1.0\n"
                                       "  (3,): 3 1.0\n")
    assert not allclose(np.arange(4), np.ones(4), print_fail=0)
    assert capsys.readouterr().out == ""
    assert allclose(np.arange(4), np.arange(4), print_fail=5)
    assert capsys.readouterr().out == ""
