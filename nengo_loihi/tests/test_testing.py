import pytest


def test_seed(request, seed):
    if request.config.getoption("seed_offset"):
        pytest.skip()
    assert seed == 2004919645
