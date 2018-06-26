from nengo.conftest import plt, seed  # noqa: F401
import pytest

import nengo_loihi


@pytest.fixture(scope="session")
def Simulator(request):
    """Simulator class to be used in tests"""

    def get_sim(*args, **kwargs):
        kwargs.setdefault("target", request.config.getoption("--target"))
        return nengo_loihi.Simulator(*args, **kwargs)

    return get_sim


def pytest_addoption(parser):
    parser.addoption("--target", type=str, default="sim",
                     help="Platform on which to run tests (e.g. 'sim' "
                          "or 'loihi')")
