from nengo.conftest import plt, seed, TestConfig  # noqa: F401
import pytest

import nengo_loihi

# This ensures that all plots go to the right directory
TestConfig.RefSimulator = TestConfig.Simulator = nengo_loihi.Simulator


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
