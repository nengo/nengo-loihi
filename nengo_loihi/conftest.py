from nengo.conftest import plt, seed  # noqa: F401
import pytest

import nengo_loihi


@pytest.fixture(scope="session")
def Simulator(request):
    """Simulator class to be used in tests"""
    try:
        import nxsdk
        return nengo_loihi.Simulator
    except ImportError:
        return nengo_loihi.NumpySimulator
