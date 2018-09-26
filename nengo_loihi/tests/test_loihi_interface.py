import pytest

from nengo_loihi import loihi_interface


class MockNxsdk:
    def __init__(self):
        self.__version__ = None


def test_error_on_old_version(monkeypatch):
    mock = MockNxsdk()
    mock.__version__ = "0.5.5"

    monkeypatch.setattr(loihi_interface, 'nxsdk', mock)
    with pytest.raises(ImportError):
        loihi_interface.LoihiSimulator.check_nxsdk_version()


def test_no_warn_on_current_version(monkeypatch):
    mock = MockNxsdk()
    mock.__version__ = "0.7.0"

    monkeypatch.setattr(loihi_interface, 'nxsdk', mock)
    with pytest.warns(None) as record:
        loihi_interface.LoihiSimulator.check_nxsdk_version()
    assert len(record) == 0


def test_warn_on_future_version(monkeypatch):
    mock = MockNxsdk()
    mock.__version__ = "0.7.1"

    monkeypatch.setattr(loihi_interface, 'nxsdk', mock)
    with pytest.warns(UserWarning):
        loihi_interface.LoihiSimulator.check_nxsdk_version()
