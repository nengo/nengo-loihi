import pytest

from nengo_loihi.hardware import interface as hardware_interface


class MockNxsdk:
    def __init__(self):
        self.__version__ = None


def test_error_on_old_version(monkeypatch):
    mock = MockNxsdk()
    mock.__version__ = "0.5.5"

    monkeypatch.setattr(hardware_interface, 'nxsdk', mock)
    with pytest.raises(ImportError):
        hardware_interface.HardwareInterface.check_nxsdk_version()


def test_no_warn_on_current_version(monkeypatch):
    mock = MockNxsdk()
    mock.__version__ = "0.7.0"

    monkeypatch.setattr(hardware_interface, 'nxsdk', mock)
    monkeypatch.setattr(hardware_interface, 'assert_nxsdk', lambda: True)
    with pytest.warns(None) as record:
        hardware_interface.HardwareInterface.check_nxsdk_version()
    assert len(record) == 0


def test_warn_on_future_version(monkeypatch):
    mock = MockNxsdk()
    mock.__version__ = "0.7.6"

    monkeypatch.setattr(hardware_interface, 'nxsdk', mock)
    monkeypatch.setattr(hardware_interface, 'assert_nxsdk', lambda: True)
    with pytest.warns(UserWarning):
        hardware_interface.HardwareInterface.check_nxsdk_version()
