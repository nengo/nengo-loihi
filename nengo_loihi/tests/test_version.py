import nengo.version
import pytest

from nengo_loihi.version import check_nengo_version


def test_nengo_version(Simulator, monkeypatch):
    # test nengo version below minimum
    monkeypatch.setattr(nengo.version, "version_info", (2, 0, 0))
    with pytest.raises(ValueError, match="nengo-loihi does not support Nengo version"):
        check_nengo_version()

    # test nengo version newer than latest
    monkeypatch.setattr(nengo.version, "version_info", (99, 0, 0))
    with pytest.warns(UserWarning):
        check_nengo_version()
