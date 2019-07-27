import nengo.version
import pytest

from nengo_loihi import version
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


def test_nengo_version_check():
    # note: we rely on travis-ci to test this against different nengo versions

    if version.dev is not None or nengo.version.dev is None:
        # nengo_loihi should be compatible with all non-development nengo
        # versions, and a nengo_loihi dev version should be compatible with all
        # (dev or non-dev) nengo versions
        assert nengo.version.version_info <= version.latest_nengo_version_info
        with pytest.warns(None) as w:
            check_nengo_version()

        assert len(w) == 0, "Should not warn about version: %s" % [
            str(x.message) for x in w
        ]
    else:  # version.dev is None and nengo.version.dev is not None
        # a development version of nengo with a non-development nengo_loihi
        # version should cause a warning (we don't want to mark a nengo_loihi
        # release as compatible with a nengo dev version, since the nengo
        # version may change and no longer be compatible with our nengo_loihi
        # release).

        # note: we assume that a nengo dev version means the latest dev version
        assert nengo.version.version_info >= version.latest_nengo_version_info

        # if no warning is issued here, it may mean you're forgetting to set
        # the `latest_nengo_version` back to the last released nengo version
        # when nengo_loihi is being released.
        with pytest.warns(UserWarning, match="This version.*not been tested"):
            check_nengo_version()
