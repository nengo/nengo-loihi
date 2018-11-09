import logging

import pytest

from nengo_loihi.builder import nengo_dl as builder_nengo_dl


def test_installer_with_no_dl(monkeypatch):
    """Ensures that the installer works as expected with no Nengo DL.

    If Nengo DL is installed, other tests will test the installer as
    it is used in ``Simulator.__init__``.
    """
    monkeypatch.setattr(builder_nengo_dl, "HAS_DL", False)
    install = builder_nengo_dl.Installer()
    with pytest.warns(UserWarning, match="nengo_dl cannot be imported"):
        install()


def test_installer_called_twice(caplog, monkeypatch):
    """Ensures that the installer prints debug messages when called twice."""
    monkeypatch.setattr(builder_nengo_dl, "HAS_DL", True)
    install = builder_nengo_dl.Installer()
    install.installed = True
    with caplog.at_level(logging.DEBUG):
        install()
    assert [rec.message for rec in caplog.records] == [
        "NengoDL neuron builders already installed",
    ]


def test_register_twice_warning():
    class MockNoiseBuilder:
        pass

    # First time does not warn
    with pytest.warns(None) as record:
        builder_nengo_dl.NoiseBuilder.register(int)(MockNoiseBuilder)
    assert len(record) == 0

    # Second time warns
    with pytest.warns(UserWarning, match="already has a builder"):
        builder_nengo_dl.NoiseBuilder.register(int)(MockNoiseBuilder)
