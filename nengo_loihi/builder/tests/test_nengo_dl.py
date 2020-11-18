import logging

import pytest

from nengo_loihi.builder import nengo_dl as builder_nengo_dl
from nengo_loihi.neurons import LoihiLIF


def test_installer_with_no_dl(caplog, monkeypatch):
    """Ensures that the installer works as expected with no NengoDL.

    If NengoDL is installed, other tests will test the installer as
    it is used in ``Simulator.__init__``.
    """
    monkeypatch.setattr(builder_nengo_dl, "HAS_DL", False)
    install = builder_nengo_dl.Installer()
    with caplog.at_level(logging.INFO):
        install()
    messages = [rec.message for rec in caplog.records]
    assert len(messages) == 1
    assert messages[0].startswith("nengo_dl cannot be imported")


def test_installer_called_twice(caplog, monkeypatch):
    """Ensures that the installer prints no messages when called twice."""
    monkeypatch.setattr(builder_nengo_dl, "HAS_DL", True)
    install = builder_nengo_dl.Installer()
    install.installed = True
    with caplog.at_level(logging.DEBUG):
        install()
    assert len(caplog.records) == 0


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


def test_install_on_instantiation():
    nengo_dl = pytest.importorskip("nengo_dl")

    if builder_nengo_dl.install_dl_builders.installed:
        # undo any installation that happened in other tests
        del nengo_dl.neuron_builders.SimNeuronsBuilder.TF_NEURON_IMPL[LoihiLIF]
        builder_nengo_dl.install_dl_builders.installed = False

    assert LoihiLIF not in nengo_dl.neuron_builders.SimNeuronsBuilder.TF_NEURON_IMPL

    LoihiLIF()

    assert LoihiLIF in nengo_dl.neuron_builders.SimNeuronsBuilder.TF_NEURON_IMPL
