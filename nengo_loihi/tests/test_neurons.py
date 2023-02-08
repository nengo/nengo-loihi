import logging

import nengo
import numpy as np
import pytest

from nengo_loihi import neurons
from nengo_loihi.compat import HAS_DL, nengo_dl
from nengo_loihi.neurons import (
    Installer,
    LoihiLIF,
    LoihiSpikingRectifiedLinear,
    install_dl_builders,
    loihi_rates,
    nengo_rates,
)

v0_arg = {"initial_state": {"voltage": nengo.dists.Choice([0])}}


@pytest.mark.parametrize("dt", [3e-4, 1e-3])
@pytest.mark.parametrize(
    "neuron_type",
    [
        nengo.LIF(**v0_arg),
        nengo.LIF(tau_ref=0.001, tau_rc=0.07, amplitude=0.34, **v0_arg),
        nengo.SpikingRectifiedLinear(**v0_arg),
        nengo.SpikingRectifiedLinear(amplitude=0.23, **v0_arg),
        nengo.RegularSpiking(nengo.LIFRate(), **v0_arg),
        nengo.RegularSpiking(
            nengo.LIFRate(tau_ref=0.001, tau_rc=0.03), amplitude=0.31, **v0_arg
        ),
        nengo.RegularSpiking(nengo.RectifiedLinear(), **v0_arg),
        nengo.RegularSpiking(nengo.RectifiedLinear(), amplitude=0.46, **v0_arg),
    ],
)
def test_loihi_rates(dt, neuron_type, Simulator, plt, allclose):
    n = 256
    x = np.linspace(-0.1, 1, n)

    encoders = np.ones((n, 1))
    max_rates = 400 * np.ones(n)
    intercepts = 0 * np.ones(n)
    gain, bias = neuron_type.gain_bias(max_rates, intercepts)
    j = x * gain + bias

    with nengo.Network() as model:
        a = nengo.Ensemble(
            n, 1, neuron_type=neuron_type, encoders=encoders, gain=gain, bias=j
        )
        ap = nengo.Probe(a.neurons)

    with Simulator(model, dt=dt) as sim:
        sim.run(1.0)

    est_rates = sim.data[ap].mean(axis=0)
    ref_rates = loihi_rates(neuron_type, x[np.newaxis, :], gain, bias, dt=dt).squeeze(
        axis=0
    )

    ref_rates2 = None
    if isinstance(neuron_type, nengo.RegularSpiking):
        if isinstance(neuron_type.base_type, nengo.LIFRate):
            neuron_type2 = nengo.LIF(
                tau_rc=neuron_type.base_type.tau_rc,
                tau_ref=neuron_type.base_type.tau_ref,
                amplitude=neuron_type.amplitude,
            )
        elif isinstance(neuron_type.base_type, nengo.RectifiedLinear):
            neuron_type2 = nengo.SpikingRectifiedLinear(
                amplitude=neuron_type.amplitude,
            )

        ref_rates2 = loihi_rates(
            neuron_type2, x[np.newaxis, :], gain, bias, dt=dt
        ).squeeze(axis=0)

    plt.plot(x, ref_rates, "k", label="predicted")
    if ref_rates2 is not None:
        plt.plot(x, ref_rates2, "b", label="predicted-base")
    plt.plot(x, est_rates, "g", label="measured")
    plt.legend(loc="best")

    assert ref_rates.shape == est_rates.shape
    assert allclose(est_rates, ref_rates, atol=1, rtol=0, xtol=1)
    if ref_rates2 is not None:
        assert allclose(ref_rates2, ref_rates)


@pytest.mark.parametrize(
    "neuron_type",
    [
        nengo.Sigmoid(),
        nengo.RegularSpiking(nengo.Sigmoid()),
    ],
)
def test_loihi_rates_other_type(neuron_type, allclose):
    """Test using a neuron type that has no Loihi-specific implementation."""
    x = np.linspace(-7, 10)
    gain, bias = 0.2, 0.4
    dt = 0.002
    ref_rates = nengo_rates(neuron_type, x, gain, bias)
    rates = loihi_rates(neuron_type, x, gain, bias, dt)
    assert ref_rates.shape == rates.shape
    assert allclose(rates, ref_rates)


@pytest.mark.parametrize("neuron_type", [LoihiLIF(), LoihiSpikingRectifiedLinear()])
def test_loihi_neurons(neuron_type, Simulator, plt, allclose):
    dt = 0.0007

    n = 256
    encoders = np.ones((n, 1))
    gain = np.zeros(n)
    if isinstance(neuron_type, nengo.SpikingRectifiedLinear):
        bias = np.linspace(0, 1001, n)
    else:
        bias = np.linspace(0, 30, n)

    with nengo.Network() as model:
        ens = nengo.Ensemble(
            n, 1, neuron_type=neuron_type, encoders=encoders, gain=gain, bias=bias
        )
        probe = nengo.Probe(ens.neurons)

    t_final = 1.0
    with nengo.Simulator(model, dt=dt) as nengo_sim:
        nengo_sim.run(t_final)

    with Simulator(model, dt=dt) as loihi_sim:
        loihi_sim.run(t_final)

    rates_nengosim = np.sum(nengo_sim.data[probe] > 0, axis=0) / t_final
    rates_loihisim = np.sum(loihi_sim.data[probe] > 0, axis=0) / t_final

    rates_ref = neuron_type.rates(0.0, gain, bias, dt=dt).squeeze()
    plt.plot(bias, rates_loihisim, "r", label="loihi sim")
    plt.plot(bias, rates_nengosim, "b-.", label="nengo sim")
    plt.plot(bias, rates_ref, "k--", label="ref")
    plt.legend(loc="best")

    assert rates_ref.shape == rates_nengosim.shape == rates_loihisim.shape
    atol = 1.0 / t_final  # the fundamental unit for our rates
    assert allclose(rates_nengosim, rates_ref, atol=atol, rtol=0, xtol=1)
    assert allclose(rates_loihisim, rates_ref, atol=atol, rtol=0, xtol=1)


def test_lif_min_voltage(Simulator, plt, allclose):
    neuron_type = nengo.LIF(min_voltage=-0.5, **v0_arg)
    t_final = 0.4

    with nengo.Network() as model:
        u = nengo.Node(lambda t: np.sin(4 * np.pi * t / t_final))
        a = nengo.Ensemble(
            1,
            1,
            neuron_type=neuron_type,
            encoders=np.ones((1, 1)),
            max_rates=[100],
            intercepts=[0.5],
        )
        nengo.Connection(u, a, synapse=None)
        ap = nengo.Probe(a.neurons, "voltage")

    with nengo.Simulator(model) as nengo_sim:
        nengo_sim.run(t_final)

    with Simulator(model) as loihi_sim:
        loihi_sim.run(t_final)

    nengo_voltage = nengo_sim.data[ap]
    loihi_voltage = loihi_sim.data[ap]
    loihi_voltage = loihi_voltage / loihi_voltage.max()
    plt.plot(nengo_sim.trange(), nengo_voltage)
    plt.plot(loihi_sim.trange(), loihi_voltage)

    nengo_min_voltage = nengo_voltage.min()
    loihi_min_voltage = loihi_voltage.min()

    # Close, but not exact, because loihi min voltage rounded to power of 2
    assert allclose(loihi_min_voltage, nengo_min_voltage, atol=0.2)


def test_no_extras(Simulator, monkeypatch):
    # check that things still work without nengo_dl / nengo_extras / tf
    monkeypatch.setattr(neurons, "HAS_TF", False)

    with nengo.Network() as net:
        a = nengo.Ensemble(10, 1, neuron_type=LoihiLIF())
        nengo.Probe(a)

    with Simulator(net) as sim:
        sim.step()


def test_installer_called_twice(caplog, monkeypatch):
    """Ensures that the installer prints no messages when called twice."""
    monkeypatch.setattr(neurons, "HAS_DL", True)
    install = Installer()
    install.installed = True
    with caplog.at_level(logging.DEBUG):
        install()
    assert len(caplog.records) == 0


@pytest.mark.skipif(not HAS_DL, reason="requires nengo-dl")
def test_install_on_instantiation():
    tf_neuron_impl = nengo_dl.neuron_builders.SimNeuronsBuilder.TF_NEURON_IMPL

    # undo any installation that happened in other tests
    if LoihiLIF in tf_neuron_impl:
        del tf_neuron_impl[LoihiLIF]
    install_dl_builders.installed = False

    assert LoihiLIF not in tf_neuron_impl
    LoihiLIF()
    assert LoihiLIF in tf_neuron_impl
