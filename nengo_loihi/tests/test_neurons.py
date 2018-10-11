import numpy as np
import nengo
import pytest

from nengo_loihi.neurons import (
    loihi_rates, LoihiLIF, LoihiSpikingRectifiedLinear)


@pytest.mark.parametrize('dt', [3e-4, 1e-3])
@pytest.mark.parametrize('neuron_type', [
    nengo.LIF(),
    nengo.LIF(tau_ref=0.001, tau_rc=0.07, amplitude=0.34),
    nengo.SpikingRectifiedLinear(),
    nengo.SpikingRectifiedLinear(amplitude=0.23),
])
def test_loihi_rates(dt, neuron_type, Simulator, plt, allclose):
    n = 256
    x = np.linspace(-0.1, 1, n)

    encoders = np.ones((n, 1))
    max_rates = 400 * np.ones(n)
    intercepts = 0 * np.ones(n)
    gain, bias = neuron_type.gain_bias(max_rates, intercepts)
    j = x * gain + bias

    with nengo.Network() as model:
        a = nengo.Ensemble(n, 1,
                           neuron_type=neuron_type,
                           encoders=encoders,
                           gain=gain,
                           bias=j)
        ap = nengo.Probe(a.neurons)

    with Simulator(model, dt=dt) as sim:
        sim.run(1.0)

    est_rates = sim.data[ap].mean(axis=0)
    ref_rates = loihi_rates(neuron_type, x, gain, bias, dt=dt)

    plt.plot(x, ref_rates, "k", label="predicted")
    plt.plot(x, est_rates, "g", label="measured")
    plt.legend(loc='best')

    assert allclose(est_rates, ref_rates, atol=1, rtol=0, xtol=1)


@pytest.mark.parametrize('neuron_type', [
    LoihiLIF(),
    LoihiSpikingRectifiedLinear(),
])
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
        a = nengo.Ensemble(n, 1, neuron_type=neuron_type,
                           encoders=encoders, gain=gain, bias=bias)
        ap = nengo.Probe(a.neurons)

    t_final = 1.0
    with nengo.Simulator(model, dt=dt) as nengo_sim:
        nengo_sim.run(t_final)

    with Simulator(model, dt=dt) as loihi_sim:
        loihi_sim.run(t_final)

    nengo_rates = (nengo_sim.data[ap] > 0).sum(axis=0) / t_final
    loihi_rates = (loihi_sim.data[ap] > 0).sum(axis=0) / t_final

    ref = neuron_type.rates(0., gain, bias, dt=dt)
    plt.plot(bias, loihi_rates, 'r', label='loihi sim')
    plt.plot(bias, nengo_rates, 'b-.', label='nengo sim')
    plt.plot(bias, ref, 'k--', label='ref')
    plt.legend(loc='best')

    atol = 1. / t_final  # the fundamental unit for our rates
    assert allclose(nengo_rates, ref, atol=atol, rtol=0, xtol=0)
    assert allclose(loihi_rates, ref, atol=atol, rtol=0, xtol=0)
