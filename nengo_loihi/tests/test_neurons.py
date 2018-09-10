import numpy as np
import nengo
import pytest

from nengo_loihi.neurons import loihi_rates


@pytest.mark.parametrize("neuron_type", [
    nengo.LIF(),
    nengo.LIF(tau_ref=0.001, tau_rc=0.07, amplitude=0.34),
    nengo.SpikingRectifiedLinear(),
    nengo.SpikingRectifiedLinear(amplitude=0.23),
])
def test_loihi_rates(neuron_type, Simulator, plt, allclose):
    n = 256
    x = np.linspace(0, 1, n)

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

    dt = 0.001
    with Simulator(model, dt=dt) as sim:
        sim.run(1.0)

    est_rates = sim.data[ap].mean(axis=0)
    ref_rates = loihi_rates(neuron_type, x, gain, bias, dt=dt)

    plt.plot(x, ref_rates, "k", label="predicted")
    plt.plot(x, est_rates, "g", label="measured")
    plt.legend(loc='best')

    assert allclose(est_rates, ref_rates, atol=1, rtol=0, xtol=1)
