import numpy as np
import nengo
import pytest


@pytest.mark.parametrize("tau_ref", [0.001, 0.003, 0.005])
def test_lif_response_curves(tau_ref, Simulator, plt):
    n = 256
    encoders = np.ones((n, 1))
    gain = np.zeros(n)
    bias = np.linspace(1, 30, n)

    with nengo.Network() as model:
        a = nengo.Ensemble(n, 1,
                           neuron_type=nengo.LIF(tau_ref=tau_ref),
                           encoders=encoders,
                           gain=gain,
                           bias=bias)
        ap = nengo.Probe(a.neurons)

    dt = 0.001
    with Simulator(model, dt=dt) as sim:
        sim.run(1.0)

    scount = np.sum(sim.data[ap] > 0, axis=0)

    upper_bound = nengo.LIF(tau_ref=tau_ref).rates(0., gain, bias)
    lower_bound = nengo.LIF(tau_ref=tau_ref + dt).rates(0., gain, bias)
    mid = nengo.LIF(tau_ref=tau_ref + 0.5*dt).rates(0., gain, bias)
    plt.title("tau_ref=%.3f" % tau_ref)
    plt.plot(bias, upper_bound, "k")
    plt.plot(bias, lower_bound, "k")
    plt.plot(bias, mid, "b")
    plt.plot(bias, scount, "g", label="Spike count on Loihi")
    plt.xlabel("Bias current")
    plt.ylabel("Firing rate (Hz)")
    plt.legend(loc="best")

    assert np.all(scount <= upper_bound + 1)
    assert np.all(scount >= lower_bound - 1)


def test_relu_response_curves(Simulator, plt):
    n = 256
    encoders = np.ones((n, 1))
    gain = np.zeros(n)
    bias = np.linspace(0, 1.01, n)

    with nengo.Network() as model:
        a = nengo.Ensemble(n, 1,
                           neuron_type=nengo.RectifiedLinear(),
                           encoders=encoders,
                           gain=gain,
                           bias=bias)
        ap = nengo.Probe(a.neurons)

    dt = 0.001
    t_final = 1.0
    with Simulator(model, dt=dt) as sim:
        sim.run(t_final)

    scount = np.sum(sim.data[ap] > 0, axis=0)
    actual = nengo.RectifiedLinear().rates(0., gain, bias / dt)
    plt.plot(bias, actual, "b", label="Ideal")
    plt.plot(bias, scount, "g", label="Loihi")
    plt.xlabel("Bias current")
    plt.ylabel("Firing rate (Hz)")
    plt.legend(loc="best")

    assert np.all(actual >= scount)
