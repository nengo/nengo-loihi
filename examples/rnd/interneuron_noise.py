import matplotlib.pyplot as plt
import numpy as np
import nengo
from nengo.processes import WhiteNoise, WhiteSignal
from nengo.dists import Gaussian, Uniform
from nengo.synapses import Lowpass, Alpha
from nengo.utils.numpy import rms

from loihi_neurons import LoihiLIF


def test_interneuron_noise(n, gain_dist=None, noise_dist=None, synapse=None,
                           plot=False, rng=np.random):
    # tlen = 5
    tlen = 1

    dt = 0.001
    relu = LoihiLIF(tau_rc=np.inf, tau_ref=0.000)

    # max_rate = 1000
    # max_rate = 999
    # max_rate = 500
    # max_rate = 250
    # max_rate = 200
    max_rate = 100
    gain = 0.5 * dt * max_rate
    bias = gain

    if gain_dist is None:
        gain_dist = Uniform(0.5, 1.0)

    if noise_dist is None:
        noise_dist = Uniform(-0.001, 0.001)

    if synapse is None:
        # synapse = nengo.synapses.Alpha(0.005)
        synapse = nengo.synapses.Alpha(0.03)
        # synapse = nengo.synapses.Lowpass(0.02)

    # synlong = None
    synlong = nengo.synapses.Lowpass(0.0)
    # synlong = nengo.synapses.Lowpass(0.1)

    # print(relu.rates(0.99, [gain, -gain], [bias, bias]))

    with nengo.Network(seed=1) as model:
        # u = nengo.Node(0.75)
        # u = nengo.Node(1.0)
        # u = nengo.Node(1.01)
        # u = nengo.Node(WhiteNoise(dist=Gaussian(mean=1., std=0.1), scale=False))
        # u = nengo.Node(WhiteSignal(period=tlen, high=3, rms=0.4))
        u = nengo.Node(WhiteSignal(period=tlen, high=5))
        # u = nengo.Node(WhiteSignal(period=tlen, high=30))
        up = nengo.Probe(u, synapse=synapse.combine(synlong))

        # --- no noise
        a = nengo.Ensemble(2*n, 1, neuron_type=relu,
                           # encoders=[[1], [-1]], max_rates=[max_rate] * 2, intercepts=[-1] * 2)
                           encoders=[[1], [-1]] * n,
                           gain=[gain, gain] * n,
                           bias=[bias, bias] * n)
        # ap = nengo.Probe(a, synapse=synapse)
        an = nengo.Node(size_in=1)
        anp = nengo.Probe(an, synapse=synapse)
        nengo.Connection(u, a, synapse=synlong)

        # ac = nengo.Connection(a, an, synapse=None)
        dec = 1. / (n * max_rate)
        ac = nengo.Connection(a.neurons, an, synapse=None,
                              transform=[[dec, -dec] * n])

        neura = nengo.Probe(a.neurons)

        # --- noise
        b = nengo.Ensemble(2*n, 1, neuron_type=relu,
                           encoders=[[1], [-1]] * n,
                           gain=[gain, gain] * n,
                           bias=[bias, bias] * n,
                           noise=WhiteNoise(dist=noise_dist, scale=True))
        bp = nengo.Probe(b, synapse=synapse)
        bn = nengo.Node(size_in=1)
        bnp = nengo.Probe(bn, synapse=synapse)
        nengo.Connection(u, b, synapse=synlong)
        # bc = nengo.Connection(b, bn, synapse=None)

        dec = 1. / (n * max_rate)
        bc = nengo.Connection(b.neurons, bn, synapse=None,
                              transform=[[dec, -dec] * n])

        neurb = nengo.Probe(b.neurons)

        # --- heterogain
        gains = gain * np.linspace(0.5, 1.0, n)
        # gains = gain * gain_dist.sample(n, rng=rng)
        # gains = gain * np.array([0.6180339850173578, 0.7320508075654991, 0.36602540378274956, 0.41421356237309487, 0.2637626158259655, 0.7912878474778965, 0.3027756377319946, 0.2909944487358056, 0.4364916731037085, 0.20710678118654743])
        biases = gains
        d = nengo.Ensemble(2*n, 1, neuron_type=relu,
                           encoders=[[1], [-1]] * n,
                           gain=gains.repeat(2),
                           bias=biases.repeat(2))
        dp = nengo.Probe(d, synapse=synapse)
        dn = nengo.Node(size_in=1)
        dnp = nengo.Probe(dn, synapse=synapse)
        nengo.Connection(u, d, synapse=synlong)
        # bc = nengo.Connection(b, bn, synapse=None)

        # dc = nengo.Connection(d, dn, synapse=None)
        dec = (0.5 * dt) / gains.sum()
        print("Heterogain effective max rate: %s" % (1. / dec, ))
        dc = nengo.Connection(d.neurons, dn, synapse=None,
                              transform=[[dec, -dec] * n])

        neurd = nengo.Probe(d.neurons)



    with nengo.Simulator(model, dt=dt) as sim:
        sim.run(tlen)


    if plot:
        # print(sim.data[a].gain)
        # print(sim.data[a].bias)
        print(sim.data[ac].weights)
        print(sim.data[dc].weights)

        print(sim.data[neura].sum(axis=0) * (dt / tlen))
        print(sim.data[neurb].sum(axis=0) * (dt / tlen))
        print(sim.data[neurd].sum(axis=0) * (dt / tlen))

        plt.plot(sim.trange(), sim.data[up], label='input')
        # plt.plot(sim.trange(), sim.data[ap] / max_rate)
        plt.plot(sim.trange(), sim.data[anp], label='clean')
        plt.plot(sim.trange(), sim.data[bnp], label='noise')
        plt.plot(sim.trange(), sim.data[dnp], label='gains')
        plt.legend()
        plt.show()


    x = sim.data[up]
    error = lambda probe: rms((probe - x)[100:])
    a_error = error(sim.data[anp])
    b_error = error(sim.data[bnp])
    d_error = error(sim.data[dnp])
    return a_error, b_error, d_error


def find_best_gain_dist():
    n = 10
    rng = np.random.RandomState(3)

    # gain_mins = np.linspace(0, 0.99, 5)
    gain_mins = np.linspace(0, 0.99, 16)
    # gain_mins = np.linspace(0, 0.99, 31)

    # synapse = Lowpass(0.005)
    # synapse = Alpha(0.005)
    synapse = Alpha(0.015)

    errors = []
    for gain_min in gain_mins:
        gain_dist = Uniform(gain_min, 1.)
        errs = test_interneuron_noise(
            n=n, gain_dist=gain_dist, synapse=synapse, rng=rng)
        errors.append(errs)

    plt.plot(gain_mins, errors)
    plt.legend(['clean', 'noise', 'hetero'])
    plt.show()


def find_best_noise_dist():
    n = 10
    rng = np.random.RandomState(3)

    # noise_rs = np.logspace(-32, -2, 16)
    noise_rs = np.logspace(-5, -2, 16)

    synapse = Lowpass(0.005)
    # synapse = Alpha(0.005)
    # synapse = Alpha(0.015)

    errors = []
    for noise_r in noise_rs:
        noise_dist = Uniform(-noise_r, noise_r)
        errs = test_interneuron_noise(
            n=n, noise_dist=noise_dist, synapse=synapse, rng=rng)
        errors.append(errs)

    plt.semilogx(noise_rs, errors)
    plt.legend(['clean', 'noise', 'hetero'])
    plt.show()


if __name__ == '__main__':
    # test_interneuron_noise(n=10, gain_dist=Uniform(0.5, 1.0), plot=True)
    # test_interneuron_noise(n=10, gain_dist=Uniform(0.5, 1.0), plot=True, synapse=Lowpass(0.005))
    # test_interneuron_noise(n=10, gain_dist=Uniform(0.5, 1.0), plot=True, synapse=Alpha(0.005))

    # find_best_gain_dist()
    find_best_noise_dist()
