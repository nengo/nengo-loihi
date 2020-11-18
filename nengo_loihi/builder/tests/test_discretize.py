import logging

import nengo
import numpy as np
import pytest
from nengo.exceptions import BuildError
from nengo.utils.numpy import rms

from nengo_loihi.block import LoihiBlock, SynapseConfig
from nengo_loihi.builder import Model
from nengo_loihi.builder.discretize import (
    decay_int,
    decay_magnitude,
    discretize_block,
    discretize_weights,
    overflow_signed,
)
from nengo_loihi.decode_neurons import NoisyDecodeNeurons


@pytest.mark.parametrize("b", (8, 16, 17, 23))
def test_overflow_signed(b, rng):
    x = rng.uniform(-(2 ** (b + 2)), 2 ** (b + 2), size=2 ** (b + 1)).astype(np.int32)
    x = np.concatenate((x, 2 ** np.arange(0, b + 2), -(2 ** np.arange(0, b + 2)), [0]))

    # if you want to be extra careful you could run this test with all possible
    # values, i.e.
    # x = np.arange(-2 ** (b + 2), 2 ** (b + 2), dtype=np.int32)

    # compute check values
    b2 = 2 ** b
    z = x % b2
    zmask = np.right_shift(x, b) % 2  # sign bit, the b-th bit

    # if the sign bit is set, subtract 2**b to make it negative
    z -= np.left_shift(zmask, b)

    # compute whether it's overflowed
    q = (x < -b2) | (x >= b2)

    y, o = overflow_signed(x, bits=b)
    assert np.array_equal(y, z)
    assert np.array_equal(o, q)


@pytest.mark.parametrize("offset", (0, 1))
def test_decay_magnitude(offset, plt):
    bits = 12
    decays = np.arange(1, 2 ** bits, 7)
    ref = []
    emp = []
    est = []

    for decay in decays:

        def empirical_decay_magnitude(decay, x0):
            x = x0 * np.ones(decay.shape, dtype=np.int32)
            y = x
            y_sum = np.zeros(decay.shape, dtype=np.int64)
            for i in range(100000):
                y_sum += y
                if (y <= 0).all():
                    break
                y = decay_int(y, decay, bits=bits, offset=offset)
            else:
                raise RuntimeError("Exceeded max number of iterations")

            return y_sum / x0

        x0 = np.arange(2 ** 21 - 1000, 2 ** 21, step=41, dtype=np.int32)

        m = empirical_decay_magnitude(decay * np.ones_like(x0), x0)
        m0 = m.mean()
        emp.append(m0)

        # reference (naive) method, not accounting for truncation loss
        r = (2 ** bits - offset - decay) / 2 ** bits
        Sx1 = (1 - r / x0) / (1 - r)  # sum_i^n x_i/x_0, where x_n = r**n*x_0 = 1
        ref.append(Sx1.mean())

        m = decay_magnitude(decay, x0, bits=bits, offset=offset)
        m2 = m.mean()
        est.append(m2)

    ref = np.array(ref)
    emp = np.array(emp)
    est = np.array(est)
    rms_ref = rms(ref - emp) / rms(emp)
    rms_est = rms(est - emp) / rms(emp)
    logging.info(
        "Ref rel RMSE: %0.3e, decay_magnitude rel RMSE: %0.3e" % (rms_ref, rms_est)
    )

    abs_ref = np.abs(ref - emp)
    abs_est = np.abs(est - emp)

    # find places where ref is better than est
    relative_diff = (abs_est - abs_ref) / emp

    ax = plt.subplot(211)
    plt.plot(abs_ref, "b")
    plt.plot(abs_est, "g")
    ax.set_yscale("log")

    ax = plt.subplot(212)
    plt.plot(relative_diff.clip(0, None))

    assert np.all(relative_diff < 1e-6)


@pytest.mark.parametrize("lossy_shift", (True, False))
def test_lossy_shift(lossy_shift, rng):
    wgt_bits = 6
    w = rng.uniform(-100, 100, size=(10, 10))
    fmt = SynapseConfig(weight_bits=wgt_bits, weight_exp=0, fanout_type=0)

    w2 = discretize_weights(fmt, w, lossy_shift=lossy_shift)

    clipped = np.round(w / 4).clip(-(2 ** wgt_bits), 2 ** wgt_bits).astype(np.int32)

    assert np.allclose(w2, np.left_shift(clipped, 8))


def test_bad_weight_exponent_error(Simulator):
    with nengo.Network() as net:
        a = nengo.Ensemble(5, 1)
        b = nengo.Ensemble(5, 1, neuron_type=nengo.LIF(tau_rc=5.0))
        nengo.Connection(
            a.neurons, b.neurons, transform=1e-8 * np.ones((5, 5)), synapse=5.0
        )

    with pytest.raises(BuildError, match="[Cc]ould not find.*weight exp"):
        with Simulator(net):
            pass


def test_bad_bias_scaling_error(Simulator):
    block = LoihiBlock(10)
    block.compartment.configure_lif(tau_rc=5.0, vth=1e8)
    block.compartment.bias[:] = 1000.0

    with pytest.raises(BuildError, match="[Cc]ould not find.*bias scaling"):
        discretize_block(block)


def test_noise_amplitude_warnings(Simulator, seed):
    with nengo.Network(seed=seed) as net:
        a = nengo.Ensemble(5, 1)
        b = nengo.Ensemble(5, 1)
        nengo.Connection(a, b)

    model = Model()
    model.decode_neurons = NoisyDecodeNeurons(10, noise_exp=5)
    with pytest.warns(UserWarning, match="[Nn]oise.*exceeds.*upper"):
        with Simulator(net, model=model):
            pass

    model = Model()
    model.decode_neurons = NoisyDecodeNeurons(10, noise_exp=-7)
    with pytest.warns(UserWarning, match="[Nn]oise.*below.*lower"):
        with Simulator(net, model=model):
            pass
