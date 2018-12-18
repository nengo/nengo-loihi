import numpy as np


class BinaryEncoder(object):
    """Node function for encoding a (-1, 1) vector in binary."""

    def __init__(self, n_bits=8):
        self.n_bits = n_bits

    def get_size_out(self, d):
        return 2 * self.n_bits * d

    def __call__(self, dummy_time, x):
        spiked = np.zeros((2, self.n_bits, len(x)), dtype=bool)
        for i, x_i in enumerate(x):
            sign_bit = 0 if x_i >= 0 else 1
            f = np.abs(x_i)
            v = 0.5  # to represent [0, 1)
            for j in range(self.n_bits):
                if f >= v:
                    f -= v
                    spiked[sign_bit, j, i] = True
                v /= 2.
        return spiked.flatten()

    def make_weights(self, encoders):
        weights = np.zeros(
            (2, self.n_bits, encoders.shape[1], encoders.shape[0]))

        for i, sign in enumerate([+1, -1]):
            for bit in range(self.n_bits):
                weights[i, bit, :, :] = sign / 2.**(1 + bit) * encoders.T

        flat_weights = weights.reshape(-1, weights.shape[-1])
        assert flat_weights.shape == (
            self.get_size_out(encoders.shape[1]),
            encoders.shape[0])

        return flat_weights
