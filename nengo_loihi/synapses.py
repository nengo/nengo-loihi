import warnings

from nengo.utils.compat import is_integer, range
import numpy as np


def tracing_mag_int_frac(synapses):
    mag = synapses.tracing_mag
    mag = mag / (synapses.size / 100)

    mag_int = int(mag)
    # TODO: how does mag_frac actually work???
    #  It's the x in x/128, I believe
    mag_frac = int(128 * (mag - mag_int))
    # mag_frac = min(int(round(1./mag_frac)), 128)

    return mag_int, mag_frac


def shift(x, s, **kwargs):
    if s < 0:
        return np.right_shift(x, -s, **kwargs)
    else:
        return np.left_shift(x, s, **kwargs)


class SynapseFmt(object):
    INDEX_BITS_MAP = [0, 6, 7, 8, 9, 10, 11, 12]
    WEIGHT_BITS_MAP = [0, 1, 2, 3, 4, 5, 6, 8]

    def __init__(self):
        self.wgtLimitMant = 0
        self.wgtLimitExp = 0
        self.wgtExp = 0
        self.discMaxWgt = 0
        self.learningCfg = 0
        self.tagBits = 0
        self.dlyBits = 0
        self.wgtBits = 0
        self.reuseSynData = 0
        self.n_synapses = 0
        self.cIdxOffset = 0
        self.cIdxMult = 0
        self.skipBits = 0
        self.idxBits = 0
        self.synType = 0
        self.fanoutType = 0
        self.compression = 0
        self.stdpProfile = 0
        self.ignoreDly = 0

    @classmethod
    def get_scale(cls, wgtExp):
        # realWgtExp is 6 + wgtExp
        return 2 ** (6 + wgtExp)

    @property
    def isMixed(self):
        return self.fanoutType == 1

    @property
    def realWgtExp(self):
        return 6 + self.wgtExp

    @property
    def realWgtBits(self):
        return self.WEIGHT_BITS_MAP[self.wgtBits]

    @property
    def realIdxBits(self):
        return self.INDEX_BITS_MAP[self.idxBits]

    @property
    def scale(self):
        return self.get_scale(self.wgtExp)

    def bits_per_axon(self, n_weights):
        """For an axon with n weights, compute the weight memory bits used"""
        bits_per_weight = self.realWgtBits + self.dlyBits + self.tagBits
        if self.compression == 0:
            bits_per_weight += self.realIdxBits
        elif self.compression == 3:
            pass
        else:
            raise NotImplementedError("Compression %s" % (self.compression,))

        SYNAPSE_FMT_IDX_BITS = 4
        N_SYNAPSES_BITS = 6
        bits = 0
        synapses_per_group = self.n_synapses + 1
        for i in range(0, n_weights, synapses_per_group):
            n = min(n_weights - i, synapses_per_group)
            bits_i = n*bits_per_weight + SYNAPSE_FMT_IDX_BITS + N_SYNAPSES_BITS
            bits_i = -64 * (-bits_i // 64)
            # ^ round up to nearest 64 (size of one int64 memory unit)
            bits += bits_i

        return bits

    def discretize_weights(self, w, dtype=np.int32):
        s = 8 - self.realWgtBits + self.isMixed
        m = 2**(8 - s) - 1

        w = np.round(w / 2.**s).clip(-m, m).astype(dtype)
        s2 = s + self.wgtExp
        shift(w, s2, out=w)
        np.left_shift(w, 6, out=w)

        if s2 < 0:
            warnings.warn("Lost %d extra bits in weight rounding" % (-s2,))

        ws = w // self.scale
        assert np.all(ws <= 255) and np.all(ws >= -256)

        return w

    def validate(self):
        assert -7 <= self.wgtExp <= 7
        assert 0 <= self.tagBits < 4
        assert 0 <= self.dlyBits < 8
        assert 1 <= self.wgtBits < 8
        assert 0 <= self.cIdxOffset < 16
        assert 0 <= self.cIdxMult < 16
        assert 0 <= self.idxBits < 8
        assert 1 <= self.fanoutType < 4


class Synapses(object):
    def __init__(self, n_axons):
        self.n_axons = n_axons
        self.fmt = SynapseFmt()
        self.weights = None
        self.indices = None
        self.tracing_tau = None
        self.tracing_mag = None

    @property
    def learning(self):
        return self.tracing_tau is not None

    @property
    def max_abs_weight(self):
        return max(np.abs(w).max() if len(w) > 0 else -np.inf
                   for w in self.weights)

    @property
    def n_bits(self):
        return sum(self.fmt.bits_per_axon(len(w))
                   for w in self.weights)

    @property
    def size(self):
        return sum(len(w) for w in self.weights)

    def _set_weights(self):
        max_idx = max(i.max() if len(i) > 0 else -1 for i in self.indices)
        idx_bits = int(np.ceil(np.log2(max_idx + 1)))
        assert idx_bits <= SynapseFmt.INDEX_BITS_MAP[-1]
        idx_bits = next(i for i, v in enumerate(SynapseFmt.INDEX_BITS_MAP)
                        if v >= idx_bits)
        self.fmt.compression = 3
        self.fmt.idxBits = idx_bits
        self.fmt.fanoutType = 1
        self.fmt.n_synapses = 63
        self.fmt.wgtBits = 7

    def set_full_weights(self, weights):
        assert weights.shape[0] == self.n_axons
        self.weights = [w.astype(np.float32) for w in weights]
        self.indices = [np.arange(w.size) for w in weights]
        self._set_weights()

    def set_diagonal_weights(self, diag):
        diag = diag.ravel()
        self.weights = [d.reshape(1).astype(np.float32) for d in diag]
        self.indices = [np.array([i]) for i in range(len(diag))]
        assert len(self.weights) == self.n_axons
        self._set_weights()

    def set_learning(self, tracing_tau=2, tracing_mag=1.0):
        assert is_integer(tracing_tau)
        self.tracing_tau = int(tracing_tau)
        self.tracing_mag = tracing_mag
        self.fmt.learningCfg = 1
        self.fmt.stdpProfile = 0
        # ^ stdpProfile hard-coded for now (see loihi_interface)

        mag_int, _ = tracing_mag_int_frac(self)
        assert int(mag_int) < 2**7
