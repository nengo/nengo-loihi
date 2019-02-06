from __future__ import division

import warnings

from nengo.exceptions import BuildError
from nengo.utils.compat import is_iterable
import numpy as np

from nengo_loihi.block import SynapseFmt

VTH_MAN_MAX = 2**17 - 1
VTH_EXP = 6
VTH_MAX = VTH_MAN_MAX * 2**VTH_EXP

BIAS_MAN_MAX = 2**12 - 1
BIAS_EXP_MAX = 2**3 - 1
BIAS_MAX = BIAS_MAN_MAX * 2**BIAS_EXP_MAX

Q_BITS = 21  # number of bits for synapse accumulator
U_BITS = 23  # number of bits for cx input (u)

LEARN_BITS = 15  # number of bits in learning accumulator (not incl. sign)
LEARN_FRAC = 7  # extra least-significant bits added to weights for learning


def array_to_int(array, value):
    assert array.dtype == np.float32
    new = np.round(value).astype(np.int32)
    array.dtype = np.int32
    array[:] = new


def learn_overflow_bits(n_factors):
    """Compute number of bits with which learning will overflow.

    Parameters
    ----------
    n_factors : int
        The number of learning factors (pre/post terms in the learning rule).
    """
    factor_bits = 7  # number of bits per factor
    mantissa_bits = 3  # number of bits for learning rate mantissa
    return factor_bits*n_factors + mantissa_bits - LEARN_BITS


def overflow_signed(x, bits=7, out=None):
    """Compute overflow on an array of signed integers.

    For example, the Loihi chip uses 23 bits plus sign to represent U.
    We can store them as 32-bit integers, and use this function to compute
    how they would overflow if we only had 23 bits plus sign.

    Parameters
    ----------
    x : array
        Integer values for which to compute values after overflow.
    bits : int
        Number of bits, not including sign, to compute overflow for.
    out : array, optional (Default: None)
        Output array to put computed overflow values in.

    Returns
    -------
    y : array
        Values of x overflowed as would happen with limited bit representation.
    overflowed : array
        Boolean array indicating which values of ``x`` actually overflowed.
    """
    if out is None:
        out = np.array(x)
    else:
        assert isinstance(out, np.ndarray)
        out[:] = x

    assert np.issubdtype(out.dtype, np.integer)

    x1 = np.array(1, dtype=out.dtype)
    smask = np.left_shift(x1, bits)  # mask for the sign bit (2**bits)
    xmask = smask - 1  # mask for all bits <= `bits`

    # find whether we've overflowed
    overflowed = (out < -smask) | (out >= smask)

    zmask = out & smask  # if `out` has negative sign bit, == 2**bits
    out &= xmask  # mask out all bits > `bits`
    out -= zmask  # subtract 2**bits if negative sign bit

    return out, overflowed


def vth_to_manexp(vth):
    exp = VTH_EXP * np.ones(vth.shape, dtype=np.int32)
    man = np.round(vth / 2**exp).astype(np.int32)
    assert (man > 0).all()
    assert (man <= VTH_MAN_MAX).all()
    return man, exp


def bias_to_manexp(bias):
    r = np.maximum(np.abs(bias) / BIAS_MAN_MAX, 1)
    exp = np.ceil(np.log2(r)).astype(np.int32)
    man = np.round(bias / 2**exp).astype(np.int32)
    assert (exp >= 0).all()
    assert (exp <= BIAS_EXP_MAX).all()
    assert (np.abs(man) <= BIAS_MAN_MAX).all()
    return man, exp


def tracing_mag_int_frac(mag):
    """Split trace magnitude into integer and fractional components for chip"""
    mag_int = int(mag)
    mag_frac = int(128 * (mag - mag_int))
    return mag_int, mag_frac


def decay_int(x, decay, bits=12, offset=0, out=None):
    """Decay integer values using a decay constant.

    The decayed value is given by::

        sign(x) * floor(abs(x) * (2**bits - offset - decay) / 2**bits)
    """
    if out is None:
        out = np.zeros_like(x)
    r = (2**bits - offset - np.asarray(decay)).astype(np.int64)
    np.right_shift(np.abs(x) * r, bits, out=out)
    return np.sign(x) * out


def decay_magnitude(decay, x0=2**21, bits=12, offset=0):
    """Estimate the sum of the series of rounded integer decays of ``x0``.

    This can be used to estimate the total input current or voltage (summed
    over time) caused by an input of magnitude ``x0``. In real values, this is
    easy to calculate as the integral of an exponential. In integer values,
    we need to account for the rounding down that happens each time the decay
    is computed.

    Specifically, we estimate the sum of the series::

        x_i = floor(r x_{i-1})

    where ``r = (2**bits - offset - decay)``.

    To simulate the effects of rounding in decay, we subtract an expected loss
    due to rounding (``q``) each iteration. Our estimated series is therefore::

        y_i = r * y_{i-1} - q
            = r^i * x_0 - sum_k^{i-1} q * r^k
    """
    # q: Expected loss per time step (found by empirical simulations). If the
    # value being rounded down were uniformly distributed between 0 and 1, this
    # should be 0.5 exactly, but empirically this does not appear to be the
    # case and this value is better (see `test_decay_magnitude`).
    q = 0.494

    r = (2**bits - offset - np.asarray(decay)) / 2**bits  # decay ratio
    n = -np.log1p(x0 * (1 - r) / q) / np.log(r)  # solve y_n = 0 for n

    # principal_sum = (1./x0) sum_i^n x0 * r^i
    # loss_sum = (1./x0) sum_i^n sum_k^{i-1} q * r^k
    principal_sum = (1 - r**(n + 1)) / (1 - r)
    loss_sum = q / ((1 - r) * x0) * (n + 1 - (1 - r**(n+1))/(1 - r))
    return principal_sum - loss_sum


def scale_pes_errors(error, scale=1.):
    """Scale PES errors based on a scaling factor, round and clip."""
    error = scale * error
    error = np.round(error).astype(np.int32)
    q = error > 127
    if np.any(q):
        warnings.warn("Max PES error (%0.2e) greater than chip max (%0.2e). "
                      "Clipping." % (error.max() / scale, 127. / scale))
        error[q] = 127
    q = error < -127
    if np.any(q):
        warnings.warn("Min PES error (%0.2e) less than chip min (%0.2e). "
                      "Clipping." % (error.min() / scale, -127. / scale))
        error[q] = -127
    return error


def shift(x, s, **kwargs):
    if s < 0:
        return np.right_shift(x, -s, **kwargs)
    else:
        return np.left_shift(x, s, **kwargs)


def discretize_model(model):
    """Discretize a `.Model` in-place."""
    for block in model.blocks:
        discretize_block(block)


def discretize_block(block):
    w_maxs = [s.max_abs_weight() for s in block.synapses]
    w_max = max(w_maxs) if len(w_maxs) > 0 else 0

    p = discretize_compartment(block.compartment, w_max)
    for synapse in block.synapses:
        discretize_synapse(synapse, w_max, p['w_scale'], p['w_exp'])
    for probe in block.probes:
        discretize_probe(probe, p['v_scale'][0])


def discretize_compartment(comp, w_max):
    # --- discretize decayU and decayV
    # subtract 1 from decayU here because it gets added back by the chip
    decayU = comp.decayU * (2**12 - 1) - 1
    array_to_int(comp.decayU, np.clip(decayU, 0, 2**12 - 1))
    array_to_int(comp.decayV, comp.decayV * (2**12 - 1))

    # Compute factors for current and voltage decay. These factors
    # counteract the fact that for longer decays, the current (or voltage)
    # created by a single spike has a larger integral.
    u_infactor = (1. / decay_magnitude(comp.decayU, x0=2**21, offset=1)
                  if comp.scaleU else np.ones(comp.decayU.shape))
    v_infactor = (1. / decay_magnitude(comp.decayV, x0=2**21)
                  if comp.scaleV else np.ones(comp.decayV.shape))
    comp.scaleU = False
    comp.scaleV = False

    # --- discretize weights and vth
    # To avoid overflow, we can either lower vth_max or lower w_exp_max.
    # Lowering vth_max is more robust, but has the downside that it may
    # force smaller w_exp on connections than necessary, potentially
    # leading to lost weight bits (see discretize_weights).
    # Lowering w_exp_max can let us keep vth_max higher, but overflow
    # is still be possible on connections with many small inputs (uncommon)
    vth_max = VTH_MAX
    w_exp_max = 0

    b_max = np.abs(comp.bias).max()
    w_exp = 0

    if w_max > 1e-8:
        w_scale = (255. / w_max)
        s_scale = 1. / (u_infactor * v_infactor)

        for w_exp in range(w_exp_max, -8, -1):
            v_scale = s_scale * w_scale * SynapseFmt.get_scale(w_exp)
            b_scale = v_scale * v_infactor
            vth = np.round(comp.vth * v_scale)
            bias = np.round(comp.bias * b_scale)
            if (vth <= vth_max).all() and (np.abs(bias) <= BIAS_MAX).all():
                break
        else:
            raise BuildError("Could not find appropriate weight exponent")
    elif b_max > 1e-8:
        b_scale = BIAS_MAX / b_max
        while b_scale*b_max > 1:
            v_scale = b_scale / v_infactor
            w_scale = b_scale * u_infactor / SynapseFmt.get_scale(w_exp)
            vth = np.round(comp.vth * v_scale)
            bias = np.round(comp.bias * b_scale)
            if np.all(vth <= vth_max):
                break

            b_scale /= 2.
        else:
            raise BuildError("Could not find appropriate bias scaling")
    else:
        # reduce vth_max in this case to avoid overflow since we're setting
        # all vth to vth_max (esp. in learning with zeroed initial weights)
        vth_max = min(vth_max, 2**Q_BITS - 1)
        v_scale = np.array([vth_max / (comp.vth.max() + 1)])
        vth = np.round(comp.vth * v_scale)
        b_scale = v_scale * v_infactor
        bias = np.round(comp.bias * b_scale)
        w_scale = (v_scale * v_infactor * u_infactor
                   / SynapseFmt.get_scale(w_exp))

    vth_man, vth_exp = vth_to_manexp(vth)
    array_to_int(comp.vth, vth_man * 2**vth_exp)

    bias_man, bias_exp = bias_to_manexp(bias)
    array_to_int(comp.bias, bias_man * 2**bias_exp)

    # --- noise
    assert (v_scale[0] == v_scale).all()
    enable_noise = np.any(comp.enableNoise)
    noiseExp0 = np.round(np.log2(10.**comp.noiseExp0 * v_scale[0]))
    if enable_noise and noiseExp0 < 0:
        warnings.warn("Noise amplitude falls below lower limit")
    if enable_noise and noiseExp0 > 23:
        warnings.warn(
            "Noise amplitude exceeds upper limit (%d > 23)" % (noiseExp0,))
    comp.noiseExp0 = int(np.clip(noiseExp0, 0, 23))
    comp.noiseMantOffset0 = int(np.round(2*comp.noiseMantOffset0))

    # --- vmin and vmax
    assert (v_scale[0] == v_scale).all()
    vmin = v_scale[0] * comp.vmin
    vmax = v_scale[0] * comp.vmax
    vmine = np.clip(np.round(np.log2(-vmin + 1)), 0, 2 ** 5 - 1)
    comp.vmin = -2 ** vmine + 1
    vmaxe = np.clip(np.round((np.log2(vmax + 1) - 9) * 0.5), 0, 2 ** 3 - 1)
    comp.vmax = 2 ** (9 + 2 * vmaxe) - 1

    return dict(w_max=w_max,
                w_scale=w_scale,
                w_exp=w_exp,
                v_scale=v_scale)


def discretize_synapse(synapse, w_max, w_scale, w_exp):
    w_max_i = synapse.max_abs_weight()
    if synapse.learning:
        w_exp2 = synapse.learning_wgt_exp
        dw_exp = w_exp - w_exp2
    elif w_max_i > 1e-16:
        dw_exp = int(np.floor(np.log2(w_max / w_max_i)))
        assert dw_exp >= 0
        w_exp2 = max(w_exp - dw_exp, -6)
    else:
        w_exp2 = -6
        dw_exp = w_exp - w_exp2
    synapse.format(wgtExp=w_exp2)
    for w, idxs in zip(synapse.weights, synapse.indices):
        ws = w_scale[idxs] if is_iterable(w_scale) else w_scale
        array_to_int(w, discretize_weights(
            synapse.synapse_fmt, w * ws * 2 ** dw_exp))

    # discretize learning
    if synapse.learning:
        synapse.tracing_tau = int(np.round(synapse.tracing_tau))

        if is_iterable(w_scale):
            assert np.all(w_scale == w_scale[0])
        w_scale_i = w_scale[0] if is_iterable(w_scale) else w_scale

        # incorporate weight scale and difference in weight exponents
        # to learning rate, since these affect speed at which we learn
        ws = w_scale_i * 2 ** dw_exp
        synapse.learning_rate *= ws

        # Loihi down-scales learning factors based on the number of
        # overflow bits. Increasing learning rate maintains true rate.
        synapse.learning_rate *= 2 ** learn_overflow_bits(2)

        # TODO: Currently, Loihi learning rate fixed at 2**-7.
        # We should explore adjusting it for better performance.
        lscale = 2 ** -7 / synapse.learning_rate
        synapse.learning_rate *= lscale
        synapse.tracing_mag /= lscale

        # discretize learning rate into mantissa and exponent
        lr_exp = int(np.floor(np.log2(synapse.learning_rate)))
        lr_int = int(np.round(synapse.learning_rate * 2 ** (-lr_exp)))
        synapse.learning_rate = lr_int * 2 ** lr_exp
        synapse._lr_int = lr_int
        synapse._lr_exp = lr_exp
        assert lr_exp >= -7

        # discretize tracing mag into integer and fractional components
        mag_int, mag_frac = tracing_mag_int_frac(synapse.tracing_mag)
        if mag_int > 127:
            warnings.warn("Trace increment exceeds upper limit "
                          "(learning rate may be too large)")
            mag_int = 127
            mag_frac = 127
        synapse.tracing_mag = mag_int + mag_frac / 128.


def discretize_weights(
        synapse_fmt, w, dtype=np.int32, lossy_shift=True, check_result=True):
    """Takes weights and returns their quantized values with wgtExp.

    The actual weight to be put on the chip is this returned value
    divided by the ``scale`` attribute.

    Parameters
    ----------
    w : float ndarray
        Weights to be discretized, in the range -255 to 255.
    dtype : np.dtype, optional (Default: np.int32)
        Data type for discretized weights.
    lossy_shift : bool, optional (Default: True)
        Whether to mimic the two-part weight shift that currently happens
        on the chip, which can lose information for small wgtExp.
    check_results : bool, optional (Default: True)
        Whether to check that the discretized weights fall in
        the valid range for weights on the chip (-256 to 255).
    """
    s = synapse_fmt.shift_bits
    m = 2**(8 - s) - 1

    w = np.round(w / 2.**s).clip(-m, m).astype(dtype)
    s2 = s + synapse_fmt.wgtExp

    if lossy_shift:
        if s2 < 0:
            warnings.warn("Lost %d extra bits in weight rounding" % (-s2,))

            # Round before `s2` right shift. Just shifting would floor
            # everything resulting in weights biased towards being smaller.
            w = (np.round(w * 2.**s2) / 2**s2).clip(-m, m).astype(dtype)

        shift(w, s2, out=w)
        np.left_shift(w, 6, out=w)
    else:
        shift(w, 6 + s2, out=w)

    if check_result:
        ws = w // synapse_fmt.scale
        assert np.all(ws <= 255) and np.all(ws >= -256)

    return w


def discretize_probe(probe, v_scale):
    if probe.key == 'voltage' and probe.weights is not None:
        probe.weights /= v_scale
