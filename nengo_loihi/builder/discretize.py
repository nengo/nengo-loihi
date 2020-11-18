import logging
import warnings

import numpy as np
from nengo.exceptions import BuildError
from nengo.utils.numpy import is_iterable

from nengo_loihi.block import SynapseConfig
from nengo_loihi.nxsdk_obfuscation import d

VTH_MAN_MAX = d(b"MTMxMDcx", int)
VTH_EXP = d(b"Ng==", int)
VTH_MAX = VTH_MAN_MAX * 2 ** VTH_EXP

BIAS_MAN_MAX = d(b"NDA5NQ==", int)
BIAS_EXP_MAX = d(b"Nw==", int)
BIAS_MAX = BIAS_MAN_MAX * 2 ** BIAS_EXP_MAX

# number of bits for synapse accumulator
Q_BITS = d(b"MjE=", int)
# number of bits for compartment input (u)
U_BITS = d(b"MjM=", int)
# number of bits in learning accumulator (not incl. sign)
LEARN_BITS = d(b"MTU=", int)
# extra least-significant bits added to weights for learning
LEARN_FRAC = d(b"Nw==", int)

logger = logging.getLogger(__name__)


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
    return factor_bits * n_factors + mantissa_bits - LEARN_BITS


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
    man = np.round(vth / 2 ** exp).astype(np.int32)
    assert (man > 0).all()
    assert (man <= VTH_MAN_MAX).all()
    return man, exp


def bias_to_manexp(bias):
    r = np.maximum(np.abs(bias) / BIAS_MAN_MAX, 1)
    exp = np.ceil(np.log2(r)).astype(np.int32)
    man = np.round(bias / 2 ** exp).astype(np.int32)
    assert (exp >= 0).all()
    assert (exp <= BIAS_EXP_MAX).all()
    assert (np.abs(man) <= BIAS_MAN_MAX).all()
    return man, exp


def tracing_mag_int_frac(mag):
    """Split trace magnitude into integer and fractional components for chip"""
    mag_int = int(mag)
    mag_frac = int(d(b"MTI4", int) * (mag - mag_int))
    return mag_int, mag_frac


def decay_int(x, decay, bits=None, offset=0, out=None):
    """Decay integer values using a decay constant.

    The decayed value is given by::

        sign(x) * floor(abs(x) * (2**bits - offset - decay) / 2**bits)
    """
    if out is None:
        out = np.zeros_like(x)
    if bits is None:
        bits = d(b"MTI=", int)
    r = (2 ** bits - offset - np.asarray(decay)).astype(np.int64)
    np.right_shift(np.abs(x) * r, bits, out=out)
    return np.sign(x) * out


def decay_magnitude(decay, x0=2 ** 21, bits=12, offset=0):
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

    r = (2 ** bits - offset - np.asarray(decay)) / 2 ** bits  # decay ratio
    n = -(np.log1p(x0 * (1 - r) / q)) / np.log(r)  # solve y_n = 0 for n

    # principal_sum = (1./x0) sum_i^n x0 * r^i
    # loss_sum = (1./x0) sum_i^n sum_k^{i-1} q * r^k
    principal_sum = (1 - r ** (n + 1)) / (1 - r)
    loss_sum = q / ((1 - r) * x0) * (n + 1 - (1 - r ** (n + 1)) / (1 - r))
    return principal_sum - loss_sum


def scale_pes_errors(error, scale=1.0):
    """Scale PES errors based on a scaling factor, round and clip."""
    error = scale * error
    error = np.round(error).astype(np.int32)
    max_err = d(b"MTI3", int)
    q = error > max_err
    if np.any(q):
        warnings.warn(
            "Received PES error greater than chip max (%0.2e). "
            "Consider changing `Model.pes_error_scale`." % (max_err / scale,)
        )
        logger.debug(
            "PES error %0.2e > %0.2e (chip max)", np.max(error) / scale, max_err / scale
        )
        error[q] = max_err
    q = error < -max_err
    if np.any(q):
        warnings.warn(
            "Received PES error less than chip min (%0.2e). "
            "Consider changing `Model.pes_error_scale`." % (-max_err / scale,)
        )
        logger.debug(
            "PES error %0.2e < %0.2e (chip min)",
            np.min(error) / scale,
            -max_err / scale,
        )
        error[q] = -max_err
    return error


def shift(x, s, **kwargs):
    if s < 0:
        return np.right_shift(x, -s, **kwargs)
    else:
        return np.left_shift(x, s, **kwargs)


def discretize_model(model):
    """Discretize a `.Model` in-place.

    Turns a floating-point `.Model` into a discrete (integer) model
    appropriate for Loihi.

    Parameters
    ----------
    model : `.Model`
        The model to discretize.
    """
    v_scale = {}

    for block in model.blocks:
        v_scale[block] = discretize_block(block)

    for probe in model.probes:
        for i, block in enumerate(probe.target):
            discretize_probe(probe, i, v_scale[block])


def discretize_block(block):
    """Discretize a `.LoihiBlock` in-place.

    Turns a floating-point `.LoihiBlock` into a discrete (integer)
    block appropriate for Loihi.

    Parameters
    ----------
    block : `.LoihiBlock`
        The block to discretize.
    """
    w_maxs = [s.max_abs_weight() for s in block.synapses]
    w_max = max(w_maxs) if len(w_maxs) > 0 else 0

    p = discretize_compartment(block.compartment, w_max)
    for synapse in block.synapses:
        discretize_synapse(synapse, w_max, p["w_scale"], p["w_exp"])
    return p["v_scale"]


def discretize_compartment(comp, w_max):
    """Discretize a `.Compartment` in-place.

    Turns a floating-point `.Compartment` into a discrete (integer)
    block appropriate for Loihi.

    Parameters
    ----------
    comp : `.Compartment`
        The compartment to discretize.
    w_max : float
        The largest connection weight in the `.LoihiBlock` containing
        ``comp``. Used to set several scaling factors.
    """

    # --- discretize decay_u and decay_v
    # subtract 1 from decay_u here because it gets added back by the chip
    decay_u = comp.decay_u * d(b"NDA5NQ==", int) - 1
    array_to_int(comp.decay_u, np.clip(decay_u, 0, d(b"NDA5NQ==", int)))
    array_to_int(comp.decay_v, comp.decay_v * d(b"NDA5NQ==", int))

    # Compute factors for current and voltage decay. These factors
    # counteract the fact that for longer decays, the current (or voltage)
    # created by a single spike has a larger integral.
    u_infactor = (
        1.0 / decay_magnitude(comp.decay_u, x0=d(b"MjA5NzE1Mg==", int), offset=1)
        if comp.scale_u
        else np.ones(comp.decay_u.shape)
    )
    v_infactor = (
        1.0 / decay_magnitude(comp.decay_v, x0=d(b"MjA5NzE1Mg==", int))
        if comp.scale_v
        else np.ones(comp.decay_v.shape)
    )
    comp.scale_u = False
    comp.scale_v = False

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
        w_scale = d(b"MjU1", float) / w_max
        s_scale = 1.0 / (u_infactor * v_infactor)

        for w_exp in range(w_exp_max, d(b"LTg=", int), d(b"LTE=", int)):
            v_scale = s_scale * w_scale * SynapseConfig.get_scale(w_exp)
            b_scale = v_scale * v_infactor
            vth = np.round(comp.vth * v_scale)
            bias = np.round(comp.bias * b_scale)
            if (vth <= vth_max).all() and (np.abs(bias) <= BIAS_MAX).all():
                break
        else:
            raise BuildError("Could not find appropriate weight exponent")
    elif b_max > 1e-8:
        b_scale = BIAS_MAX / b_max
        while b_scale * b_max > 1:
            v_scale = b_scale / v_infactor
            w_scale = b_scale * u_infactor / SynapseConfig.get_scale(w_exp)
            vth = np.round(comp.vth * v_scale)
            bias = np.round(comp.bias * b_scale)
            if np.all(vth <= vth_max):
                break

            b_scale /= 2.0
        else:
            raise BuildError("Could not find appropriate bias scaling")
    else:
        # reduce vth_max in this case to avoid overflow since we're setting
        # all vth to vth_max (esp. in learning with zeroed initial weights)
        vth_max = min(vth_max, 2 ** Q_BITS - 1)
        v_scale = np.array([vth_max / (comp.vth.max() + 1)])
        vth = np.round(comp.vth * v_scale)
        b_scale = v_scale * v_infactor
        bias = np.round(comp.bias * b_scale)
        w_scale = v_scale * v_infactor * u_infactor / SynapseConfig.get_scale(w_exp)

    vth_man, vth_exp = vth_to_manexp(vth)
    array_to_int(comp.vth, vth_man * 2 ** vth_exp)

    bias_man, bias_exp = bias_to_manexp(bias)
    array_to_int(comp.bias, bias_man * 2 ** bias_exp)

    assert (v_scale[0] == v_scale).all()
    v_scale = v_scale[0]

    # --- noise
    enable_noise = np.any(comp.enable_noise)
    noise_exp = np.round(np.log2(10.0 ** comp.noise_exp * v_scale))
    if enable_noise and noise_exp < d(b"MQ==", int):
        warnings.warn("Noise amplitude falls below lower limit")
        enable_noise = False
    if enable_noise and noise_exp > d(b"MjM=", int):
        warnings.warn("Noise amplitude exceeds upper limit (%d > 23)" % (noise_exp,))
    comp.noise_exp = int(np.clip(noise_exp, d(b"MQ==", int), d(b"MjM=", int)))
    comp.noise_offset = int(np.round(2 * comp.noise_offset))

    # --- vmin and vmax
    vmin = v_scale * comp.vmin
    vmax = v_scale * comp.vmax
    vmine = np.clip(np.round(np.log2(-vmin + 1)), 0, 2 ** 5 - 1)
    comp.vmin = -(2 ** vmine) + 1
    vmaxe = np.clip(np.round((np.log2(vmax + 1) - 9) * 0.5), 0, 2 ** 3 - 1)
    comp.vmax = 2 ** (9 + 2 * vmaxe) - 1

    return dict(w_max=w_max, w_scale=w_scale, w_exp=w_exp, v_scale=v_scale)


def discretize_synapse(synapse, w_max, w_scale, w_exp):
    """Discretize a `.Synapse` in-place.

    Turns a floating-point `.Synapse` into a discrete (integer)
    block appropriate for Loihi.

    Parameters
    ----------
    synapse : `.Synapse`
        The synapse to discretize.
    w_max : float
        The largest connection weight in the `.LoihiBlock` containing
        ``synapse``. Used to scale weights appropriately.
    w_scale : float
        Connection weight scaling factor. Usually computed by
        `.discretize_compartment`.
    w_exp : float
        Exponent on the connection weight scaling factor. Usually computed by
        `.discretize_compartment`.
    """
    w_max_i = synapse.max_abs_weight()
    if synapse.learning:
        w_exp2 = synapse.learning_wgt_exp
        dw_exp = w_exp - w_exp2
    elif w_max_i > 1e-16:
        dw_exp = int(np.floor(np.log2(w_max / w_max_i)))
        assert dw_exp >= 0
        w_exp2 = max(w_exp - dw_exp, d(b"LTY=", int))
    else:
        w_exp2 = d(b"LTY=", int)
        dw_exp = w_exp - w_exp2
    synapse.format(weight_exp=w_exp2)
    for w, idxs in zip(synapse.weights, synapse.indices):
        ws = w_scale[idxs] if is_iterable(w_scale) else w_scale
        array_to_int(w, discretize_weights(synapse.synapse_cfg, w * ws * 2 ** dw_exp))

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
        assert lr_exp >= d(b"LTc=", int)

        # discretize tracing mag into integer and fractional components
        mag_int, mag_frac = tracing_mag_int_frac(synapse.tracing_mag)
        if mag_int > d(b"MTI3", int):
            warnings.warn(
                "Trace increment exceeds upper limit "
                "(learning rate may be too large)"
            )
            mag_int = d(b"MTI3", int)
            mag_frac = d(b"MTI3", int)
        synapse.tracing_mag = mag_int + mag_frac / d(b"MTI4", float)


def discretize_weights(
    synapse_cfg, w, dtype=np.int32, lossy_shift=True, check_result=True
):
    """Takes weights and returns their quantized values with weight_exp.

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
        on the chip, which can lose information for small weight_exp.
    check_results : bool, optional (Default: True)
        Whether to check that the discretized weights fall in
        the valid range for weights on the chip (-256 to 255).
    """
    s = synapse_cfg.shift_bits
    m = 2 ** (d(b"OA==", int) - s) - 1

    w = np.round(w / 2.0 ** s).clip(-m, m).astype(dtype)
    s2 = s + synapse_cfg.weight_exp

    if lossy_shift:
        if s2 < 0:
            warnings.warn("Lost %d extra bits in weight rounding" % (-s2,))

            # Round before `s2` right shift. Just shifting would floor
            # everything resulting in weights biased towards being smaller.
            w = (np.round(w * 2.0 ** s2) / 2 ** s2).clip(-m, m).astype(dtype)

        shift(w, s2, out=w)
        np.left_shift(w, d(b"Ng==", int), out=w)
    else:
        shift(w, d(b"Ng==", int) + s2, out=w)

    if check_result:
        ws = w // synapse_cfg.scale
        assert np.all(ws <= d(b"MjU1", int)) and np.all(ws >= d(b"LTI1Ng==", int))

    return w


def discretize_probe(probe, i, v_scale):
    if probe.key == "voltage" and probe.weights[i] is not None:
        probe.weights[i] /= v_scale
