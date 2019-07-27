from nengo.neurons import LIF, SpikingRectifiedLinear
import numpy as np

from nengo_loihi.compat import HAS_TF, tf


def discretize_tau_rc(tau_rc, dt):
    """Discretize tau_rc as per discretize_compartment.

    Parameters
    ----------
    tau_rc : float
        The neuron membrane time constant.
    dt : float
        The simulator time step.
    """
    lib = tf if HAS_TF and isinstance(tau_rc, tf.Tensor) else np

    decay_rc = -(lib.expm1(-dt / tau_rc))
    decay_rc = lib.round(decay_rc * (2 ** 12 - 1)) / (2 ** 12 - 1)
    return -dt / lib.log1p(-decay_rc)


def discretize_tau_ref(tau_ref, dt):
    """Discretize tau_ref as per Compartment.configure_lif.

    Parameters
    ----------
    tau_rc : float
        The neuron membrane time constant.
    dt : float
        The simulator time step.
    """
    lib = tf if HAS_TF and isinstance(tau_ref, tf.Tensor) else np

    return dt * lib.round(tau_ref / dt)


def loihi_lif_rates(neuron_type, x, gain, bias, dt):
    tau_ref = discretize_tau_ref(neuron_type.tau_ref, dt)
    tau_rc = discretize_tau_rc(neuron_type.tau_rc, dt)

    j = neuron_type.current(x, gain, bias) - 1
    out = np.zeros_like(j)
    period = tau_ref + tau_rc * np.log1p(1.0 / j[j > 0])
    out[j > 0] = (neuron_type.amplitude / dt) / np.ceil(period / dt)
    return out


def loihi_spikingrectifiedlinear_rates(neuron_type, x, gain, bias, dt):
    j = neuron_type.current(x, gain, bias)

    out = np.zeros_like(j)
    period = 1.0 / j[j > 0]
    out[j > 0] = (neuron_type.amplitude / dt) / np.ceil(period / dt)
    return out


def _broadcast_rates_inputs(x, gain, bias):
    x = np.array(x, dtype=float, copy=False, ndmin=1)
    gain = np.array(gain, dtype=float, copy=False, ndmin=1)
    bias = np.array(bias, dtype=float, copy=False, ndmin=1)
    if x.ndim == 1:
        x = x[:, np.newaxis] * np.ones(gain.shape[-1])
    return x, gain, bias


def loihi_rates(neuron_type, x, gain, bias, dt):
    x, gain, bias = _broadcast_rates_inputs(x, gain, bias)
    for cls in type(neuron_type).__mro__:
        if cls in loihi_rate_functions:
            return loihi_rate_functions[cls](neuron_type, x, gain, bias, dt)
    return neuron_type.rates(x, gain, bias)


def nengo_rates(neuron_type, x, gain, bias):
    """Call NeuronType.rates with Nengo 3.0 broadcasting rules"""
    x, gain, bias = _broadcast_rates_inputs(x, gain, bias)
    return neuron_type.rates(x, gain, bias)


loihi_rate_functions = {
    LIF: loihi_lif_rates,
    SpikingRectifiedLinear: loihi_spikingrectifiedlinear_rates,
}


class LoihiLIF(LIF):
    """Simulate LIF neurons as done by Loihi.

    On Loihi, the inter-spike interval has to be an integer. This causes
    aliasing the firing rates where a wide variety of inputs can produce the
    same output firing rate. This class reproduces this effect, as well as
    the discretization of some of the neuron parameters. It can be used in
    e.g. ``nengo`` or ``nengo_dl`` to reproduce these unique Loihi effects.

    Parameters
    ----------
    nengo_dl_noise : NeuronOutputNoise
        Noise added to the rate-neuron output when training with this neuron
        type in ``nengo_dl``.
    """

    def __init__(
        self,
        tau_rc=0.02,
        tau_ref=0.002,
        min_voltage=0,
        amplitude=1,
        nengo_dl_noise=None,
    ):
        super(LoihiLIF, self).__init__(
            tau_rc=tau_rc, tau_ref=tau_ref, min_voltage=min_voltage, amplitude=amplitude
        )
        self.nengo_dl_noise = nengo_dl_noise

    @property
    def _argreprs(self):
        args = super(LoihiLIF, self)._argreprs
        if self.nengo_dl_noise is not None:
            args.append("nengo_dl_noise=%s" % self.nengo_dl_noise)
        return args

    def rates(self, x, gain, bias, dt=0.001):
        return loihi_lif_rates(self, x, gain, bias, dt)

    def step_math(self, dt, J, spiked, voltage, refractory_time):
        tau_ref = discretize_tau_ref(self.tau_ref, dt)

        refractory_time -= dt
        delta_t = (dt - refractory_time).clip(0, dt)
        voltage -= (J - voltage) * np.expm1(-delta_t / self.tau_rc)

        spiked_mask = voltage > 1
        spiked[:] = spiked_mask * (self.amplitude / dt)

        voltage[voltage < self.min_voltage] = self.min_voltage
        voltage[spiked_mask] = 0
        refractory_time[spiked_mask] = tau_ref + dt


class LoihiSpikingRectifiedLinear(SpikingRectifiedLinear):
    """Simulate spiking rectified linear neurons as done by Loihi.

    On Loihi, the inter-spike interval has to be an integer. This causes
    aliasing in the firing rates such that a wide variety of inputs produce the
    same output firing rate. This class reproduces this effect. It can be used
    in e.g. ``nengo`` or ``nengo_dl`` to reproduce these unique Loihi effects.
    """

    def rates(self, x, gain, bias, dt=0.001):
        return loihi_spikingrectifiedlinear_rates(self, x, gain, bias, dt)

    def step_math(self, dt, J, spiked, voltage):
        voltage += J * dt

        spiked_mask = voltage > 1
        spiked[:] = spiked_mask * (self.amplitude / dt)

        voltage[voltage < 0] = 0
        voltage[spiked_mask] = 0


class NeuronOutputNoise:
    """Noise added to the output of a rate neuron.

    Often used when training deep networks with rate neurons for final
    implementation in spiking neurons to simulate the variability
    caused by spiking.
    """

    pass


class LowpassRCNoise(NeuronOutputNoise):
    """Noise model combining Lowpass synapse and neuron membrane filters.

    Samples "noise" (i.e. variability) from a regular spike train filtered
    by the following transfer function, where :math:`\\tau_{rc}` is the
    membrane time constant and :math:`\\tau_s` is the synapse time constant:

    .. math::

        H(s) = [(\\tau_s s + 1) (\\tau_{rc} s + 1)]^{-1}

    See [1]_ for background and derivations.

    Attributes
    ----------
    tau_s : float
        Time constant for Lowpass synaptic filter.

    References
    ----------
    .. [1] E. Hunsberger (2018) "Spiking Deep Neural Networks: Engineered and
       Biological Approaches to Object Recognition." PhD thesis. pp. 106--113.
       (http://compneuro.uwaterloo.ca/publications/hunsberger2018.html)
    """

    def __init__(self, tau_s):
        self.tau_s = tau_s

    def __repr__(self):
        return "%s(tau_s=%s)" % (type(self).__name__, self.tau_s)


class AlphaRCNoise(NeuronOutputNoise):
    """Noise model combining Alpha synapse and neuron membrane filters.

    Samples "noise" (i.e. variability) from a regular spike train filtered
    by the following transfer function, where :math:`\\tau_{rc}` is the
    membrane time constant and :math:`\\tau_s` is the synapse time constant:

    .. math::

        H(s) = [(\\tau_s s + 1)^2 (\\tau_{rc} s + 1)]^{-1}

    See [1]_ for background and derivations.

    Attributes
    ----------
    tau_s : float
        Time constant for Alpha synaptic filter.

    References
    ----------
    .. [1] E. Hunsberger (2018) "Spiking Deep Neural Networks: Engineered and
       Biological Approaches to Object Recognition." PhD thesis. pp. 106--113.
       (http://compneuro.uwaterloo.ca/publications/hunsberger2018.html)
    """

    def __init__(self, tau_s):
        self.tau_s = tau_s

    def __repr__(self):
        return "%s(tau_s=%s)" % (type(self).__name__, self.tau_s)
