import warnings

import numpy as np

import nengo
from nengo.exceptions import ValidationError
from nengo.neurons import NeuronType
from nengo.params import NumberParam


def loihi_lif_rates(neuron_type, x, gain, bias, dt):
    # discretize tau_ref as per CxGroup.configure_lif
    tau_ref = dt * np.round(neuron_type.tau_ref / dt)
    j = neuron_type.current(x, gain, bias) - 1

    out = np.zeros_like(j)
    period = tau_ref + neuron_type.tau_rc * np.log1p(1. / j[j > 0])
    out[j > 0] = (neuron_type.amplitude / dt) / np.ceil(period / dt)
    return out


def loihi_spikingrectifiedlinear_rates(neuron_type, x, gain, bias, dt):
    j = neuron_type.current(x, gain, bias)

    out = np.zeros_like(j)
    period = 1. / j[j > 0]
    out[j > 0] = (neuron_type.amplitude / dt) / np.ceil(period / dt)
    return out


def loihi_rates(neuron_type, x, gain, bias, dt):
    for cls in type(neuron_type).__mro__:
        if cls in loihi_rate_functions:
            return loihi_rate_functions[cls](neuron_type, x, gain, bias, dt)
    return neuron_type.rates(x, gain, bias)


loihi_rate_functions = {
    nengo.LIF: loihi_lif_rates,
    nengo.SpikingRectifiedLinear: loihi_spikingrectifiedlinear_rates,
}


class LoihiLIF(nengo.LIF):
    def rates(self, x, gain, bias, dt=0.001):
        return loihi_lif_rates(self, x, gain, bias, dt)

    def step_math(self, dt, J, spiked, voltage, refractory_time):
        tau_ref = dt * np.round(self.tau_ref / dt)
        refractory_time -= dt

        delta_t = (dt - refractory_time).clip(0, dt)
        voltage -= (J - voltage) * np.expm1(-delta_t / self.tau_rc)

        spiked_mask = voltage > 1
        spiked[:] = spiked_mask * (self.amplitude / dt)

        voltage[voltage < self.min_voltage] = self.min_voltage
        voltage[spiked_mask] = 0
        refractory_time[spiked_mask] = tau_ref + dt


class LoihiSpikingRectifiedLinear(nengo.SpikingRectifiedLinear):
    def rates(self, x, gain, bias, dt=0.001):
        return loihi_spikingrectifiedlinear_rates(self, x, gain, bias, dt)

    def step_math(self, dt, J, spiked, voltage):
        voltage += J * dt

        spiked_mask = voltage > 1
        spiked[:] = spiked_mask * (self.amplitude / dt)

        voltage[voltage < 0] = 0
        voltage[spiked_mask] = 0


class NIFRate(NeuronType):
    """Non-spiking version of the non-leaky integrate-and-fire (NIF) model.

    Parameters
    ----------
    tau_ref : float
        Absolute refractory period, in seconds. This is how long the
        membrane voltage is held at zero after a spike.
    amplitude : float
        Scaling factor on the neuron output. Corresponds to the relative
        amplitude of the output spikes of the neuron.
    """

    probeable = ('rates',)

    tau_ref = NumberParam('tau_ref', low=0)
    amplitude = NumberParam('amplitude', low=0, low_open=True)

    def __init__(self, tau_ref=0.002, amplitude=1):
        super(NIFRate, self).__init__()
        self.tau_ref = tau_ref
        self.amplitude = amplitude

    @property
    def _argreprs(self):
        args = []
        if self.tau_ref != 0.002:
            args.append("tau_ref=%s" % self.tau_ref)
        if self.amplitude != 1:
            args.append("amplitude=%s" % self.amplitude)
        return args

    def gain_bias(self, max_rates, intercepts):
        """Analytically determine gain, bias."""
        max_rates = np.array(max_rates, dtype=float, copy=False, ndmin=1)
        intercepts = np.array(intercepts, dtype=float, copy=False, ndmin=1)

        inv_tau_ref = 1. / self.tau_ref if self.tau_ref > 0 else np.inf
        if np.any(max_rates > inv_tau_ref):
            raise ValidationError("Max rates must be below the inverse "
                                  "refractory period (%0.3f)" % inv_tau_ref,
                                  attr='max_rates', obj=self)

        x = 1.0 / (1.0/max_rates - self.tau_ref)
        gain = x / (1 - intercepts)
        bias = 1 - gain * intercepts
        return gain, bias

    def max_rates_intercepts(self, gain, bias):
        """Compute the inverse of gain_bias."""
        intercepts = (1 - bias) / gain
        max_rates = 1.0 / (self.tau_ref + 1.0/(gain + bias - 1))
        if not np.all(np.isfinite(max_rates)):
            warnings.warn("Non-finite values detected in `max_rates`; this "
                          "probably means that `gain` was too small.")
        return max_rates, intercepts

    def rates(self, x, gain, bias):
        """Always use NIFRate to determine rates."""
        J = self.current(x, gain, bias)
        out = np.zeros_like(J)
        # Use NIFRate's step_math explicitly to ensure rate approximation
        NIFRate.step_math(self, dt=1, J=J, output=out)
        return out

    def step_math(self, dt, J, output):
        """Implement the NIFRate nonlinearity."""
        j = J - 1
        output[:] = 0  # faster than output[j <= 0] = 0
        output[j > 0] = self.amplitude / (self.tau_ref + 1./j[j > 0])


# class NIF(NIFRate):
#     """Spiking version of non-leaky integrate-and-fire (NIF) neuron model.

#     Parameters
#     ----------
#     tau_ref : float
#         Absolute refractory period, in seconds. This is how long the
#         membrane voltage is held at zero after a spike.
#     min_voltage : float
#         Minimum value for the membrane voltage. If ``-np.inf``, the voltage
#         is never clipped.
#     amplitude : float
#         Scaling factor on the neuron output. Corresponds to the relative
#         amplitude of the output spikes of the neuron.
#     """

#     probeable = ('spikes', 'voltage', 'refractory_time')

#     min_voltage = NumberParam('min_voltage', high=0)

#     def __init__(self, tau_ref=0.002, min_voltage=0, amplitude=1):
#         super(NIF, self).__init__(tau_ref=tau_ref, amplitude=amplitude)
#         self.min_voltage = min_voltage

#     def step_math(self, dt, J, spiked, voltage, refractory_time):
#         refractory_time -= dt
#         delta_t = (dt - refractory_time).clip(0, dt)
#         voltage += J * delta_t

#         # determine which neurons spiked (set them to 1/dt, else 0)
#         spiked_mask = voltage > 1
#         spiked[:] = spiked_mask * (self.amplitude / dt)

#         # set v(0) = 1 and solve for t to compute the spike time
#         t_spike = dt - (voltage[spiked_mask] - 1) / J[spiked_mask]

#         # set spiked voltages to zero, refractory times to tau_ref, and
#         # rectify negative voltages to a floor of min_voltage
#         voltage[voltage < self.min_voltage] = self.min_voltage
#         voltage[spiked_mask] = 0
#         refractory_time[spiked_mask] = self.tau_ref + t_spike


class NIF(NIFRate):
    probeable = ('spikes', 'voltage', 'refractory_time')

    min_voltage = NumberParam('min_voltage', high=0)

    def __init__(self, tau_ref=0.002, min_voltage=0, amplitude=1):
        super(NIF, self).__init__(tau_ref=tau_ref, amplitude=amplitude)
        self.min_voltage = min_voltage

    def step_math(self, dt, J, spiked, voltage, refractory_time):
        refractory_time -= dt
        delta_t = (dt - refractory_time).clip(0, dt)
        voltage += J * delta_t

        # determine which neurons spiked (set them to 1/dt, else 0)
        spiked_mask = voltage > 1
        spiked[:] = spiked_mask * (self.amplitude / dt)

        voltage[voltage < self.min_voltage] = self.min_voltage
        voltage[spiked_mask] -= 1
        refractory_time[spiked_mask] = self.tau_ref + dt


@nengo.builder.Builder.register(NIFRate)
def nengo_build_nif_rate(model, nif_rate, neurons):
    return nengo.builder.neurons.build_lif(model, nif_rate, neurons)


@nengo.builder.Builder.register(NIF)
def nengo_build_nif(model, nif, neurons):
    return nengo.builder.neurons.build_lif(model, nif, neurons)
