"""
TODO:
- Loihi refractory periods get rounded down. For example, if input to a neuron
is enough to make it fire every timestep, and it has a ref of dt, then it will
fire every timestep. This is currently covered in the spiking neuron here,
but not in the rate one.
- Fix gain and bias computation to accommodate the rounded down ref periods
- Allow for leaky neurons
"""

import numpy as np

import nengo
import nengo_loihi
# from nengo.builder import Builder
# from nengo.builder.neurons import build_lif
# from nengo.neurons import LIFRate


class LoihiLIFRate(nengo.neurons.LIFRate):
    TAU_RC_MAX = 1e30

    # def gain_bias(self, max_rates, intercepts):
    #     """Analytically determine gain, bias."""
    #     max_rates = np.array(max_rates, dtype=float, copy=False, ndmin=1)
    #     intercepts = np.array(intercepts, dtype=float, copy=False, ndmin=1)

    #     inv_tau_ref = 1. / self.tau_ref if self.tau_ref > 0 else np.inf
    #     if np.any(max_rates > inv_tau_ref):
    #         raise ValidationError("Max rates must be below the inverse "
    #                               "refractory period (%0.3f)" % inv_tau_ref,
    #                               attr='max_rates', obj=self)

    #     x = 1.0 / (1.0/max_rates - self.tau_ref)
    #     gain = x / (1 - intercepts)
    #     bias = 1 - gain * intercepts
    #     return gain, bias

    # def max_rates_intercepts(self, gain, bias):
    #     """Compute the inverse of gain_bias."""
    #     intercepts = (1 - bias) / gain
    #     max_rates = 1.0 / (self.tau_ref + 1.0/(gain + bias - 1))
    #     if not np.all(np.isfinite(max_rates)):
    #         warnings.warn("Non-finite values detected in `max_rates`; this "
    #                       "probably means that `gain` was too small.")
    #     return max_rates, intercepts

    def rates(self, x, gain, bias):
        J = self.current(x, gain, bias)
        out = np.zeros_like(J)
        LoihiLIFRate.step_math(self, dt=0.001, J=J, output=out)
        return out

    def step_math(self, dt, J, output):
        """Implement the NIFRate nonlinearity."""
        output[:] = 0  # faster than output[j <= 0] = 0

        if self.tau_rc > self.TAU_RC_MAX:
            j = J
            p = self.tau_ref + dt / j[j > 0]
            output[j > 0] = (self.amplitude / dt) / np.ceil(p / dt)
        else:
            j = J - 1
            p = self.tau_ref + self.tau_rc * np.log1p(1. / j[j > 0])
            output[j > 0] = (self.amplitude / dt) / np.ceil(p / dt)


class LoihiLIF(LoihiLIFRate, nengo.neurons.LIF):
    def step_math(self, dt, J, spiked, voltage, refractory_time):
        refractory_time -= dt
        if self.tau_rc > self.TAU_RC_MAX:
            delta_t = (1 - refractory_time/dt).clip(0, 1)
            voltage += J * delta_t
        else:
            delta_t = (dt - refractory_time).clip(0, dt)
            voltage -= (J - voltage) * np.expm1(-delta_t / self.tau_rc)

        # determine which neurons spiked (set them to 1/dt, else 0)
        spiked_mask = voltage > 1
        spiked[:] = spiked_mask * (self.amplitude / dt)

        # set v(0) = 1 and solve for t to compute the spike time
        # t_spike = dt - (voltage[spiked_mask] - 1) / J[spiked_mask]

        # set spiked voltages to zero, refractory times to tau_ref, and
        # rectify negative voltages to a floor of min_voltage
        voltage[voltage < self.min_voltage] = self.min_voltage
        voltage[spiked_mask] = 0
        refractory_time[spiked_mask] = self.tau_ref + dt


@nengo.builder.Builder.register(LoihiLIFRate)
def nengo_build_loihi_lif_rate(model, lif, neurons):
    return nengo.builder.neurons.build_lif(model, lif, neurons)


@nengo.builder.Builder.register(LoihiLIF)
def nengo_build_loihi_lif(model, lif, neurons):
    return nengo.builder.neurons.build_lif(model, lif, neurons)


@nengo_loihi.builder.Builder.register(LoihiLIF)
def nengo_loihi_build_loihi_lif(model, lif, neurons, group):
    nengo_loihi.builder.build_lif(model, lif, neurons, group)
    # group.configure_lif(
    #     tau_rc=lif.tau_rc,
    #     tau_ref=lif.tau_ref,
    #     dt=model.dt)
