import logging
import warnings

import numpy as np

from nengo_loihi.compat import HAS_DL
from nengo_loihi.neurons import (
    AlphaRCNoise,
    discretize_tau_rc,
    discretize_tau_ref,
    LoihiLIF,
    LoihiSpikingRectifiedLinear,
    LowpassRCNoise,
)

if HAS_DL:
    import nengo_dl
    import nengo_dl.neuron_builders
    import tensorflow as tf
    from tensorflow.python.keras.utils import tf_utils
else:  # pragma: no cover
    # Empty classes so that we can define the subclasses even though
    # we will never use them, as they are only used in the `install`
    # function that can only run if nengo_dl is importable.
    class nengo_dl:
        class neuron_builders:
            class LIFBuilder:
                pass

            class SpikingRectifiedLinearBuilder:
                pass


logger = logging.getLogger(__name__)


class NoiseBuilder:
    """Build noise classes in ``nengo_dl``.

    Attributes
    ----------
    noise_models : list of NeuronOutputNoise
        The noise models used for each op/signal.
    """

    builders = {}

    def __init__(self, ops, signals, config, noise_models):
        self.noise_models = noise_models
        self.dtype = signals.dtype
        self.np_dtype = self.dtype.as_numpy_dtype()

    @classmethod
    def build(cls, ops, signals, config):
        """Create a NoiseBuilder for the provided ops."""

        noise_models = [getattr(op.neurons, "nengo_dl_noise", None) for op in ops]
        model_type = type(noise_models[0]) if len(noise_models) > 0 else None
        equal_types = all(type(m) is model_type for m in noise_models)

        if not equal_types:
            raise NotImplementedError(
                "Multiple noise models for the same neuron type is not supported"
            )

        return cls.builders[model_type](ops, signals, config, noise_models)

    @classmethod
    def register(cls, noise_builder):
        """A decorator for adding a class to the list of builders.

        Raises a warning if a builder already exists for the class.

        Parameters
        ----------
        noise_builder : NoiseBuilder
            The NoiseBuilder subclass that the decorated class builds.
        """

        def register_builder(build_cls):
            if noise_builder in cls.builders:
                warnings.warn(
                    "Type '%s' already has a builder. Overwriting." % noise_builder
                )
            cls.builders[noise_builder] = build_cls
            return build_cls

        return register_builder

    def generate(self, period, tau_rc=None):
        """Generate TensorFlow code to implement these noise models.

        Parameters
        ----------
        period : tf.Tensor
            The inter-spike periods of the neurons to add noise to.
        tau_rc : tf.Tensor
            The membrane time constant of the neurons (used by some noise
            models).
        """
        raise NotImplementedError("Subclass must implement")


@NoiseBuilder.register(type(None))
class NoNoiseBuilder(NoiseBuilder):
    """nengo_dl builder for if there is no noise model."""

    def generate(self, period, tau_rc=None):
        return tf.math.reciprocal(period)


@NoiseBuilder.register(LowpassRCNoise)
class LowpassRCNoiseBuilder(NoiseBuilder):
    """nengo_dl builder for the LowpassRCNoise model."""

    def __init__(self, ops, signals, *args, **kwargs):
        super(LowpassRCNoiseBuilder, self).__init__(ops, signals, *args, **kwargs)

        # tau_s is the time constant of the synaptic filter
        tau_s = np.concatenate(
            [
                model.tau_s * np.ones((1, op.J.shape[0]), dtype=self.np_dtype)
                for model, op in zip(self.noise_models, ops)
            ]
        )
        self.tau_s = tf.constant(tau_s, dtype=self.dtype)

    def generate(self, period, tau_rc=None):
        d = tau_rc - self.tau_s
        u01 = tf.random.uniform(tf.shape(period))
        t = u01 * period
        q_rc = tf.exp(-t / tau_rc)
        q_s = tf.exp(-t / self.tau_s)
        r_rc1 = -(tf.math.expm1(-period / tau_rc))  # 1 - exp(-period/tau_rc)
        r_s1 = -(tf.math.expm1(-period / self.tau_s))  # 1 - exp(-period/tau_s)
        return (1.0 / d) * (q_rc / r_rc1 - q_s / r_s1)


@NoiseBuilder.register(AlphaRCNoise)
class AlphaRCNoiseBuilder(NoiseBuilder):
    """nengo_dl builder for the AlphaRCNoise model."""

    def __init__(self, ops, signals, *args, **kwargs):
        super(AlphaRCNoiseBuilder, self).__init__(ops, signals, *args, **kwargs)

        # tau_s is the time constant of the synaptic filter
        tau_s = np.concatenate(
            [
                model.tau_s * np.ones((1, op.J.shape[0]), dtype=self.np_dtype)
                for model, op in zip(self.noise_models, ops)
            ]
        )
        self.tau_s = tf.constant(tau_s, dtype=self.dtype)

    def generate(self, period, tau_rc=None):
        d = tau_rc - self.tau_s
        u01 = tf.random.uniform(tf.shape(period))
        t = u01 * period
        q_rc = tf.exp(-t / tau_rc)
        q_s = tf.exp(-t / self.tau_s)
        r_rc1 = -(tf.math.expm1(-period / tau_rc))  # 1 - exp(-period/tau_rc)
        r_s1 = -(tf.math.expm1(-period / self.tau_s))  # 1 - exp(-period/tau_s)

        pt = tf.where(
            period < 100 * self.tau_s, (period - t) * (1 - r_s1), tf.zeros_like(period)
        )
        qt = tf.where(t < 100 * self.tau_s, q_s * (t + pt), tf.zeros_like(t))
        rt = qt / (self.tau_s * d * r_s1 ** 2)
        rn = tau_rc * (q_rc / (d * d * r_rc1) - q_s / (d * d * r_s1)) - rt
        return rn


class LoihiLIFBuilder(nengo_dl.neuron_builders.LIFBuilder):
    """nengo_dl builder for the LoihiLIF neuron type.

    Attributes
    ----------
    spike_noise : NoiseBuilder
        Generator for any output noise associated with these neurons.
    """

    def __init__(self, ops, signals, config):
        super(LoihiLIFBuilder, self).__init__(ops, signals, config)
        self.spike_noise = NoiseBuilder.build(ops, signals, config)

    def _rate_step(self, J, dt):
        tau_ref = discretize_tau_ref(self.tau_ref, dt)
        tau_rc = discretize_tau_rc(self.tau_rc, dt)
        # Since LoihiLIF takes `ceil(period/dt)` the firing rate is
        # always below the LIF rate. Using `tau_ref1` in LIF curve makes
        # it the average of the LoihiLIF curve (rather than upper bound).
        tau_ref1 = tau_ref + 0.5 * dt

        J -= self.one

        # --- compute Loihi rates (for forward pass)
        period = tau_ref + tau_rc * tf.math.log1p(
            tf.math.reciprocal(tf.maximum(J, self.epsilon))
        )
        period = dt * tf.math.ceil(period / dt)
        loihi_rates = self.spike_noise.generate(period, tau_rc=tau_rc)
        loihi_rates = tf.where(J > self.zero, self.amplitude * loihi_rates, self.zeros)

        # --- compute LIF rates (for backward pass)
        if self.config.lif_smoothing:
            js = J / self.sigma
            j_valid = js > -20
            js_safe = tf.where(j_valid, js, self.zeros)

            # softplus(js) = log(1 + e^js)
            z = tf.nn.softplus(js_safe) * self.sigma

            # as z->0
            #   z = s*log(1 + e^js) = s*e^js
            #   log(1 + 1/z) = log(1/z) = -log(s*e^js) = -js - log(s)
            q = tf.where(
                j_valid,
                tf.math.log1p(tf.math.reciprocal(z)),
                -js - tf.math.log(self.sigma),
            )

            rates = self.amplitude / (tau_ref1 + tau_rc * q)
        else:
            rates = self.amplitude / (
                tau_ref1
                + tau_rc
                * tf.math.log1p(tf.math.reciprocal(tf.maximum(J, self.epsilon)))
            )
            rates = tf.where(J > self.zero, rates, self.zeros)

        # rates + stop_gradient(loihi_rates - rates) =
        #     loihi_rates on forward pass, rates on backwards
        return rates + tf.stop_gradient(loihi_rates - rates)

    def _step(self, J, voltage, refractory, dt):
        tau_ref = discretize_tau_ref(self.tau_ref, dt)
        tau_rc = discretize_tau_rc(self.tau_rc, dt)

        delta_t = tf.clip_by_value(dt - refractory, self.zero, dt)
        voltage -= (J - voltage) * tf.math.expm1(-delta_t / tau_rc)

        spiked = voltage > self.one
        spikes = tf.cast(spiked, J.dtype) * self.alpha

        refractory = tf.where(spiked, tau_ref + self.zeros, refractory - dt)
        voltage = tf.where(spiked, self.zeros, tf.maximum(voltage, self.min_voltage))

        # we use stop_gradient to avoid propagating any nans (those get
        # propagated through the cond even if the spiking version isn't
        # being used at all)
        return (
            tf.stop_gradient(spikes),
            tf.stop_gradient(voltage),
            tf.stop_gradient(refractory),
        )

    def build_step(self, signals):
        J = signals.gather(self.J_data)
        voltage = signals.gather(self.voltage_data)
        refractory = signals.gather(self.refractory_data)

        spike_out, spike_voltage, spike_ref = self._step(
            J, voltage, refractory, signals.dt
        )

        if self.config.inference_only:
            spikes, voltage, refractory = (spike_out, spike_voltage, spike_ref)
        else:
            rate_out = self._rate_step(J, signals.dt)

            spikes, voltage, refractory = tf_utils.smart_cond(
                self.config.training,
                lambda: (rate_out, voltage, refractory),
                lambda: (spike_out, spike_voltage, spike_ref),
            )

        signals.scatter(self.output_data, spikes)
        signals.scatter(self.refractory_data, refractory)
        signals.scatter(self.voltage_data, voltage)


class LoihiSpikingRectifiedLinearBuilder(
    nengo_dl.neuron_builders.SpikingRectifiedLinearBuilder
):
    """nengo_dl builder for the LoihiSpikingRectifiedLinear neuron type.
    """

    def __init__(self, ops, signals, config):
        super(LoihiSpikingRectifiedLinearBuilder, self).__init__(ops, signals, config)

        self.amplitude = signals.op_constant(
            [op.neurons for op in ops],
            [op.J.shape[0] for op in ops],
            "amplitude",
            signals.dtype,
        )

        self.zeros = tf.zeros(
            (signals.minibatch_size,) + self.J_data.shape, signals.dtype
        )

        self.epsilon = tf.constant(1e-15, dtype=signals.dtype)

        # copy these so that they're easily accessible in _step functions
        self.zero = signals.zero
        self.one = signals.one

    def _rate_step(self, J, dt):
        # Since LoihiLIF takes `ceil(period/dt)` the firing rate is
        # always below the LIF rate. Using `tau_ref1` in LIF curve makes
        # it the average of the LoihiLIF curve (rather than upper bound).
        tau_ref1 = 0.5 * dt

        # --- compute Loihi rates (for forward pass)
        period = tf.math.reciprocal(tf.maximum(J, self.epsilon))
        loihi_rates = self.alpha / tf.math.ceil(period / dt)
        loihi_rates = tf.where(J > self.zero, loihi_rates, self.zeros)

        # --- compute RectifiedLinear rates (for backward pass)
        rates = self.amplitude / (
            tau_ref1 + tf.math.reciprocal(tf.maximum(J, self.epsilon))
        )
        rates = tf.where(J > self.zero, rates, self.zeros)

        # rates + stop_gradient(loihi_rates - rates) =
        #     loihi_rates on forward pass, rates on backwards
        return rates + tf.stop_gradient(loihi_rates - rates)

    def _step(self, J, voltage, dt):
        voltage += J * dt
        spiked = voltage > self.one
        spikes = tf.cast(spiked, J.dtype) * self.alpha
        voltage = tf.where(spiked, self.zeros, tf.maximum(voltage, 0))

        # we use stop_gradient to avoid propagating any nans (those get
        # propagated through the cond even if the spiking version isn't
        # being used at all)
        return tf.stop_gradient(spikes), tf.stop_gradient(voltage)

    def build_step(self, signals):
        J = signals.gather(self.J_data)
        voltage = signals.gather(self.voltage_data)

        spike_out, spike_voltage = self._step(J, voltage, signals.dt)

        if self.config.inference_only:
            out, voltage = spike_out, spike_voltage
        else:
            rate_out = self._rate_step(J, signals.dt)

            out, voltage = tf_utils.smart_cond(
                self.config.training,
                lambda: (rate_out, voltage),
                lambda: (spike_out, spike_voltage),
            )

        signals.scatter(self.output_data, out)
        signals.scatter(self.voltage_data, voltage)


class Installer:
    def __init__(self):
        self.installed = False

    def __call__(self):
        if self.installed:
            pass
        elif not HAS_DL:
            logger.info("nengo_dl cannot be imported, so not installing builders")
        else:
            logger.info("Installing NengoDL neuron builders")
            nengo_dl.neuron_builders.SimNeuronsBuilder.TF_NEURON_IMPL[
                LoihiLIF
            ] = LoihiLIFBuilder
            nengo_dl.neuron_builders.SimNeuronsBuilder.TF_NEURON_IMPL[
                LoihiSpikingRectifiedLinear
            ] = LoihiSpikingRectifiedLinearBuilder
            self.installed = True


install_dl_builders = Installer()
