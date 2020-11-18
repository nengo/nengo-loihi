import logging
import warnings

import numpy as np

from nengo_loihi.compat import HAS_DL
from nengo_loihi.neurons import (
    AlphaRCNoise,
    LoihiLIF,
    LoihiSpikingRectifiedLinear,
    LowpassRCNoise,
    discretize_tau_rc,
    discretize_tau_ref,
)

if HAS_DL:
    import nengo_dl
    import tensorflow as tf
    from nengo_dl.neuron_builders import LIFBuilder, SpikingRectifiedLinearBuilder
else:  # pragma: no cover
    # Empty classes so that we can define the subclasses even though
    # we will never use them, as they are only used in the `install`
    # function that can only run if nengo_dl is importable.
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

    def __init__(self, ops, noise_models):
        self.ops = ops
        self.noise_models = noise_models

    @classmethod
    def build(cls, ops):
        """Create a NoiseBuilder for the provided ops."""

        noise_models = [getattr(op.neurons, "nengo_dl_noise", None) for op in ops]
        model_type = type(noise_models[0]) if len(noise_models) > 0 else None
        if not all(type(m) is model_type for m in noise_models):
            raise NotImplementedError(
                "Multiple noise models for the same neuron type is not supported"
            )

        return cls.builders[model_type](ops, noise_models)

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

    def build_pre(self, signals, config):
        self.dtype = signals.dtype
        self.np_dtype = self.dtype.as_numpy_dtype()

    def build_step(self, period, tau_rc=None):
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

    def build_step(self, period, tau_rc=None):
        return tf.math.reciprocal(period)


@NoiseBuilder.register(LowpassRCNoise)
class LowpassRCNoiseBuilder(NoiseBuilder):
    """nengo_dl builder for the LowpassRCNoise model."""

    def build_pre(self, signals, config):
        super().build_pre(signals, config)

        # tau_s is the time constant of the synaptic filter
        tau_s = np.concatenate(
            [
                model.tau_s * np.ones((1, op.J.shape[0]), dtype=self.np_dtype)
                for model, op in zip(self.noise_models, self.ops)
            ]
        )
        self.tau_s = tf.constant(tau_s, dtype=self.dtype)

    def build_step(self, period, tau_rc=None):
        d = tau_rc - self.tau_s
        u01 = tf.random.uniform(tf.shape(period))
        t = u01 * period
        q_rc = tf.exp(-t / tau_rc)
        q_s = tf.exp(-t / self.tau_s)
        r_rc1 = -tf.math.expm1(-period / tau_rc)  # 1 - exp(-period/tau_rc)
        r_s1 = -tf.math.expm1(-period / self.tau_s)  # 1 - exp(-period/tau_s)
        return (1.0 / d) * (q_rc / r_rc1 - q_s / r_s1)


@NoiseBuilder.register(AlphaRCNoise)
class AlphaRCNoiseBuilder(NoiseBuilder):
    """nengo_dl builder for the AlphaRCNoise model."""

    def build_pre(self, signals, config):
        super().build_pre(signals, config)

        # tau_s is the time constant of the synaptic filter
        tau_s = np.concatenate(
            [
                model.tau_s * np.ones((1, op.J.shape[0]), dtype=self.np_dtype)
                for model, op in zip(self.noise_models, self.ops)
            ]
        )
        self.tau_s = tf.constant(tau_s, dtype=self.dtype)

    def build_step(self, period, tau_rc=None):
        d = tau_rc - self.tau_s
        u01 = tf.random.uniform(tf.shape(period))
        t = u01 * period
        q_rc = tf.exp(-t / tau_rc)
        q_s = tf.exp(-t / self.tau_s)
        r_rc1 = -tf.math.expm1(-period / tau_rc)  # 1 - exp(-period/tau_rc)
        r_s1 = -tf.math.expm1(-period / self.tau_s)  # 1 - exp(-period/tau_s)

        pt = tf.where(
            period < 100 * self.tau_s, (period - t) * (1 - r_s1), tf.zeros_like(period)
        )
        qt = tf.where(t < 100 * self.tau_s, q_s * (t + pt), tf.zeros_like(t))
        rt = qt / (self.tau_s * d * r_s1 ** 2)
        rn = tau_rc * (q_rc / (d * d * r_rc1) - q_s / (d * d * r_s1)) - rt
        return rn


class LoihiLIFBuilder(LIFBuilder):
    """nengo_dl builder for the LoihiLIF neuron type.

    Attributes
    ----------
    spike_noise : NoiseBuilder
        Generator for any output noise associated with these neurons.
    """

    def __init__(self, ops):
        super().__init__(ops)
        self.spike_noise = NoiseBuilder.build(ops)

    def build_pre(self, signals, config):
        super().build_pre(signals, config)
        self.spike_noise.build_pre(signals, config)

    def training_step(self, J, dt, **state):
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
        loihi_rates = self.spike_noise.build_step(period, tau_rc=tau_rc)
        loihi_rates = tf.where(J > self.zero, loihi_rates, self.zeros)
        if self.amplitude is not None:
            loihi_rates *= self.amplitude

        # --- compute LIF rates (for backward pass)
        amplitude = self.one if self.amplitude is None else self.amplitude
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

            rates = amplitude / (tau_ref1 + tau_rc * q)
        else:
            rates = amplitude / (
                tau_ref1
                + tau_rc
                * tf.math.log1p(tf.math.reciprocal(tf.maximum(J, self.epsilon)))
            )
            rates = tf.where(J > self.zero, rates, self.zeros)

        # rates + stop_gradient(loihi_rates - rates) =
        #     loihi_rates on forward pass, rates on backwards
        return rates + tf.stop_gradient(loihi_rates - rates)

    def step(self, J, dt, voltage, refractory_time):
        tau_ref = discretize_tau_ref(self.tau_ref, dt)
        tau_rc = discretize_tau_rc(self.tau_rc, dt)

        delta_t = tf.clip_by_value(dt - refractory_time, self.zero, dt)
        voltage -= (J - voltage) * tf.math.expm1(-delta_t / tau_rc)

        spiked = voltage > self.one
        spikes = tf.cast(spiked, J.dtype) * self.alpha

        refractory_time = tf.where(spiked, tau_ref + self.zeros, refractory_time - dt)
        voltage = tf.where(spiked, self.zeros, tf.maximum(voltage, self.min_voltage))

        # we use stop_gradient to avoid propagating any nans (those get
        # propagated through the cond even if the spiking version isn't
        # being used at all)
        return (
            tf.stop_gradient(spikes),
            tf.stop_gradient(voltage),
            tf.stop_gradient(refractory_time),
        )


class LoihiSpikingRectifiedLinearBuilder(SpikingRectifiedLinearBuilder):
    """nengo_dl builder for the LoihiSpikingRectifiedLinear neuron type."""

    def build_pre(self, signals, config):
        super().build_pre(signals, config)

        self.zeros = tf.zeros(
            (signals.minibatch_size,) + self.J_data.shape, signals.dtype
        )

        self.epsilon = tf.constant(1e-15, dtype=signals.dtype)

        # copy these so that they're easily accessible in _step functions
        self.zero = signals.zero
        self.one = signals.one

    def training_step(self, J, dt, **state):
        # Since LoihiLIF takes `ceil(period/dt)` the firing rate is
        # always below the LIF rate. Using `tau_ref1` in LIF curve makes
        # it the average of the LoihiLIF curve (rather than upper bound).
        tau_ref1 = 0.5 * dt

        # --- compute Loihi rates (for forward pass)
        period = tf.math.reciprocal(tf.maximum(J, self.epsilon))
        loihi_rates = self.alpha / tf.math.ceil(period / dt)
        loihi_rates = tf.where(J > self.zero, loihi_rates, self.zeros)

        # --- compute RectifiedLinear rates (for backward pass)
        rates = (self.one if self.amplitude is None else self.amplitude) / (
            tau_ref1 + tf.math.reciprocal(tf.maximum(J, self.epsilon))
        )
        rates = tf.where(J > self.zero, rates, self.zeros)

        # rates + stop_gradient(loihi_rates - rates) =
        #     loihi_rates on forward pass, rates on backwards
        return rates + tf.stop_gradient(loihi_rates - rates)

    def step(self, J, dt, voltage):
        voltage += J * dt
        spiked = voltage > self.one
        spikes = tf.cast(spiked, J.dtype) * self.alpha
        voltage = tf.where(spiked, self.zeros, tf.maximum(voltage, 0))

        # we use stop_gradient to avoid propagating any nans (those get
        # propagated through the cond even if the spiking version isn't
        # being used at all)
        return tf.stop_gradient(spikes), tf.stop_gradient(voltage)


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
