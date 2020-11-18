import warnings

import nengo
import nengo.utils.numpy as npext
import numpy as np
from nengo import Ensemble
from nengo.builder.ensemble import BuiltEnsemble, gen_eval_points
from nengo.dists import Distribution, get_samples
from nengo.exceptions import BuildError

from nengo_loihi.block import LoihiBlock
from nengo_loihi.builder.builder import Builder


def get_gain_bias(ens, rng=np.random, intercept_limit=1.0):
    # Modified from the Nengo version to handle `intercept_limit`

    if ens.gain is not None and ens.bias is not None:
        gain = get_samples(ens.gain, ens.n_neurons, rng=rng)
        bias = get_samples(ens.bias, ens.n_neurons, rng=rng)
        max_rates, intercepts = ens.neuron_type.max_rates_intercepts(gain, bias)
    elif ens.gain is not None or ens.bias is not None:
        # TODO: handle this instead of error
        raise NotImplementedError(
            "gain or bias set for %s, but not both. "
            "Solving for one given the other is not "
            "implemented yet." % ens
        )
    else:
        int_distorarray = ens.intercepts
        if isinstance(int_distorarray, nengo.dists.Uniform):
            if int_distorarray.high > intercept_limit:
                warnings.warn(
                    "Intercepts are larger than intercept limit (%g). "
                    "High intercept values cause issues when discretizing "
                    "the model for running on Loihi." % intercept_limit
                )
                int_distorarray = nengo.dists.Uniform(
                    min(int_distorarray.low, intercept_limit),
                    min(int_distorarray.high, intercept_limit),
                )

        max_rates = get_samples(ens.max_rates, ens.n_neurons, rng=rng)
        intercepts = get_samples(int_distorarray, ens.n_neurons, rng=rng)

        if np.any(intercepts > intercept_limit):
            intercepts[intercepts > intercept_limit] = intercept_limit
            warnings.warn(
                "Intercepts are larger than intercept limit (%g). "
                "High intercept values cause issues when discretizing "
                "the model for running on Loihi." % intercept_limit
            )

        gain, bias = ens.neuron_type.gain_bias(max_rates, intercepts)
        if gain is not None and (not np.all(np.isfinite(gain)) or np.any(gain <= 0.0)):
            raise BuildError(
                "The specified intercepts for %s lead to neurons with "
                "negative or non-finite gain. Please adjust the intercepts so "
                "that all gains are positive. For most neuron types (e.g., "
                "LIF neurons) this is achieved by reducing the maximum "
                "intercept value to below 1." % ens
            )

    return gain, bias, max_rates, intercepts


@Builder.register(Ensemble)
def build_ensemble(model, ens):
    if isinstance(ens.neuron_type, nengo.Direct):
        raise NotImplementedError("Direct neurons not implemented")

    # Create random number generator
    rng = np.random.RandomState(model.seeds[ens])

    eval_points = gen_eval_points(ens, ens.eval_points, rng=rng)

    # Set up encoders
    if isinstance(ens.encoders, Distribution):
        encoders = get_samples(ens.encoders, ens.n_neurons, ens.dimensions, rng=rng)
    else:
        encoders = npext.array(ens.encoders, min_dims=2, dtype=np.float64)

    if ens.normalize_encoders:
        encoders /= npext.norm(encoders, axis=1, keepdims=True)

    if np.any(np.isnan(encoders)):
        raise BuildError(
            "NaNs detected in %r encoders. This usually means that you had zero-length "
            "encoders that were normalized, resulting in NaNs. Ensure all encoders "
            "have non-zero length, or set `normalize_encoders=False`." % ens
        )

    # Build the neurons
    gain, bias, max_rates, intercepts = get_gain_bias(ens, rng, model.intercept_limit)

    block = LoihiBlock(ens.n_neurons, label="%s" % ens)
    block.compartment.bias[:] = bias

    # build the neuron_type (see builders below)
    model.build(ens.neuron_type, ens.neurons, block)

    # set default filter just in case no other filter gets set
    block.compartment.configure_default_filter(model.decode_tau, dt=model.dt)

    if ens.noise is not None:
        raise NotImplementedError("Ensemble noise not implemented")

    # Scale the encoders
    # we exclude the radius to keep scaling reasonable for decode neurons
    scaled_encoders = encoders * gain[:, np.newaxis]

    # add instructions for splitting
    model.block_shapes[block] = model.config[ens].block_shape

    model.add_block(block)

    model.objs[ens]["in"] = block
    model.objs[ens]["out"] = block
    model.objs[ens.neurons]["in"] = block
    model.objs[ens.neurons]["out"] = block
    model.params[ens] = BuiltEnsemble(
        eval_points=eval_points,
        encoders=encoders,
        intercepts=intercepts,
        max_rates=max_rates,
        scaled_encoders=scaled_encoders,
        gain=gain,
        bias=bias,
    )


@Builder.register(nengo.neurons.NeuronType)
def build_neurons(model, neurontype, neurons, block):
    # If we haven't registered a builder for a specific type, then it cannot
    # be simulated on Loihi.
    raise BuildError(
        "The neuron type %r cannot be simulated on Loihi. Please either "
        "switch to a supported neuron type like LIF or "
        "SpikingRectifiedLinear, or explicitly mark ensembles using this "
        "neuron type as off-chip with\n"
        "  net.config[ensembles].on_chip = False" % type(neurontype).__name__
    )


def check_state_zero(model, neuron_type, neurons, block):
    seed = model.seeds[neurons.ensemble]
    seed = seed if seed is None else (seed + 1)
    rng = np.random.RandomState(seed)

    state_init = neuron_type.make_state(block.n_neurons, rng=rng)
    for key, value in state_init.items():
        value = np.asarray(value)
        if not np.all(value == 0):
            warnings.warn(
                "NengoLoihi does not support initial values for %r being non-zero on "
                "%s neurons. On the chip, all values will be initialized to zero."
                % (key, type(neuron_type).__name__)
            )


@Builder.register(nengo.LIF)
def build_lif(model, lif, neurons, block):
    check_state_zero(model, lif, neurons, block)
    block.compartment.configure_lif(
        tau_rc=lif.tau_rc, tau_ref=lif.tau_ref, min_voltage=lif.min_voltage, dt=model.dt
    )


@Builder.register(nengo.SpikingRectifiedLinear)
def build_relu(model, relu, neurons, block):
    check_state_zero(model, relu, neurons, block)
    block.compartment.configure_relu(
        vth=1.0 / model.dt,  # so input == 1 -> neuron fires 1/dt steps -> 1 Hz
        dt=model.dt,
    )


@Builder.register(nengo.RegularSpiking)
def build_regularspiking(model, regularspiking, neurons, block):
    base = regularspiking.base_type
    if type(base) not in (nengo.LIFRate, nengo.RectifiedLinear):
        raise BuildError(
            "RegularSpiking neurons with %r as a base type cannot be simulated on "
            "Loihi. Please either switch to a supported base neuron type like "
            "LIFRate or RectifiedLinear, or explicitly mark ensembles using this "
            "neuron type as off-chip with\n"
            "  net.config[ensembles].on_chip = False" % type(base).__name__
        )

    if base.amplitude != 1:
        raise BuildError(
            "Amplitude is not supported on RegularSpiking base types on Loihi, since "
            "this effectively modifies the `dt` for individual neurons. To change the "
            "amplitude of output spikes, set `amplitude` on the `RegularSpiking` "
            "instance instead of the base type instance."
        )

    check_state_zero(model, regularspiking, neurons, block)
    if type(base) is nengo.LIFRate:
        block.compartment.configure_lif(
            tau_rc=base.tau_rc, tau_ref=base.tau_ref, dt=model.dt
        )
    elif type(base) is nengo.RectifiedLinear:
        block.compartment.configure_relu(
            vth=1.0 / model.dt,  # so input == 1 -> neuron fires 1/dt steps -> 1 Hz
            dt=model.dt,
        )
