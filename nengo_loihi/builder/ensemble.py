import nengo
from nengo.builder.ensemble import (
    BuiltEnsemble, gen_eval_points, get_gain_bias)
from nengo.dists import Distribution, get_samples
import nengo.utils.numpy as npext
import numpy as np

from nengo_loihi.builder import Builder, CoreGroup, INTER_N, INTER_RATE
from nengo_loihi.synapses import Synapses

# Filter on intermediary neurons
INTER_TAU = 0.005
# ^TODO: how to choose this filter? Need it since all input will be spikes,
#   but maybe don't want double filtering if connection has a filter


@Builder.register(nengo.Ensemble)
def build_ensemble(model, ens):

    # Create random number generator
    rng = np.random.RandomState(model.seeds[ens])

    eval_points = gen_eval_points(ens, ens.eval_points, rng=rng)

    # Set up encoders
    if isinstance(ens.neuron_type, nengo.Direct):
        encoders = np.identity(ens.dimensions)
    elif isinstance(ens.encoders, Distribution):
        encoders = get_samples(
            ens.encoders, ens.n_neurons, ens.dimensions, rng=rng)
    else:
        encoders = npext.array(ens.encoders, min_dims=2, dtype=np.float64)
    if ens.normalize_encoders:
        encoders /= npext.norm(encoders, axis=1, keepdims=True)

    # Build the neurons
    gain, bias, max_rates, intercepts = get_gain_bias(ens, rng)

    if isinstance(ens.neuron_type, nengo.Direct):
        raise NotImplementedError()
    else:
        group = CoreGroup(ens.n_neurons, label=str(ens))
        group.compartments.bias[...] = bias
        model.build(ens.neuron_type, ens.neurons, group.compartments)

    group.compartments.configure_filter(INTER_TAU, dt=model.dt)

    # Scale the encoders
    if isinstance(ens.neuron_type, nengo.Direct):
        raise NotImplementedError("Direct neurons not implemented")
    else:
        # to keep scaling reasonable, we don't include the radius
        scaled_encoders = encoders * gain[:, np.newaxis]

    # --- encoders for interneurons
    synapses = Synapses(2*scaled_encoders.shape[1])
    inter_scale = 1. / (model.dt * INTER_RATE * INTER_N)
    interscaled_encoders = scaled_encoders * inter_scale
    synapses.set_full_weights(
        np.vstack([interscaled_encoders.T, -interscaled_encoders.T]))
    group.synapses.add(synapses, name='encoders2')

    model.add_group(group)

    model.objs[ens]['in'] = group
    model.objs[ens]['out'] = group
    model.objs[ens.neurons]['in'] = group
    model.objs[ens.neurons]['out'] = group
    model.params[ens] = BuiltEnsemble(
        eval_points=eval_points,
        encoders=encoders,
        intercepts=intercepts,
        max_rates=max_rates,
        scaled_encoders=scaled_encoders,
        gain=gain,
        bias=bias)


@Builder.register(nengo.LIF)
def build_lif(model, lif, neurons, cx_group):
    cx_group.configure_lif(
        tau_rc=lif.tau_rc,
        tau_ref=lif.tau_ref,
        dt=model.dt)


@Builder.register(nengo.RectifiedLinear)
def build_relu(model, relu, neurons, cx_group):
    cx_group.configure_relu(dt=model.dt)
