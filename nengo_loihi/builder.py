import collections
import logging
import warnings

import numpy as np

import nengo
from nengo import Network, Ensemble, Connection, Node, Probe
from nengo.builder.connection import BuiltConnection
from nengo.dists import Distribution, get_samples
from nengo.connection import LearningRule
from nengo.ensemble import Neurons
from nengo.exceptions import BuildError, ValidationError
from nengo.solvers import NoSolver, Solver
from nengo.utils.builder import default_n_eval_points
import nengo.utils.numpy as npext

from nengo_loihi import conv
from nengo_loihi.loihi_cx import (
    ChipReceiveNeurons,
    ChipReceiveNode,
    CxAxons,
    CxGroup,
    CxModel,
    CxProbe,
    CxSpikeInput,
    CxSynapses,
)
from nengo_loihi.neurons import (
    loihi_rates,
    LoihiSpikingRectifiedLinear,
    NIF,
)

logger = logging.getLogger(__name__)


class DecodeNeurons(object):
    """Defines parameters for a group of decode neurons.

    DecodeNeurons are used on the chip to facilitate NEF-style connections,
    where activities from a neural ensemble are first transformed into a
    decoded value (which is stored in the activities and synapses of the
    spiking decode neurons), before being passed on to another ensemble
    (via that ensemble's encoders).

    Parameters
    ----------
    dt : float
        Time step used by the simulator.
    """
    def __init__(self, dt=0.001):
        self.dt = dt

    def __str__(self):
        return "%s(dt=%0.3g)" % (type(self).__name__, self.dt)

    def get_cx(self, weights, cx_label=None, syn_label=None):
        """Get a CxGroup for implementing neurons on the chip.

        Parameters
        ----------
        weights : (d, n) ndarray
            Weights that project the ``n`` inputs to the ``d`` dimensions
            represented by these neurons. Typically, the inputs will be neurons
            belonging to an Ensemble, and these weights will be decoders.
        cx_label : string (Default: None)
            Optional label for the CxGroup.
        syn_label : string (Default: None)
            Optional label for the CxSynapses.

        Returns
        -------
        cx : CxGroup
            The neurons on the chip.
        syn : CxSynapses
            The synapses connecting into the chip neurons.
        """
        raise NotImplementedError()

    def get_ensemble(self, dim):
        """Get a Nengo Ensemble for implementing neurons on the host.

        Parameters
        ----------
        dim : int
            Number of dimensions to be represented by these neurons.

        Returns
        -------
        ens : Ensemble
            An Ensemble for implementing these neurons in a Nengo network.
        """
        raise NotImplementedError()

    def get_post_encoders(self, encoders):
        """Encoders for post population that these neurons connect in to.

        Parameters
        ----------
        encoders : (n, d) ndarray
            Regular scaled encoders for the ensemble, which map the ensemble's
            ``d`` input dimensions to its ``n`` neurons.

        Returns
        -------
        decode_neuron_encoders : (?, n) ndarray
            Encoders for mapping these neurons to the post-ensemble's neurons.
            The number of rows depends on how ``get_post_inds`` is being used
            (i.e. there could be one row per neuron in this group, or there
            could be fewer rows with ``get_post_inds`` mapping multiple neurons
            to each row).
        """
        raise NotImplementedError()

    def get_post_inds(self, inds, d):
        """Indices for mapping neurons to post-encoder dimensions.

        Parameters
        ----------
        inds : list of ints
            Indices for mapping decode neuron dimensions to post-ensemble
            dimensions. Usually, this will be determined by a slice on the
            post ensemble in a connection (which maps the output of the
            transform/function to select dimensions on the post ensemble).
        d : int
            Number of dimensions in the post-ensemble.
        """
        raise NotImplementedError()


class OnOffDecodeNeurons(DecodeNeurons):
    """One or more pairs of on/off neurons per dimension.

    In this class itself, all the pairs in a dimension are identical. It can
    still be advantageous to have more than one pair per dimension, though,
    since this can allow all neurons to have lower firing rates and thus
    act more linearly (due to period aliasing at high firing rates). Subclasses
    may use pairs that are not identical (by adding noise or heterogeneity).

    Parameters
    ----------
    pairs_per_dim : int
        Number of repeated neurons per dimension. Currently, all DecodeNeuron
        classes use separate on/off neuron pairs for each dimension. This is
        the number of such pairs per dimension.
    dt : float
        Time step used by the simulator.
    rate : float (Default: None)
        Max firing rate of each neuron. By default, this is chosen so that
        the sum of all repeated neuron rates is ``1. / dt``, and thus as a
        group the neurons average one spike per timestep.
    """

    def __init__(self, pairs_per_dim=1, dt=0.001, rate=None):
        super(OnOffDecodeNeurons, self).__init__(dt=dt)

        self.pairs_per_dim = pairs_per_dim

        self.rate = (1. / (self.dt * self.pairs_per_dim) if rate is None
                     else rate)
        self.scale = 1. / (self.dt * self.rate * self.pairs_per_dim)
        self.neuron_type = LoihiSpikingRectifiedLinear()

        gain = 0.5 * self.rate * np.ones(self.pairs_per_dim)
        bias = gain  # intercept of -1
        self.gain = gain.repeat(2)
        self.bias = bias.repeat(2)
        # ^ repeat for on/off neurons

    def __str__(self):
        return "%s(pairs_per_dim=%d, dt=%0.3g, rate=%0.3g)" % (
            type(self).__name__, self.pairs_per_dim, self.dt, self.rate)

    def get_cx(self, weights, cx_label=None, syn_label=None):
        gain = self.gain * self.dt
        bias = self.bias * self.dt

        d, n = weights.shape
        n_neurons = 2 * d * self.pairs_per_dim
        cx = CxGroup(n_neurons, label=cx_label, location='core')
        cx.configure_relu(dt=self.dt)
        cx.bias[:] = bias.repeat(d)

        syn = CxSynapses(n, label=syn_label)
        weights2 = []
        for ga, gb in gain.reshape(self.pairs_per_dim, 2):
            weights2.extend([ga*weights.T, -gb*weights.T])
        weights2 = np.hstack(weights2)
        syn.set_full_weights(weights2)
        cx.add_synapses(syn)

        return cx, syn

    def get_ensemble(self, dim):
        if self.pairs_per_dim != 1:
            # To support this, we need to figure out how to deal with the
            # `post_inds` that map neurons to axons. Either we can do this
            # on the host, in which case we'd have inputs going to the chip
            # where we can have multiple spikes per axon per timestep, or we
            # need to do it on the chip with one input axon per neuron.
            raise NotImplementedError(
                "Input neurons with more than one neuron per dimension")

        n_neurons = 2 * dim * self.pairs_per_dim
        encoders = np.vstack([np.eye(dim), -np.eye(dim)] * self.pairs_per_dim)
        ens = nengo.Ensemble(
            n_neurons, dim,
            neuron_type=NIF(tau_ref=0.0),
            encoders=encoders,
            gain=self.gain.repeat(dim),
            bias=self.bias.repeat(dim),
            add_to_container=False)
        return ens

    def get_post_encoders(self, encoders):
        encoders = encoders * self.scale
        return np.vstack([encoders.T, -encoders.T])

    def get_post_inds(self, inds, d):
        return np.concatenate([inds, inds + d] * self.pairs_per_dim)


class NoisyDecodeNeurons(OnOffDecodeNeurons):
    """Uses multiple on/off neuron pairs per dimension, plus noise.

    The noise allows each on-off neuron pair to do something different. The
    population average is a better representation of the encoded value
    than can be achieved with a single on/off neuron pair (if the magnitude
    of the noise is correctly calibrated).

    Parameters
    ----------
    pairs_per_dim : int
        Number of repeated neurons per dimension. Currently, all DecodeNeuron
        classes use separate on/off neuron pairs for each dimension. This is
        the number of such pairs per dimension.
    dt : float
        Time step used by the simulator.
    rate : float (Default: None)
        Max firing rate of each neuron. By default, this is chosen so that
        the sum of all repeated neuron rates is ``1. / dt``, and thus as a
        group the neurons average one spike per timestep.
    noise_exp : float, optional (Default: -2.)
        Base-10 exponent for noise added to neuron voltages.
    """

    def __init__(self, pairs_per_dim, dt=0.001, rate=None, noise_exp=-2.):
        super(NoisyDecodeNeurons, self).__init__(
            pairs_per_dim=pairs_per_dim, dt=dt, rate=rate)
        self.noise_exp = noise_exp  # noise exponent for added voltage noise

    def __str__(self):
        return (
            "%s(pairs_per_dim=%d, dt=%0.3g, rate=%0.3g, noise_exp=%0.3g)" % (
                type(self).__name__,
                self.pairs_per_dim,
                self.dt,
                self.rate,
                self.noise_exp,
            )
        )

    def get_cx(self, weights, cx_label=None, syn_label=None):
        cx, syn = super(NoisyDecodeNeurons, self).get_cx(
            weights, cx_label=cx_label, syn_label=syn_label)

        if self.noise_exp > -30:
            cx.enableNoise[:] = 1
            cx.noiseExp0 = self.noise_exp
            cx.noiseAtDendOrVm = 1

        return cx, syn


class Preset5DecodeNeurons(OnOffDecodeNeurons):
    """Uses five heterogeneous on/off pairs with pre-set values per dimension.

    The script for configuring these values can be found at:
        nengo-loihi-sandbox/utils/interneuron_unidecoder_design.py
    """

    def __init__(self, dt=0.001, rate=None):
        super(Preset5DecodeNeurons, self).__init__(
            pairs_per_dim=5, dt=dt, rate=rate)

        assert self.pairs_per_dim == 5
        intercepts = np.linspace(-0.8, 0.8, self.pairs_per_dim)
        max_rates = np.linspace(160, 70, self.pairs_per_dim)
        gain, bias = self.neuron_type.gain_bias(max_rates, intercepts)

        target_point = 0.85
        target_rate = np.sum(self.neuron_type.rates(target_point, gain, bias))
        self.scale = 1.08 * target_point / (self.dt * target_rate)
        # ^ TODO: why does this 1.08 factor help? found it empirically in
        # test_decode_neurons.test_add_inputs

        self.gain = gain.repeat(2)
        self.bias = bias.repeat(2)
        # ^ repeat for on/off neurons

    def __str__(self):
        return "%s(dt=%0.3g, rate=%0.3g)" % (
            type(self).__name__, self.dt, self.rate)


class Preset10DecodeNeurons(OnOffDecodeNeurons):
    """Uses ten heterogeneous on/off pairs with pre-set values per dimension.

    The script for configuring these values can be found at:
        nengo-loihi-sandbox/utils/interneuron_unidecoder_design.py
    """

    def __init__(self, dt=0.001, rate=None):
        super(Preset10DecodeNeurons, self).__init__(
            pairs_per_dim=10, dt=dt, rate=rate)

        # Parameters determined by hyperopt
        assert self.pairs_per_dim == 10
        intercepts = np.linspace(-1.171, 0.484, self.pairs_per_dim)
        max_rates = np.linspace(171.186, 74.620, self.pairs_per_dim)
        gain, bias = self.neuron_type.gain_bias(max_rates, intercepts)

        target_point = 1.0
        target_rate = np.sum(self.neuron_type.rates(target_point, gain, bias))
        self.scale = 1.05 * target_point / (self.dt * target_rate)
        # ^ TODO: why does this 1.05 factor help? found it empirically in
        # test_decode_neurons.test_add_inputs

        self.gain = gain.repeat(2)
        self.bias = bias.repeat(2)
        # ^ repeat for on/off neurons

    def __str__(self):
        return "%s(dt=%0.3g, rate=%0.3g)" % (
            type(self).__name__, self.dt, self.rate)


class Model(CxModel):
    """The data structure for the chip/simulator.

    This is a subclass of CxModel, which defines methods for adding ensembles,
    discretizing, and tracking the simulator. This class handles build
    functions and keeping track of chip/host communication.

    Parameters
    ----------
    dt : float, optional (Default: 0.001)
        The length of a simulator timestep, in seconds.
    label : str, optional (Default: None)
        A name or description to differentiate models.
    builder : Builder, optional (Default: None)
        A `.Builder` instance to keep track of build functions.
        If None, the default builder will be used.

    Attributes
    ----------
    builder : Builder
        The build functions used by this model.
    dt : float
        The length of a simulator timestep, in seconds.
    label : str or None
        A name or description to differentiate models.
    objs : dict
        Dictionary mapping from Nengo objects to Nengo Loihi objects.
    params : dict
        Mapping from objects to namedtuples containing parameters generated
        in the build process.
    probes : list
        List of all probes. Probes must be added to this list in the build
        process, as this list is used by Simulator.
    seeded : dict
        All objects are assigned a seed, whether the user defined the seed
        or it was automatically generated. 'seeded' keeps track of whether
        the seed is user-defined. We consider the seed to be user-defined
        if it was set directly on the object, or if a seed was set on the
        network in which the object resides, or if a seed was set on any
        ancestor network of the network in which the object resides.
    seeds : dict
        Mapping from objects to the integer seed assigned to that object.
    """
    def __init__(self, dt=0.001, label=None, builder=None):
        super(Model, self).__init__(dt=dt, label=label)

        self.objs = collections.defaultdict(dict)
        self.params = {}  # Holds data generated when building objects
        self.probes = []
        self.probe_conns = {}

        self.seeds = {}
        self.seeded = {}

        self.builder = Builder() if builder is None else builder
        self.build_callback = None

        # --- other (typically standard) parameters
        # Filter on decode neurons
        self.decode_tau = 0.005
        # ^TODO: how to choose this filter? Even though the input is spikes,
        # it may not be absolutely necessary since tau_rc provides a filter,
        # and maybe we don't want double filtering if connection has a filter

        self.decode_neurons = Preset10DecodeNeurons(dt=dt)
        self.node_neurons = OnOffDecodeNeurons(dt=dt)

        # voltage threshold for non-spiking neurons (i.e. voltage decoders)
        self.vth_nonspiking = 10

        # limit for clipping intercepts, to avoid neurons with high gains
        self.intercept_limit = 0.95

        # scaling for PES errors, before rounding and clipping to -127..127
        self.pes_error_scale = 100.

        # learning weight exponent for PES (controls the maximum weight
        # magnitude/weight resolution)
        self.pes_wgt_exp = 4

        # Will be provided by Simulator
        self.chip2host_params = {}

    def __getstate__(self):
        raise NotImplementedError("Can't pickle nengo_loihi.builder.Model")

    def __setstate__(self, state):
        raise NotImplementedError("Can't pickle nengo_loihi.builder.Model")

    def __str__(self):
        return "Model: %s" % self.label

    def build(self, obj, *args, **kwargs):
        built = self.builder.build(self, obj, *args, **kwargs)
        if self.build_callback is not None:
            self.build_callback(obj)
        return built

    def has_built(self, obj):
        return obj in self.params


class Builder(object):
    """Fills in the Loihi Model object based on the Nengo Network."""

    builders = {}  # Methods that build different components

    @classmethod
    def build(cls, model, obj, *args, **kwargs):
        if model.has_built(obj):
            warnings.warn("Object %s has already been built." % obj)
            return None

        for obj_cls in type(obj).__mro__:
            if obj_cls in cls.builders:
                break
        else:
            raise BuildError(
                "Cannot build object of type %r" % type(obj).__name__)

        return cls.builders[obj_cls](model, obj, *args, **kwargs)

    @classmethod
    def register(cls, nengo_class):
        """Register methods to build Nengo objects into Model."""

        def register_builder(build_fn):
            if nengo_class in cls.builders:
                warnings.warn("Type '%s' already has a builder. Overwriting."
                              % nengo_class)
            cls.builders[nengo_class] = build_fn
            return build_fn
        return register_builder


@Builder.register(Network)
def build_network(model, network):
    def get_seed(obj, rng):
        return (rng.randint(npext.maxint)
                if not hasattr(obj, 'seed') or obj.seed is None else obj.seed)

    if network not in model.seeds:
        model.seeded[network] = getattr(network, 'seed', None) is not None
        model.seeds[network] = get_seed(network, np.random)

    # # Set config
    # old_config = model.config
    # model.config = network.config

    # assign seeds to children
    rng = np.random.RandomState(model.seeds[network])
    # Put probes last so that they don't influence other seeds
    sorted_types = (Connection, Ensemble, Network, Node, Probe)
    assert all(tp in sorted_types for tp in network.objects)
    for obj_type in sorted_types:
        for obj in network.objects[obj_type]:
            model.seeded[obj] = (model.seeded[network]
                                 or getattr(obj, 'seed', None) is not None)
            model.seeds[obj] = get_seed(obj, rng)

    logger.debug("Network step 1: Building ensembles and nodes")
    for obj in network.ensembles + network.nodes:
        model.build(obj)

    logger.debug("Network step 2: Building subnetworks")
    for subnetwork in network.networks:
        model.build(subnetwork)

    logger.debug("Network step 3: Building connections")
    for conn in network.connections:
        model.build(conn)

    logger.debug("Network step 4: Building probes")
    for probe in network.probes:
        model.build(probe)

    # # Unset config
    # model.config = old_config
    model.params[network] = None


def gen_eval_points(ens, eval_points, rng, scale_eval_points=True):
    if isinstance(eval_points, Distribution):
        n_points = ens.n_eval_points
        if n_points is None:
            n_points = default_n_eval_points(ens.n_neurons, ens.dimensions)
        eval_points = eval_points.sample(n_points, ens.dimensions, rng)
    else:
        if (ens.n_eval_points is not None
                and eval_points.shape[0] != ens.n_eval_points):
            warnings.warn("Number of eval_points doesn't match "
                          "n_eval_points. Ignoring n_eval_points.")
        eval_points = np.array(eval_points, dtype=np.float64)
        assert eval_points.ndim == 2

    if scale_eval_points:
        eval_points *= ens.radius  # scale by ensemble radius
    return eval_points


def get_gain_bias(ens, rng=np.random, intercept_limit=1.0):
    if ens.gain is not None and ens.bias is not None:
        gain = get_samples(ens.gain, ens.n_neurons, rng=rng)
        bias = get_samples(ens.bias, ens.n_neurons, rng=rng)
        max_rates, intercepts = ens.neuron_type.max_rates_intercepts(
            gain, bias)
    elif ens.gain is not None or ens.bias is not None:
        # TODO: handle this instead of error
        raise NotImplementedError("gain or bias set for %s, but not both. "
                                  "Solving for one given the other is not "
                                  "implemented yet." % ens)
    else:
        int_distorarray = ens.intercepts
        if isinstance(int_distorarray, nengo.dists.Uniform):
            if int_distorarray.high > intercept_limit:
                warnings.warn(
                    "Intercepts are larger than intercept limit (%g). "
                    "High intercept values cause issues when discretizing "
                    "the model for running on Loihi." % intercept_limit)
                int_distorarray = nengo.dists.Uniform(
                    min(int_distorarray.low, intercept_limit),
                    min(int_distorarray.high, intercept_limit))

        max_rates = get_samples(ens.max_rates, ens.n_neurons, rng=rng)
        intercepts = get_samples(int_distorarray, ens.n_neurons, rng=rng)

        if np.any(intercepts > intercept_limit):
            intercepts[intercepts > intercept_limit] = intercept_limit
            warnings.warn(
                "Intercepts are larger than intercept limit (%g). "
                "High intercept values cause issues when discretizing "
                "the model for running on Loihi." % intercept_limit)

        gain, bias = ens.neuron_type.gain_bias(max_rates, intercepts)
        if gain is not None and (
                not np.all(np.isfinite(gain)) or np.any(gain <= 0.)):
            raise BuildError(
                "The specified intercepts for %s lead to neurons with "
                "negative or non-finite gain. Please adjust the intercepts so "
                "that all gains are positive. For most neuron types (e.g., "
                "LIF neurons) this is achieved by reducing the maximum "
                "intercept value to below 1." % ens)

    return gain, bias, max_rates, intercepts


BuiltEnsemble = collections.namedtuple(
    'BuiltEnsemble',
    ('eval_points',
     'encoders',
     'intercepts',
     'max_rates',
     'scaled_encoders',
     'gain',
     'bias'))


@Builder.register(Ensemble)
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
    gain, bias, max_rates, intercepts = get_gain_bias(
        ens, rng, model.intercept_limit)

    group = CxGroup(ens.n_neurons, label='%s' % ens)
    group.bias[:] = bias
    model.build(ens.neuron_type, ens.neurons, group)

    # set default filter just in case no other filter gets set
    group.configure_default_filter(model.decode_tau, dt=model.dt)

    if ens.noise is not None:
        raise NotImplementedError("Ensemble noise not implemented")

    # Scale the encoders
    if isinstance(ens.neuron_type, nengo.Direct):
        raise NotImplementedError("Direct neurons not implemented")
        # scaled_encoders = encoders
    else:
        # to keep scaling reasonable, we don't include the radius
        # scaled_encoders = encoders * (gain / ens.radius)[:, np.newaxis]
        scaled_encoders = encoders * gain[:, np.newaxis]

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


def build_decode_neuron_encoders(model, ens, kind='decode_neuron_encoders'):
    """Build encoders accepting decode neuron input."""
    group = model.objs[ens.neurons]['in']
    scaled_encoders = model.params[ens].scaled_encoders
    if kind == 'node_encoders':
        encoders = model.node_neurons.get_post_encoders(scaled_encoders)
    elif kind == 'decode_neuron_encoders':
        encoders = model.decode_neurons.get_post_encoders(scaled_encoders)

    synapses = CxSynapses(encoders.shape[0], label=kind)
    synapses.set_full_weights(encoders)
    group.add_synapses(synapses, name=kind)


@Builder.register(nengo.neurons.NeuronType)
def build_neurons(model, neurontype, neurons, group):
    # If we haven't registered a builder for a specific type, then it cannot
    # be simulated on Loihi.
    raise BuildError(
        "The neuron type %r cannot be simulated on Loihi. Please either "
        "switch to a supported neuron type like LIF or "
        "SpikingRectifiedLinear, or explicitly mark ensembles using this "
        "neuron type as off-chip with\n"
        "  net.config[ensembles].on_chip = False")


@Builder.register(nengo.LIF)
def build_lif(model, lif, neurons, group):
    group.configure_lif(
        tau_rc=lif.tau_rc,
        tau_ref=lif.tau_ref,
        dt=model.dt)


@Builder.register(nengo.SpikingRectifiedLinear)
def build_relu(model, relu, neurons, group):
    group.configure_relu(
        vth=1./model.dt,  # so input == 1 -> neuron fires 1/dt steps -> 1 Hz
        dt=model.dt)


@Builder.register(Node)
def build_node(model, node):
    if isinstance(node, ChipReceiveNode):
        spike_input = CxSpikeInput(node.raw_dimensions)
        model.add_input(spike_input)
        model.objs[node]['out'] = spike_input
        node.cx_spike_input = spike_input
    else:
        raise NotImplementedError()


def get_eval_points(model, conn, rng):
    if conn.eval_points is None:
        view = model.params[conn.pre_obj].eval_points.view()
        view.setflags(write=False)
        return view
    else:
        return gen_eval_points(
            conn.pre_obj, conn.eval_points, rng, conn.scale_eval_points)


def get_targets(conn, eval_points):
    if conn.function is None:
        targets = eval_points[:, conn.pre_slice]
    elif isinstance(conn.function, np.ndarray):
        targets = conn.function
    else:
        targets = np.zeros((len(eval_points), conn.size_mid))
        for i, ep in enumerate(eval_points[:, conn.pre_slice]):
            out = conn.function(ep)
            if out is None:
                raise BuildError("Building %s: Connection function returned "
                                 "None. Cannot solve for decoders." % (conn,))
            targets[i] = out

    return targets


def build_decoders(model, conn, rng, transform):
    encoders = model.params[conn.pre_obj].encoders
    gain = model.params[conn.pre_obj].gain
    bias = model.params[conn.pre_obj].bias

    eval_points = get_eval_points(model, conn, rng)
    targets = get_targets(conn, eval_points)

    x = np.dot(eval_points, encoders.T / conn.pre_obj.radius)
    E = None
    if conn.solver.weights:
        E = model.params[conn.post_obj].scaled_encoders.T[conn.post_slice]
        # include transform in solved weights
        targets = multiply(targets, transform.T)

    # wrapped_solver = (model.decoder_cache.wrap_solver(solve_for_decoders)
    #                   if model.seeded[conn] else solve_for_decoders)
    # decoders, solver_info = wrapped_solver(
    decoders, solver_info = solve_for_decoders(
        conn, gain, bias, x, targets, rng=rng, dt=model.dt, E=E)

    weights = (decoders.T if conn.solver.weights else
               multiply(transform, decoders.T))
    return eval_points, weights, solver_info


def solve_for_decoders(conn, gain, bias, x, targets, rng, dt, E=None):
    activities = loihi_rates(conn.pre_obj.neuron_type, x, gain, bias, dt)
    if np.count_nonzero(activities) == 0:
        raise BuildError(
            "Building %s: 'activities' matrix is all zero for %s. "
            "This is because no evaluation points fall in the firing "
            "ranges of any neurons." % (conn, conn.pre_obj))

    decoders, solver_info = conn.solver(activities, targets, rng=rng, E=E)
    return decoders, solver_info


def multiply(x, y):
    if x.ndim <= 2 and y.ndim < 2:
        return x * y
    elif x.ndim < 2 and y.ndim == 2:
        return x.reshape(-1, 1) * y
    elif x.ndim == 2 and y.ndim == 2:
        return np.dot(x, y)
    else:
        raise BuildError("Tensors not supported (x.ndim = %d, y.ndim = %d)"
                         % (x.ndim, y.ndim))


@Builder.register(Solver)
def build_solver(model, solver, conn, rng, transform):
    return build_decoders(model, conn, rng, transform)


@Builder.register(NoSolver)
def build_no_solver(model, solver, conn, rng, transform):
    activities = np.zeros((1, conn.pre_obj.n_neurons))
    targets = np.zeros((1, conn.size_mid))
    E = np.zeros((1, conn.post_obj.n_neurons)) if solver.weights else None
    # No need to invoke the cache for NoSolver
    decoders, solver_info = conn.solver(activities, targets, rng=rng, E=E)
    weights = (decoders.T if conn.solver.weights else
               multiply(transform, decoders.T))
    return None, weights, solver_info


@Builder.register(Connection)  # noqa: C901
def build_connection(model, conn):
    if isinstance(conn.transform, conv.Conv2D):
        # TODO: integrate these into the same function
        conv.build_conv2d_connection(model, conn)
        return

    # Create random number generator
    rng = np.random.RandomState(model.seeds[conn])

    pre_cx = model.objs[conn.pre_obj]['out']
    post_cx = model.objs[conn.post_obj]['in']
    assert isinstance(pre_cx, (CxGroup, CxSpikeInput))
    assert isinstance(post_cx, (CxGroup, CxProbe))

    weights = None
    eval_points = None
    solver_info = None
    neuron_type = None

    # Sample transform if given a distribution
    transform = get_samples(
        conn.transform, conn.size_out, d=conn.size_mid, rng=rng)

    tau_s = 0.0  # `synapse is None` gets mapped to `tau_s = 0.0`
    if isinstance(conn.synapse, nengo.synapses.Lowpass):
        tau_s = conn.synapse.tau
    elif conn.synapse is not None:
        raise NotImplementedError("Cannot handle non-Lowpass synapses")

    needs_decode_neurons = False
    target_encoders = None
    if isinstance(conn.pre_obj, Node):
        assert conn.pre_slice == slice(None)

        if np.array_equal(transform, np.array(1.)):
            # TODO: this identity transform may be avoidable
            transform = np.eye(conn.pre.size_out)
        else:
            assert transform.ndim == 2, "transform shape not handled yet"
            assert transform.shape[1] == conn.pre.size_out

        assert transform.shape[1] == conn.pre.size_out
        if isinstance(conn.pre_obj, ChipReceiveNeurons):
            weights = transform / model.dt
            neuron_type = conn.pre_obj.neuron_type
        else:
            # input is on-off neuron encoded, so double/flip transform
            weights = np.column_stack([transform, -transform])
            target_encoders = 'node_encoders'
    elif (isinstance(conn.pre_obj, Ensemble)
          and isinstance(conn.pre_obj.neuron_type, nengo.Direct)):
        raise NotImplementedError()
    elif isinstance(conn.pre_obj, Ensemble):  # Normal decoded connection
        eval_points, weights, solver_info = model.build(
            conn.solver, conn, rng, transform)

        # the decoder solver assumes a spike height of 1/dt; that isn't the
        # case on loihi, so we need to undo that scaling
        weights = weights / model.dt

        neuron_type = conn.pre_obj.neuron_type

        if not conn.solver.weights:
            needs_decode_neurons = True
    elif isinstance(conn.pre_obj, Neurons):
        assert conn.pre_slice == slice(None)
        assert transform.ndim == 2, "transform shape not handled yet"
        weights = transform / model.dt
        neuron_type = conn.pre_obj.ensemble.neuron_type
    else:
        raise NotImplementedError("Connection from type %r" % (
            type(conn.pre_obj),))

    if neuron_type is not None and hasattr(neuron_type, 'amplitude'):
        weights = weights * neuron_type.amplitude

    mid_cx = pre_cx
    mid_axon_inds = None
    post_tau = tau_s
    if needs_decode_neurons and not isinstance(conn.post_obj, Neurons):
        # --- add decode neurons
        assert weights.ndim == 2
        d, n = weights.shape

        if isinstance(post_cx, CxProbe):
            # use non-spiking decode neurons for voltage probing
            assert post_cx.target is None
            assert conn.post_slice == slice(None)

            # use the same scaling as the ensemble does, to get good
            #  decodes.  Note that this assumes that the decoded value
            #  is in the range -radius to radius, which is usually true.
            weights = weights / conn.pre_obj.radius

            gain = 1
            dec_cx = CxGroup(2 * d, label='%s' % conn, location='core')
            dec_cx.configure_nonspiking(dt=model.dt, vth=model.vth_nonspiking)
            dec_cx.bias[:] = 0
            model.add_group(dec_cx)
            model.objs[conn]['decoded'] = dec_cx

            dec_syn = CxSynapses(n, label="probe_decoders")
            weights2 = gain * np.vstack([weights, -weights]).T

            dec_syn.set_full_weights(weights2)
            dec_cx.add_synapses(dec_syn)
            model.objs[conn]['decoders'] = dec_syn
        else:
            # use spiking decode neurons for on-chip connection
            if isinstance(conn.post_obj, Ensemble):
                # loihi encoders don't include radius, so handle scaling here
                weights = weights / conn.post_obj.radius

            post_d = conn.post_obj.size_in
            post_inds = np.arange(post_d, dtype=np.int32)[conn.post_slice]
            assert weights.shape[0] == len(post_inds) == conn.size_out == d
            mid_axon_inds = model.decode_neurons.get_post_inds(
                post_inds, post_d)

            target_encoders = 'decode_neuron_encoders'
            dec_cx, dec_syn = model.decode_neurons.get_cx(
                weights, cx_label="%s" % conn, syn_label="decoders")

            model.add_group(dec_cx)
            model.objs[conn]['decoded'] = dec_cx
            model.objs[conn]['decoders'] = dec_syn

        # use tau_s for filter into decode neurons, decode_tau for filter out
        dec_cx.configure_filter(tau_s, dt=model.dt)
        post_tau = model.decode_tau

        dec_ax0 = CxAxons(n, label="decoders")
        dec_ax0.target = dec_syn
        pre_cx.add_axons(dec_ax0)
        model.objs[conn]['decode_axons'] = dec_ax0

        if conn.learning_rule_type is not None:
            rule_type = conn.learning_rule_type
            if isinstance(rule_type, nengo.PES):
                if not isinstance(rule_type.pre_synapse,
                                  nengo.synapses.Lowpass):
                    raise ValidationError(
                        "Loihi only supports `Lowpass` pre-synapses for "
                        "learning rules", attr='pre_synapse', obj=rule_type)

                tracing_tau = rule_type.pre_synapse.tau / model.dt

                # Nengo builder scales PES learning rate by `dt / n_neurons`
                n_neurons = (conn.pre_obj.n_neurons
                             if isinstance(conn.pre_obj, Ensemble)
                             else conn.pre_obj.size_in)
                learning_rate = rule_type.learning_rate * model.dt / n_neurons

                # Account for scaling to put integer error in range [-127, 127]
                learning_rate /= model.pes_error_scale

                # Tracing mag set so that the magnitude of the pre trace
                # is independent of the pre tau. `dt` factor accounts for
                # Nengo's `dt` spike scaling. Where is the second `dt` from?
                # Maybe the fact that post decode neurons have `vth = 1/dt`?
                tracing_mag = -np.expm1(-1. / tracing_tau) / model.dt**2

                # learning weight exponent controls the maximum weight
                # magnitude/weight resolution
                wgt_exp = model.pes_wgt_exp

                dec_syn.set_learning(
                    learning_rate=learning_rate,
                    tracing_mag=tracing_mag,
                    tracing_tau=tracing_tau,
                    wgt_exp=wgt_exp,
                )
            else:
                raise NotImplementedError()

        mid_cx = dec_cx

    if isinstance(post_cx, CxProbe):
        assert post_cx.target is None
        assert conn.post_slice == slice(None)
        post_cx.target = mid_cx
        mid_cx.add_probe(post_cx)
    elif isinstance(conn.post_obj, Neurons):
        assert isinstance(post_cx, CxGroup)
        assert conn.post_slice == slice(None)
        if weights is None:
            raise NotImplementedError("Need weights for connection to neurons")
        else:
            assert weights.ndim == 2
            n2, n1 = weights.shape
            assert post_cx.n == n2

            syn = CxSynapses(n1, label="neuron_weights")
            gain = model.params[conn.post_obj.ensemble].gain
            syn.set_full_weights(weights.T * gain)
            post_cx.add_synapses(syn)
            model.objs[conn]['weights'] = syn

        ax = CxAxons(mid_cx.n, label="neuron_weights")
        ax.target = syn
        mid_cx.add_axons(ax)

        post_cx.configure_filter(post_tau, dt=model.dt)

        if conn.learning_rule_type is not None:
            raise NotImplementedError()
    elif isinstance(conn.post_obj, Ensemble) and conn.solver.weights:
        assert isinstance(post_cx, CxGroup)
        assert weights.ndim == 2
        n2, n1 = weights.shape
        assert post_cx.n == n2

        # loihi encoders don't include radius, so handle scaling here
        weights = weights / conn.post_obj.radius

        syn = CxSynapses(n1, label="%s::decoder_weights" % conn)
        syn.set_full_weights(weights.T)
        post_cx.add_synapses(syn)
        model.objs[conn]['weights'] = syn

        ax = CxAxons(n1, label="decoder_weights")
        ax.target = syn
        mid_cx.add_axons(ax)

        post_cx.configure_filter(post_tau, dt=model.dt)

        if conn.learning_rule_type is not None:
            raise NotImplementedError()
    elif isinstance(conn.post_obj, Ensemble):
        assert target_encoders is not None
        if target_encoders not in post_cx.named_synapses:
            build_decode_neuron_encoders(
                model, conn.post_obj, kind=target_encoders)

        mid_ax = CxAxons(mid_cx.n, label="encoders")
        mid_ax.target = post_cx.named_synapses[target_encoders]
        mid_ax.set_axon_map(mid_axon_inds)
        mid_cx.add_axons(mid_ax)
        model.objs[conn]['mid_axons'] = mid_ax

        post_cx.configure_filter(post_tau, dt=model.dt)
    elif isinstance(conn.post_obj, Node):
        raise NotImplementedError()
    else:
        raise NotImplementedError()

    model.params[conn] = BuiltConnection(
        eval_points=eval_points,
        solver_info=solver_info,
        transform=transform,
        weights=weights)


def conn_probe(model, probe):
    # Connection probes create a connection from the target, and probe
    # the resulting signal (used when you want to probe the default
    # output of an object, which may not have a predefined signal)

    synapse = 0  # Removed internal filtering

    # get any extra arguments if this probe was created to send data
    #  to an off-chip Node via the splitter

    kwargs = model.chip2host_params.get(probe, None)
    if kwargs is not None:
        # this probe is for sending data to a Node

        # determine the dimensionality
        input_dim = probe.target.size_out
        func = kwargs['function']
        if func is not None:
            if callable(func):
                input_dim = np.asarray(
                    func(np.zeros(input_dim, dtype=np.float64))).size
            else:
                input_dim = len(func[0])
        transform = kwargs['transform']
        transform = np.asarray(transform, dtype=np.float64)
        if transform.ndim <= 1:
            output_dim = input_dim
        elif transform.ndim == 2:
            assert transform.shape[1] == input_dim
            output_dim = transform.shape[0]
        else:
            raise NotImplementedError()

        target = nengo.Node(size_in=output_dim, add_to_container=False)

        conn = Connection(probe.target, target, synapse=synapse,
                          solver=probe.solver, add_to_container=False,
                          **kwargs
                          )
        model.probe_conns[probe] = conn
    else:
        conn = Connection(probe.target, probe, synapse=synapse,
                          solver=probe.solver, add_to_container=False,
                          )
        target = probe

    # Set connection's seed to probe's (which isn't used elsewhere)
    model.seeded[conn] = model.seeded[probe]
    model.seeds[conn] = model.seeds[probe]

    d = conn.size_out
    if isinstance(probe.target, Ensemble):
        # probed values are scaled by the target ensemble's radius
        scale = probe.target.radius
        w = np.diag(scale * np.ones(d))
        weights = np.vstack([w, -w])
    else:
        raise NotImplementedError(
            "Nodes cannot be onchip, connections not yet probeable")

    cx_probe = CxProbe(key='v', weights=weights, synapse=probe.synapse)
    model.objs[target]['in'] = cx_probe
    model.objs[target]['out'] = cx_probe

    # add an extra entry for simulator.run_steps to read data out
    model.objs[probe]['out'] = cx_probe

    # Build the connection
    model.build(conn)


def signal_probe(model, key, probe):
    kwargs = model.chip2host_params.get(probe, None)
    weights = None
    if kwargs is not None:
        if kwargs['function'] is not None:
            raise BuildError("Functions not supported for signal probe")
        weights = kwargs['transform'].T / model.dt

    if isinstance(probe.target, nengo.ensemble.Neurons):
        if probe.attr == 'output':
            if weights is None:
                # spike probes should give values of 1.0/dt on spike events
                weights = 1.0 / model.dt

            if hasattr(probe.target.ensemble.neuron_type, 'amplitude'):
                weights = weights * probe.target.ensemble.neuron_type.amplitude

    # Signal probes directly probe a target signal
    target = model.objs[probe.obj]['out']

    cx_probe = CxProbe(
        target=target, key=key, slice=probe.slice,
        synapse=probe.synapse, weights=weights)
    target.add_probe(cx_probe)
    model.objs[probe]['in'] = target
    model.objs[probe]['out'] = cx_probe


probemap = {
    Ensemble: {'decoded_output': None,
               'input': 'q'},
    Neurons: {'output': 's',
              'spikes': 's',
              'voltage': 'v',
              'input': 'u'},
    Node: {'output': None},
    Connection: {'output': 'weighted',
                 'input': 'in'},
    LearningRule: {},  # make LR signals probeable, but no mapping required
}


@Builder.register(Probe)
def build_probe(model, probe):
    # find the right parent class in `objtypes`, using `isinstance`
    for nengotype, probeables in probemap.items():
        if isinstance(probe.obj, nengotype):
            break
    else:
        raise BuildError(
            "Type %r is not probeable" % type(probe.obj).__name__)

    key = probeables[probe.attr] if probe.attr in probeables else probe.attr
    if key is None:
        conn_probe(model, probe)
    else:
        signal_probe(model, key, probe)

    model.probes.append(probe)

    # Simulator will fill this list with probe data during simulation
    model.params[probe] = []
