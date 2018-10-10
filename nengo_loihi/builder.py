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
from nengo.exceptions import BuildError
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
from nengo_loihi.neurons import loihi_rates

logger = logging.getLogger(__name__)


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
        # Filter on intermediary neurons
        self.inter_tau = 0.005
        # ^TODO: how to choose this filter? Even though the input is spikes,
        # it may not be absolutely necessary since tau_rc provides a filter,
        # and maybe we don't want double filtering if connection has a filter

        # firing rate of inter neurons
        self._inter_rate = None

        # number of inter neurons
        self.inter_n = 10

        # noise exponent for inter neurons
        self.inter_noise_exp = -2

        # voltage threshold for non-spiking neurons (i.e. voltage decoders)
        self.vth_nonspiking = 10

        # limit for clipping intercepts, to avoid neurons with high gains
        self.intercept_limit = 0.95

        # Will be provided by Simulator
        self.chip2host_params = {}

    @property
    def inter_rate(self):
        return (1. / (self.dt * self.inter_n) if self._inter_rate is None else
                self._inter_rate)

    @inter_rate.setter
    def inter_rate(self, inter_rate):
        self._inter_rate = inter_rate

    @property
    def inter_scale(self):
        """Scaling applied to input from interneurons.

        Such that if all ``inter_n`` interneurons are firing at
        their max rate ``inter_rate``, then the total output when
        averaged over time will be 1.
        """
        return 1. / (self.dt * self.inter_rate * self.inter_n)

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
    group.configure_default_filter(model.inter_tau, dt=model.dt)

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


def build_interencoders(model, ens):
    """Build encoders accepting on/off interneuron input."""
    group = model.objs[ens.neurons]['in']
    scaled_encoders = model.params[ens].scaled_encoders

    synapses = CxSynapses(2*scaled_encoders.shape[1], label="inter_encoders")
    interscaled_encoders = scaled_encoders * model.inter_scale
    synapses.set_full_weights(
        np.vstack([interscaled_encoders.T, -interscaled_encoders.T]))
    group.add_synapses(synapses, name='inter_encoders')


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

    needs_interneurons = False
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

            # (max_rate = INTER_RATE * INTER_N) is the spike rate we
            # use to represent a value of +/- 1
            weights = weights * model.inter_scale
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
            needs_interneurons = True
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
    if needs_interneurons and not isinstance(conn.post_obj, Neurons):
        # --- add interneurons
        assert weights.ndim == 2
        d, n = weights.shape

        if isinstance(post_cx, CxProbe):
            # use non-spiking interneurons for voltage probing
            assert post_cx.target is None
            assert conn.post_slice == slice(None)

            # use the same scaling as the ensemble does, to get good
            #  decodes.  Note that this assumes that the decoded value
            #  is in the range -radius to radius, which is usually true.
            weights = weights / conn.pre_obj.radius

            gain = 1  # model.dt * INTER_RATE(=1000)
            dec_cx = CxGroup(2 * d, label='%s' % conn, location='core')
            dec_cx.configure_nonspiking(dt=model.dt, vth=model.vth_nonspiking)
            dec_cx.bias[:] = 0
            model.add_group(dec_cx)
            model.objs[conn]['decoded'] = dec_cx

            dec_syn = CxSynapses(n, label="probe_decoders")
            weights2 = gain * np.vstack([weights, -weights]).T
        else:
            # use spiking interneurons for on-chip connection
            post_d = conn.post_obj.size_in
            post_inds = np.arange(post_d, dtype=np.int32)[conn.post_slice]
            assert len(post_inds) == conn.size_out == d
            mid_axon_inds = np.hstack([post_inds,
                                       post_inds + post_d] * model.inter_n)

            gain = model.dt * model.inter_rate
            dec_cx = CxGroup(2 * d * model.inter_n, label='%s' % conn,
                             location='core')
            dec_cx.configure_relu(dt=model.dt)
            dec_cx.bias[:] = 0.5 * gain * np.array(
                ([1.] * d + [1.] * d) * model.inter_n)
            if model.inter_noise_exp > -30:
                dec_cx.enableNoise[:] = 1
                dec_cx.noiseExp0 = model.inter_noise_exp
                dec_cx.noiseAtDendOrVm = 1
            model.add_group(dec_cx)
            model.objs[conn]['decoded'] = dec_cx

            if isinstance(conn.post_obj, Ensemble):
                # loihi encoders don't include radius, so handle scaling here
                weights = weights / conn.post_obj.radius

            dec_syn = CxSynapses(n, label="decoders")
            weights2 = 0.5 * gain * np.vstack([weights,
                                               -weights] * model.inter_n).T

        # use tau_s for filter into interneurons, and INTER_TAU for filter out
        dec_cx.configure_filter(tau_s, dt=model.dt)
        post_tau = model.inter_tau

        dec_syn.set_full_weights(weights2)
        dec_cx.add_synapses(dec_syn)
        model.objs[conn]['decoders'] = dec_syn

        dec_ax0 = CxAxons(n, label="decoders")
        dec_ax0.target = dec_syn
        pre_cx.add_axons(dec_ax0)
        model.objs[conn]['decode_axons'] = dec_ax0

        if conn.learning_rule_type is not None:
            if isinstance(conn.learning_rule_type, nengo.PES):
                pes_learn_rate = conn.learning_rule_type.learning_rate
                # scale learning rates to roughly match Nengo
                # 1e-4 is the Nengo core default learning rate
                pes_learn_rate *= 4 / 1e-4
                assert isinstance(conn.learning_rule_type.pre_synapse,
                                  nengo.synapses.Lowpass)
                pes_pre_syn = conn.learning_rule_type.pre_synapse.tau
                # scale pre_syn.tau from s to ms
                pes_pre_syn *= 1e3
                dec_syn.set_learning(tracing_tau=pes_pre_syn,
                                     tracing_mag=pes_learn_rate)
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
        if 'inter_encoders' not in post_cx.named_synapses:
            build_interencoders(model, conn.post_obj)

        mid_ax = CxAxons(mid_cx.n, label="encoders")
        mid_ax.target = post_cx.named_synapses['inter_encoders']
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
    if isinstance(probe.target, Node):
        w = np.diag(model.inter_scale * np.ones(d))
        weights = np.vstack([w, -w])
    else:
        # probed values are scaled by the target ensemble's radius
        scale = probe.target.radius
        w = np.diag(scale * np.ones(d))
        weights = np.vstack([w, -w])
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
