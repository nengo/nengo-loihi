import collections
import warnings

import numpy as np

import nengo
from nengo import Network, Ensemble, Connection, Node, Probe
from nengo.dists import Distribution, get_samples
from nengo.connection import LearningRule
from nengo.ensemble import Neurons
from nengo.exceptions import BuildError
from nengo.neurons import Direct, LIF, RectifiedLinear
from nengo.solvers import NoSolver, Solver
from nengo.utils.builder import default_n_eval_points
from nengo.utils.compat import iteritems
import nengo.utils.numpy as npext

from nengo_loihi.loihi_cx import (
    CxModel, CxGroup, CxSynapses, CxAxons, CxProbe, CxSpikeInput)
from . import splitter


# Filter on intermediary neurons
INTER_TAU = 0.005
# ^TODO: how to choose this filter? Need it since all input will be spikes,
#   but maybe don't want double filtering if connection has a filter

# firing rate of inter neurons
INTER_RATE = 100

# number of inter neurons
INTER_N = 10

# noise exponent for inter neurons
INTER_NOISE_EXP = -2

# voltage threshold for non-spiking neurons (i.e. voltage decoders)
VTH_NONSPIKING = 10


class Model(CxModel):
    """The data structure for the chip/simulator.

    CxModel defines adding ensembles, discretizing, and tracks the simulator
    """
    def __init__(self, dt=0.001, label=None, builder=None):
        super(Model, self).__init__()

        self.dt = dt
        self.label = label

        self.objs = collections.defaultdict(dict)
        self.params = {}  # Holds data generated when building objects
        self.probes = []
        self.chip2host_params = {}

        self.seeds = {}
        self.seeded = {}

        self.builder = Builder() if builder is None else builder
        self.build_callback = None

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
            model.seeded[obj] = (model.seeded[network] or
                                 getattr(obj, 'seed', None) is not None)
            model.seeds[obj] = get_seed(obj, rng)

    # logger.debug("Network step 1: Building ensembles and nodes")
    for obj in network.ensembles + network.nodes:
        model.build(obj)

    # logger.debug("Network step 2: Building subnetworks")
    for subnetwork in network.networks:
        model.build(subnetwork)

    # logger.debug("Network step 3: Building connections")
    for conn in network.connections:
        model.build(conn)

    # logger.debug("Network step 4: Building probes")
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


def get_gain_bias(ens, rng=np.random):
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
        max_rates = get_samples(ens.max_rates, ens.n_neurons, rng=rng)
        intercepts = get_samples(ens.intercepts, ens.n_neurons, rng=rng)
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
    if isinstance(ens.neuron_type, Direct):
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

    if isinstance(ens.neuron_type, Direct):
        raise NotImplementedError()
    else:
        group = CxGroup(ens.n_neurons, label='%s' % ens)
        group.bias[:] = bias
        model.build(ens.neuron_type, ens.neurons, group)

    group.configure_filter(INTER_TAU, dt=model.dt)

    # Scale the encoders
    if isinstance(ens.neuron_type, Direct):
        raise NotImplementedError("Direct neurons not implemented")
        # scaled_encoders = encoders
    else:
        assert ens.radius == 1
        scaled_encoders = encoders * (gain / ens.radius)[:, np.newaxis]

    # synapses = CxSynapses(scaled_encoders.shape[1])
    # synapses.set_full_weights(scaled_encoders.T)
    # group.add_synapses(synapses, name='encoders')

    synapses2 = CxSynapses(2*scaled_encoders.shape[1])
    inter_scale = 1. / (model.dt * INTER_RATE * INTER_N)
    interscaled_encoders = scaled_encoders * inter_scale
    synapses2.set_full_weights(
        np.vstack([interscaled_encoders.T, -interscaled_encoders.T]))
    group.add_synapses(synapses2, name='encoders2')

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


@Builder.register(LIF)
def build_lif(model, lif, neurons, group):
    assert lif.amplitude == 1
    group.configure_lif(
        tau_rc=lif.tau_rc,
        tau_ref=lif.tau_ref,
        dt=model.dt)


@Builder.register(RectifiedLinear)
def build_relu(model, relu, neurons, group):
    assert relu.amplitude == 1
    group.configure_relu(dt=model.dt)


@Builder.register(Node)
def build_node(model, node):
    if isinstance(node, splitter.ChipReceiveNode):
        cx_spiker = node.cx_spike_input
        model.add_input(cx_spiker)
        model.objs[node]['out'] = cx_spiker
        return
    else:
        raise NotImplementedError()


BuiltConnection = collections.namedtuple(
    'BuiltConnection',
    ('eval_points', 'solver_info', 'weights', 'transform'))


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
        conn, gain, bias, x, targets, rng=rng, E=E)

    weights = (decoders.T if conn.solver.weights else
               multiply(transform, decoders.T))
    return eval_points, weights, solver_info


def solve_for_decoders(conn, gain, bias, x, targets, rng, E=None):
    activities = conn.pre_obj.neuron_type.rates(x, gain, bias)
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
    # Create random number generator
    rng = np.random.RandomState(model.seeds[conn])

    pre_cx = model.objs[conn.pre_obj]['out']
    post_cx = model.objs[conn.post_obj]['in']
    assert isinstance(pre_cx, (CxGroup, CxSpikeInput))
    assert isinstance(post_cx, (CxGroup, CxProbe))

    weights = None
    eval_points = None
    solver_info = None

    # Sample transform if given a distribution
    transform = get_samples(
        conn.transform, conn.size_out, d=conn.size_mid, rng=rng)

    tau_s = 0.0
    if isinstance(conn.synapse, nengo.synapses.Lowpass):
        tau_s = conn.synapse.tau
    elif conn.synapse is not None:
        raise NotImplementedError("Cannot handle non-Lowpass synapses")

    # post_cx.configure_filter(tau_s, dt=model.dt)
    # ^ TODO: check that all conns into post use same filter

    if isinstance(conn.pre_obj, Node):
        assert conn.pre_slice == slice(None)
        assert conn.post_slice == slice(None)

        # node is using on/off neuron coding
        assert np.array_equal(conn.transform, np.array(1.))
        # if tau_s != 0.0:
        #     raise NotImplementedError()

        if isinstance(post_cx, CxProbe):
            assert post_cx.target is None
            assert conn.post_slice == slice(None)
            post_cx.target = pre_cx
            pre_cx.add_probe(post_cx)
        else:
            d = conn.size_out
            ax = CxAxons(2*d)
            ax.target = post_cx.named_synapses['encoders2']
            pre_cx.add_axons(ax)
            model.objs[conn]['encode_axons'] = ax
    elif (isinstance(conn.pre_obj, Ensemble) and
          isinstance(conn.pre_obj.neuron_type, Direct)):
        raise NotImplementedError()
    elif isinstance(conn.pre_obj, Ensemble):  # Normal decoded connection
        eval_points, weights, solver_info = model.build(
            conn.solver, conn, rng, transform)

        weights = weights / model.dt
        # ^ scale, since nengo spikes have 1/dt scaling

        if conn.solver.weights:
            assert isinstance(post_cx, CxGroup)
            # post_slice = None  # don't apply slice later

            assert weights.ndim == 2
            n2, n1 = weights.shape
            assert post_cx.n == n2

            syn = CxSynapses(n1)
            syn.set_full_weights(weights.T)
            post_cx.add_synapses(syn)
            model.objs[conn]['weights'] = syn

            ax = CxAxons(n1)
            ax.target = syn
            pre_cx.add_axons(ax)

            post_cx.configure_filter(tau_s, dt=model.dt)
            # ^ TODO: check that all conns into post use same filter
        else:
            # on/off neuron coding for decoded values
            assert weights.ndim == 2
            d, n = weights.shape

            if isinstance(post_cx, CxProbe):
                assert post_cx.target is None
                assert conn.post_slice == slice(None)

                gain = 1  # model.dt * INTER_RATE(=1000)
                dec_cx = CxGroup(2*d, label='%s' % conn, location='core')
                dec_cx.configure_nonspiking(dt=model.dt, vth=VTH_NONSPIKING)
                dec_cx.configure_filter(tau_s, dt=model.dt)
                dec_cx.bias[:] = 0
                model.add_group(dec_cx)
                model.objs[conn]['decoded'] = dec_cx

                dec_syn = CxSynapses(n)
                weights2 = gain * np.vstack([weights, -weights]).T
                dec_syn.set_full_weights(weights2)
                dec_cx.add_synapses(dec_syn)
                model.objs[conn]['decoders'] = dec_syn

                dec_ax0 = CxAxons(n)
                dec_ax0.target = dec_syn
                pre_cx.add_axons(dec_ax0)
                model.objs[conn]['decode_axons'] = dec_ax0

                post_cx.target = dec_cx
                dec_cx.add_probe(post_cx)

            else:
                post_d = conn.post_obj.size_in
                post_inds = np.arange(post_d, dtype=np.int32)[conn.post_slice]
                assert len(post_inds) == d

                gain = model.dt * INTER_RATE
                dec_cx = CxGroup(2 * d * INTER_N, label='%s' % conn,
                                 location='core')
                dec_cx.configure_relu(dt=model.dt)
                dec_cx.configure_filter(tau_s, dt=model.dt)
                dec_cx.bias[:] = 0.5 * gain * np.array(([1.] * d +
                                                        [1.] * d) * INTER_N)
                if INTER_NOISE_EXP > -30:
                    dec_cx.enableNoise[:] = 1
                    dec_cx.noiseExp0 = INTER_NOISE_EXP
                    dec_cx.noiseAtDendOrVm = 1
                model.add_group(dec_cx)
                model.objs[conn]['decoded'] = dec_cx

                dec_syn = CxSynapses(n)
                weights2 = 0.5 * gain * np.vstack([weights,
                                                   -weights] * INTER_N).T
                dec_syn.set_full_weights(weights2)
                dec_cx.add_synapses(dec_syn)
                model.objs[conn]['decoders'] = dec_syn

                dec_ax0 = CxAxons(n)
                dec_ax0.target = dec_syn
                pre_cx.add_axons(dec_ax0)
                model.objs[conn]['decode_axons'] = dec_ax0

                dec_ax1 = CxAxons(2*d*INTER_N)
                dec_ax1.target = post_cx.named_synapses['encoders2']
                dec_ax1.target_inds = np.hstack(
                    [post_inds, post_d+post_inds] * INTER_N)
                dec_cx.add_axons(dec_ax1)
                model.objs[conn]['encode_axons'] = dec_ax1
    else:
        assert conn.pre_slice == slice(None)
        assert conn.post_slice == slice(None)
        weights = transform

        assert weights.ndim == 2
        n2, n1 = weights.shape
        assert post_cx.n == n2

        syn = CxSynapses(n1)
        syn.set_full_weights(weights.T)
        post_cx.add_synapses(syn)

        ax = CxAxons(n1)
        ax.target = syn
        pre_cx.add_axons(ax)

    # tau_s = 0.0
    # if isinstance(conn.synapse, nengo.synapses.Lowpass):
    #     tau_s = conn.synapse.tau
    # elif conn.synapse is not None:
    #     raise NotImplementedError("Cannot handle non-Lowpass synapses")

    # post_cx.configure_filter(tau_s, dt=model.dt)
    # ^ TODO: check that all conns into post use same filter

    if isinstance(conn.post_obj, Neurons):
        raise NotImplementedError()
        # # Apply neuron gains (we don't need to do this if we're connecting to
        # # an Ensemble, because the gains are rolled into the encoders)
        # gains = Signal(model.params[conn.post_obj.ensemble].gain[post_slice],
        #                name="%s.gains" % conn)
        # model.add_op(ElementwiseInc(
        #     gains, signal, model.sig[conn]['out'][post_slice],
        #     tag="%s.gains_elementwiseinc" % conn))
    else:
        pass
        # if post_slice is not None:
        #     raise NotImplementedError()

    # Build learning rules
    if conn.learning_rule is not None:
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

        target = nengo.Node(None, size_in=output_dim,
                            add_to_container=False)

        conn = Connection(probe.target, target, synapse=synapse,
                          solver=probe.solver, add_to_container=False,
                          **kwargs
                          )
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
        inter_scale = 1. / (model.dt * INTER_RATE * INTER_N)
        w = np.diag(inter_scale * np.ones(d))
        weights = np.vstack([w, -w])
    else:
        w = np.diag(np.ones(d))
        weights = np.vstack([w, -w])
    cx_probe = CxProbe(key='v', weights=weights, synapse=probe.synapse)
    model.objs[target]['in'] = cx_probe
    model.objs[target]['out'] = cx_probe

    # add an extra entry for simulator.run_steps to read data out
    model.objs[probe]['out'] = cx_probe

    # Build the connection
    model.build(conn)


def signal_probe(model, key, probe):
    # Signal probes directly probe a target signal
    target = model.objs[probe.obj]['out']

    weights = None
    # spike probes should give values of 1.0/dt on spike events
    if isinstance(probe.target, nengo.ensemble.Neurons):
        if probe.attr == 'output':
            weights = 1.0 / model.dt

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
    for nengotype, probeables in iteritems(probemap):
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
