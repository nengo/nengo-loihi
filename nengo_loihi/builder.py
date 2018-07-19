import collections
import logging

import numpy as np

import nengo
from nengo import Network, Ensemble, Connection, Node, Probe
from nengo.builder.builder import Builder as NengoBuilder
from nengo.builder.connection import (
    build_no_solver, build_solver, BuiltConnection)
from nengo.builder.ensemble import (
    BuiltEnsemble, gen_eval_points, get_gain_bias)
from nengo.builder.network import build_network
from nengo.cache import NoDecoderCache
from nengo.dists import Distribution, get_samples
from nengo.connection import LearningRule
from nengo.ensemble import Neurons
from nengo.exceptions import BuildError
from nengo.neurons import Direct, LIF, RectifiedLinear
from nengo.solvers import NoSolver, Solver
from nengo.utils.compat import iteritems
import nengo.utils.numpy as npext

from nengo_loihi.cx import (
    CxModel, CxGroup, CxSynapses, CxAxons, CxProbe, CxSpikeInput)
from nengo_loihi.splitter import ChipReceiveNeurons, ChipReceiveNode

logger = logging.getLogger(__name__)

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
        self.probe_conns = {}

        self.seeds = {}
        self.seeded = {}

        self.toplevel = None
        self.config = None
        self.decoder_cache = NoDecoderCache()

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


class Builder(NengoBuilder):
    """Fills in the Loihi Model object based on the Nengo Network.

    We cannot use the Nengo builder as is because we make normal Nengo
    networks for host-to-chip and chip-to-host communication. To keep
    Nengo and Nengo Loihi builders separate, we make a blank subclass,
    which effectively copies the class.
    """

    builders = {}


Builder.register(Network)(build_network)


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
        # to keep scaling reasonable, we don't include the radius
        # scaled_encoders = encoders * (gain / ens.radius)[:, np.newaxis]
        scaled_encoders = encoders * gain[:, np.newaxis]

    # --- encoders
    # synapses = CxSynapses(scaled_encoders.shape[1])
    # synapses.set_full_weights(scaled_encoders.T)
    # group.add_synapses(synapses, name='encoders')

    # --- encoders for interneurons
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
    group.configure_lif(
        tau_rc=lif.tau_rc,
        tau_ref=lif.tau_ref,
        dt=model.dt)


@Builder.register(RectifiedLinear)
def build_relu(model, relu, neurons, group):
    group.configure_relu(dt=model.dt)


@Builder.register(Node)
def build_node(model, node):
    if isinstance(node, ChipReceiveNode):
        cx_spiker = node.cx_spike_input
        model.add_input(cx_spiker)
        model.objs[node]['out'] = cx_spiker
        return
    else:
        raise NotImplementedError()


Builder.register(Solver)(build_solver)
Builder.register(NoSolver)(build_no_solver)


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
    neuron_type = None

    # Sample transform if given a distribution
    transform = get_samples(
        conn.transform, conn.size_out, d=conn.size_mid, rng=rng)

    tau_s = 0.0
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
            assert transform.ndim == 2
            assert transform.shape[1] == conn.pre.size_out

        assert transform.shape[1] == conn.pre.size_out
        if isinstance(conn.pre_obj, ChipReceiveNeurons):
            weights = transform
        else:
            # input is on-off neuron encoded, so double/flip transform
            weights = np.column_stack([transform, -transform])

            # remove rate factor added by pre-compute nodes;
            # (max_rate = INTER_RATE * INTER_N) is the spike rate we
            # use to represent a value of +/- 1
            weights = weights / (INTER_RATE * INTER_N * model.dt)
    elif (isinstance(conn.pre_obj, Ensemble) and
          isinstance(conn.pre_obj.neuron_type, Direct)):
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
        assert transform.ndim == 2
        weights = transform
        neuron_type = conn.pre_obj.ensemble.neuron_type
    else:
        raise NotImplementedError("Connection from type %r" % (
            type(conn.pre_obj),))

    if neuron_type is not None and hasattr(neuron_type, 'amplitude'):
        weights = weights * neuron_type.amplitude

    mid_cx = pre_cx
    mid_axon_inds = slice(None)
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
            dec_cx.configure_nonspiking(dt=model.dt, vth=VTH_NONSPIKING)
            dec_cx.configure_filter(tau_s, dt=model.dt)
            dec_cx.bias[:] = 0
            model.add_group(dec_cx)
            model.objs[conn]['decoded'] = dec_cx

            dec_syn = CxSynapses(n)
            weights2 = gain * np.vstack([weights, -weights]).T
        else:
            # use spiking interneurons for on-chip connection
            post_d = conn.post_obj.size_in
            post_inds = np.arange(post_d, dtype=np.int32)[conn.post_slice]
            assert len(post_inds) == conn.size_out == d
            mid_axon_inds = np.hstack([post_inds, post_d+post_inds] * INTER_N)

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

            if isinstance(conn.post_obj, Ensemble):
                # loihi encoders don't include radius, so handle scaling here
                weights = weights / conn.post_obj.radius

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

        if conn.learning_rule_type is not None:
            if isinstance(conn.learning_rule_type, nengo.PES):
                pes_learn_rate = conn.learning_rule_type.learning_rate
                # scale learning rates such that the default would be 10
                pes_learn_rate *= 10 / nengo.PES.learning_rate.default
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

            syn = CxSynapses(n1)
            gain = model.params[conn.post_obj.ensemble].gain
            syn.set_full_weights(weights.T * gain)
            post_cx.add_synapses(syn)
            model.objs[conn]['weights'] = syn

        ax = CxAxons(pre_cx.n)
        ax.target = syn
        pre_cx.add_axons(ax)

        post_cx.configure_filter(tau_s, dt=model.dt)
        # ^ TODO: check that all conns into post use same filter

        if conn.learning_rule_type is not None:
            raise NotImplementedError()
    elif isinstance(conn.post_obj, Ensemble) and conn.solver.weights:
        assert isinstance(post_cx, CxGroup)
        assert weights.ndim == 2
        n2, n1 = weights.shape
        assert post_cx.n == n2

        # loihi encoders don't include radius, so handle scaling here
        weights = weights / conn.post_obj.radius

        syn = CxSynapses(n1)
        syn.set_full_weights(weights.T)
        post_cx.add_synapses(syn)
        model.objs[conn]['weights'] = syn

        ax = CxAxons(n1)
        ax.target = syn
        pre_cx.add_axons(ax)

        post_cx.configure_filter(tau_s, dt=model.dt)
        # ^ TODO: check that all conns into post use same filter

        if conn.learning_rule_type is not None:
            raise NotImplementedError()
    elif isinstance(conn.post_obj, Ensemble):
        mid_ax = CxAxons(mid_cx.n)
        mid_ax.target = post_cx.named_synapses['encoders2']
        mid_ax.target_inds = mid_axon_inds
        mid_cx.add_axons(mid_ax)
        model.objs[conn]['mid_axons'] = mid_ax
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

        target = nengo.Node(None, size_in=output_dim,
                            add_to_container=False)

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
        inter_scale = 1. / (model.dt * INTER_RATE * INTER_N)
        w = np.diag(inter_scale * np.ones(d))
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
            raise ValueError("Functions not supported for signal probe")
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


# TODO: why q, s, v, u ?
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
    # This is a copy of Nengo's build_probe, but since conn_probe
    # and signal_probe are different, we have to include it here.

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
