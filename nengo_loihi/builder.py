import collections
import warnings

import numpy as np

from nengo import Ensemble, Connection
from nengo.exceptions import BuildError


class Model(object):
    def __init__(self, dt=0.001, label=None, decoder_cache=None, builder=None):
        self.dt = dt
        self.label = label
        self.decoder_cache = (NoDecoderCache() if decoder_cache is None
                              else decoder_cache)

        # Will be filled in by the network builder
        self.toplevel = None
        self.config = None

        # Resources used by the build process
        self.operators = []
        self.params = {}
        self.probes = []
        self.seeds = {}
        self.seeded = {}

        self.sig = collections.defaultdict(dict)
        self.sig['common'][0] = Signal(0., readonly=True, name='ZERO')
        self.sig['common'][1] = Signal(1., readonly=True, name='ONE')

        self.step = Signal(np.array(0, dtype=np.int64), name='step')
        self.time = Signal(np.array(0, dtype=np.float64), name='time')
        self.add_op(TimeUpdate(self.step, self.time))

        self.builder = Builder() if builder is None else builder
        self.build_callback = None

    def __str__(self):
        return "Model: %s" % self.label

    def add_op(self, op):
        """Add an operator to the model.

        In addition to adding the operator, this method performs additional
        error checking by calling the operator's ``make_step`` function.
        Calling ``make_step`` catches errors early, such as when signals are
        not properly initialized, which aids debugging. For that reason,
        we recommend calling this method over directly accessing
        the ``operators`` attribute.
        """
        self.operators.append(op)
        # Fail fast by trying make_step with a temporary sigdict
        signals = SignalDict()
        op.init_signals(signals)
        op.make_step(signals, self.dt, np.random)

    def build(self, obj, *args, **kwargs):
        """Build an object into this model.

        See `.Builder.build` for more details.

        Parameters
        ----------
        obj : object
            The object to build into this model.
        """
        built = self.builder.build(self, obj, *args, **kwargs)
        if self.build_callback is not None:
            self.build_callback(obj)
        return built

    def has_built(self, obj):
        """Returns true if the object has already been built in this model.

        .. note:: Some objects (e.g. synapses) can be built multiple times,
                  and therefore will always result in this method returning
                  ``False`` even though they have been built.

        This check is implemented by checking if the object is in the
        ``params`` dictionary. Build function should therefore add themselves
        to ``model.params`` if they cannot be built multiple times.

        Parameters
        ----------
        obj : object
            The object to query.
        """
        return obj in self.params


class Builder(object):

    builders = {}

    @classmethod
    def build(cls, model, obj, *args, **kwargs):
        if model.has_built(obj):
            # TODO: Prevent this at pre-build validation time.
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
        """A decorator for adding a class to the build function registry.

        Raises a warning if a build function already exists for the class.

        Parameters
        ----------
        nengo_class : Class
            The type associated with the build function being decorated.
        """
        def register_builder(build_fn):
            if nengo_class in cls.builders:
                warnings.warn("Type '%s' already has a builder. Overwriting."
                              % nengo_class)
            cls.builders[nengo_class] = build_fn
            return build_fn
        return register_builder


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
        group = CxGroup(ens.n_neurons)
        group.bias = bias

    # Scale the encoders
    if isinstance(ens.neuron_type, Direct):
        scaled_encoders = encoders
    else:
        scaled_encoders = encoders * (gain / ens.radius)[:, np.newaxis]

    synapses = CxSynapses(scaled_encoders.shape[0])
    synapses.set_weights(scaled_encoders)
    group.add_synapses(synapses)

    model.add_group(group)

    # # Inject noise if specified
    # if ens.noise is not None:
    #     model.build(ens.noise, sig_out=model.sig[ens.neurons]['in'], inc=True)


    # Output is neural output
    # model.sig[ens]['out'] = model.sig[ens.neurons]['out']

    # model.params[ens] = BuiltEnsemble(eval_points=eval_points,
    #                                   encoders=encoders,
    #                                   intercepts=intercepts,
    #                                   max_rates=max_rates,
    #                                   scaled_encoders=scaled_encoders,
    #                                   gain=gain,
    #                                   bias=bias)


@Builder.register(Connection)
def build_connection(model, conn):

    # Create random number generator
    rng = np.random.RandomState(model.seeds[conn])

    pre_cx = model.cx_map[conn.pre_obj]
    post_cx = model.cx_map[conn.post_obj]

    weights = None
    eval_points = None
    solver_info = None
    signal_size = conn.size_out
    post_slice = conn.post_slice

    # Sample transform if given a distribution
    transform = get_samples(
        conn.transform, conn.size_out, d=conn.size_mid, rng=rng)

    if (isinstance(conn.pre_obj, Node) or
            (isinstance(conn.pre_obj, Ensemble) and
             isinstance(conn.pre_obj.neuron_type, Direct))):
        raise NotImplementedError()
    elif isinstance(conn.pre_obj, Ensemble):  # Normal decoded connection
        eval_points, weights, solver_info = model.build(
            conn.solver, conn, rng, transform)
        if conn.solver.weights:
            model.sig[conn]['out'] = model.sig[conn.post_obj.neurons]['in']
            signal_size = conn.post_obj.neurons.size_in
            post_slice = None  # don't apply slice later
    else:
        weights = transform
        in_signal = slice_signal(model, in_signal, conn.pre_slice)

    # Add operator for applying weights
    model.sig[conn]['weights'] = Signal(
        weights, name="%s.weights" % conn, readonly=True)
    signal = Signal(np.zeros(signal_size), name="%s.weighted" % conn)
    model.add_op(Reset(signal))
    op = ElementwiseInc if weights.ndim < 2 else DotInc
    model.add_op(op(model.sig[conn]['weights'],
                    in_signal,
                    signal,
                    tag="%s.weights_elementwiseinc" % conn))

    # Add operator for filtering
    if conn.synapse is not None:
        signal = model.build(conn.synapse, signal)

    # Store the weighted-filtered output in case we want to probe it
    model.sig[conn]['weighted'] = signal

    if isinstance(conn.post_obj, Neurons):
        # Apply neuron gains (we don't need to do this if we're connecting to
        # an Ensemble, because the gains are rolled into the encoders)
        gains = Signal(model.params[conn.post_obj.ensemble].gain[post_slice],
                       name="%s.gains" % conn)
        model.add_op(ElementwiseInc(
            gains, signal, model.sig[conn]['out'][post_slice],
            tag="%s.gains_elementwiseinc" % conn))
    else:
        # Copy to the proper slice
        model.add_op(Copy(
            signal, model.sig[conn]['out'], dst_slice=post_slice,
            inc=True, tag="%s" % conn))

    # Build learning rules
    if conn.learning_rule is not None:
        rule = conn.learning_rule
        rule = [rule] if not is_iterable(rule) else rule
        targets = []
        for r in itervalues(rule) if isinstance(rule, dict) else rule:
            model.build(r)
            targets.append(r.modifies)

        if 'encoders' in targets:
            encoder_sig = model.sig[conn.post_obj]['encoders']
            encoder_sig.readonly = False
        if 'decoders' in targets or 'weights' in targets:
            if weights.ndim < 2:
                raise BuildError(
                    "'transform' must be a 2-dimensional array for learning")
            model.sig[conn]['weights'].readonly = False

    model.params[conn] = BuiltConnection(eval_points=eval_points,
                                         solver_info=solver_info,
                                         transform=transform,
                                         weights=weights)
