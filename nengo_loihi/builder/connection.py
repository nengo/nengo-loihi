import nengo
from nengo import Ensemble, Connection, Node
from nengo.builder.connection import (
    build_no_solver as _build_no_solver,
    BuiltConnection,
    get_eval_points,
    get_targets,
    multiply,
)
from nengo.ensemble import Neurons
from nengo.exceptions import BuildError, ValidationError
from nengo.solvers import NoSolver, Solver
import numpy as np

from nengo_loihi import conv
from nengo_loihi.block import Axon, LoihiBlock, Probe, Synapse
from nengo_loihi.builder.builder import Builder
from nengo_loihi.builder.inputs import ChipReceiveNeurons
from nengo_loihi.compat import (
    nengo_transforms, sample_transform, conn_solver)
from nengo_loihi.inputs import LoihiInput
from nengo_loihi.neurons import loihi_rates


def build_decoders(model, conn, rng, sampled_transform):
    # Copied from Nengo, except where noted below

    encoders = model.params[conn.pre_obj].encoders
    gain = model.params[conn.pre_obj].gain
    bias = model.params[conn.pre_obj].bias

    eval_points = get_eval_points(model, conn, rng)
    targets = get_targets(conn, eval_points)

    if conn.solver.weights and not conn.solver.compositional:
        # solver is solving for the whole weight matrix, so apply
        # transform/encoders to targets

        # CHANGE: backwards compatibility with nengo<=2.8.0
        # if not isinstance(conn.transform, Dense):
        #     raise BuildError(
        #         "Non-compositional solvers only work with Dense transforms")
        # transform = conn.transform.sample(rng=rng)
        # targets = np.dot(targets, transform.T)
        if nengo_transforms is not None and not isinstance(
                conn.transform, nengo_transforms.Dense):  # pragma: no cover
            raise BuildError(
                "Non-compositional solvers only work with Dense transforms")
        targets = np.dot(targets, sampled_transform.T)

        # weight solvers only allowed on ensemble->ensemble connections
        assert isinstance(conn.post_obj, Ensemble)
        post_enc = model.params[conn.post_obj].scaled_encoders
        targets = np.dot(targets, post_enc.T[conn.post_slice])

    x = np.dot(eval_points, encoders.T / conn.pre_obj.radius)

    # CHANGE: we pass `dt` to `solve_for_decoders`,
    # and do not support the decoder cache.
    # wrapped_solver = (model.decoder_cache.wrap_solver(solve_for_decoders)
    #                   if model.seeded[conn] else solve_for_decoders)
    # decoders, solver_info = wrapped_solver(
    #     conn, gain, bias, x, targets, rng=rng)
    decoders, solver_info = solve_for_decoders(
        conn, gain, bias, x, targets, rng=rng, dt=model.dt)

    return eval_points, decoders.T, solver_info


def solve_for_decoders(conn, gain, bias, x, targets, rng, dt):
    # Copied from Nengo, except where noted below

    # CHANGE: we use `loihi_rates` to get activities
    # activities = conn.pre_obj.neuron_type.rates(x, gain, bias)
    activities = loihi_rates(conn.pre_obj.neuron_type, x, gain, bias, dt)

    if np.count_nonzero(activities) == 0:
        raise BuildError(
            "Building %s: 'activities' matrix is all zero for %s. "
            "This is because no evaluation points fall in the firing "
            "ranges of any neurons." % (conn, conn.pre_obj))

    # CHANGE: backwards compatibility for solvers
    # decoders, solver_info = conn.solver(activities, targets, rng=rng)
    decoders, solver_info = conn_solver(
        conn.solver, activities, targets, rng=rng)

    return decoders, solver_info


def build_decode_neuron_encoders(model, ens, kind='decode_neuron_encoders'):
    """Build encoders accepting decode neuron input."""
    block = model.objs[ens.neurons]['in']
    scaled_encoders = model.params[ens].scaled_encoders
    if kind == 'node_encoders':
        encoders = model.node_neurons.get_post_encoders(scaled_encoders)
    elif kind == 'decode_neuron_encoders':
        encoders = model.decode_neurons.get_post_encoders(scaled_encoders)

    synapse = Synapse(encoders.shape[0], label=kind)
    synapse.set_full_weights(encoders)
    block.add_synapse(synapse, name=kind)


@Builder.register(Solver)
def build_solver(model, solver, conn, rng, sampled_transform):
    return build_decoders(model, conn, rng, sampled_transform)


@Builder.register(NoSolver)
def build_no_solver(model, solver, conn, rng, sampled_transform):
    if nengo_transforms is None:
        return _build_no_solver(model, solver, conn, rng, sampled_transform)
    else:
        return _build_no_solver(model, solver, conn, rng)


@Builder.register(Connection)  # noqa: C901
def build_connection(model, conn):
    if nengo_transforms is not None:
        if isinstance(conn.transform, nengo_transforms.Convolution):
            # TODO: integrate these into the same function
            conv.build_conv2d_connection(model, conn)
            return
        elif not isinstance(conn.transform, nengo_transforms.Dense):
            raise NotImplementedError(
                "nengo-loihi does not yet support %s transforms"
                % conn.transform)

    # Create random number generator
    rng = np.random.RandomState(model.seeds[conn])

    pre_cx = model.objs[conn.pre_obj]['out']
    post_cx = model.objs[conn.post_obj]['in']
    assert isinstance(pre_cx, (LoihiBlock, LoihiInput))
    assert isinstance(post_cx, (LoihiBlock, Probe))

    weights = None
    eval_points = None
    solver_info = None
    neuron_type = None
    post_slice = conn.post_slice

    # sample transform (if using a distribution)
    transform = sample_transform(conn, rng=rng)

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
        eval_points, decoders, solver_info = model.build(
            conn.solver, conn, rng, transform)

        if conn.solver.weights and not conn.solver.compositional:
            weights = decoders
        else:
            weights = multiply(transform, decoders)

        # the decoder solver assumes a spike height of 1/dt; that isn't the
        # case on loihi, so we need to undo that scaling
        weights = weights / model.dt

        neuron_type = conn.pre_obj.neuron_type

        if conn.solver.weights:
            # weight solvers only allowed on ensemble->ensemble connections
            assert isinstance(conn.post_obj, Ensemble)

            if conn.solver.compositional:
                encoders = model.params[conn.post_obj].scaled_encoders.T
                encoders = encoders[post_slice]
                weights = multiply(encoders.T, weights)

            # post slice already applied to encoders (either here or in
            # `build_decoders`), so don't apply later
            post_slice = None
        else:
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

        if isinstance(post_cx, Probe):
            # use non-spiking decode neurons for voltage probing
            assert post_cx.target is None
            assert post_slice == slice(None)

            # use the same scaling as the ensemble does, to get good
            #  decodes.  Note that this assumes that the decoded value
            #  is in the range -radius to radius, which is usually true.
            weights = weights / conn.pre_obj.radius

            gain = 1
            dec_cx = LoihiBlock(2 * d, label='%s' % conn)
            dec_cx.compartment.configure_nonspiking(
                dt=model.dt, vth=model.vth_nonspiking)
            dec_cx.compartment.bias[:] = 0
            model.add_block(dec_cx)
            model.objs[conn]['decoded'] = dec_cx

            dec_syn = Synapse(n, label="probe_decoders")
            weights2 = gain * np.vstack([weights, -weights]).T

            dec_syn.set_full_weights(weights2)
            dec_cx.add_synapse(dec_syn)
            model.objs[conn]['decoders'] = dec_syn
        else:
            # use spiking decode neurons for on-chip connection
            if isinstance(conn.post_obj, Ensemble):
                # loihi encoders don't include radius, so handle scaling here
                weights = weights / conn.post_obj.radius

            post_d = conn.post_obj.size_in
            post_inds = np.arange(post_d, dtype=np.int32)[post_slice]
            assert weights.shape[0] == len(post_inds) == conn.size_out == d
            mid_axon_inds = model.decode_neurons.get_post_inds(
                post_inds, post_d)

            target_encoders = 'decode_neuron_encoders'
            dec_cx, dec_syn = model.decode_neurons.get_block(
                weights, block_label="%s" % conn, syn_label="decoders")

            model.add_block(dec_cx)
            model.objs[conn]['decoded'] = dec_cx
            model.objs[conn]['decoders'] = dec_syn

        # use tau_s for filter into decode neurons, decode_tau for filter out
        dec_cx.compartment.configure_filter(tau_s, dt=model.dt)
        post_tau = model.decode_tau

        dec_ax0 = Axon(n, label="decoders")
        dec_ax0.target = dec_syn
        pre_cx.add_axon(dec_ax0)
        model.objs[conn]['decode_axon'] = dec_ax0

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

    if isinstance(post_cx, Probe):
        assert post_cx.target is None
        assert post_slice == slice(None)
        post_cx.target = mid_cx
        mid_cx.add_probe(post_cx)
    elif isinstance(conn.post_obj, Neurons):
        assert isinstance(post_cx, LoihiBlock)
        assert post_slice == slice(None)
        if weights is None:
            raise NotImplementedError("Need weights for connection to neurons")
        else:
            assert weights.ndim == 2
            n2, n1 = weights.shape
            assert post_cx.n_neurons == n2

            syn = Synapse(n1, label="neuron_weights")
            gain = model.params[conn.post_obj.ensemble].gain
            syn.set_full_weights(weights.T * gain)
            post_cx.add_synapse(syn)
            model.objs[conn]['weights'] = syn

        ax = Axon(mid_cx.n_neurons, label="neuron_weights")
        ax.target = syn
        mid_cx.add_axon(ax)

        post_cx.compartment.configure_filter(post_tau, dt=model.dt)

        if conn.learning_rule_type is not None:
            raise NotImplementedError()
    elif isinstance(conn.post_obj, Ensemble) and conn.solver.weights:
        assert isinstance(post_cx, LoihiBlock)
        assert weights.ndim == 2
        n2, n1 = weights.shape
        assert post_cx.n_neurons == n2

        # loihi encoders don't include radius, so handle scaling here
        weights = weights / conn.post_obj.radius

        syn = Synapse(n1, label="%s::decoder_weights" % conn)
        syn.set_full_weights(weights.T)
        post_cx.add_synapse(syn)
        model.objs[conn]['weights'] = syn

        ax = Axon(n1, label="decoder_weights")
        ax.target = syn
        mid_cx.add_axon(ax)

        post_cx.compartment.configure_filter(post_tau, dt=model.dt)

        if conn.learning_rule_type is not None:
            raise NotImplementedError()
    elif isinstance(conn.post_obj, Ensemble):
        assert target_encoders is not None
        if target_encoders not in post_cx.named_synapses:
            build_decode_neuron_encoders(
                model, conn.post_obj, kind=target_encoders)

        mid_ax = Axon(mid_cx.n_neurons, label="encoders")
        mid_ax.target = post_cx.named_synapses[target_encoders]
        mid_ax.set_axon_map(mid_axon_inds)
        mid_cx.add_axon(mid_ax)
        model.objs[conn]['mid_axon'] = mid_ax

        post_cx.compartment.configure_filter(post_tau, dt=model.dt)
    else:
        # This includes Node, since nodes can't be targets on-chip
        raise NotImplementedError()

    model.params[conn] = BuiltConnection(
        eval_points=eval_points,
        solver_info=solver_info,
        transform=transform,
        weights=weights)
