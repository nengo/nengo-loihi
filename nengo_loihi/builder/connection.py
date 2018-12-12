import copy
import logging

import nengo
from nengo import Ensemble, Connection, Node, Probe as NengoProbe
from nengo.builder.connection import (
    build_no_solver as _build_no_solver,
    BuiltConnection,
    get_eval_points,
    get_targets,
    multiply,
)
from nengo.connection import LearningRule
from nengo.ensemble import Neurons
from nengo.exceptions import BuildError, ValidationError
from nengo.solvers import NoSolver, Solver
import numpy as np

from nengo_loihi.block import Axon, LoihiBlock, Probe, Synapse
from nengo_loihi.builder.builder import Builder
from nengo_loihi.builder.inputs import (
    ChipReceiveNode,
    ChipReceiveNeurons,
    HostSendNode,
    HostReceiveNode,
    PESModulatoryTarget,
)
from nengo_loihi.compat import (
    nengo_transforms, sample_transform, conn_solver)
from nengo_loihi.conv import channel_idxs, conv2d_loihi_weights, pixel_idxs
from nengo_loihi.inputs import LoihiInput
from nengo_loihi.neurons import loihi_rates
from nengo_loihi.passthrough import base_obj

logger = logging.getLogger(__name__)


def _inherit_seed(dest_model, dest_obj, src_model, src_obj):
    dest_model.seeded[dest_obj] = src_model.seeded[src_obj]
    dest_model.seeds[dest_obj] = src_model.seeds[src_obj]


@Builder.register(Connection)
def build_connection(model, conn):
    pre_onchip = model.split.on_chip(base_obj(conn.pre))

    if isinstance(conn.post_obj, LearningRule):
        assert not pre_onchip
        return build_host_to_learning_rule(model, conn)

    post_onchip = model.split.on_chip(base_obj(conn.post))

    if pre_onchip and post_onchip:
        build_chip_connection(model, conn)

    elif not pre_onchip and post_onchip:
        if isinstance(conn.pre_obj, Neurons):
            build_host_neurons_to_chip(model, conn)
        else:
            build_host_to_chip(model, conn)

    elif pre_onchip and not post_onchip:
        build_chip_to_host(model, conn)

    else:
        assert not pre_onchip and not post_onchip
        host = model.host_model(base_obj(conn.pre))
        assert host is model.host_model(base_obj(conn.post))
        _inherit_seed(host, conn, model, conn)
        host.build(conn)


def build_host_neurons_to_chip(model, conn):
    """Send spikes over and do the rest of the connection on-chip"""

    assert not isinstance(conn.post, LearningRule)
    dim = conn.size_in
    host = model.host_model(base_obj(conn.pre))

    logger.debug("Creating ChipReceiveNeurons for %s", conn)
    receive = ChipReceiveNeurons(
        dim,
        neuron_type=conn.pre_obj.ensemble.neuron_type,
        label=None if conn.label is None else "%s_neurons" % conn.label,
        add_to_container=False,
    )
    _inherit_seed(model, receive, model, conn)
    model.builder.build(model, receive)

    receive2post = Connection(
        receive,
        conn.post,
        transform=conn.transform,
        synapse=conn.synapse,
        label=None if conn.label is None else "%s_chip" % conn.label,
        add_to_container=False,
    )
    _inherit_seed(model, receive2post, model, conn)
    build_chip_connection(model, receive2post)

    logger.debug("Creating HostSendNode for %s", conn)
    send = HostSendNode(
        dim,
        label=None if conn.label is None else "%s_send" % conn.label,
        add_to_container=False,
    )
    host.build(send)

    pre2send = Connection(
        conn.pre,
        send,
        synapse=None,
        label=None if conn.label is None else "%s_host" % conn.label,
        add_to_container=False,
    )
    model.host2chip_senders[send] = receive
    _inherit_seed(host, pre2send, model, conn)
    host.build(pre2send)


def build_host_to_chip(model, conn):
    rng = np.random.RandomState(model.seeds[conn])
    dim = conn.size_out
    host = model.host_model(base_obj(conn.pre))

    logger.debug("Creating ChipReceiveNode for %s", conn)
    receive = ChipReceiveNode(
        dim * 2,
        size_out=dim,
        label=None if conn.label is None else "%s_node" % conn.label,
        add_to_container=False,
    )
    model.builder.build(model, receive)

    receive2post = Connection(
        receive,
        conn.post,
        synapse=model.decode_tau,
        label=None if conn.label is None else "%s_chip" % conn.label,
        add_to_container=False,
    )
    _inherit_seed(model, receive2post, model, conn)
    build_chip_connection(model, receive2post)

    logger.debug("Creating DecodeNeuron ensemble for %s", conn)
    ens = model.node_neurons.get_ensemble(dim)
    ens.label = None if conn.label is None else "%s_ens" % conn.label
    _inherit_seed(host, ens, model, conn)
    host.build(ens)

    if nengo_transforms is not None and isinstance(
            conn.transform, nengo_transforms.Convolution):
        raise BuildError(
            "Conv2D transforms not supported for off-chip to "
            "on-chip connections where `pre` is not a Neurons object.")

    # Scale the input spikes based on the radius of the target ensemble
    weights = sample_transform(conn, rng=rng)

    if isinstance(conn.post_obj, Ensemble):
        weights = weights / conn.post_obj.radius

    if nengo_transforms is None:
        transform = weights
    else:
        # copy the Transform information, setting `init` to the sampled weights
        transform = copy.copy(conn.transform)
        type(transform).init.data[transform] = weights

    pre2ens = Connection(
        conn.pre,
        ens,
        function=conn.function,
        solver=conn.solver,
        eval_points=conn.eval_points,
        scale_eval_points=conn.scale_eval_points,
        synapse=conn.synapse,
        transform=transform,
        label=None if conn.label is None else "%s_enc" % conn.label,
        add_to_container=False,
    )
    _inherit_seed(host, pre2ens, model, conn)
    host.build(pre2ens)

    logger.debug("Creating HostSendNode for %s", conn)
    send = HostSendNode(
        dim * 2,
        label=None if conn.label is None else "%s_send" % conn.label,
        add_to_container=False,
    )
    host.build(send)

    ensneurons2send = Connection(
        ens.neurons,
        send,
        synapse=None,
        label=None if conn.label is None else "%s_host" % conn.label,
        add_to_container=False,
    )
    _inherit_seed(host, ensneurons2send, model, conn)
    model.host2chip_senders[send] = receive
    host.build(ensneurons2send)


def build_chip_to_host(model, conn):
    rng = np.random.RandomState(model.seeds[conn])
    dim = conn.size_out
    host = model.host_model(base_obj(conn.post))

    logger.debug("Creating HostReceiveNode for %s", conn)
    receive = HostReceiveNode(
        dim,
        label=None if conn.label is None else "%s_receive" % conn.label,
        add_to_container=False,
    )
    host.build(receive)

    receive2post = Connection(
        receive,
        conn.post,
        synapse=conn.synapse,
        label=None if conn.label is None else "%s_host" % conn.label,
        add_to_container=False,
    )
    _inherit_seed(host, receive2post, model, conn)
    host.build(receive2post)

    logger.debug("Creating Probe for %s", conn)
    transform = sample_transform(conn, rng=rng)

    probe = NengoProbe(conn.pre,
                       synapse=None,
                       solver=conn.solver,
                       add_to_container=False)
    model.chip2host_params[probe] = dict(
        learning_rule_type=conn.learning_rule_type,
        function=conn.function,
        eval_points=conn.eval_points,
        scale_eval_points=conn.scale_eval_points,
        transform=transform,
        label=None if conn.label is None else "%s_probe" % conn.label,
    )
    model.chip2host_receivers[probe] = receive
    _inherit_seed(model, probe, model, conn)
    model.builder.build(model, probe)

    if conn.learning_rule_type is not None:
        if not isinstance(conn.pre_obj, Ensemble):
            raise NotImplementedError(
                "Learning rule presynaptic object must be an Ensemble "
                "(got %r)" % type(conn.pre_obj).__name__)
        model.needs_sender[conn.learning_rule] = PESModulatoryTarget(probe)


def build_host_to_learning_rule(model, conn):
    dim = conn.size_out
    host = model.host_model(base_obj(conn.pre))

    logger.debug("Creating HostSendNode for %s", conn)
    send = HostSendNode(
        dim,
        label=None if conn.label is None else "%s_send" % conn.label,
        add_to_container=False,
    )
    host.build(send)

    pre2send = Connection(
        conn.pre,
        send,
        function=conn.function,
        solver=conn.solver,
        eval_points=conn.eval_points,
        scale_eval_points=conn.scale_eval_points,
        synapse=conn.synapse,
        transform=conn.transform,
        label=conn.label,
        add_to_container=False,
    )
    pes_target = model.needs_sender[conn.post_obj]
    model.host2chip_senders[send] = pes_target
    _inherit_seed(host, pre2send, model, conn)
    host.build(pre2send)


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
    args = (model, solver, conn, rng)
    if nengo_transforms is None:
        args += (sampled_transform,)
    return _build_no_solver(*args)


def expand_to_2d(weights, pre_size, post_size):
    if weights.ndim == 0:
        assert pre_size == post_size
        weights2d = weights * np.eye(pre_size)
    elif weights.ndim == 1:
        assert pre_size == post_size
        assert weights.size == pre_size
        weights2d = np.diag(weights)
    else:
        assert weights.ndim == 2
        weights2d = weights

    assert weights2d.shape[0] == post_size
    assert weights2d.shape[1] == pre_size
    return weights2d


def build_chip_connection(model, conn):  # noqa: C901
    if nengo_transforms is not None:
        if isinstance(conn.transform, nengo_transforms.Convolution):
            return build_conv2d_connection(model, conn)
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
    if (isinstance(conn.pre_obj, Node)
            and not isinstance(conn.pre_obj, ChipReceiveNeurons)):
        assert conn.pre_slice == slice(None)

        transform = expand_to_2d(transform, conn.pre.size_out,
                                 conn.post.size_in)

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
    elif isinstance(conn.pre_obj, (Neurons, ChipReceiveNeurons)):
        assert conn.pre_slice == slice(None)
        weights = expand_to_2d(transform, conn.pre.size_out, conn.post.size_in)
        weights = weights / model.dt
        neuron_type = (conn.pre_obj.neuron_type
                       if isinstance(conn.pre_obj, ChipReceiveNeurons)
                       else conn.pre_obj.ensemble.neuron_type)

        if isinstance(conn.post_obj, Ensemble):
            needs_decode_neurons = True
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


def build_conv2d_connection(model, conn):
    if nengo_transforms is None:
        # It should not be possible to reach this, because this function is
        # only called for a Convolution transform, which can exist only if
        # nengo_transforms exists.
        raise NotImplementedError("Convolution requires newer Nengo")

    if conn.transform.dimensions != 2:
        raise NotImplementedError("nengo-loihi only supports 2D convolution")
    if conn.transform.padding != "valid":
        raise NotImplementedError(
            "nengo-loihi only supports convolution with 'valid' padding")

    # Create random number generator
    rng = np.random.RandomState(model.seeds[conn])

    pre_cx = model.objs[conn.pre_obj]['out']
    post_cx = model.objs[conn.post_obj]['in']
    assert isinstance(pre_cx, (LoihiInput, LoihiBlock))
    assert isinstance(post_cx, LoihiBlock)

    tau_s = 0.0
    if isinstance(conn.synapse, nengo.synapses.Lowpass):
        tau_s = conn.synapse.tau
    elif conn.synapse is not None:
        raise NotImplementedError("Cannot handle non-Lowpass synapses")

    # --- pre
    assert isinstance(conn.pre_obj, (Neurons, ChipReceiveNeurons))
    assert conn.pre_slice == slice(None)

    assert isinstance(conn.transform, nengo_transforms.Convolution)

    weights = conn.transform.sample(rng=rng)
    input_shape = conn.transform.input_shape

    # Account for nengo spike height of 1/dt
    weights = weights / model.dt

    if isinstance(conn.pre_obj, ChipReceiveNeurons):
        neuron_type = conn.pre_obj.neuron_type
    elif isinstance(conn.pre_obj, Neurons):
        neuron_type = conn.pre_obj.ensemble.neuron_type

    if neuron_type is not None and hasattr(neuron_type, 'amplitude'):
        weights = weights * neuron_type.amplitude

    # --- post
    assert isinstance(conn.post_obj, Neurons)
    assert conn.post_slice == slice(None)

    gain = model.params[conn.post_obj.ensemble].gain
    if not np.all(gain == gain[0]):
        # TODO: support this?
        raise ValidationError(
            "All neurons targeted by a Convolution connection must "
            "have the same gain", "gain", obj=conn.post_obj.ensemble)
    weights = weights * gain[0]

    pop_type = 32  # TODO: pick this
    new_transform = copy.copy(conn.transform)
    type(new_transform).init.data[new_transform] = weights
    weights, indices, axon_to_weight_map, cx_bases = conv2d_loihi_weights(
        new_transform)

    synapse = Synapse(np.prod(input_shape.spatial_shape),
                      label="conv2d_weights")
    synapse.set_population_weights(
        weights, indices, axon_to_weight_map, cx_bases, pop_type=pop_type)
    post_cx.add_synapse(synapse)
    model.objs[conn]['weights'] = synapse

    ax = Axon(np.prod(input_shape.spatial_shape), label="conv2d_weights")
    ax.target = synapse
    ax.cx_to_axon_map = pixel_idxs(input_shape)
    ax.cx_atoms = channel_idxs(input_shape)
    pre_cx.add_axon(ax)

    post_cx.compartment.configure_filter(tau_s, dt=model.dt)

    model.params[conn] = BuiltConnection(
        eval_points=None,
        solver_info=None,
        transform=None,
        weights=weights)
