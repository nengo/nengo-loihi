import nengo
from nengo.builder.connection import (
    build_no_solver, build_solver, BuiltConnection)
from nengo.dists import get_samples
from nengo.solvers import NoSolver, Solver
import numpy as np

from nengo_loihi.axons import Axons
from nengo_loihi.builder import (
    Builder, CoreGroup, INTER_N, INTER_RATE, SpikeInput)
from nengo_loihi.probes import Probe
from nengo_loihi.splitter import ChipReceiveNeurons
from nengo_loihi.synapses import Synapses

# noise exponent for inter neurons
INTER_NOISE_EXP = -2

# voltage threshold for non-spiking neurons (i.e. voltage decoders)
VTH_NONSPIKING = 10


@Builder.register(nengo.Connection)  # noqa: C901
def build_connection(model, conn):
    # Create random number generator
    rng = np.random.RandomState(model.seeds[conn])

    pre_cx = model.objs[conn.pre_obj]['out']
    post_cx = model.objs[conn.post_obj]['in']
    assert isinstance(pre_cx, (CoreGroup, SpikeInput))
    assert isinstance(post_cx, (CoreGroup, Probe))

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
    if isinstance(conn.pre_obj, nengo.Node):
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
    elif (isinstance(conn.pre_obj, nengo.Ensemble) and
          isinstance(conn.pre_obj.neuron_type, nengo.Direct)):
        raise NotImplementedError()
    elif isinstance(conn.pre_obj, nengo.Ensemble):  # Normal decoded connection
        eval_points, weights, solver_info = model.build(
            conn.solver, conn, rng, transform)

        # the decoder solver assumes a spike height of 1/dt; that isn't the
        # case on loihi, so we need to undo that scaling
        weights = weights / model.dt

        neuron_type = conn.pre_obj.neuron_type

        if not conn.solver.weights:
            needs_interneurons = True
    elif isinstance(conn.pre_obj, nengo.ensemble.Neurons):
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
    if needs_interneurons and (
            not isinstance(conn.post_obj, nengo.ensemble.Neurons)):
        # --- add interneurons
        assert weights.ndim == 2
        d, n = weights.shape

        if isinstance(post_cx, Probe):
            # use non-spiking interneurons for voltage probing
            assert post_cx.target is None
            assert conn.post_slice == slice(None)

            # use the same scaling as the ensemble does, to get good
            #  decodes.  Note that this assumes that the decoded value
            #  is in the range -radius to radius, which is usually true.
            weights = weights / conn.pre_obj.radius

            gain = 1  # model.dt * INTER_RATE(=1000)
            dec_group = CoreGroup(2 * d, label=str(conn))
            dec_group.compartments.configure_nonspiking(
                dt=model.dt, vth=VTH_NONSPIKING)
            dec_group.compartments.configure_filter(tau_s, dt=model.dt)
            dec_group.compartments.bias[...] = 0

            weights2 = gain * np.vstack([weights, -weights]).T
        else:
            # use spiking interneurons for on-chip connection
            post_d = conn.post_obj.size_in
            post_inds = np.arange(post_d, dtype=np.int32)[conn.post_slice]
            assert len(post_inds) == conn.size_out == d
            mid_axon_inds = np.hstack([post_inds, post_d+post_inds] * INTER_N)

            gain = model.dt * INTER_RATE
            dec_group = CoreGroup(2 * d * INTER_N, label=str(conn))
            dec_group.compartments.configure_relu(dt=model.dt)
            dec_group.compartments.configure_filter(tau_s, dt=model.dt)
            # TODO: is 2 * d * INTER_N necessary here?
            dec_group.compartments.bias[...] = (
                0.5 * gain * np.ones(2 * d * INTER_N))
            if INTER_NOISE_EXP > -30:
                dec_group.compartments.enableNoise[:] = 1
                dec_group.compartments.noiseExp0 = INTER_NOISE_EXP
                dec_group.compartments.noiseAtDendOrVm = 1

            if isinstance(conn.post_obj, nengo.Ensemble):
                # loihi encoders don't include radius, so handle scaling here
                weights = weights / conn.post_obj.radius

            weights2 = 0.5 * gain * np.vstack([weights, -weights] * INTER_N).T

        dec_syn = Synapses(n)
        dec_syn.set_full_weights(weights2)
        dec_group.synapses.add(dec_syn)
        model.objs[conn]['decoders'] = dec_syn

        dec_ax0 = Axons(n)
        dec_ax0.target = dec_syn  # TODO: handle target better
        pre_cx.axons.add(dec_ax0)
        model.objs[conn]['decode_axons'] = dec_ax0

        model.add_group(dec_group)
        model.objs[conn]['decoded'] = dec_group

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
                assert int(pes_pre_syn) == pes_pre_syn
                dec_syn.set_learning(tracing_tau=int(pes_pre_syn),
                                     tracing_mag=pes_learn_rate)
            else:
                raise NotImplementedError()

        mid_cx = dec_group

    if isinstance(post_cx, Probe):
        assert post_cx.target is None
        assert conn.post_slice == slice(None)
        post_cx.target = mid_cx
        mid_cx.probes.add(post_cx)
    elif isinstance(conn.post_obj, nengo.ensemble.Neurons):
        assert isinstance(post_cx, CoreGroup)
        assert conn.post_slice == slice(None)
        if weights is None:
            raise NotImplementedError("Need weights for connection to neurons")
        else:
            assert weights.ndim == 2
            n2, n1 = weights.shape
            assert post_cx.n_compartments == n2

            syn = Synapses(n1)
            gain = model.params[conn.post_obj.ensemble].gain
            syn.set_full_weights(weights.T * gain)
            post_cx.synapses.add(syn)
            model.objs[conn]['weights'] = syn

        if isinstance(pre_cx, SpikeInput):
            ax = Axons(pre_cx.n)
            pre_cx.add_axons(ax)
        elif isinstance(pre_cx, CoreGroup):
            ax = Axons(pre_cx.n_compartments)
            pre_cx.axons.add(ax)
        ax.target = syn

        post_cx.compartments.configure_filter(tau_s, dt=model.dt)
        # ^ TODO: check that all conns into post use same filter

        if conn.learning_rule_type is not None:
            raise NotImplementedError()
    elif isinstance(conn.post_obj, nengo.Ensemble) and conn.solver.weights:
        assert isinstance(post_cx, CoreGroup)
        assert weights.ndim == 2
        n2, n1 = weights.shape
        assert post_cx.n_compartments == n2

        # loihi encoders don't include radius, so handle scaling here
        weights = weights / conn.post_obj.radius

        syn = Synapses(n1)
        syn.set_full_weights(weights.T)
        post_cx.synapses.add(syn)
        model.objs[conn]['weights'] = syn

        ax = Axons(n1)
        ax.target = syn
        pre_cx.axons.add(ax)

        post_cx.compartments.configure_filter(tau_s, dt=model.dt)
        # ^ TODO: check that all conns into post use same filter

        if conn.learning_rule_type is not None:
            raise NotImplementedError()
    elif isinstance(conn.post_obj, nengo.Ensemble):
        if isinstance(mid_cx, SpikeInput):
            mid_ax = Axons(mid_cx.n)
            mid_cx.add_axons(mid_ax)
        elif isinstance(mid_cx, CoreGroup):
            mid_ax = Axons(mid_cx.n_compartments)
            mid_cx.axons.add(mid_ax)
        mid_ax.target = post_cx.synapses.named_synapses['encoders2']
        mid_ax.target_inds = mid_axon_inds
        model.objs[conn]['mid_axons'] = mid_ax
    elif isinstance(conn.post_obj, nengo.Node):
        raise NotImplementedError()
    else:
        raise NotImplementedError()

    model.params[conn] = BuiltConnection(
        eval_points=eval_points,
        solver_info=solver_info,
        transform=transform,
        weights=weights)


Builder.register(Solver)(build_solver)
Builder.register(NoSolver)(build_no_solver)
