import logging
import warnings
from collections import OrderedDict

import numpy as np
from nengo.exceptions import SimulationError, ValidationError

from nengo_loihi.builder.discretize import (
    LEARN_FRAC,
    Q_BITS,
    U_BITS,
    decay_int,
    learn_overflow_bits,
    overflow_signed,
    scale_pes_errors,
    shift,
)
from nengo_loihi.compat import make_process_step
from nengo_loihi.probe import LoihiProbe

logger = logging.getLogger(__name__)


class EmulatorInterface:
    """Software emulator for Loihi chip behaviour.

    Parameters
    ----------
    model : Model
        Model specification that will be simulated.
    seed : int, optional (Default: None)
        A seed for all stochastic operations done in this simulator.
    """

    strict = False

    def __init__(self, model, seed=None):
        self.closed = True

        if seed is None:
            seed = np.random.randint(2 ** 31 - 1)
        self.seed = seed
        logger.debug("EmulatorInterface seed: %d", seed)
        self.rng = np.random.RandomState(self.seed)

        self.block_info = BlockInfo(model.blocks)
        self.inputs = list(model.inputs)
        logger.debug("EmulatorInterface dtype: %s", self.block_info.dtype)

        self.compartment = CompartmentState(self.block_info, strict=self.strict)
        self.synapses = SynapseState(
            self.block_info,
            pes_error_scale=getattr(model, "pes_error_scale", 1.0),
            strict=self.strict,
        )
        self.axons = AxonState(self.block_info)
        self.probes = ProbeState(self.block_info, list(model.probes), model.dt)

        self.t = 0
        self._chip2host_sent_steps = 0
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def close(self):
        self.closed = True

        # remove references to states to free memory (except probes)
        self.block_info = None
        self.inputs = None
        self.compartment = None
        self.synapses = None
        self.axons = None

    def chip2host(self, probes_receivers):
        increment = 0
        for probe, receiver in probes_receivers.items():
            inc = self.probes.send(probe, self._chip2host_sent_steps, receiver)
            increment = inc if increment == 0 else increment
            assert inc == 0 or increment == inc

        self._chip2host_sent_steps += increment

    def host2chip(self, spikes, errors):
        for spike_input, t, spike_idxs in spikes:
            spike_input.add_spikes(t, spike_idxs)

        self.synapses.update_pes_errors(errors)

    def run_steps(self, steps):
        """Simulate for the given number of ``dt`` steps.

        Parameters
        ----------
        steps : int
            Number of steps to run the simulation for.
        """
        for _ in range(steps):
            self.step()

    def step(self):
        """Advance the simulation by 1 step (``dt`` seconds)."""
        self.t += 1
        self.compartment.advance_input()
        self.synapses.inject_current(
            self.t, self.inputs, self.axons, self.compartment.spiked
        )
        self.synapses.update_input(self.compartment.input)
        self.synapses.update_traces(self.t, self.rng)
        self.synapses.update_weights(self.t, self.rng)
        self.compartment.update(self.rng)
        self.probes.update(self.t, self.compartment)

    def get_probe_output(self, probe):
        return self.probes[probe]


class BlockInfo:
    """Provide information about all the LoihiBlocks in the model.

    Attributes
    ----------
    dtype : dtype
        Datatype of the blocks. Either ``np.float32`` if the blocks are not
        discretized or ``np.int32`` if they are. All blocks are the same.
    blocks : list of LoihiBlock
        List of all the blocks in the model.
    n_compartments : int
        Total number of compartments across all blocks.
    slices : dict of {LoihiBlock: slice}
        Maps each block to a slice for that block's compartments with
        respect to all compartments. Used to slice into any array storing
        data across all compartments.
    """

    def __init__(self, blocks):
        self.blocks = list(blocks)
        self.slices = OrderedDict()

        assert self.dtype in (np.float32, np.int32)

        start_ix = end_ix = 0
        for block in self.blocks:
            end_ix += block.n_neurons
            self.slices[block] = slice(start_ix, end_ix)
            assert block.compartment.vth.dtype == self.dtype
            assert block.compartment.bias.dtype == self.dtype
            start_ix = end_ix

        self.n_compartments = end_ix

    @property
    def dtype(self):
        return self.blocks[0].compartment.vth.dtype


class IterableState:
    """Base class for aspects of the emulator state.

    This class takes the name of a LoihiBlock attribute as the
    ``block_key`` and maps these objects to their parent blocks and slices.

    Attributes
    ----------
    dtype : dtype
        Datatype of the state elements (given by the BlockInfo datatype).
    block_map : dict of {item: block}
        Maps an item (determined by ``block_key``) to the block
        it belongs to.
    n_compartments : int
        The total number of neuron compartments (given by BlockInfo).
    slices : dict of {item: slice}
        Maps an item to the ``block_info.slice`` for the block
        it belongs to.
    strict : bool (Default: True)
        Whether "undesired" chip effects (ex. overflow) raise errors (``True``)
        or whether they only raise warnings (``False``).
    """

    def __init__(self, block_info, block_key, strict=True):
        self.n_compartments = block_info.n_compartments
        self.dtype = block_info.dtype
        self.strict = strict

        blocks_items = list(self._blocks_items(block_info.blocks, block_key))
        self.block_map = OrderedDict((item, block) for block, item in blocks_items)
        self.slices = OrderedDict(
            (item, block_info.slices[block]) for block, item in blocks_items
        )

    @staticmethod
    def _blocks_items(blocks, block_key):
        for block in blocks:
            if block_key == "compartment":
                # one item per block
                yield block, getattr(block, block_key)
            else:
                # multiple items per block (attribute is iterable)
                for item in getattr(block, block_key):
                    yield block, item

    def error(self, msg):
        if self.strict:
            raise SimulationError(msg)
        else:
            warnings.warn(msg)

    def items(self):
        return self.slices.items()


class CompartmentState(IterableState):
    """State representing the Compartments of all blocks."""

    MAX_DELAY = 1  # delay not yet implemented

    def __init__(self, block_info, strict=True):
        super().__init__(block_info, "compartment", strict=strict)

        # Initialize NumPy arrays to store compartment-related data
        self.input = np.zeros((self.MAX_DELAY, self.n_compartments), dtype=self.dtype)
        self.current = np.zeros(self.n_compartments, dtype=self.dtype)
        self.voltage = np.zeros(self.n_compartments, dtype=self.dtype)
        self.spiked = np.zeros(self.n_compartments, dtype=bool)
        self.spike_count = np.zeros(self.n_compartments, dtype=np.int32)
        self.ref_count = np.zeros(self.n_compartments, dtype=np.int32)

        self.decay_u = np.full(self.n_compartments, np.nan, dtype=self.dtype)
        self.decay_v = np.full(self.n_compartments, np.nan, dtype=self.dtype)
        self.scale_u = np.ones(self.n_compartments, dtype=self.dtype)
        self.scale_v = np.ones(self.n_compartments, dtype=self.dtype)

        self.vth = np.full(self.n_compartments, np.nan, dtype=self.dtype)
        self.vmin = np.full(self.n_compartments, np.nan, dtype=self.dtype)
        self.vmax = np.full(self.n_compartments, np.nan, dtype=self.dtype)

        self.bias = np.full(self.n_compartments, np.nan, dtype=self.dtype)
        self.ref = np.full(self.n_compartments, np.nan, dtype=self.dtype)

        # Fill in arrays with parameters from CompartmentSegments
        for compartment, sl in self.items():
            self.decay_u[sl] = compartment.decay_u
            self.decay_v[sl] = compartment.decay_v
            if compartment.scale_u:
                self.scale_u[sl] = compartment.decay_u
            if compartment.scale_v:
                self.scale_v[sl] = compartment.decay_v
            self.vth[sl] = compartment.vth
            self.vmin[sl] = compartment.vmin
            self.vmax[sl] = compartment.vmax
            self.bias[sl] = compartment.bias
            self.ref[sl] = compartment.refract_delay

        assert not np.any(np.isnan(self.decay_u))
        assert not np.any(np.isnan(self.decay_v))
        assert not np.any(np.isnan(self.vth))
        assert not np.any(np.isnan(self.vmin))
        assert not np.any(np.isnan(self.vmax))
        assert not np.any(np.isnan(self.bias))
        assert not np.any(np.isnan(self.ref))

        if self.dtype == np.int32:
            assert (self.scale_u == 1).all()
            assert (self.scale_v == 1).all()
            self._decay_current = lambda x, u: decay_int(x, self.decay_u, offset=1) + u
            self._decay_voltage = lambda x, u: decay_int(x, self.decay_v) + u

            def overflow(x, bits, name=None):
                _, o = overflow_signed(x, bits=bits, out=x)
                if np.any(o):
                    self.error("Overflow" + (" in %s" % name if name else ""))

        elif self.dtype == np.float32:

            def decay_float(x, u, d, s):
                return (1 - d) * x + s * u

            self._decay_current = lambda x, u: decay_float(
                x, u, d=self.decay_u, s=self.scale_u
            )
            self._decay_voltage = lambda x, u: decay_float(
                x, u, d=self.decay_v, s=self.scale_v
            )

            def overflow(x, bits, name=None):
                pass  # do not do overflow in floating point

        else:
            raise ValidationError(
                "dtype %r not supported" % self.dtype, attr="dtype", obj=block_info
            )

        self._overflow = overflow

        self.noise = NoiseState(block_info)

    def advance_input(self):
        self.input[:-1] = self.input[1:]
        self.input[-1] = 0

    def update(self, rng):
        noise = self.noise.sample(rng)
        q0 = self.input[0, :]
        q0[~(self.noise.target_u)] += noise[~(self.noise.target_u)]
        self._overflow(q0, bits=Q_BITS, name="q0")

        self.current[:] = self._decay_current(self.current, q0)
        self._overflow(self.current, bits=U_BITS, name="current")
        u2 = self.current + self.bias
        u2[self.noise.target_u] += noise[self.noise.target_u]
        self._overflow(u2, bits=U_BITS, name="u2")

        self.voltage[:] = self._decay_voltage(self.voltage, u2)
        # We have not been able to create V overflow on the chip, so we do
        # not include it here. See github.com/nengo/nengo-loihi/issues/130
        # self.overflow(self.v, bits=V_BIT, name="V")

        np.clip(self.voltage, self.vmin, self.vmax, out=self.voltage)
        self.voltage[self.ref_count > 0] = 0
        # TODO^: don't zero voltage in case neuron is saving overshoot

        self.spiked[:] = self.voltage > self.vth
        self.voltage[self.spiked] = 0
        self.ref_count[self.spiked] = self.ref[self.spiked]
        # decrement ref_count
        np.clip(self.ref_count - 1, 0, None, out=self.ref_count)

        self.spike_count[self.spiked] += 1


class NoiseState(IterableState):
    """State representing the noise parameters for all compartments."""

    def __init__(self, block_info):
        super().__init__(block_info, "compartment")
        self.enabled = np.full(self.n_compartments, np.nan, dtype=bool)
        self.exp = np.full(self.n_compartments, np.nan, dtype=self.dtype)
        self.mant_offset = np.full(self.n_compartments, np.nan, dtype=self.dtype)
        self.target_u = np.full(self.n_compartments, np.nan, dtype=bool)

        # Fill in arrays with parameters from Compartment
        for compartment, sl in self.items():
            self.enabled[sl] = compartment.enable_noise
            self.exp[sl] = compartment.noise_exp
            self.mant_offset[sl] = compartment.noise_offset
            self.target_u[sl] = compartment.noise_at_membrane

        if self.dtype == np.int32:
            # TODO: if we could do this mult with shifts, it'd be faster, but
            # numpy has no function taking a vector of positive/negative shifts
            self.mult = np.where(self.enabled, 2.0 ** (self.exp - 7), 0)
            self.mant_offset *= 64

            def uniform(rng, n=self.n_compartments):
                return rng.randint(-127, 128, size=n, dtype=np.int32)

        elif self.dtype == np.float32:
            self.mult = np.where(self.enabled, 10.0 ** self.exp, 0)

            def uniform(rng, n=self.n_compartments):
                return rng.uniform(-1, 1, size=n).astype(np.float32)

        else:
            raise ValidationError(
                "dtype %r not supported" % self.dtype, attr="dtype", obj=block_info
            )

        assert not np.any(np.isnan(self.enabled))
        assert not np.any(np.isnan(self.exp))
        assert not np.any(np.isnan(self.mant_offset))
        assert not np.any(np.isnan(self.target_u))
        assert not np.any(np.isnan(self.mult))

        self._uniform = uniform

    def sample(self, rng):
        x = self._uniform(rng)
        return ((x + self.mant_offset) * self.mult).astype(self.dtype)


class SynapseState(IterableState):
    """State representing all synapses.

    Attributes
    ----------
    pes_error_scale : float
        Scaling for the errors of PES learning rules.
    pes_errors : {Synapse: ndarray(n_neurons / 2)}
        Maps synapse to PES learning rule errors for those synapses.
    spikes_in : {Synapse: list}
        Maps synapse to a queue of input spikes targeting those synapses.
    traces : {Synapse: ndarray(Synapse.n_axons)}
        Maps synapse to trace values for each of their axons.
    trace_spikes : {Synapse: set}
        Maps synapse to a queue of input spikes waiting to be added to those
        synapse traces.
    """

    def __init__(self, block_info, pes_error_scale=1.0, strict=True):  # noqa: C901
        super().__init__(block_info, "synapses", strict=strict)

        self.pes_error_scale = pes_error_scale

        self.spikes_in = OrderedDict()
        self.traces = OrderedDict()
        self.trace_spikes = OrderedDict()
        self.pes_errors = OrderedDict()
        for synapse in self.slices:
            n = synapse.n_axons
            self.spikes_in[synapse] = []

            if synapse.learning:
                self.traces[synapse] = np.zeros(n, dtype=self.dtype)
                self.trace_spikes[synapse] = set()
                self.pes_errors[synapse] = np.zeros(
                    self.block_map[synapse].n_neurons // 2, dtype=self.dtype
                )
                # ^ Currently, PES learning only happens on Nodes, where we
                # have pairs of on/off neurons. Therefore, the number of error
                # dimensions is half the number of neurons.

        if self.dtype == np.int32:

            def stochastic_round(
                x, dtype=self.dtype, rng=None, clip=None, name="values"
            ):
                x_sign = np.sign(x).astype(dtype)
                x_frac, x_int = np.modf(np.abs(x))
                p = rng.rand(*x.shape)
                y = x_int.astype(dtype) + (x_frac > p)
                if clip is not None:
                    q = y > clip
                    if np.any(q):
                        warnings.warn("Clipping %s" % name)
                    y[q] = clip
                return x_sign * y

            def trace_round(x, rng=None):
                return stochastic_round(x, rng=rng, clip=127, name="synapse trace")

            def weight_update(synapse, delta_ws, rng=None):
                synapse_cfg = synapse.synapse_cfg
                wgt_exp = synapse_cfg.real_weight_exp
                shift_bits = synapse_cfg.shift_bits
                overflow = learn_overflow_bits(n_factors=2)
                for w, delta_w in zip(synapse.weights, delta_ws):
                    product = shift(
                        delta_w * synapse._lr_int,
                        LEARN_FRAC + synapse._lr_exp - overflow,
                    )
                    learn_w = shift(w, LEARN_FRAC - wgt_exp) + product
                    learn_w[:] = stochastic_round(
                        learn_w * 2 ** (-LEARN_FRAC - shift_bits),
                        clip=2 ** (8 - shift_bits) - 1,
                        rng=rng,
                        name="learning weights",
                    )
                    w[:] = np.left_shift(learn_w, wgt_exp + shift_bits)

        elif self.dtype == np.float32:

            def trace_round(x, rng=None):
                return x  # no rounding

            def weight_update(synapse, delta_ws, rng=None):
                for w, delta_w in zip(synapse.weights, delta_ws):
                    w += synapse.learning_rate * delta_w

        else:
            raise ValidationError(
                "dtype %r not supported" % self.dtype, attr="dtype", obj=block_info
            )

        self._trace_round = trace_round
        self._weight_update = weight_update

    def inject_current(self, t, spike_inputs, all_axons, spiked):
        # --- clear spikes going in to each synapse
        for spike_queue in self.spikes_in.values():
            spike_queue.clear()

        # --- inputs pass spikes to synapses
        if t >= 2:  # input spikes take one time-step to arrive
            for spike_input in spike_inputs:
                compartment_idxs = spike_input.spike_idxs(t - 1)
                for axon in spike_input.axons:
                    spikes = axon.map_spikes(compartment_idxs)
                    self.spikes_in[axon.target].extend(
                        s for s in spikes if s is not None
                    )

        # --- axons pass spikes to synapses
        for axon, a_idx in all_axons.items():
            compartment_idxs = spiked[a_idx].nonzero()[0]
            spikes = axon.map_spikes(compartment_idxs)
            self.spikes_in[axon.target].extend(s for s in spikes if s is not None)

    def update_input(self, input):
        for synapse, s_slice in self.items():
            qb = input[:, s_slice]

            for spike in self.spikes_in[synapse]:
                base = synapse.axon_compartment_base(spike.axon_idx)
                if base is None:
                    continue

                weights, indices = synapse.axon_weights_indices(
                    spike.axon_idx, atom=spike.atom
                )
                qb[0, base + indices] += weights

    def update_pes_errors(self, errors):
        # TODO: these are sent every timestep, but learning only happens every
        # `tepoch * 2**learn_k` timesteps (see Synapse). Need to average.
        for pes_errors in self.pes_errors.values():
            pes_errors[:] = 0

        for synapse, _, e in errors:
            pes_errors = self.pes_errors[synapse]
            assert pes_errors.shape == e.shape
            pes_errors += scale_pes_errors(e, scale=self.pes_error_scale)

    def update_weights(self, t, rng):
        for synapse, pes_error in self.pes_errors.items():
            if t % synapse.learn_epoch == 0:
                trace = self.traces[synapse]
                e = np.hstack([-pes_error, pes_error])
                delta_w = np.outer(trace, e)
                self._weight_update(synapse, delta_w, rng=rng)

    def update_traces(self, t, rng):
        for synapse in self.traces:
            trace_spikes = self.trace_spikes.get(synapse, None)
            if trace_spikes is not None:
                for spike in self.spikes_in[synapse]:
                    if spike.axon_idx in trace_spikes:
                        self.error("Synaptic trace spikes lost")
                    trace_spikes.add(spike.axon_idx)

            trace = self.traces.get(synapse, None)
            if trace is not None and t % synapse.train_epoch == 0:
                tau = synapse.tracing_tau
                decay = np.exp(-synapse.train_epoch / tau)
                trace1 = decay * trace
                trace1[list(trace_spikes)] += synapse.tracing_mag
                trace[:] = self._trace_round(trace1, rng=rng)
                trace_spikes.clear()


class AxonState(IterableState):
    """State representing all (output) Axons."""

    def __init__(self, block_info):
        super().__init__(block_info, "axons")


class ProbeState:
    """State representing all probes.

    Attributes
    ----------
    dt : float
        Time constant of the Emulator.
    filters : {nengo_loihi.Probe: function}
        Maps Probes to the filtering function for that probe.
    filter_pos : {nengo_loihi.Probe: int}
        Maps Probes to the position of their filter in the data.
    block_probes : {nengo_loihi.Probe: slice}
        Maps Probes to the BlockInfo slice for the block they are probing.
    input_probes : {nengo_loihi.Probe: SpikeInput}
        Maps Probes to the SpikeInput that they are probing.
    """

    def __init__(self, block_info, probes, dt):
        self.dt = dt

        self.probes = OrderedDict()
        for probe in probes:
            block_slices = OrderedDict(
                [(block, block_info.slices[block]) for block in probe.target]
            )
            self.probes[probe] = block_slices

        self.filters = {}
        self.filter_pos = {}
        for probe, block_slices in self.probes.items():
            if probe.synapse is not None:
                size = probe.output_size
                self.filters[probe] = make_process_step(
                    probe.synapse,
                    shape_in=(size,),
                    shape_out=(size,),
                    dt=self.dt,
                    rng=None,
                    dtype=np.float32,
                )
                self.filter_pos[probe] = 0

        self.outputs = {}
        for probe, block_slices in self.probes.items():
            self.outputs[probe] = [[] for block in block_slices]

    def __getitem__(self, probe):
        assert isinstance(probe, LoihiProbe)
        out = probe.weight_outputs(self.outputs[probe])
        # TODO: if this is called multiple times, it will change the filter state
        return self._filter(probe, out) if probe in self.filters else out

    def _filter(self, probe, data):
        dt = self.dt
        i = self.filter_pos[probe]
        step = self.filters[probe]
        filt_data = np.zeros_like(data)
        for k, x in enumerate(data):
            filt_data[k] = step((i + k) * dt, x)
        self.filter_pos[probe] = i + k
        return filt_data

    def send(self, probe, already_sent, receiver):
        """Send probed data to the receiver node.

        Returns
        -------
        steps : int
            The number of steps sent to the receiver.
        """
        # we don't currently filter here, so make sure the probe isn't expecting it
        assert probe.synapse is None, "Filtering should be done on host-side connection"

        outputs = [block_output[already_sent:] for block_output in self.outputs[probe]]

        n_timesteps = len(outputs[0])
        if n_timesteps > 0:
            x = probe.weight_outputs(outputs)
            for j, xx in enumerate(x):
                receiver.receive(self.dt * (already_sent + j + 2), xx)

        return n_timesteps

    def update(self, t, compartment):
        for probe, block_slices in self.probes.items():
            assert hasattr(compartment, probe.key)
            target_attr = getattr(compartment, probe.key)

            for k, (block, block_slice) in enumerate(block_slices.items()):
                output = target_attr[block_slice][probe.slice[k]]
                if output.base is not None:
                    # if not already copied by the probe slice, then copy
                    output = output.copy()

                self.outputs[probe][k].append(output)
