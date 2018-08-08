from __future__ import division

import collections
import logging

from nengo.utils.compat import iteritems, itervalues, range
import numpy as np

from nengo_loihi.probes import Probe

logger = logging.getLogger(__name__)


class Emulator(object):
    """Numerical simulation of chip behaviour given a Model."""

    def __init__(self, model, seed=None):
        if seed is None:
            seed = np.random.randint(2**31 - 1)
        self.seed = seed
        logger.debug("Emulator seed: %d", seed)
        self.rng = np.random.RandomState(self.seed)

        self.group_info = GroupInfo(model.groups)
        self.inputs = list(model.spike_inputs)
        logger.debug("Emulator dtype: %s", self.group_info.dtype)

        self.compartments = CompartmentState(self.group_info)
        self.synapses = SynapseState(self.group_info)
        self.axons = AxonState(self.group_info)
        self.probes = ProbeState(
            model.objs, model.dt, self.inputs, self.group_info)

        self.t = 0

    def run_steps(self, steps):
        for _ in range(steps):
            self.step()

    def step(self):
        self.compartments.advance_input()
        self.synapses.inject_current(
            self.t, self.inputs, self.axons, self.compartments.spiked)
        self.compartments.update_input(self.synapses)
        self.synapses.update_traces()
        self.compartments.update(self.rng)
        self.probes.update(self.t, self.compartments)
        self.t += 1


class GroupInfo(object):
    def __init__(self, groups):
        self.groups = list(groups)
        self.slices = {}

        assert self.dtype in (np.float32, np.int32)

        start_ix = end_ix = 0
        for group in self.groups:
            end_ix += group.n_compartments
            self.slices[group] = slice(start_ix, end_ix)
            assert group.compartments.vth.dtype == self.dtype
            assert group.compartments.bias.dtype == self.dtype
            start_ix = end_ix

    @property
    def dtype(self):
        return self.groups[0].compartments.vth.dtype

    @property
    def n_compartments(self):
        return sum(group.n_compartments for group in self.groups)


class IterableState(object):
    def __init__(self, group_info, group_key):
        self.n_compartments = group_info.n_compartments
        self.dtype = group_info.dtype

        if group_key == "compartments":
            self.slices = {
                getattr(group, group_key): group_info.slices[group]
                for group in group_info.groups
            }
        else:
            self.slices = {
                item: group_info.slices[core_group]
                for core_group in group_info.groups
                for item in getattr(getattr(core_group, group_key), group_key)
            }

    def __contains__(self, item):
        return item in self.slices

    def __getitem__(self, key):
        return self.slices[key]

    def __iter__(self):
        for obj in self.slices:
            yield obj

    def __len__(self):
        return len(self.slices)

    def items(self):
        return iteritems(self.slices)


class CompartmentState(IterableState):
    MAX_DELAY = 1  # don't do delay yet

    def __init__(self, group_info):
        super(CompartmentState, self).__init__(group_info, "compartments")

        # Initialize NumPy arrays to store compartment-related data
        self.input = np.zeros(
            (self.MAX_DELAY, self.n_compartments), dtype=self.dtype)
        self.current = np.zeros(self.n_compartments, dtype=self.dtype)
        self.voltage = np.zeros(self.n_compartments, dtype=self.dtype)
        self.spiked = np.zeros(self.n_compartments, dtype=bool)
        self.spike_count = np.zeros(self.n_compartments, dtype=np.int32)
        self.ref_count = np.zeros(self.n_compartments, dtype=np.int32)

        self.decay_u = np.full(self.n_compartments, np.nan, dtype=np.float32)
        self.decay_v = np.full(self.n_compartments, np.nan, dtype=np.float32)
        self.scale_u = np.ones(self.n_compartments, dtype=np.float32)
        self.scale_v = np.ones(self.n_compartments, dtype=np.float32)

        self.vth = np.full(self.n_compartments, np.nan, dtype=self.dtype)
        self.vmin = np.full(self.n_compartments, np.nan, dtype=self.dtype)
        self.vmax = np.full(self.n_compartments, np.nan, dtype=self.dtype)

        self.bias = np.full(self.n_compartments, np.nan, dtype=self.dtype)
        self.ref = np.full(self.n_compartments, np.nan, dtype=self.dtype)

        # Fill in arrays with parameters from CompartmentGroups
        for compartment, sl in self.items():
            self.decay_u[sl] = compartment.decayU
            self.decay_v[sl] = compartment.decayV
            if compartment.scaleU:
                self.scale_u[sl] = compartment.decayU
            if compartment.scaleV:
                self.scale_v[sl] = compartment.decayV
            self.vth[sl] = compartment.vth
            self.vmin[sl] = compartment.vmin
            self.vmax[sl] = compartment.vmax
            self.bias[sl] = compartment.bias
            self.ref[sl] = compartment.refractDelay

        assert not np.any(np.isnan(self.decay_u))
        assert not np.any(np.isnan(self.decay_v))
        assert not np.any(np.isnan(self.vth))
        assert not np.any(np.isnan(self.vmin))
        assert not np.any(np.isnan(self.vmax))
        assert not np.any(np.isnan(self.bias))
        assert not np.any(np.isnan(self.ref))

        if self.dtype == np.float32:

            def _decay_float(x, u, d, s):
                return (1 - d)*x + s*u

            self._decay = _decay_float
        else:
            assert self.dtype == np.int32

            def _decay_int(x, u, d, s, a=12, b=0):
                r = (2**a - b - np.asarray(d)).astype(np.int64)
                # round to zero
                x = np.sign(x) * np.right_shift(np.abs(x) * r, a)
                return x + u  # no scaling on u

            self._decay = _decay_int

        self.noise = NoiseState(group_info)

    def advance_input(self):
        self.input[:-1] = self.input[1:]
        self.input[-1] = 0

    def update_input(self, all_synapses):
        for synapses, s_slice in all_synapses.items():
            activity_in = all_synapses.activity_in[synapses]
            weights = synapses.weights
            indices = synapses.indices
            qb = self.input[:, s_slice]

            for i in activity_in.nonzero()[0]:
                # faster than mult since likely 1
                for _ in range(activity_in[i]):
                    qb[0, indices[i]] += weights[i]
                # qb[inputs[indices[i]], indices[i]] += weights[i]

    def update(self, rng):
        noise = self.noise.sample(rng)
        q0 = self.input[0, :]
        q0[~self.noise.target_u] += noise[~self.noise.target_u]

        self.current[:] = self._decay(
            x=self.current[:], u=q0, d=self.decay_u, s=self.scale_u, b=1)
        u2 = self.current[:] + self.bias
        u2[self.noise.target_u] += noise[self.noise.target_u]

        self.voltage[:] = self._decay(
            x=self.voltage, u=u2, d=self.decay_v, s=self.scale_v)
        np.clip(self.voltage, self.vmin, self.vmax, out=self.voltage)
        self.voltage[self.ref_count > 0] = 0
        # TODO^: don't zero voltage in case neuron is saving overshoot

        self.spiked[:] = (self.voltage > self.vth)
        self.voltage[self.spiked] = 0
        self.ref_count[self.spiked] = self.ref[self.spiked]
        # decrement ref_count
        np.clip(self.ref_count - 1, 0, None, out=self.ref_count)

        self.spike_count[self.spiked] += 1


class NoiseState(IterableState):
    def __init__(self, group_info):
        super(NoiseState, self).__init__(group_info, "compartments")
        self.enabled = np.full(self.n_compartments, np.nan, dtype=bool)
        self.exp = np.full(self.n_compartments, np.nan, dtype=self.dtype)
        self.mant_offset = np.full(self.n_compartments, np.nan, dtype=self.dtype)
        self.target_u = np.full(self.n_compartments, np.nan, dtype=bool)

        # Fill in arrays with parameters from CompartmentGroups
        for compartment, sl in self.items():
            self.enabled[sl] = compartment.enableNoise
            self.exp[sl] = compartment.noiseExp0
            self.mant_offset[sl] = compartment.noiseMantOffset0
            self.target_u[sl] = compartment.noiseAtDendOrVm

        if self.dtype == np.float32:
            self.mult = np.where(self.enabled, 10.**self.exp, 0)
            self.r_scale = 1
            self.mant_scale = 1
        else:
            assert self.dtype == np.int32
            self.mult = np.where(self.enabled, 2**(self.exp - 7), 0)
            self.r_scale = 128
            self.mant_scale = 64

        assert not np.any(np.isnan(self.enabled))
        assert not np.any(np.isnan(self.exp))
        assert not np.any(np.isnan(self.mant_offset))
        assert not np.any(np.isnan(self.target_u))
        assert not np.any(np.isnan(self.mult))

    def sample(self, rng):
        x = rng.uniform(-self.r_scale, self.r_scale, size=self.n_compartments)
        x = x.astype(self.dtype)
        return (x + self.mant_scale * self.mant_offset) * self.mult


class SynapseState(IterableState):
    def __init__(self, group_info):
        super(SynapseState, self).__init__(group_info, "synapses")

        self.activity_in = {}
        self.traces = {}
        for synapses in self.slices:
            n = synapses.n_axons
            self.activity_in[synapses] = np.zeros(n, dtype=np.int32)
            if synapses.learning:
                self.traces[synapses] = np.zeros(n, dtype=np.float64)

    def inject_current(self, t, spike_inputs, all_axons, spiked):
        for activity_in in itervalues(self.activity_in):
            activity_in[...] = 0

        for spike_input in spike_inputs:
            for axons in spike_input.axons:
                synapses = axons.target
                assert axons.target_inds == slice(None)
                self.activity_in[synapses] += spike_input.spikes[t]

        for axons, a_idx in all_axons.items():
            synapses = axons.target
            # Use add.at to allow repeated indices
            np.add.at(
                self.activity_in[synapses], axons.target_inds, spiked[a_idx])

    def update_weights(self, synapses, x):
        assert synapses.learning
        traces = self.traces[synapses]
        delta_w = np.outer(traces, x).astype('int32')
        for i, w in enumerate(synapses.weights):
            w += delta_w[i]

    def update_traces(self):
        for synapses in self.traces:
            activity_in = self.activity_in[synapses]
            traces = self.traces[synapses]
            tau = synapses.tracing_tau
            mag = synapses.tracing_mag
            decay = np.exp(-1.0 / tau)
            traces *= decay
            traces += mag * activity_in


class AxonState(IterableState):
    def __init__(self, group_info):
        super(AxonState, self).__init__(group_info, "axons")


class ProbeState(object):
    def __init__(self, objs, dt, inputs, group_info):
        self.objs = objs
        self.dt = dt
        self.input_probes = {}
        for spike_input in inputs:
            for probe in spike_input.probes:
                assert probe.key == 'spiked'
                self.input_probes[probe] = spike_input
        self.other_probes = {}
        for group in group_info.groups:
            for probe in group.probes.probes:
                self.other_probes[probe] = group_info.slices[group]

        self.filters = {}
        self.filter_pos = {}
        for probe, spike_input in iteritems(self.input_probes):
            if probe.synapse is not None:
                self.filters[probe] = probe.synapse.make_step(
                    shape=spike_input.spikes[0][probe.slice].shape[0],
                    dt=self.dt,
                    rng=None,
                    dtype=spike_input.spikes.dtype,
                )
                self.filter_pos[probe] = 0

        for probe, sl in iteritems(self.other_probes):
            if probe.synapse is not None:
                size = (sl.stop - sl.start if probe.weights is None
                        else probe.weights.shape[1])
                self.filters[probe] = probe.synapse.make_step(
                    shape_in=(size,),
                    shape_out=(size,),
                    dt=self.dt,
                    rng=None,
                    dtype=np.float32,
                )
                self.filter_pos[probe] = 0

        self.outputs = collections.defaultdict(list)

    def __getitem__(self, nengo_probe):
        probe = self.objs[nengo_probe]['out']
        assert isinstance(probe, Probe)
        out = np.asarray(self.outputs[probe], dtype=np.float32)
        out = out if probe.weights is None else np.dot(out, probe.weights)
        if probe in self.filters:
            return self._filter(probe, out)
        else:
            return out

    def _filter(self, probe, data):
        dt = self.dt
        i = self.filter_pos[probe]
        step = self.filters[probe]
        filt_data = np.zeros_like(data)
        for k, x in enumerate(data):
            filt_data[k] = step((i + k) * dt, x)
        self.filter_pos[probe] = i + k
        return filt_data

    def send(self, nengo_probe, already_sent, receiver):
        """Send probed data to the receiver node.

        Returns
        -------
        steps : int
            The number of steps sent to the receiver.
        """
        probe = self.objs[nengo_probe]['out']
        x = self.outputs[probe][already_sent:]

        if len(x) > 0:
            if probe.weights is not None:
                x = np.dot(x, probe.weights)
            for j, xx in enumerate(x):
                receiver.receive(self.dt * (already_sent + j + 2), xx)
        return len(x)

    def update(self, t, compartments):
        for probe, spike_input in iteritems(self.input_probes):
            output = spike_input.spikes[t][probe.slice].copy()
            self.outputs[probe].append(output)

        for probe, out_idx in iteritems(self.other_probes):
            p_slice = probe.slice
            assert hasattr(compartments, probe.key)
            output = getattr(compartments, probe.key)[out_idx][p_slice].copy()
            self.outputs[probe].append(output)
