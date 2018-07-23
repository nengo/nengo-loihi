from __future__ import division

import collections
import logging

import numpy as np

from nengo_loihi.model import CxProbe

logger = logging.getLogger(__name__)


class Emulator(object):
    """Numerical simulation of chip behaviour given a CxModel"""

    def __init__(self, model, seed=None):
        self.build(model, seed=seed)

        self._probe_filters = {}
        self._probe_filter_pos = {}

    def build(self, model, seed=None):  # noqa: C901
        if seed is None:
            seed = np.random.randint(2**31 - 1)

        logger.debug("CxSimulator seed: %d", seed)
        self.seed = seed
        self.rng = np.random.RandomState(seed)

        self.t = 0

        self.model = model
        self.inputs = list(self.model.cx_inputs)
        self.groups = list(self.model.groups)
        self.probe_outputs = collections.defaultdict(list)

        self.n_cx = sum(group.n_compartments for group in self.groups)
        self.group_cxs = {}
        i0 = 0
        for group in self.groups:
            i1 = i0 + group.n_compartments
            self.group_cxs[group] = slice(i0, i1)
            i0 = i1

        self.cx_slice = slice(0, i0)

        # --- allocate group memory
        group_dtype = self.groups[0].compartments.vth.dtype
        assert group_dtype in (np.float32, np.int32)
        for group in self.groups:
            assert group.compartments.vth.dtype == group_dtype
            assert group.compartments.bias.dtype == group_dtype

        logger.debug("CxSimulator dtype: %s", group_dtype)

        MAX_DELAY = 1  # don't do delay yet
        self.q = np.zeros((MAX_DELAY, self.n_cx), dtype=group_dtype)
        self.u = np.zeros(self.n_cx, dtype=group_dtype)
        self.v = np.zeros(self.n_cx, dtype=group_dtype)
        self.s = np.zeros(self.n_cx, dtype=bool)  # spiked
        self.c = np.zeros(self.n_cx, dtype=np.int32)  # spike counter
        self.w = np.zeros(self.n_cx, dtype=np.int32)  # ref period counter

        # --- allocate group parameters
        self.decayU = np.hstack([
            group.compartments.decayU for group in self.groups])
        self.decayV = np.hstack([
            group.compartments.decayV for group in self.groups])
        self.scaleU = np.hstack([
            group.compartments.decayU if group.compartments.scaleU
            else np.ones_like(group.compartments.decayU)
            for group in self.groups])
        self.scaleV = np.hstack([
            group.compartments.decayV if group.compartments.scaleV
            else np.ones_like(group.compartments.decayV)
            for group in self.groups])

        def decay_float(x, u, d, s):
            return (1 - d)*x + s*u

        def decay_int(x, u, d, s, a=12, b=0):
            r = (2**a - b - np.asarray(d)).astype(np.int64)
            x = np.sign(x) * np.right_shift(np.abs(x) * r, a)  # round to zero
            return x + u  # no scaling on u

        if group_dtype == np.int32:
            assert (self.scaleU == 1).all()
            assert (self.scaleV == 1).all()
            self.decayU_fn = lambda x, u: decay_int(
                x, u, d=self.decayU, s=self.scaleU, b=1)
            self.decayV_fn = lambda x, u: decay_int(
                x, u, d=self.decayV, s=self.scaleV)
        elif group_dtype == np.float32:
            self.decayU_fn = lambda x, u: decay_float(
                x, u, d=self.decayU, s=self.scaleU)
            self.decayV_fn = lambda x, u: decay_float(
                x, u, d=self.decayV, s=self.scaleV)

        ones = lambda n: np.ones(n, dtype=group_dtype)
        self.vth = np.hstack([group.compartments.vth for group in self.groups])
        self.vmin = np.hstack([
            group.compartments.vmin*ones(group.n_compartments)
            for group in self.groups])
        self.vmax = np.hstack([
            group.compartments.vmax*ones(group.n_compartments)
            for group in self.groups])

        self.bias = np.hstack([group.compartments.bias
                               for group in self.groups])
        self.ref = np.hstack([group.compartments.refractDelay
                              for group in self.groups])

        # --- allocate synapse memory
        self.a_in = {synapses: np.zeros(synapses.n_axons, dtype=np.int32)
                     for group in self.groups
                     for synapses in group.synapses.synapses}
        self.z = {synapses: np.zeros(synapses.n_axons, dtype=np.float64)
                  for group in self.groups
                  for synapses in group.synapses.synapses
                  if synapses.learning}

        # --- noise
        enableNoise = np.hstack([
            group.compartments.enableNoise*ones(group.n_compartments)
            for group in self.groups])
        noiseExp0 = np.hstack([
            group.compartments.noiseExp0*ones(group.n_compartments)
            for group in self.groups])
        noiseMantOffset0 = np.hstack([
            group.compartments.noiseMantOffset0*ones(group.n_compartments)
            for group in self.groups])
        noiseTarget = np.hstack([
            group.compartments.noiseAtDendOrVm*ones(group.n_compartments)
            for group in self.groups])
        if group_dtype == np.int32:
            noiseMult = np.where(enableNoise, 2**(noiseExp0 - 7), 0)

            def noiseGen(n=self.n_cx, rng=self.rng):
                x = rng.randint(-128, 128, size=n)
                return (x + 64*noiseMantOffset0) * noiseMult
        elif group_dtype == np.float32:
            noiseMult = np.where(enableNoise, 10.**noiseExp0, 0)

            def noiseGen(n=self.n_cx, rng=self.rng):
                x = rng.uniform(-1, 1, size=n)
                return (x + noiseMantOffset0) * noiseMult

        self.noiseGen = noiseGen
        self.noiseTarget = noiseTarget

    def step(self):  # noqa: C901
        # --- connections
        self.q[:-1] = self.q[1:]  # advance delays
        self.q[-1] = 0

        for a_in in self.a_in.values():
            a_in[:] = 0

        for input in self.inputs:
            for axons in input.axons:
                synapses = axons.target
                assert axons.target_inds == slice(None)
                self.a_in[synapses] += input.spikes[self.t]

        for group in self.groups:
            for axons in group.axons.axons:
                synapses = axons.target
                s_in = self.a_in[synapses]

                a_slice = self.group_cxs[group]
                sa = self.s[a_slice]
                np.add.at(s_in, axons.target_inds, sa)  # allows repeat inds

        for group in self.groups:
            for synapses in group.synapses.synapses:
                s_in = self.a_in[synapses]

                b_slice = self.group_cxs[group]
                weights = synapses.weights
                indices = synapses.indices
                qb = self.q[:, b_slice]
                # delays = np.zeros(qb.shape[1], dtype=np.int32)

                for i in s_in.nonzero()[0]:
                    for _ in range(s_in[i]):  # faster than mult since likely 1
                        qb[0, indices[i]] += weights[i]
                    # qb[delays[indices[i]], indices[i]] += weights[i]

                if synapses.learning:
                    z = self.z[synapses]
                    tau = synapses.tracing_tau
                    mag = synapses.tracing_mag

                    decay = np.exp(-1.0 / tau)
                    z *= decay

                    z += mag * s_in

        # --- updates
        q0 = self.q[0, :]

        noise = self.noiseGen()
        q0[self.noiseTarget == 0] += noise[self.noiseTarget == 0]

        # self.U[:] = self.decayU_fn(self.U, self.decayU, a=12, b=1)
        self.u[:] = self.decayU_fn(self.u[:], q0)
        u2 = self.u[:] + self.bias
        u2[self.noiseTarget == 1] += noise[self.noiseTarget == 1]

        # self.V[:] = self.decayV_fn(v, self.decayV, a=12) + u2
        self.v[:] = self.decayV_fn(self.v, u2)
        np.clip(self.v, self.vmin, self.vmax, out=self.v)
        self.v[self.w > 0] = 0
        # TODO^: don't zero voltage in case neuron is saving overshoot

        self.s[:] = (self.v > self.vth)

        cx = self.cx_slice
        self.v[cx][self.s[cx]] = 0

        self.w[self.s] = self.ref[self.s]
        np.clip(self.w - 1, 0, None, out=self.w)  # decrement w

        self.c[self.s] += 1

        # --- probes
        for input in self.inputs:
            for probe in input.probes:
                assert probe.key == 's'
                p_slice = probe.slice
                x = input.spikes[self.t][p_slice].copy()
                self.probe_outputs[probe].append(x)

        for group in self.groups:
            for probe in group.probes.probes:
                x_slice = self.group_cxs[group]
                p_slice = probe.slice
                assert hasattr(self, probe.key)
                x = getattr(self, probe.key)[x_slice][p_slice].copy()
                self.probe_outputs[probe].append(x)

        self.t += 1

    def run_steps(self, steps):
        for _ in range(steps):
            self.step()

    def _filter_probe(self, cx_probe, data):
        dt = self.model.dt
        i = self._probe_filter_pos.get(cx_probe, 0)
        if i == 0:
            shape = data[0].shape
            synapse = cx_probe.synapse
            rng = None
            step = (synapse.make_step(shape, shape, dt, rng, dtype=data.dtype)
                    if synapse is not None else None)
            self._probe_filters[cx_probe] = step
        else:
            step = self._probe_filters[cx_probe]

        if step is None:
            self._probe_filter_pos[cx_probe] = i + len(data)
            return data
        else:
            filt_data = np.zeros_like(data)
            for k, x in enumerate(data):
                filt_data[k] = step((i + k) * dt, x)

            self._probe_filter_pos[cx_probe] = i + k
            return filt_data

    def get_probe_output(self, probe):
        cx_probe = self.model.objs[probe]['out']
        assert isinstance(cx_probe, CxProbe)
        x = np.asarray(self.probe_outputs[cx_probe], dtype=np.float32)
        x = x if cx_probe.weights is None else np.dot(x, cx_probe.weights)
        return self._filter_probe(cx_probe, x)
