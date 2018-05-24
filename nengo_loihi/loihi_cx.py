from __future__ import division

import collections

import numpy as np

from nengo_loihi.loihi_api import (
    VTH_MAX, vth_to_manexp, BIAS_MAX, bias_to_manexp, SynapseFmt)


class CxGroup(object):
    def __init__(self, n, label=None, location='core'):
        self.n = n
        self.label = label

        # self.cxProfiles = []
        # self.vthProfiles = []

        # self.outputAxonMap = None
        # self.outputAxons = None

        self.decayU = np.zeros(n, dtype=np.float32)
        self.decayV = np.zeros(n, dtype=np.float32)
        self.refractDelay = np.zeros(n, dtype=np.int32)
        self.vth = np.zeros(n, dtype=np.float32)
        self.vmin = 0
        self.vmax = np.inf
        self.bias = np.zeros(n, dtype=np.float32)

        self.synapses = []
        self.named_synapses = {}
        self.axons = []
        self.named_axons = {}
        self.probes = []

        assert location in ('core', 'cpu')
        self.location = location

    def __str__(self):
        return "%s(%s)" % (
            type(self).__name__, self.label if self.label else '')

    def add_synapses(self, synapses, name=None):
        assert synapses.parent is None
        synapses.parent = self
        self.synapses.append(synapses)
        if name is not None:
            assert name not in self.named_synapses
            self.named_synapses[name] = synapses

        AXONS_MAX = 4096
        MAX_MEM_LEN = 16384
        n_axons = sum(s.n_axons for s in self.synapses)
        if n_axons > AXONS_MAX:
            raise ValueError("Total axons (%d) exceeded max (%d)" % (
                n_axons, AXONS_MAX))

        n_synapses = sum(s.size() for s in self.synapses)
        max_synapses = 4*(MAX_MEM_LEN - len(self.synapses))
        if n_synapses > max_synapses:
            raise ValueError("Total synapses (%d) exceeded max (%d)" % (
                n_synapses, max_synapses))

    def add_axons(self, axons, name=None):
        self.axons.append(axons)
        if name is not None:
            assert name not in self.named_axons
            self.named_axons[name] = axons

        assert axons.n_axons == self.n, "Axons currently only one-to-one"

    def add_probe(self, probe):
        if probe.target is None:
            probe.target = self
        assert probe.target is self
        self.probes.append(probe)

    def configure_filter(self, tau_s, dt=0.001):
        self.decayU[:] = -np.expm1(-dt/np.asarray(tau_s))

    # def configure_linear(self, tau_s=0.0, dt=0.001):
    #     self.decayU[:] = -np.expm1(-dt/np.asarray(tau_s))
    #     self.decayV[:] = 0.
    #     self.refractDelay[:] = 0.
    #     self.vth[:] = np.inf
    #     self.vmin = -np.inf
    #     self.vmax = np.inf
    #     self.scaleU = True
    #     self.scaleV = False

    def configure_lif(
            self, tau_s=0.005, tau_rc=0.02, tau_ref=0.001, vth=1, dt=0.001):
        self.decayU[:] = -np.expm1(-dt/np.asarray(tau_s))
        self.decayV[:] = -np.expm1(-dt/np.asarray(tau_rc))
        self.refractDelay[:] = np.round(tau_ref / dt) + 1
        self.vth[:] = vth
        self.vmin = 0
        self.vmax = np.inf
        self.scaleU = True
        self.scaleV = True

    def configure_relu(self, tau_s=0.0, tau_ref=0.0, vth=1, dt=0.001):
        self.decayU[:] = -np.expm1(-dt/np.asarray(tau_s))
        self.decayV[:] = 0.
        self.refractDelay[:] = np.round(tau_ref / dt) + 1
        self.vth[:] = vth
        self.vmin = 0
        self.vmax = np.inf
        self.scaleU = True
        self.scaleV = False

    def discretize(self):  # noqa C901
        def discretize(target, value):
            assert target.dtype == np.float32
            # new = np.round(target * scale).astype(np.int32)
            new = np.round(value).astype(np.int32)
            target.dtype = np.int32
            target[:] = new

        # --- discretize decayU and decayV
        u_infactor = (
            self.decayU.copy() if self.scaleU else np.ones_like(self.decayU))
        v_infactor = (
            self.decayV.copy() if self.scaleV else np.ones_like(self.decayV))
        discretize(self.decayU, self.decayU * (2**12 - 1))
        discretize(self.decayV, self.decayV * (2**12 - 1))
        self.scaleU = False
        self.scaleV = False

        # --- vmin and vmax
        vmine = np.clip(np.round(np.log2(-self.vmin + 1)), 0, 2**5-1)
        self.vmin = -2**vmine + 1
        vmaxe = np.clip(np.round((np.log2(self.vmax + 1) - 9)*0.5), 0, 2**3-1)
        self.vmax = 2**(9 + 2*vmaxe) - 1

        # --- discretize weights and vth
        w_maxs = [max(np.abs(w).max() for w in s.weights)
                  for s in self.synapses]
        w_max = max(w_maxs) if len(w_maxs) > 0 else 0
        b_max = np.abs(self.bias).max()
        wgtExp = -7

        if w_max > 1e-8:
            w_scale = (255. / w_max)
            s_scale = 1. / (u_infactor * v_infactor)

            for wgtExp in range(7, -8, -1):
                v_scale = s_scale * w_scale * SynapseFmt.get_scale(wgtExp)
                b_scale = v_scale * v_infactor
                vth = np.round(self.vth * v_scale)
                bias = np.round(self.bias * b_scale)
                if (vth <= VTH_MAX).all() and (np.abs(bias) <= BIAS_MAX).all():
                    break
            else:
                raise ValueError("Could not find appropriate wgtExp")
        elif b_max > 1e-8:
            b_scale = BIAS_MAX / b_max
            while b_scale*b_max > 1:
                v_scale = b_scale / v_infactor
                w_scale = b_scale * u_infactor / SynapseFmt.get_scale(wgtExp)
                vth = np.round(self.vth * v_scale)
                bias = np.round(self.bias * b_scale)
                if np.all(vth <= VTH_MAX):
                    break

                b_scale /= 2.
            else:
                raise ValueError("Could not find appropriate bias scaling")
        else:
            v_scale = VTH_MAX / (self.vth.max() + 1)
            vth = np.round(self.vth * v_scale)
            b_scale = v_scale * v_infactor
            bias = np.round(self.bias * b_scale)
            w_scale = (v_scale * v_infactor * u_infactor
                       / SynapseFmt.get_scale(wgtExp))

        vth_man, vth_exp = vth_to_manexp(vth)
        discretize(self.vth, vth_man * 2**vth_exp)

        bias_man, bias_exp = bias_to_manexp(bias)
        discretize(self.bias, bias_man * 2**bias_exp)

        for i, synapse in enumerate(self.synapses):
            if w_maxs[i] > 1e-16:
                dWgtExp = int(np.floor(np.log2(w_max / w_maxs[i])))
                assert dWgtExp >= 0
                wgtExp2 = max(wgtExp - dWgtExp, -6)
            else:
                wgtExp2 = -6
                dWgtExp = wgtExp - wgtExp2
            synapse.format(wgtExp=wgtExp2)
            for w in synapse.weights:
                discretize(w, synapse.synapse_fmt.discretize_weights(
                    w * w_scale * 2**dWgtExp))


class CxSynapses(object):
    def __init__(self, n_axons):
        self.n_axons = n_axons
        self.parent = None
        self.synapse_fmt = None
        self.weights = None
        self.indices = None

    def size(self):
        return sum(len(w) for w in self.weights)

    def set_full_weights(self, weights):
        self.weights = [w.astype(np.float32) for w in weights]
        self.indices = [np.arange(w.size) for w in weights]
        assert weights.shape[0] == self.n_axons

        idxBits = int(np.ceil(np.log2(max(i.max() for i in self.indices) + 1)))
        idxBits = next(i for i, v in enumerate(SynapseFmt.INDEX_BITS_MAP)
                       if v >= idxBits)
        self.format(compression=3, idxBits=idxBits, fanoutType=1,
                    numSynapses=63, wgtBits=7)

    def format(self, **kwargs):
        if self.synapse_fmt is None:
            self.synapse_fmt = SynapseFmt()
        self.synapse_fmt.set(**kwargs)


class CxAxons(object):
    def __init__(self, n_axons):
        self.n_axons = n_axons

        self.target = None


class CxProbe(object):
    _slice = slice

    def __init__(self, target=None, key=None, slice=None):
        self.target = target
        self.key = key
        self.slice = slice if slice is not None else self._slice(None)


class CxModel(object):

    def __init__(self):
        self.cx_groups = collections.OrderedDict()

    def add_group(self, group):
        assert isinstance(group, CxGroup)
        assert group not in self.cx_groups
        self.cx_groups[group] = len(self.cx_groups)

    def discretize(self):
        for group in self.cx_groups:
            group.discretize()

    def get_loihi(self):
        from nengo_loihi.loihi_interface import LoihiSimulator
        return LoihiSimulator(self)

    def get_simulator(self):
        return CxSimulator(self)


class CxSimulator(object):
    """
    TODO:
    - noise on u/v
    - compartment mixing (omega)
    """

    def __init__(self, model):
        self.build(model)

    def build(self, model):
        self.model = model
        # self.groups = list(self.model.cx_groups)
        self.groups = sorted(self.model.cx_groups,
                             key=lambda g: g.location == 'cpu')
        self.probes = list(self.model.probes)
        self.probe_outputs = collections.defaultdict(list)

        self.n_cx = sum(group.n for group in self.groups)
        self.group_slices = {}
        self.synapse_slices = {}
        self.axon_slices = {}
        cx_slice = None
        i0 = 0
        for group in self.groups:
            if group.location == 'cpu' and cx_slice is None:
                cx_slice = slice(0, i0)

            i1 = i0 + group.n
            self.group_slices[group] = slice(i0, i1)
            for synapse in group.synapses:
                self.synapse_slices[synapse] = slice(i0, i1)
            for axon in group.axons:
                self.axon_slices[axon] = slice(i0, i1)
                # ^TODO: allow non one-to-one axons
            i0 = i1

        self.cx_slice = slice(0, i0) if cx_slice is None else cx_slice
        self.cpu_slice = slice(self.cx_slice.stop, i1)

        # --- allocate group memory
        group_dtype = self.groups[0].vth.dtype
        assert group_dtype in (np.float32, np.int32)
        for group in self.groups:
            assert group.vth.dtype == group_dtype
            assert group.bias.dtype == group_dtype

        print("Simulator dtype: %s" % group_dtype)

        MAX_DELAY = 1  # don't do delay yet
        self.q = np.zeros((MAX_DELAY, self.n_cx), dtype=group_dtype)
        self.u = np.zeros(self.n_cx, dtype=group_dtype)
        self.v = np.zeros(self.n_cx, dtype=group_dtype)
        self.s = np.zeros(self.n_cx, dtype=bool)  # spiked
        self.c = np.zeros(self.n_cx, dtype=np.int32)  # spike counter
        self.w = np.zeros(self.n_cx, dtype=np.int32)  # ref period counter

        # --- allocate weights
        self.decayU = np.hstack([group.decayU for group in self.groups])
        self.decayV = np.hstack([group.decayV for group in self.groups])
        self.scaleU = np.hstack([
            group.decayU if group.scaleU else np.ones_like(group.decayU)
            for group in self.groups])
        self.scaleV = np.hstack([
            group.decayV if group.scaleV else np.ones_like(group.decayV)
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
        self.vth = np.hstack([group.vth for group in self.groups])
        self.vmin = np.hstack([
            group.vmin*ones(group.n) for group in self.groups])
        self.vmax = np.hstack([
            group.vmax*ones(group.n) for group in self.groups])

        self.bias = np.hstack([group.bias for group in self.groups])
        self.ref = np.hstack([group.refractDelay for group in self.groups])

    def step(self):
        # --- connections
        self.q[:-1] = self.q[1:]  # advance delays
        self.q[-1] = 0

        for group in self.groups:
            for axon in group.axons:
                synapse = axon.target
                a_slice = self.axon_slices[axon]
                sa = self.s[a_slice]

                b_slice = self.synapse_slices[synapse]
                weights = synapse.weights
                indices = synapse.indices
                qb = self.q[:, b_slice]
                delays = np.zeros(qb.shape[1], dtype=np.int32)

                for i in sa.nonzero()[0]:
                    qb[delays, indices[i]] += weights[i]

        # --- updates
        q0 = self.q[0, :]

        # self.U[:] = self.decayU_fn(self.U, self.decayU, a=12, b=1)
        self.u[:] = self.decayU_fn(self.u[:], q0)
        u2 = self.u[:] + self.bias

        # self.V[:] = self.decayV_fn(v, self.decayV, a=12) + u2
        self.v[:] = self.decayV_fn(self.v, u2)
        np.clip(self.v, self.vmin, self.vmax, out=self.v)
        self.v[self.w > 0] = 0
        # TODO^: don't zero voltage in case neuron is saving overshoot

        self.s[:] = (self.v > self.vth)

        cx = self.cx_slice
        cpu = self.cpu_slice
        self.v[cx][self.s[cx]] = 0
        self.v[cpu][self.s[cpu]] -= self.vth[cpu][self.s[cpu]]

        self.w[self.s] = self.ref[self.s]
        np.clip(self.w - 1, 0, None, out=self.w)  # decrement w

        self.c[self.s] += 1

        # --- probes
        for group in self.groups:
            for probe in group.probes:
                x_slice = self.group_slices[probe.target]
                p_slice = probe.slice
                if probe.key == 'x':
                    x = (self.u[x_slice][p_slice] /
                         self.vth[x_slice][p_slice].astype(np.float32))
                else:
                    assert hasattr(self, probe.key)
                    x = getattr(self, probe.key)[x_slice][p_slice].copy()
                self.probe_outputs[probe].append(x)

    def run_steps(self, steps):
        for _ in range(steps):
            self.step()

    def get_probe_output(self, probe):
        target = self.model.objs[probe]['out']
        assert isinstance(target, CxProbe)
        return self.probe_outputs[target]
