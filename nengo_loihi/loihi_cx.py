import collections

import numpy as np

from .allocators import SynapseFmt
from .loihi_api import VTH_MAN_MAX, BIAS_MAX


def decay12_scale(decay_x):
    """The amount of effective scaling applied to a spike"""
    mult = float(2**12 - decay_x) / 2**12
    return 1. / (1 - mult)  # sum of infinite geometric series


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
        self.refDelay = np.zeros(n, dtype=np.float32)
        self.vth = np.zeros(n, dtype=np.float32)
        self.vmin = 0
        self.vmax = np.inf
        self.bias = np.zeros(n, dtype=np.float32)

        self.synapses = []
        self.named_synapses = {}
        self.axons = []
        self.named_axons = {}

        assert location in ('core', 'cpu')
        self.location = location

    def add_synapses(self, synapses, name=None):
        assert synapses.parent is None
        synapses.parent = self
        self.synapses.append(synapses)
        if name is not None:
            assert name not in self.named_synapses
            self.named_synapses[name] = synapses

        AXONS_MAX = 4096
        MAX_MEM_LEN = 16384
        assert sum(s.n_axons for s in self.synapses) < AXONS_MAX
        assert sum(s.weights.size for s in self.synapses) < 4*(
            MAX_MEM_LEN - len(self.synapses))

    def add_axons(self, axons, name=None):
        self.axons.append(axons)
        if name is not None:
            assert name not in self.named_axons
            self.named_axons[name] = axons

    def configure_filter(self, tau_s, dt=0.001):
        self.decayU[:] = -np.expm1(-dt/np.asarray(tau_s))

    # def configure_linear(self, tau_s=0.0, dt=0.001):
    #     self.decayU[:] = -np.expm1(-dt/np.asarray(tau_s))
    #     self.decayV[:] = 0.
    #     self.refDelay[:] = 0.
    #     self.vth[:] = np.inf
    #     self.vmin = -np.inf
    #     self.vmax = np.inf
    #     self.scaleU = True
    #     self.scaleV = False

    def configure_lif(
            self, tau_s=0.005, tau_rc=0.02, tau_ref=0.001, vth=1, dt=0.001):
        self.decayU[:] = -np.expm1(-dt/np.asarray(tau_s))
        self.decayV[:] = -np.expm1(-dt/np.asarray(tau_rc))
        self.refDelay[:] = np.round(tau_ref / dt)
        self.vth[:] = vth
        self.vmin = 0
        self.vmax = np.inf
        self.scaleU = True
        self.scaleV = True

    def configure_relu(self, tau_s=0.0, tau_ref=0.0, vth=1, dt=0.001):
        self.decayU[:] = -np.expm1(-dt/np.asarray(tau_s))
        self.decayV[:] = 0.
        self.refDelay[:] = np.round(tau_ref / dt)
        self.vth[:] = vth
        self.vmin = 0
        self.vmax = np.inf
        self.scaleU = True
        self.scaleV = False

    def discretize(self):
        def discretize(target, value):
            assert target.dtype == np.float32
            # new = np.round(target * scale).astype(np.int32)
            new = np.round(value).astype(np.int32)
            target.dtype = np.int32
            target[:] = new

        # --- discretize decayU and decayV
        u_scale = (
            self.decayU.copy() if self.scaleU else np.ones_like(self.decayU))
        v_scale = (
            self.decayV.copy() if self.scaleV else np.ones_like(self.decayV))
        discretize(self.decayU, self.decayU * (2**12 - 1))
        discretize(self.decayV, self.decayV * (2**12 - 1))

        # --- vmin and vmax
        vmine = np.clip(np.round(np.log2(-self.vmin + 1)), 0, 2**5-1)
        self.vmin = -2**vmine + 1
        vmaxe = np.clip(np.round((np.log2(self.vmax + 1) - 9)*0.5), 0, 2**3-1)
        self.vmax = 2**(9 + 2*vmaxe) - 1

        # --- discretize weights and vth
        w_maxs = [np.abs(s.weights).max() for s in self.synapses]
        b_max = np.abs(self.bias).max()
        # dt = 0.001

        if len(w_maxs) > 0:
            w_maxi = np.argmax(w_maxs)
            w_max = w_maxs[w_maxi]
            w_scale = (127. / w_max)

            self.synapses[w_maxi].format(wgtExp=0)
            synapse_fmt = self.synapses[w_maxi].synapse_fmt
            # wgtExpBase = self.synapses[w_maxi].synapse_fmt.Wscale

            # s_scale = decay12_scale(self.decayU)
            s_scale = 1. / (u_scale * v_scale)

            # for wgtExp in range(0, -8, -1):
            for wgtExp in range(7, -8, -1):
                synapse_fmt.set(wgtExp=wgtExp)
                x_scale = s_scale * w_scale * 2**synapse_fmt.Wscale
                b_scale = x_scale * v_scale
                vth = self.vth * x_scale
                vth2 = np.round(vth / 2**6)
                bias = np.round(self.bias * b_scale)
                if np.all(vth2 <= VTH_MAN_MAX) and np.all(bias <= BIAS_MAX):
                    vth = vth2 * 2**6
                    break
            else:
                raise ValueError("Could not find appropriate wgtExp")

        else:
            s_scale = 1. / v_scale
            b_scale = BIAS_MAX / b_max
            while b_scale*b_max > 1:
                x_scale = s_scale * b_scale
                vth = self.vth * x_scale
                vth2 = np.round(vth / 2**6)
                bias = np.round(self.bias * b_scale * v_scale)
                if np.all(vth2 <= VTH_MAN_MAX):
                    vth = vth2 * 2**6
                    break

                b_scale /= 2.
            else:
                raise ValueError("Could not find appropriate bias scaling")

        discretize(self.vth, vth)
        discretize(self.bias, bias)  # TODO: round bias to fit mant/exp
        for i, synapse in enumerate(self.synapses):
            dWgtExp = np.floor(np.log2(w_max / w_maxs[i]))
            assert dWgtExp >= 0
            wgtExp2 = max(wgtExp - dWgtExp, -7)
            dWgtExp = wgtExp - wgtExp2
            synapse.format(WgtExp=wgtExp2)
            discretize(
                synapse.weights,
                synapse.weights * w_scale * 2**synapse.synapse_fmt.Wscale)


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

    def format(self, **kwargs):
        if self.synapse_fmt is None:
            self.synapse_fmt = SynapseFmt()
        self.synapse_fmt.set(**kwargs)


class CxAxons(object):
    def __init__(self, n_axons):
        self.n_axons = n_axons

        self.target = None


# class CxCpuTarget(object):
#     def __init__(self, n):
#         self.n = n

#         self.synapses = []
#         self.named_synapses = {}

#     def add_synapses(self, synapses, name=None):
#         assert synapses.parent is None
#         synapses.parent = self
#         self.synapses.append(synapses)
#         if name is not None:
#             assert name not in self.named_synapses
#             self.named_synapses[name] = synapses


class CxModel(object):

    def __init__(self):
        self.groups = collections.OrderedDict()

    def add_group(self, group):
        assert isinstance(group, CxGroup)
        assert group not in self.groups
        self.groups[group] = len(self.groups)

    def discretize(self):
        for group in self.groups:
            group.discretize()


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
        # self.groups = list(self.model.groups)
        self.groups = sorted(self.model.groups,
                             key=lambda g: g.location == 'cpu')

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

        MAX_DELAY = 1  # don't do delay yet
        self.q = np.zeros((MAX_DELAY, self.n_cx), dtype=group_dtype)
        self.U = np.zeros(self.n_cx, dtype=group_dtype)
        self.V = np.zeros(self.n_cx, dtype=group_dtype)
        self.S = np.zeros(self.n_cx, dtype=bool)  # spiked
        self.C = np.zeros(self.n_cx, dtype=np.int32)  # spike counter
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
            self.decayU_fn = (
                lambda x, u: decay_int(x, u, d=self.decayU, s=self.scaleU))
            self.decayV_fn = (
                lambda x, u: decay_int(x, u, d=self.decayV, s=self.scaleV))
        elif group_dtype == np.float32:
            self.decayU_fn = (
                lambda x, u: decay_float(x, u, d=self.decayU, s=self.scaleU))
            self.decayV_fn = (
                lambda x, u: decay_float(x, u, d=self.decayV, s=self.scaleV))

        ones = lambda n: np.ones(n, dtype=group_dtype)
        self.vth = np.hstack([group.vth for group in self.groups])
        self.vmin = np.hstack([
            group.vmin*ones(group.n) for group in self.groups])
        self.vmax = np.hstack([
            group.vmax*ones(group.n) for group in self.groups])

        self.bias = np.hstack([group.bias for group in self.groups])

        # self.ref = np.hstack([group.refDelay for group in self.groups])
        self.ref = np.hstack([
            np.round(group.refDelay).astype(np.int32)
            for group in self.groups])

    def step(self):
        # --- connections
        self.q[:-1] = self.q[1:]  # advance delays
        self.q[-1] = 0

        for group in self.groups:
            for axon in group.axons:
                synapse = axon.target
                a_slice = self.axon_slices[axon]
                Sa = self.S[a_slice]

                b_slice = self.synapse_slices[synapse]
                weights = synapse.weights
                indices = synapse.indices
                qb = self.q[:, b_slice]
                delays = np.zeros(qb.shape[1], dtype=np.int32)

                for i in Sa.nonzero()[0]:
                    qb[delays, indices[i]] += weights[i]

        # --- updates
        q0 = self.q[0, :]

        # self.U[:] = self.decayU_fn(self.U, self.decayU, a=12, b=1)
        self.U[:] = self.decayU_fn(self.U[:], q0)
        u2 = self.U[:] + self.bias

        # self.V[:] = self.decayV_fn(v, self.decayV, a=12) + u2
        self.V[:] = self.decayV_fn(self.V, u2)
        np.clip(self.V, self.vmin, self.vmax, out=self.V)
        self.V[self.w > 0] = 0

        self.S[:] = (self.V > self.vth)

        cx = self.cx_slice
        cpu = self.cpu_slice
        self.V[cx][self.S[cx]] = 0
        self.V[cpu][self.S[cpu]] -= self.vth[cpu][self.S[cpu]]

        self.w[self.S] = self.ref[self.S]
        np.clip(self.w - 1, 0, None, out=self.w)  # decrement w

        self.C[self.S] += 1

    def get_probe_value(self, probe):
        target = self.model.objs[probe]['in'].named_synapses['encoders2']
        b_slice = self.synapse_slices[target]
        qb = self.q[0, b_slice] / self.vth[b_slice].astype(float)
        return qb.copy()
