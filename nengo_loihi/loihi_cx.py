import collections

import numpy as np


class CxGroup(object):

    def __init__(self, n, label=None):
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
        self.vmin = np.zeros(n, dtype=np.float32)
        self.vmax = np.zeros(n, dtype=np.float32)
        self.bias = np.zeros(n, dtype=np.float32)

        self.synapses = []
        self.named_synapses = {}
        self.axons = []
        self.named_axons = {}

    def add_synapses(self, synapses, name=None):
        assert synapses.parent is None
        synapses.parent = self
        self.synapses.append(synapses)
        if name is not None:
            assert name not in self.named_synapses
            self.named_synapses[name] = synapses

    def add_axons(self, axons, name=None):
        self.axons.append(axons)
        if name is not None:
            assert name not in self.named_axons
            self.named_axons[name] = axons

    def configure_filter(self, tau_s, dt=0.001):
        self.decayU[:] = -np.expm1(-dt/np.asarray(tau_s))

    def configure_lif(
            self, tau_s=0.005, tau_rc=0.02, tau_ref=0.001, vth=1, dt=0.001):
        self.decayU[:] = -np.expm1(-dt/np.asarray(tau_s))
        self.decayV[:] = -np.expm1(-dt/np.asarray(tau_rc))
        self.refDelay[:] = tau_ref / dt
        self.vth[:] = vth
        self.vmin[:] = 0
        self.vmax[:] = np.inf
        self.scaleU = True
        self.scaleV = True

    def configure_relu(self, tau_s=0.0, tau_ref=0.001, vth=1, dt=0.001):
        self.decayU[:] = -np.expm1(-dt/np.asarray(tau_s))
        self.decayV[:] = 0.
        self.refDelay[:] = tau_ref / dt
        self.vth[:] = vth
        self.vmin[:] = 0
        self.vmax[:] = np.inf
        self.scaleU = True
        self.scaleV = False


class CxSynapses(object):
    def __init__(self, n_axons):
        self.n_axons = n_axons
        self.parent = None

    def set_full_weights(self, weights):
        self.weights = weights
        self.indices = [np.arange(w.size) for w in weights]
        assert weights.shape[0] == self.n_axons


class CxAxons(object):
    def __init__(self, n_axons):
        self.n_axons = n_axons

        self.target = None


class CxCpuTarget(object):
    def __init__(self, n):
        self.n = n

        self.synapses = []
        self.named_synapses = {}

    def add_synapses(self, synapses, name=None):
        assert synapses.parent is None
        synapses.parent = self
        self.synapses.append(synapses)
        if name is not None:
            assert name not in self.named_synapses
            self.named_synapses[name] = synapses



# class CxConnection(object):

#     def __init__(self, a, b):
#         self.a = a
#         self.b = b


class CxModel(object):

    def __init__(self):
        self.groups = collections.OrderedDict()
        # self.connections = collections.OrderedDict()
        self.targets = collections.OrderedDict()

    def add_group(self, group):
        assert isinstance(group, CxGroup)
        assert group not in self.groups
        self.groups[group] = len(self.groups)

    def add_target(self, target):
        assert target not in self.targets
        self.targets[target] = len(self.targets)

    # def add_connection(self, connection):
    #     assert isinstance(connection, CxConnection)
    #     assert connection not in self.connections
    #     self.connections[connection] = len(self.connections)


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
        self.groups = list(self.model.groups)
        self.targets = list(self.model.targets)
        # self.conns = list(self.model.connections)

        self.n_cx = sum(group.n for group in self.groups)
        self.group_slices = {}
        self.synapse_slices = {}
        self.axon_slices = {}
        i0 = 0
        for group in self.groups:
            i1 = i0 + group.n
            self.group_slices[group] = slice(i0, i1)
            for synapse in group.synapses:
                self.synapse_slices[synapse] = slice(i0, i1)
            for axon in group.axons:
                self.axon_slices[axon] = slice(i0, i1)
                # ^TODO: allow non one-to-one axons
            i0 = i1

        self.n_target = sum(target.n for target in self.targets)
        for target in self.targets:
            i1 = i0 + target.n
            self.synapse_slices[target] = slice(i0, i1)
            for synapse in target.synapses:
                self.synapse_slices[synapse] = slice(i0, i1)
            i0 = i1

        # --- allocate group memory
        group_dtype = np.float32
        MAX_DELAY = 1  # don't do delay yet
        # self.q = np.zeros((MAX_DELAY, self.n_cx), dtype=group_dtype)
        self.q = np.zeros((MAX_DELAY, self.n_cx + self.n_target),
                          dtype=group_dtype)
        self.U = np.zeros(self.n_cx, dtype=group_dtype)
        self.V = np.zeros(self.n_cx, dtype=group_dtype)
        self.S = np.zeros(self.n_cx, dtype=bool)  # spiked
        self.C = np.zeros(self.n_cx, dtype=np.int32)  # spike counter
        self.w = np.zeros(self.n_cx, dtype=np.int32)  # ref period counter

        # --- allocate weights
        # self.indices = {conn: conn.get_indices() for conn in self.conns}
        # self.weights = {conn: conn.get_weights() for conn in self.conns}

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

        self.decayU_fn = lambda x, u: decay_float(x, u, d=self.decayU, s=self.scaleU)
        self.decayV_fn = lambda x, u: decay_float(x, u, d=self.decayV, s=self.scaleV)

        self.vth = np.hstack([group.vth for group in self.groups])
        self.vmin = np.hstack([group.vmin for group in self.groups])
        self.vmax = np.hstack([group.vmax for group in self.groups])

        self.bias = np.hstack([group.bias for group in self.groups])

        # self.ref = np.hstack([group.refDelay for group in self.groups])
        self.ref = np.hstack([
            np.round(group.refDelay).astype(np.int32) for group in self.groups])

        # --- probe outputs
        # self._probe_outputs = {target: list() for target in self.targets}

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

        # --- compartments
        q_cx0 = self.q[:, :self.n_cx]
        print(q_cx0)

        # self.U[:] = self.decayU_fn(self.U, self.decayU, a=12, b=1)
        self.U[:] = self.decayU_fn(self.U, q_cx0)
        u2 = self.U + self.bias
        # print(self.bias)
        # print(u2)

        # self.V[:] = self.decayV_fn(v, self.decayV, a=12) + u2
        self.V[:] = self.decayV_fn(self.V, u2)
        # print(self.V)
        np.clip(self.V, self.vmin, self.vmax, out=self.V)
        # print("vmin: %s, vmax: %s" % (self.vmin, self.vmax))
        self.V[self.w > 0] = 0
        # print(self.V)

        self.S[:] = (self.V > self.vth)
        self.V[self.S] = 0
        self.w[self.S] = self.ref[self.S]
        np.clip(self.w - 1, 0, None, out=self.w)  # decrement w

        self.C[self.S] += 1

        # --- targets
        # for target in self.targets:
        #     b_slice = self.synapse_slices[target]
        #     qb = self.q[0, b_slice]
        #     self._probe_outputs[target].append(qb.copy())

    def get_probe_value(self, probe):
        target = self.model.objs[probe]['in']
        b_slice = self.synapse_slices[target]
        qb = self.q[0, b_slice]
        return qb.copy()
