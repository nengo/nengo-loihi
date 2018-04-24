

class CxGroup(object):

    def __init__(self, n):
        self.n = n

        # self.cxProfiles = []
        # self.vthProfiles = []

        # self.outputAxonMap = None
        # self.outputAxons = None

        self.synapses = []
        self.named_synapses = {}

    def add_synapses(self, synapses, name=None):
        self.synapses.append(synapses)
        if name is not None:
            self.named_synapses[name] = synapses


class CxSynapses(object):
    def __init__(self, n_axons):
        self.n_axons = n_axons

    def set_weights(self, weights):
        self.weights = weights
        assert weights.shape[0] == self.n_axons




class CxConnection(object):

    def __init__(self, a, b):
        self.a = a
        self.b = b


class CxModel(object):

    def __init__(self):
        self.groups = collections.OrderedDict()
        self.connections = collections.OrderedDict()

    def add_group(self, group):
        assert isinstance(group, CxGroup)
        assert group not in self.groups
        self.groups[group] = len(self.groups)

    def add_connection(self, connection):
        assert isinstance(connection, CxConnection)
        assert connection not in self.connections
        self.connections[connection] = len(self.connections)


class CxSimulator(object):
    """
    TODO:
    - noise on u/v
    - compartment mixing (omega)
    """

    def __init__(self, model):
        self.build(model)

    def build(self, model):
        self.groups = list(self.model.groups)
        self.conns = list(self.model.connections)

        self.n_cx = sum(group.n for group in self.groups)
        self.group_slices = {}
        i0 = 0
        for group in self.groups:
            i1 = i0 + group.n
            self.group_slices[group] = slice(i0, i1)
            i0 = i1

        # --- allocate group memory
        group_dtype = np.float32
        self.U = np.zeros(self.n_cx, dtype=group_dtype)
        self.V = np.zeros(self.n_cx, dtype=group_dtype)
        self.S = np.zeros(self.n_cx, dtype=bool)  # spiked
        self.C = np.zeros(self.n_cx, dtype=np.int32)  # spike counter
        self.w = np.zeros(self.n_cx, dtype=np.int32)  # refractory period counter


        MAX_DELAY = 1  # don't do delay yet
        self.q = np.zeros((MAX_DELAY, self.n_cx), dtype=dtype)

        # --- allocate weights
        self.indices = {conn: conn.get_indices() for conn in self.conns}
        self.weights = {conn: conn.get_weights() for conn in self.conns}

        self.decay_u = np.hstack([group.get_decay_u() for group in self.groups])
        self.decay_v = np.hstack([group.get_decay_v() for group in self.groups])
        # self.decay_u_fn =

        self.vmin = np.hstack([group.get_vmin() for group in self.groups])
        self.vmax = np.hstack([group.get_vmax() for group in self.groups])
        # self.vmaxs =

        self.bias = np.hstack([group.get_bias() for group in self.groups])


        return step

    def step(self):
        # --- connections
        self.q[:-1] = self.q[1:]  # advance delays

        for conn in self.conns:
            a_slice = self.group_slices[conn.a]
            b_slice = self.group_slices[conn.b]
            weights = self.weights[conn]
            Sa = self.S[a_slice]
            qb = self.q[:, b_slice]

            for i in Sa.nonzero():
                qb[delays, indices[i]] += weights[i]

        # --- compartments
        self.U[:] = self.decay_u_fn(u, self.decay_u, a=12, b=1) + q[0]
        u2 = self.U + self.bias

        self.V[:] = self.decay_v_fn(v, self.decay_v, a=12) + u2
        np.clip(self.V, self.vmin, self.vmax, out=self.V)
        self.V[self.ref > 0] = 0

        self.S[:] = (self.V > self.vth)
        self.V[self.S] = 0
        self.w[self.S] = self.ref[self.S]
        np.clip(self.w - 1, 0, None, out=self.w)

        self.C[self.S] += 1
