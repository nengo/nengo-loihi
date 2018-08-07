class Probe(object):
    _slice = slice

    def __init__(self, target=None, key=None, slice=None, weights=None,
                 synapse=None):
        self.target = target
        self.key = key
        self.slice = slice if slice is not None else self._slice(None)
        self.weights = weights
        self.synapse = synapse
        self.use_snip = False
        self.snip_info = None


class ProbeGroup(object):
    def __init__(self):
        self.probes = []

    def add(self, probe):
        """Add a Probe object to ensemble."""
        # if probe.target is None:
        #     probe.target = self
        # assert probe.target is self
        self.probes.append(probe)

    def discretize(self, v_scale):
        for p in self.probes:
            if p.key == 'v' and p.weights is not None:
                p.weights /= v_scale
