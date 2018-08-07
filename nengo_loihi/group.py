from nengo_loihi.axons import AxonGroup
from nengo_loihi.compartments import CompartmentGroup
from nengo_loihi.synapses import SynapseGroup


class ProbeGroup(object):
    def __init__(self):
        self.probes = []

    def add(self, probe):
        """Add a CxProbe object to ensemble."""
        # if probe.target is None:
        #     probe.target = self
        # assert probe.target is self
        self.probes.append(probe)

    def discretize(self, v_scale):
        for p in self.probes:
            if p.key == 'v' and p.weights is not None:
                p.weights /= v_scale


class CoreGroup(object):
    """A group of Loihi objects to be placed on a Loihi neuron core.

    Consists of a `.CompartmentGroup`, `.SynapseGroup` `.AxonGroup`
    and `.ProbeGroup`.
    """

    def __init__(self, n_compartments, label=None):
        self.label = label

        self.axons = AxonGroup(n_axons=n_compartments)
        self.compartments = CompartmentGroup(n_compartments=n_compartments)
        self.synapses = SynapseGroup(n_synapses=n_compartments)
        self.probes = ProbeGroup()

    @property
    def n_compartments(self):
        return self.compartments.n_compartments

    def __str__(self):
        return "%s(%s)" % (
            type(self).__name__, self.label if self.label else '')

    def discretize(self):
        w_max = self.synapses.max_weight()

        self.compartments.discretize(w_max)
        self.synapses.discretize(
            self.compartments.w_scale, self.compartments.weight_exp)
        self.probes.discretize(self.compartments.v_scale)
