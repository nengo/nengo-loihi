from nengo_loihi.axons import AxonGroup
from nengo_loihi.compartments import CompartmentGroup
from nengo_loihi.probes import ProbeGroup
from nengo_loihi.synapses import SynapseGroup


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
