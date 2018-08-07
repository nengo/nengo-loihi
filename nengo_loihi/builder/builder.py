import collections

import nengo
from nengo.builder.builder import Builder as NengoBuilder
from nengo.builder.network import build_network
from nengo.cache import NoDecoderCache

from nengo_loihi.axons import AxonGroup
from nengo_loihi.compartments import CompartmentGroup
from nengo_loihi.probes import ProbeGroup
from nengo_loihi.synapses import SynapseGroup

# firing rate of inter neurons
INTER_RATE = 100

# number of inter neurons
INTER_N = 10


class SpikeInput(object):
    def __init__(self, spikes):
        assert spikes.ndim == 2
        self.spikes = spikes
        self.axons = []
        self.probes = []

    @property
    def n(self):
        return self.spikes.shape[1]

    def add_axons(self, axons):
        assert axons.n_axons == self.n
        self.axons.append(axons)

    def add_probe(self, probe):
        if probe.target is None:
            probe.target = self
        assert probe.target is self
        self.probes.append(probe)


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


class Model(object):
    """The data structure for the chip/simulator."""
    def __init__(self, dt=0.001, label=None, builder=None):
        self.dt = dt
        self.label = label

        self.objs = collections.defaultdict(dict)
        self.params = {}  # Holds data generated when building objects
        self.probes = []
        self.chip2host_params = {}
        self.probe_conns = {}
        self.spike_inputs = collections.OrderedDict()
        self.groups = collections.OrderedDict()

        self.seeds = {}
        self.seeded = {}

        self.toplevel = None
        self.config = None
        self.decoder_cache = NoDecoderCache()

        self.builder = Builder() if builder is None else builder
        self.build_callback = None

    def __str__(self):
        return "Model: %s" % self.label

    def add_group(self, group):
        assert isinstance(group, CoreGroup)
        assert group not in self.groups
        self.groups[group] = len(self.groups)

    def add_input(self, input):
        assert isinstance(input, SpikeInput)
        assert input not in self.spike_inputs
        self.spike_inputs[input] = len(self.spike_inputs)

    def build(self, obj, *args, **kwargs):
        built = self.builder.build(self, obj, *args, **kwargs)
        if self.build_callback is not None:
            self.build_callback(obj)
        return built

    def discretize(self):
        for group in self.groups:
            group.discretize()

    def has_built(self, obj):
        return obj in self.params


class Builder(NengoBuilder):
    """Fills in the Loihi Model object based on the Nengo Network.

    We cannot use the Nengo builder as is because we make normal Nengo
    networks for host-to-chip and chip-to-host communication. To keep
    Nengo and Nengo Loihi builders separate, we make a blank subclass,
    which effectively copies the class.
    """

    builders = {}


Builder.register(nengo.Network)(build_network)
