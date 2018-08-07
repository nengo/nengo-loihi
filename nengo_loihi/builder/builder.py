import collections

import nengo
from nengo.builder.builder import Builder as NengoBuilder
from nengo.builder.network import build_network
from nengo.cache import NoDecoderCache

from nengo_loihi.model import CxModel

# firing rate of inter neurons
INTER_RATE = 100

# number of inter neurons
INTER_N = 10


class Model(CxModel):
    """The data structure for the chip/simulator.

    CxModel defines adding ensembles, discretizing, and tracks the simulator
    """
    def __init__(self, dt=0.001, label=None, builder=None):
        super(Model, self).__init__()

        self.dt = dt
        self.label = label

        self.objs = collections.defaultdict(dict)
        self.params = {}  # Holds data generated when building objects
        self.probes = []
        self.chip2host_params = {}
        self.probe_conns = {}

        self.seeds = {}
        self.seeded = {}

        self.toplevel = None
        self.config = None
        self.decoder_cache = NoDecoderCache()

        self.builder = Builder() if builder is None else builder
        self.build_callback = None

    def __str__(self):
        return "Model: %s" % self.label

    def build(self, obj, *args, **kwargs):
        built = self.builder.build(self, obj, *args, **kwargs)
        if self.build_callback is not None:
            self.build_callback(obj)
        return built

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
