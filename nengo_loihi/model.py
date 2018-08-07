from __future__ import division

import collections
import logging

logger = logging.getLogger(__name__)


class CxSpikeInput(object):
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


class CxModel(object):
    def __init__(self):
        self.cx_inputs = collections.OrderedDict()
        self.groups = collections.OrderedDict()

    def add_input(self, input):
        assert isinstance(input, CxSpikeInput)
        assert input not in self.cx_inputs
        self.cx_inputs[input] = len(self.cx_inputs)

    def add_group(self, group):
        # assert isinstance(group, CoreGroup)
        assert group not in self.groups
        self.groups[group] = len(self.groups)

    def discretize(self):
        for group in self.groups:
            group.discretize()
