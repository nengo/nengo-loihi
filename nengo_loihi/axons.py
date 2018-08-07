class Axons(object):
    def __init__(self, n_axons):
        self.n_axons = n_axons

        self.target = None
        self.target_inds = slice(None)  # which synapse inputs are targeted
        # ^ TODO: this does not allow multiple pre-cx per axon, loihi does


class AxonGroup(object):
    def __init__(self, n_axons):
        self.n_axons = n_axons
        self.axons = []
        self.named_axons = {}

    def add(self, axons, name=None):
        """Add an Axons object to this group."""
        self.axons.append(axons)
        if name is not None:
            assert name not in self.named_axons
            self.named_axons[name] = axons
