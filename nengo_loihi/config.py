import nengo
from nengo.params import Parameter


def add_params(network):
    """Create custom config options for nengo_loihi"""
    config = network.config

    cfg = config[nengo.Ensemble]
    if 'on_chip' not in cfg._extra_params:
        cfg.set_param("on_chip",
                      Parameter('on_chip', default=True, optional=True))


def set_defaults():
    nengo.Ensemble.max_rates.default = nengo.dists.Uniform(100, 120)
    nengo.Ensemble.intercepts.default = nengo.dists.Uniform(-0.5, 0.5)
