import nengo
from nengo.params import Parameter


def add_params(network):
    """Create custom config options for nengo_loihi"""
    config = network.config

    cfg = config[nengo.Ensemble]
    if 'on_chip' not in cfg._extra_params:
        cfg.set_param("on_chip",
                      Parameter('on_chip', default=True, optional=True))
