import nengo
from nengo.params import Parameter


def add_params(network):
    """Add nengo_loihi config option to *network*.

    The following options will be added:

    `nengo.Ensemble`
      * ``on_chip``: Whether the ensemble should be simulated
        on a Loihi chip. Marking specific ensembles for simulation
        off of a Loihi chip can help with debugging.

    Examples
    --------

    >>> with nengo.Network() as model:
    ...     ens = nengo.Ensemble(10, dimensions=1)
    ...     # By default, ens will be placed on a Loihi chip
    ...     nengo_loihi.add_params(model)
    ...     model.config[ens].on_chip = False
    ...     # Now it will be simulated with Nengo

    """
    config = network.config

    cfg = config[nengo.Ensemble]
    if 'on_chip' not in cfg._extra_params:
        cfg.set_param("on_chip",
                      Parameter('on_chip', default=None, optional=True))


def set_defaults():
    """Modify Nengo's default parameters for better performance with Loihi.

    The following defaults will be modified:

    `nengo.Ensemble`
      * ``max_rates``: Set to ``Uniform(low=100, high=120)``
      * ``intercepts``: Set to ``Uniform(low=-0.5, high=0.5)``

    """
    nengo.Ensemble.max_rates.default = nengo.dists.Uniform(100, 120)
    nengo.Ensemble.intercepts.default = nengo.dists.Uniform(-0.5, 0.5)
