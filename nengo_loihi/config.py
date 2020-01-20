import nengo
from nengo.params import Parameter


def add_params(network):
    """Add nengo_loihi config option to *network*.

    The following options will be added:

    `nengo.Ensemble`
      * ``on_chip``: Whether the ensemble should be simulated
        on a Loihi chip. Marking specific ensembles for simulation
        off of a Loihi chip can help with debugging.
    `nengo.Connection`
      * ``pop_type``: The axon format when using population spikes, which are only
        used for convolutional connections. By default, we use ``pop_type`` 32.
        Setting ``pop_type`` to 16 allows more axons to fit on one chip as long as
        the ``Convolution`` transform has ``channels_last=True`` and ``n_filters``
        is a multiple of 4.

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

    ens_cfg = config[nengo.Ensemble]
    if "on_chip" not in ens_cfg._extra_params:
        ens_cfg.set_param("on_chip", Parameter("on_chip", default=None, optional=True))

    conn_cfg = config[nengo.Connection]
    if "pop_type" not in conn_cfg._extra_params:
        conn_cfg.set_param("pop_type", Parameter("pop_type", default=32, optional=True))


def set_defaults():
    """Modify Nengo's default parameters for better performance with Loihi.

    The following defaults will be modified:

    `nengo.Ensemble`
      * ``max_rates``: Set to ``Uniform(low=100, high=120)``
      * ``intercepts``: Set to ``Uniform(low=-0.5, high=0.5)``

    """
    nengo.Ensemble.max_rates.default = nengo.dists.Uniform(100, 120)
    nengo.Ensemble.intercepts.default = nengo.dists.Uniform(-1.0, 0.5)
