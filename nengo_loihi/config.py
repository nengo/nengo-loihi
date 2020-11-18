import nengo
import numpy as np
from nengo import Ensemble
from nengo.config import InstanceParams
from nengo.exceptions import ValidationError
from nengo.params import Parameter
from nengo.utils.numpy import is_integer


class BlockShapeParam(Parameter):
    def coerce(self, instance, block_shape):
        if isinstance(instance, InstanceParams):
            instance = instance._configures

        assert isinstance(instance, Ensemble), "Not implemented for non-Ensembles"
        self.check_type(instance, block_shape, BlockShape)

        if instance.n_neurons != block_shape.ensemble_size:
            raise ValidationError(
                "Block shape ensemble size (`prod(%s) = %d`) must match "
                "number of ensemble neurons (%d)"
                % (
                    list(block_shape.ensemble_shape),
                    block_shape.ensemble_size,
                    instance.n_neurons,
                ),
                attr=self.name,
                obj=instance,
            )

        return super().coerce(instance, block_shape)


class BlockShape:
    """Specifies how an ensemble should be split across Loihi neuron cores.

    Each neuron core can have, at most, 1024 compartments. Ensembles with more
    than 1024 neurons are automatically split such that they will be distributed
    evenly across as few cores as possible.

    This class allows you to split a smaller ensemble onto multiple cores.
    It also allows you to control the splitting process for any ensemble,
    which is particularly useful when that ensemble participates in a
    convolutional connection.

    Parameters
    ----------
    shape : tuple of int
        The conceptual shape of the compartments on each core after splitting.
        This tuple must have the same number of elements as ``ensemble_shape``.
    ensemble_shape_or_transform : tuple of int or `nengo.Convolution`
        The conceptual shape of the neurons in the ensemble. If a `nengo.Convolution`
        instance is passed, the conceptual shape is inferred automatically.

    Attributes
    ----------
    shape : tuple of int
        The conceptual shape of the compartments on each core after splitting.
    ensemble_shape : tuple of int
        The conceptual shape of the neurons in the ensemble.

    Examples
    --------

    Split an ensemble across two Loihi blocks.

    .. testcode::

       with nengo.Network() as net:
           nengo_loihi.add_params(net)
           ens = nengo.Ensemble(10, 1)
           net.config[ens].block_shape = nengo_loihi.BlockShape((5,), (10,))
       print(net.config[ens].block_shape.n_splits)

    .. testoutput::

       2

    Interpret an ensemble as a two dimensional layer and split into four blocks.

    .. testcode::

       with nengo.Network() as net:
           nengo_loihi.add_params(net)
           ens = nengo.Ensemble(16, 1)
           net.config[ens].block_shape = nengo_loihi.BlockShape((2, 2), (4, 4))
       print(net.config[ens].block_shape.n_splits)

    .. testoutput::

       4
    """

    def __init__(self, shape, ensemble_shape_or_transform):
        self.ensemble_shape = ensemble_shape_or_transform
        if isinstance(ensemble_shape_or_transform, nengo.Convolution):
            self.ensemble_shape = ensemble_shape_or_transform.output_shape.shape
        self.shape = shape

        for attr in ["ensemble_shape", "shape"]:
            shape = getattr(self, attr)
            if not isinstance(shape, tuple):
                raise ValidationError("Must be a tuple", attr=attr)
            if any(not is_integer(el) for el in shape):
                raise ValidationError("All elements must be an int", attr=attr)
        if len(self.shape) != len(self.ensemble_shape):
            raise ValidationError(
                "`shape` and `ensemble_shape` must be the same length", attr="shape"
            )

        # Store numpy array versions of these shapes for easier manipulation
        self._ens_shape = np.asarray(self.ensemble_shape)
        self._shape = np.asarray(self.shape)

        n_splits = np.ceil(self._ens_shape / self._shape).astype(int)
        self.n_splits = np.prod(n_splits)

    @property
    def block_size(self):
        return np.prod(self._shape)

    @property
    def ensemble_size(self):
        return np.prod(self._ens_shape)

    def zip_dimensions(self):
        return zip(self.ensemble_shape, self.shape)


def add_params(network):
    """Add nengo_loihi config option to *network*.

    The following options will be added:

    `nengo.Ensemble`
      * ``on_chip``: Whether the ensemble should be simulated
        on a Loihi chip. Marking specific ensembles for simulation
        off of a Loihi chip can help with debugging.
      * ``block_shape``: Specifies how this ensemble should be split across
        Loihi neuron cores. See `.BlockShape` for more details.
    `nengo.Connection`
      * ``pop_type``: The axon format when using population spikes, which are only
        used for convolutional connections. Must be either the integer 16 or 32.
        By default, we use ``pop_type`` 32. Using 16 reduces the number of axons
        required by a factor of two, but has more constraints, including on the
        number of channels allowed per block. When using ``pop_type`` 16, we
        recommend to use ``channels_last=True`` and have ``n_filters`` (as well as
        the number of channels per block) be a multiple of 4 on your
        convolutional connections; this will reduce the required synapse memory.

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
    if "block_shape" not in ens_cfg._extra_params:
        ens_cfg.set_param(
            "block_shape", BlockShapeParam("block_shape", default=None, optional=True)
        )

    conn_cfg = config[nengo.Connection]
    if "pop_type" not in conn_cfg._extra_params:
        conn_cfg.set_param("pop_type", Parameter("pop_type", default=32, optional=True))


def set_defaults():
    """Modify Nengo's default parameters for better performance with Loihi.

    The following defaults will be modified:

    `nengo.Ensemble`
      * ``max_rates``: Set to ``Uniform(low=100, high=120)``
      * ``intercepts``: Set to ``Uniform(low=-0.5, high=0.5)``

    `nengo.LIF` and `nengo.SpikingRectifiedLinear`
      * ``initial_voltage``: Set to 0
    """
    nengo.Ensemble.max_rates.default = nengo.dists.Uniform(100, 120)
    nengo.Ensemble.intercepts.default = nengo.dists.Uniform(-1.0, 0.5)
    nengo.LIF.state["voltage"] = nengo.dists.Choice([0])
    nengo.SpikingRectifiedLinear.state["voltage"] = nengo.dists.Choice([0])
