import warnings
from collections import OrderedDict

import numpy as np
import scipy.sparse
from nengo.exceptions import BuildError

from nengo_loihi.nxsdk_obfuscation import d

MAX_COMPARTMENTS = d(b"MTAyNA==", int)
MAX_IN_AXONS = d(b"NDA5Ng==", int)
MAX_OUT_AXONS = d(b"NDA5Ng==", int)
MAX_SYNAPSE_BITS = d(b"MTA0ODU3Ng==", int)


class LoihiBlock:
    """Class holding Loihi objects that can be placed on the chip.

    This class can be thought of as a block of the Loihi board, and is how
    NengoLoihi keeps track of how Loihi Neuron cores will be configured.
    Generally, the job of the build process is to convert Nengo objects
    (ensembles, connections, and nodes) to LoihiBlocks, which will then
    be used by the `.EmulatorInterface` or `.HardwareInterface`.

    Initially, parameters in a LoihiBlock are floating point values.
    The `.discretize_block` function converts them to integer values inplace
    for use on Loihi.

    Attributes
    ----------
    axons : list of Axon
        Axon objects outputting from the compartments in the block.
    compartment : Compartment
        Compartment object representing all compartments in the block.
    n_neurons : int
        The number of neurons in the block.
    named_synapses : dict {str: Synape}
        Mapping from a name to a Synapse object.
    label : string
        A label for the block (for debugging purposes).
    synapses : list of Synapse
        Synapse objects projecting to compartments in the block.
    """

    def __init__(self, n_neurons, label=None):
        self.n_neurons = n_neurons
        self.label = label

        self.compartment = Compartment(n_compartments=n_neurons, label=label)
        self.axons = []
        self.synapses = []
        self.named_synapses = {}

    def __str__(self):
        return "%s(%s)" % (type(self).__name__, self.label if self.label else "")

    def add_synapse(self, synapse, name=None):
        self.synapses.append(synapse)
        if name is not None:
            assert name not in self.named_synapses
            self.named_synapses[name] = synapse

    def add_axon(self, axon):
        self.axons.append(axon)

    def utilization(self):
        """Measure utilization of core resources by the block.

        Returns
        -------
        utilization : OrderedDict
            "compartments": The (current, max) number of compartments used.
            "in-axons": The (current, max) number of input axons used.
            "out-axons": The (current, max) number of output axons used.
            "synapses": The (current, max) amount of synapse memory used.
        """
        n_compartments = self.compartment.n_compartments
        n_in_axons = sum(s.n_axons for s in self.synapses)
        n_out_axons = sum(a.axon_slots() for a in self.axons)
        n_synapse_bits = sum(s.bits() for s in self.synapses)
        return OrderedDict(
            [
                ("compartments", (n_compartments, MAX_COMPARTMENTS)),
                ("in-axons", (n_in_axons, MAX_IN_AXONS)),
                ("out-axons", (n_out_axons, MAX_OUT_AXONS)),
                ("synapses", (n_synapse_bits, MAX_SYNAPSE_BITS)),
            ]
        )


class Config:
    def __eq__(self, obj):
        return isinstance(obj, type(self)) and all(
            self.__dict__[key] == obj.__dict__[key] for key in self.params
        )

    def __hash__(self):
        return hash(tuple(self.__dict__[key] for key in self.params))


class Compartment:
    """Stores information for configuring Loihi compartments.

    The information stored here will be associated with some block,
    and all compartments will share certain information.
    While compartments are usually thought of neurons, we use compartments
    to implement Nengo ensembles, nodes, and connection through special
    decode neurons.

    Before `.discretize_compartment` has been called, most attributes in
    this class are floating-point values. Calling `.discretize_compartment`
    converts them to integer values inplace for use on Loihi.

    Attributes
    ----------
    bias : (n,) ndarray
        Compartment biases.
    enable_noise : (n,) ndarray
        Whether to enable noise for each compartment.
    decay_u : (n,) ndarray
        Input (synapse) decay constant for each compartment.
    decay_v : (n,) ndarray
        Voltage decay constant for each compartment.
    label : string
        A label for the block (for debugging purposes).
    n_compartments : int
        The number of compartments in the block.
    noise_at_membrane : {0, 1}
        Inject noise into current (0) or voltage (1).
    noise_exp : float or int
        Exponent for noise generation. Floating point values are base 10
        in units of current or voltage. Integer values are in base 2.
    noise_offset : float or int
        Offset for noise generation.
    refract_delay : (n,) ndarray
        Compartment refractory delays, in time steps.
    scale_u : bool
        Scale input (U) by decay_u so that the integral of U is
        the same before and after filtering.
    scale_v : bool
        Scale voltage (V) by decay_v so that the integral of V is
        the same before and after filtering.
    tau_s : float or None
        Time constant used to set decay_u. None if decay_u has not been set.
    vmax : float or int (range [2**9 - 1, 2**23 - 1])
        Maximum voltage for all compartments, in the same units as ``vth``.
    vmin : float or int (range [-2**23 + 1, 0])
        Minimum voltage for all compartments, in the same units as ``vth``.
    vth : (n,) ndarray
        Compartment voltage thresholds.
    """

    # threshold at which U/V scaling is allowed
    DECAY_SCALE_TH = 0.5 / d(b"NDA5Ng==", int)  # half of decay scaling unit

    def __init__(self, n_compartments, label=None):
        self.n_compartments = n_compartments
        self.label = label

        # parameters specific to compartments/block
        self.decay_u = np.ones(n_compartments, dtype=np.float32)
        # ^ default to no filter
        self.decay_v = np.zeros(n_compartments, dtype=np.float32)
        # ^ default to integration
        self.tau_s = None
        self.scale_u = True
        self.scale_v = False

        self.refract_delay = np.zeros(n_compartments, dtype=np.int32)
        self.vth = np.zeros(n_compartments, dtype=np.float32)
        self.bias = np.zeros(n_compartments, dtype=np.float32)
        self.enable_noise = np.zeros(n_compartments, dtype=bool)

        # parameters common to core
        self.vmin = 0
        self.vmax = np.inf
        self.noise_offset = 0
        self.noise_exp = 0
        self.noise_at_membrane = 0

    def __str__(self):
        return "%s(%s)" % (type(self).__name__, self.label if self.label else "")

    def configure_default_filter(self, tau_s, dt=0.001):
        """Set the default Lowpass synaptic input filter for compartments.

        Parameters
        ----------
        tau_s : float
            `nengo.Lowpass` synapse time constant for filtering.
        dt : float
            Simulator time step.
        """
        if self.tau_s is None:  # don't overwrite a non-default filter
            self._configure_filter(tau_s, dt=dt)

    def configure_filter(self, tau_s, dt=0.001):
        """Set Lowpass synaptic input filter for compartments.

        Parameters
        ----------
        tau_s : float
            `nengo.Lowpass` synapse time constant for filtering.
        dt : float
            Simulator time step.
        """
        if self.tau_s is not None and tau_s < self.tau_s:
            warnings.warn(
                "tau_s is already set to %g, which is larger than "
                "%g. Using %g." % (self.tau_s, tau_s, self.tau_s)
            )
            return
        elif self.tau_s is not None and tau_s > self.tau_s:
            warnings.warn(
                "tau_s is currently %g, which is smaller than %g. Overwriting "
                "tau_s with %g." % (self.tau_s, tau_s, tau_s)
            )
        self._configure_filter(tau_s, dt=dt)
        self.tau_s = tau_s

    def _configure_filter(self, tau_s, dt):
        decay_u = 1 if tau_s == 0 else -(np.expm1(-dt / np.asarray(tau_s)))
        self.decay_u[:] = decay_u
        self.scale_u = decay_u > self.DECAY_SCALE_TH
        if not self.scale_u:
            raise BuildError(
                "Current (U) scaling is required. Perhaps a synapse time "
                "constant is too large in your model."
            )

    def configure_lif(self, tau_rc=0.02, tau_ref=0.001, vth=1, dt=0.001, min_voltage=0):
        """Configure these compartments as individual LIF neurons.

        Parameters
        ----------
        tau_rc : float
            Membrane time constant (in seconds) of the neurons.
        tau_ref : float
            Refractory period (in seconds) of the neurons.
        vth : float
            Voltage firing threshold of the neurons.
        dt : float
            Simulator time step length (in seconds).
        min_voltage : float
            The minimum voltage for the neurons.
        """

        self.decay_v[:] = -(np.expm1(-dt / np.asarray(tau_rc)))
        self.refract_delay[:] = np.round(tau_ref / dt) + 1
        self.vth[:] = vth
        self.vmin = min_voltage
        self.vmax = np.inf
        self.scale_v = np.all(self.decay_v > self.DECAY_SCALE_TH)
        if not self.scale_v:
            raise BuildError(
                "Voltage (V) scaling is required with LIF neurons. Perhaps "
                "the neuron tau_rc time constant is too large."
            )

    def configure_nonspiking(self, tau_ref=0.0, vth=1, dt=0.001):
        """Configure these compartments as individual non-spiking neurons.

        Parameters
        ----------
        tau_ref : float
            Refractory period (in seconds) of the neurons.
        vth : float
            Voltage firing threshold of the neurons.
        dt : float
            Simulator time step length (in seconds).
        """

        self.decay_v[:] = 1.0
        self.refract_delay[:] = 1
        self.vth[:] = vth
        self.vmin = 0
        self.vmax = np.inf
        self.scale_v = False

    def configure_relu(self, tau_ref=0.0, vth=1, dt=0.001):
        """Configure these compartments as individual Rectified Linear neurons.

        These are also known as non-leaky integrate-and-fire neurons. The
        voltage is the integral of the input current.

        Parameters
        ----------
        tau_ref : float
            Refractory period (in seconds) of the neurons.
        vth : float
            Voltage firing threshold of the neurons.
        dt : float
            Simulator time step length (in seconds).
        """

        self.decay_v[:] = 0.0
        self.refract_delay[:] = np.round(tau_ref / dt) + 1
        self.vth[:] = vth
        self.vmin = 0
        self.vmax = np.inf
        self.scale_v = False


class Axon:
    """A group of axons targeting a specific Synapse object.

    Attributes
    ----------
    compartment_atoms : list of length ``block.n_neurons``
        Atom (weight index) associated with each block compartment.
    compartment_map : list of length ``block.n_neurons``
        Index of the axon in ``target`` targeted by each block compartment.
    n_axons : int
        The number of outgoing axons.
    target : Synapse
        Target synapses for these axons.
    """

    class Spike:
        """A spike targeting a particular axon within a Synapse object.

        The Synapse target is implicit, given by the Axon object that
        creates this Spike.

        Parameters
        ----------
        axon_idx : int
            The index of the axon within the targeted Synapse object.
        atom : int, optional (Default: 0)
            An index into the target Synapse weights. This allows spikes
            targeting a particular axon to use different weights.
        """

        __slots__ = ["axon_idx", "atom"]

        def __init__(self, axon_idx, atom=0):
            self.axon_idx = axon_idx
            self.atom = atom

        def __repr__(self):
            return "%s(axon_idx=%d, atom=%d)" % (
                type(self).__name__,
                self.axon_idx,
                self.atom,
            )

    def __init__(self, n_axons, label=None):
        self.n_axons = n_axons
        self.label = label

        self.target = None
        self.compartment_map = None
        self.compartment_atoms = None

    def __str__(self):
        return "%s(%s)" % (type(self).__name__, self.label if self.label else "")

    @property
    def pop_type(self):
        return self.target.pop_type

    @property
    def slots_per_axon(self):
        """The number of axon_cfg slots occupied by each axon."""
        return 2 if self.pop_type == 32 else 1

    def axon_slots(self):
        """The total number of axon_cfg slots used by all axons."""
        return self.slots_per_axon * self.n_axons

    def map_axon(self, compartment_idxs):
        return (
            self.compartment_map[compartment_idxs]
            if self.compartment_map is not None
            else compartment_idxs
        )

    def map_atoms(self, compartment_idxs):
        return (
            self.compartment_atoms[compartment_idxs]
            if self.compartment_atoms is not None
            else [0 for _ in compartment_idxs]
        )

    def map_spikes(self, compartment_idxs):
        axon_ids = self.map_axon(compartment_idxs)
        atoms = self.map_atoms(compartment_idxs)
        return [
            self.Spike(axon_id, atom=atom) if axon_id >= 0 else None
            for axon_id, atom in zip(axon_ids, atoms)
        ]

    def set_compartment_axon_map(self, target_axons, atoms=None):
        """Set mapping from compartments to axons in target.

        Parameters
        ----------
        target_axons : array_like (``n_compartments``,)
            Indices indicating which target axon each compartment maps to.
            If < 0, the corresponding compartment will not be used with these
            axons.
        atoms : array_like (``n_compartments``,)
            Atoms to use for each compartment. Use only if ``pop_type != 0``.
        """
        self.compartment_map = target_axons
        self.compartment_atoms = atoms


class SynapseConfig(Config):
    INDEX_BITS_MAP = d(b"WzAsIDYsIDcsIDgsIDksIDEwLCAxMSwgMTJd", "list_int")
    WEIGHT_BITS_MAP = d(b"WzAsIDEsIDIsIDMsIDQsIDUsIDYsIDhd", "list_int")

    params = (
        "weight_limit_mant",
        "weight_limit_exp",
        "weight_exp",
        "disc_max_weight",
        "learning_cfg",
        "tag_bits",
        "delay_bits",
        "weight_bits",
        "reuse_synapse_data",
        "n_synapses",
        "idx_offset",
        "idx_mult",
        "skip_bits",
        "idx_bits",
        "synapse_type",
        "fanout_type",
        "compression",
        "stdp_cfg",
        "ignore_delay",
    )

    def __init__(
        self,
        weight_limit_mant=0,
        weight_limit_exp=0,
        weight_exp=0,
        disc_max_weight=0,
        learning_cfg=0,
        tag_bits=0,
        delay_bits=0,
        weight_bits=0,
        reuse_synapse_data=0,
        n_synapses=0,
        idx_offset=0,
        idx_mult=0,
        skip_bits=0,
        idx_bits=0,
        synapse_type=0,
        fanout_type=0,
        compression=0,
        stdp_cfg=0,
        ignore_delay=0,
    ):
        self.weight_limit_mant = weight_limit_mant
        self.weight_limit_exp = weight_limit_exp
        self.weight_exp = weight_exp
        self.disc_max_weight = disc_max_weight
        self.learning_cfg = learning_cfg
        self.tag_bits = tag_bits
        self.delay_bits = delay_bits
        self.weight_bits = weight_bits
        self.reuse_synapse_data = reuse_synapse_data
        self.n_synapses = n_synapses
        self.idx_offset = idx_offset
        self.idx_mult = idx_mult
        self.skip_bits = skip_bits
        self.idx_bits = idx_bits
        self.synapse_type = synapse_type
        self.fanout_type = fanout_type
        self.compression = compression
        self.stdp_cfg = stdp_cfg
        self.ignore_delay = ignore_delay

    @classmethod
    def get_real_weight_exp(cls, weight_exp):
        return d(b"Ng==", int) + weight_exp

    @classmethod
    def get_scale(cls, weight_exp):
        return 2 ** cls.get_real_weight_exp(weight_exp)

    @property
    def is_mixed(self):
        return self.fanout_type == 1

    @property
    def real_idx_bits(self):
        return self.INDEX_BITS_MAP[self.idx_bits]

    @property
    def real_weight_bits(self):
        return self.WEIGHT_BITS_MAP[self.weight_bits]

    @property
    def real_weight_exp(self):
        return self.get_real_weight_exp(self.weight_exp)

    @property
    def scale(self):
        return self.get_scale(self.weight_exp)

    @property
    def shift_bits(self):
        """Number of bits the weight is right-shifted by."""
        return d(b"OA==", int) - self.real_weight_bits + self.is_mixed

    def bits_per_axon(self, n_weights):
        """For an axon with n weights, compute the weight memory bits used"""
        bits_per_weight = self.real_weight_bits + self.delay_bits + self.tag_bits
        if self.compression == d(b"MA==", int):
            bits_per_weight += self.real_idx_bits
        elif self.compression == d(b"Mw==", int):
            pass
        else:
            raise NotImplementedError("Compression %s" % (self.compression,))

        synapse_idx_bits = d(b"NA==", int)
        n_synapses_bits = d(b"Ng==", int)
        bits = 0
        synapses_per_block = self.n_synapses + 1
        for i in range(0, n_weights, synapses_per_block):
            n = min(n_weights - i, synapses_per_block)
            bits_i = n * bits_per_weight + synapse_idx_bits + n_synapses_bits
            # round up to nearest memory unit
            bits_i = -d(b"NjQ=", int) * (-bits_i // d(b"NjQ=", int))
            bits += bits_i

        return bits

    def set(self, **kwargs):
        for key, value in kwargs.items():
            assert hasattr(self, key)
            setattr(self, key, value)


class Synapse:
    """A group of Loihi synapses that share some properties.

    Attributes
    ----------
    axon_compartment_bases : list or None
        List providing base (compartment offset) for each input axon.
    axon_to_weight_map : dict or None
        Map from input axon index to weight index, to allow weights to be
        re-used by axons. If None, the weight index for an input axon is the
        axon index.
    indices : (population, axon, compartment) ndarray
        The synapse indices.
    learning : bool
        Whether synaptic tracing and learning is enabled for these synapses.
    learning_rate : float
        The learning rate.
    learning_wgt_exp : int
        The weight exponent used on this connection if learning is enabled.
    n_axons : int
        Number of input axons to this group of synapses.
    pop_type : int (0, 16, 32)
        Whether these synapses are discrete (0), pop16, or pop32. This
        determines the type of axons these synapses can connect to.
    synapse_cfg : SynapseConfig
        The synapse format object for these synapses.
    tracing_mag : float
        Magnitude by which the learning trace is increased for each spike.
    tracing_tau : int
        Decay time constant for the learning trace, in timesteps (not seconds).
    weights : (n_axons,) list of (n_populations, n_compartments) ndarray
        The synapse weights. Organized as a list of arrays so each axon
        can have a different number of target compartments.
    """

    def __init__(self, n_axons, label=None):
        self.n_axons = n_axons
        self.label = label
        self.synapse_cfg = None
        self.weights = None
        self.indices = None
        self.axon_compartment_bases = None
        self.axon_to_weight_map = None

        self.learning = False
        self.learning_rate = 1.0
        self.learning_wgt_exp = None
        self.tracing_tau = None
        self.tracing_mag = None
        self.pop_type = 0  # one of (0, 16, 32) for discrete, pop16, pop32

    def __str__(self):
        return "%s(%s)" % (type(self).__name__, self.label if self.label else "")

    def atom_bits(self):
        """Number of bits needed to represent the atom for incoming spikes."""
        max_populations = max(w.shape[0] for w in self.weights)
        return int(np.ceil(np.log2(max_populations)))

    def atom_bits_extra(self):
        """Number of extra bits needed for the atom for incoming pop16 spikes."""
        if self.pop_type == 16:
            atom_bits = self.atom_bits()
            assert atom_bits <= d(b"OQ==", int), "Too many atom bits"
            return max(atom_bits - d(b"NQ==", int), 0)
        else:
            return 0  # meaningless if pop_type != 16

    def axon_bits(self):
        """Number of bits available to represent the target axon on incoming spikes."""
        if self.pop_type == 16:
            return d(b"MTA=", int) - self.atom_bits_extra()
        else:
            return d(b"MTI=", int)

    def axon_compartment_base(self, axon_idx):
        """Offset for compartment indices for a particular axon.

        A return value of ``None`` indicates the axon is unused.
        """
        if self.axon_compartment_bases is None:
            return 0
        base = self.axon_compartment_bases[axon_idx]

        # negative indicates unused axon
        return base if base >= 0 else None

    def axon_populations(self, axon_idx):
        """Number of populations (atom values) for a particular axon."""
        weight_idx = self.axon_weight_idx(axon_idx)
        return self.weights[weight_idx].shape[0]

    def axon_weight_idx(self, axon_idx):
        """Index of weights in weight array for a particular axon."""
        return (
            self.axon_to_weight_map[axon_idx]
            if self.axon_to_weight_map is not None
            else axon_idx
        )

    def axon_weights_indices(self, axon_idx, atom=0):
        """The weights and indices for a particular axon (and atom, if applicable)."""
        weight_idx = self.axon_weight_idx(axon_idx)
        w = self.weights[weight_idx]
        i = self.indices[weight_idx]
        return w[atom, :], i[atom, :]

    def bits(self):
        """The total number of bits used by all weights in this Synapse."""
        return sum(self.synapse_cfg.bits_per_axon(w.size) for w in self.weights)

    def format(self, **kwargs):
        """Modify the SynapseConfig format of this Synapse."""
        if self.synapse_cfg is None:
            self.synapse_cfg = SynapseConfig()
        self.synapse_cfg.set(**kwargs)

    def idx_bits(self):
        """The number of index bits required for each weight entry."""
        bits = int(np.ceil(np.log2(self.max_ind() + 1)))
        if bits <= SynapseConfig.INDEX_BITS_MAP[-1]:
            return next(
                i for i, v in enumerate(SynapseConfig.INDEX_BITS_MAP) if v >= bits
            )
        else:  # pragma: no cover
            # number of bits is actually out of range, we need to split this ensemble
            # before it goes on the chip. Use -1 as a placeholder.
            return -1

    def idxs_per_synapse(self):
        """The number of axon indices (slots) required for each incoming axon."""
        return d(b"Mg==", int) if self.learning else d(b"MQ==", int)

    def max_abs_weight(self):
        """The maximum absolute value of all the weights in this Synapse."""
        return max(np.abs(w).max() if w.size > 0 else -np.inf for w in self.weights)

    def max_ind(self):
        """The maximum compartment index in weight memory.

        Does not include ``axon_compartment_base``.
        """
        return max(i.max() if i.size > 0 else -1 for i in self.indices)

    def _set_weights_indices(
        self,
        weights,
        indices=None,
        weight_dtype=np.float32,
        compression=d(b"MA==", int),
    ):
        weights = [
            np.array(w, copy=False, dtype=weight_dtype, ndmin=2) for w in weights
        ]
        assert all(
            w.ndim == 2 for w in weights
        ), "Weights must be shape (n_axons,) (n_populations, n_compartments)"
        assert all(
            w.shape[0] == weights[0].shape[0] for w in weights
        ), "All axon weights must have the same number of populations"
        self.weights = weights

        if indices is None:
            indices = [
                np.zeros((w.shape[0], 1), dtype=np.int32)
                + np.arange(w.shape[1], dtype=np.int32)
                for w in self.weights
            ]
        indices = [np.array(i, copy=False, dtype=np.int32, ndmin=2) for i in indices]
        assert all(
            i.ndim == 2 for i in indices
        ), "Indices must be shape (n_axons,) (n_populations, n_compartments)"
        assert all(
            i.shape == w.shape for i, w in zip(indices, weights)
        ), "Indices shapes must match weights shapes"
        assert len(weights) == len(indices)
        self.indices = indices

        self.format(
            compression=compression,
            idx_bits=self.idx_bits(),
            fanout_type=d(b"MQ==", int),
            n_synapses=d(b"NjM=", int),
            weight_bits=d(b"Nw==", int),
        )

    def set_weights(self, weights):
        """Set dense or sparse weights on this Synapse."""
        if isinstance(weights, scipy.sparse.spmatrix):
            csr = weights.tocsr()
            weights_by_row, idxs_by_row = [], []
            for i in range(weights.shape[0]):
                i0, i1 = csr.indptr[i : i + 2]
                weights_by_row.append(csr.data[i0:i1])
                idxs_by_row.append(csr.indices[i0:i1])

            weights = weights_by_row
            indices = idxs_by_row
        else:
            weights = np.array(weights, copy=False, dtype=np.float32)
            assert weights.ndim == 2
            indices = None

        assert len(weights) == self.n_axons, "Must have different weights for each axon"
        self._set_weights_indices(weights, indices=indices, compression=d(b"Mw==", int))

    def set_learning(
        self, learning_rate=1.0, tracing_tau=2, tracing_mag=1.0, wgt_exp=4
    ):
        """Set the learning parameters for this Synapse."""
        assert tracing_tau == int(tracing_tau), "tracing_tau must be integer"

        self.learning = True
        self.tracing_tau = int(tracing_tau)
        self.tracing_mag = tracing_mag
        # stdp_cfg hard-coded for now (see hardware.builder)
        self.format(learning_cfg=d(b"MQ==", int), stdp_cfg=d(b"MA==", int))

        self.train_epoch = 2
        self.learn_epoch_k = 1
        self.learn_epoch = self.train_epoch * 2 ** self.learn_epoch_k

        self.learning_rate = learning_rate * self.learn_epoch
        self.learning_wgt_exp = wgt_exp

    def set_population_weights(
        self, weights, indices, axon_to_weight_map, compartment_bases, pop_type=None
    ):
        """Set population weights on this Synapse."""
        self.axon_to_weight_map = axon_to_weight_map
        self.axon_compartment_bases = compartment_bases
        self.pop_type = 16 if pop_type is None else pop_type

        self._set_weights_indices(weights, indices=indices, compression=d(b"MA==", int))

    def size(self):
        return sum(w.size for w in self.weights)
