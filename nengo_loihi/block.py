import warnings

from nengo.exceptions import BuildError

import numpy as np


class LoihiBlock(object):
    """Class holding Loihi objects that can be placed on the chip.

    This class can be thought of as a block of the Loihi board, and is how
    Nengo Loihi keeps track of how Loihi Neuron cores will be configured.
    Generally, the job of the build process is to convert Nengo objects
    (ensembles, connections, nodes, and probes) to LoihiBlocks, which
    will then be used by the `.EmulatorInterface` or `.HardwareInterface`.

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
    named_axons : dict {str: Axon}
        Mapping from a name to an Axon object.
    named_synapses : dict {str: Synape}
        Mapping from a name to a Synapse object.
    label : string
        A label for the block (for debugging purposes).
    probes : list of Probe
        Probes recording information from objects in the block.
    synapses : list of Synapse
        Synapse objects projecting to compartments in the block.
    """
    def __init__(self, n_neurons, label=None):
        self.n_neurons = n_neurons
        self.label = label

        self.compartment = Compartment(n_compartments=n_neurons)
        self.axons = []
        self.named_axons = {}
        self.synapses = []
        self.named_synapses = {}
        self.probes = []

    def __str__(self):
        return "%s(%s)" % (
            type(self).__name__, self.label if self.label else '')

    def add_synapse(self, synapse, name=None):
        self.synapses.append(synapse)
        if name is not None:
            assert name not in self.named_synapses
            self.named_synapses[name] = synapse

    def add_axon(self, axon, name=None):
        self.axons.append(axon)
        if name is not None:
            assert name not in self.named_axons
            self.named_axons[name] = axon

    def add_probe(self, probe):
        self.probes.append(probe)


class Profile(object):
    def __eq__(self, obj):
        return isinstance(obj, type(self)) and all(
            self.__dict__[key] == obj.__dict__[key] for key in self.params)

    def __hash__(self):
        return hash(tuple(self.__dict__[key] for key in self.params))


class Compartment(object):
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
    enableNoise : (n,) ndarray
        Whether to enable noise for each compartment.
    decayU : (n,) ndarray
        Input (synapse) decay constant for each compartment.
    decayV : (n,) ndarray
        Voltage decay constant for each compartment.
    label : string
        A label for the block (for debugging purposes).
    n_compartments : int
        The number of compartments in the block.
    noiseAtDenOrVm : {0, 1}
        Inject noise into current (0) or voltage (1).
    noiseExp0 : float or int
        Exponent for noise generation. Floating point values are base 10
        in units of current or voltage. Integer values are in base 2.
    noiseMantOffset0 : float or int
        Offset for noise generation.
    refractDelay : (n,) ndarray
        Compartment refractory delays, in time steps.
    scaleU : bool
        Scale input (U) by decayU so that the integral of U is
        the same before and after filtering.
    scaleV : bool
        Scale voltage (V) by decayV so that the integral of V is
        the same before and after filtering.
    tau_s : float or None
        Time constant used to set decayU. None if decayU has not been set.
    vmax : float or int (range [2**9 - 1, 2**23 - 1])
        Maximum voltage for all compartments, in the same units as ``vth``.
    vmin : float or int (range [-2**23 + 1, 0])
        Minimum voltage for all compartments, in the same units as ``vth``.
    vth : (n,) ndarray
        Compartment voltage thresholds.
    """
    # threshold at which U/V scaling is allowed
    DECAY_SCALE_TH = 0.5 / 2**12  # half of one decay scaling unit

    def __init__(self, n_compartments, label=None):
        self.n_compartments = n_compartments
        self.label = label

        # parameters specific to compartments/block
        self.decayU = np.ones(n_compartments, dtype=np.float32)
        # ^ default to no filter
        self.decayV = np.zeros(n_compartments, dtype=np.float32)
        # ^ default to integration
        self.tau_s = None
        self.scaleU = True
        self.scaleV = False

        self.refractDelay = np.zeros(n_compartments, dtype=np.int32)
        self.vth = np.zeros(n_compartments, dtype=np.float32)
        self.bias = np.zeros(n_compartments, dtype=np.float32)
        self.enableNoise = np.zeros(n_compartments, dtype=bool)

        # parameters common to core
        self.vmin = 0
        self.vmax = np.inf
        self.noiseMantOffset0 = 0
        self.noiseExp0 = 0
        self.noiseAtDendOrVm = 0

    def __str__(self):
        return "%s(%s)" % (
            type(self).__name__, self.label if self.label else '')

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
            warnings.warn("tau_s is already set to %g, which is larger than "
                          "%g. Using %g." % (self.tau_s, tau_s, self.tau_s))
            return
        elif self.tau_s is not None and tau_s > self.tau_s:
            warnings.warn(
                "tau_s is currently %g, which is smaller than %g. Overwriting "
                "tau_s with %g." % (self.tau_s, tau_s, tau_s))
        self._configure_filter(tau_s, dt=dt)
        self.tau_s = tau_s

    def _configure_filter(self, tau_s, dt):
        decayU = 1 if tau_s == 0 else -np.expm1(-dt/np.asarray(tau_s))
        self.decayU[:] = decayU
        self.scaleU = decayU > self.DECAY_SCALE_TH
        if not self.scaleU:
            raise BuildError(
                "Current (U) scaling is required. Perhaps a synapse time "
                "constant is too large in your model.")

    def configure_lif(self, tau_rc=0.02, tau_ref=0.001, vth=1, dt=0.001,
                      min_voltage=0):
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

        self.decayV[:] = -np.expm1(-dt/np.asarray(tau_rc))
        self.refractDelay[:] = np.round(tau_ref / dt) + 1
        self.vth[:] = vth
        self.vmin = min_voltage
        self.vmax = np.inf
        self.scaleV = np.all(self.decayV > self.DECAY_SCALE_TH)
        if not self.scaleV:
            raise BuildError(
                "Voltage (V) scaling is required with LIF neurons. Perhaps "
                "the neuron tau_rc time constant is too large.")

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

        self.decayV[:] = 1.
        self.refractDelay[:] = 1
        self.vth[:] = vth
        self.vmin = 0
        self.vmax = np.inf
        self.scaleV = False

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

        self.decayV[:] = 0.
        self.refractDelay[:] = np.round(tau_ref / dt) + 1
        self.vth[:] = vth
        self.vmin = 0
        self.vmax = np.inf
        self.scaleV = False


class Axon(object):
    """A group of axons targeting a specific Synapse object.

    Attributes
    ----------
    cx_atoms : list of length ``block.n_neurons``
        Atom (weight index) associated with each block compartment.
    cx_to_axon_map : list of length ``block.n_neurons``
        Index of the axon in ``target`` targeted by each block compartment.
    n_axons : int
        The number of outgoing axons.
    target : Synapse
        Target synapses for these axons.
    """

    class Spike(object):
        """A spike targeting a particular axon within a Synapse object.

        The Synapse target is implicit, given by the Axon object that
        creates this Spike.

        Parameters
        ----------
        axon_id : int
            The index of the axon within the targeted Synapse object.
        atom : int, optional (Default: 0)
            An index into the target Synapse weights. This allows spikes
            targeting a particular axon to use different weights.
        """

        __slots__ = ['axon_id', 'atom']

        def __init__(self, axon_id, atom=0):
            self.axon_id = axon_id
            self.atom = atom

        def __repr__(self):
            return "%s(axon_id=%d, atom=%d)" % (
                type(self).__name__, self.axon_id, self.atom)

    def __init__(self, n_axons, label=None):
        self.n_axons = n_axons
        self.label = label

        self.target = None
        self.cx_to_axon_map = None
        self.cx_atoms = None

    def __str__(self):
        return "%s(%s)" % (
            type(self).__name__, self.label if self.label else '')

    @property
    def pop_type(self):
        return self.target.pop_type

    @property
    def slots_per_axon(self):
        """The number of axonCfg slots occupied by each axon."""
        return 2 if self.pop_type == 32 else 1

    def axon_slots(self):
        """The total number of axonCfg slots used by all axons."""
        return self.slots_per_axon * self.n_axons

    def map_cx_axon(self, cx_idxs):
        return (self.cx_to_axon_map[cx_idxs]
                if self.cx_to_axon_map is not None else cx_idxs)

    def map_cx_atoms(self, cx_idxs):
        return (self.cx_atoms[cx_idxs] if self.cx_atoms is not None else
                [0 for _ in cx_idxs])

    def map_cx_spikes(self, cx_idxs):
        axon_ids = self.map_cx_axon(cx_idxs)
        atoms = self.map_cx_atoms(cx_idxs)
        return [self.Spike(axon_id, atom=atom) if axon_id >= 0 else None
                for axon_id, atom in zip(axon_ids, atoms)]

    def set_axon_map(self, cx_to_axon_map, cx_atoms=None):
        self.cx_to_axon_map = cx_to_axon_map
        self.cx_atoms = cx_atoms


class SynapseFmt(Profile):
    INDEX_BITS_MAP = [0, 6, 7, 8, 9, 10, 11, 12]
    WEIGHT_BITS_MAP = [0, 1, 2, 3, 4, 5, 6, 8]

    params = ('wgtLimitMant', 'wgtLimitExp', 'wgtExp', 'discMaxWgt',
              'learningCfg', 'tagBits', 'dlyBits', 'wgtBits',
              'reuseSynData', 'numSynapses', 'cIdxOffset', 'cIdxMult',
              'skipBits', 'idxBits', 'synType', 'fanoutType',
              'compression', 'stdpProfile', 'ignoreDly')

    def __init__(self, wgtLimitMant=0, wgtLimitExp=0, wgtExp=0, discMaxWgt=0,
                 learningCfg=0, tagBits=0, dlyBits=0, wgtBits=0,
                 reuseSynData=0, numSynapses=0, cIdxOffset=0, cIdxMult=0,
                 skipBits=0, idxBits=0, synType=0, fanoutType=0,
                 compression=0, stdpProfile=0, ignoreDly=0):
        self.wgtLimitMant = wgtLimitMant
        self.wgtLimitExp = wgtLimitExp
        self.wgtExp = wgtExp
        self.discMaxWgt = discMaxWgt
        self.learningCfg = learningCfg
        self.tagBits = tagBits
        self.dlyBits = dlyBits
        self.wgtBits = wgtBits
        self.reuseSynData = reuseSynData
        self.numSynapses = numSynapses
        self.cIdxOffset = cIdxOffset
        self.cIdxMult = cIdxMult
        self.skipBits = skipBits
        self.idxBits = idxBits
        self.synType = synType
        self.fanoutType = fanoutType
        self.compression = compression
        self.stdpProfile = stdpProfile
        self.ignoreDly = ignoreDly

    @classmethod
    def get_realWgtExp(cls, wgtExp):
        return 6 + wgtExp

    @classmethod
    def get_scale(cls, wgtExp):
        return 2**cls.get_realWgtExp(wgtExp)

    @property
    def isMixed(self):
        return self.fanoutType == 1

    @property
    def realIdxBits(self):
        return self.INDEX_BITS_MAP[self.idxBits]

    @property
    def realWgtBits(self):
        return self.WEIGHT_BITS_MAP[self.wgtBits]

    @property
    def realWgtExp(self):
        return self.get_realWgtExp(self.wgtExp)

    @property
    def scale(self):
        return self.get_scale(self.wgtExp)

    @property
    def shift_bits(self):
        """Number of bits the -256..255 weight is right-shifted by."""
        return 8 - self.realWgtBits + self.isMixed

    def bits_per_axon(self, n_weights):
        """For an axon with n weights, compute the weight memory bits used"""
        bits_per_weight = self.realWgtBits + self.dlyBits + self.tagBits
        if self.compression == 0:
            bits_per_weight += self.realIdxBits
        elif self.compression == 3:
            pass
        else:
            raise NotImplementedError("Compression %s" % (self.compression,))

        SYNAPSE_FMT_IDX_BITS = 4
        N_SYNAPSES_BITS = 6
        bits = 0
        synapses_per_block = self.numSynapses + 1
        for i in range(0, n_weights, synapses_per_block):
            n = min(n_weights - i, synapses_per_block)
            bits_i = n*bits_per_weight + SYNAPSE_FMT_IDX_BITS + N_SYNAPSES_BITS
            bits_i = -64 * (-bits_i // 64)
            # ^ round up to nearest 64 (size of one int64 memory unit)
            bits += bits_i

        return bits

    def set(self, **kwargs):
        for key, value in kwargs.items():
            assert hasattr(self, key)
            setattr(self, key, value)


class Synapse(object):
    """A group of Loihi synapses that share some properties.

    Attributes
    ----------
    axon_cx_bases : list or None
        List providing ax cx_base (compartment offset) for each input axon.
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
    synapse_fmt : SynapseFmt
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
        self.synapse_fmt = None
        self.weights = None
        self.indices = None
        self.axon_cx_bases = None
        self.axon_to_weight_map = None

        self.learning = False
        self.learning_rate = 1.
        self.learning_wgt_exp = None
        self.tracing_tau = None
        self.tracing_mag = None
        self.pop_type = 0  # one of (0, 16, 32) for discrete, pop16, pop32

    def __str__(self):
        return "%s(%s)" % (
            type(self).__name__, self.label if self.label else '')

    def atom_bits(self):
        max_populations = max(w.shape[0] for w in self.weights)
        return int(np.ceil(np.log2(max_populations)))

    def atom_bits_extra(self):
        atom_bits = self.atom_bits()
        assert atom_bits <= 9, "Cannot have more than 9 atom bits"
        return max(atom_bits - 5, 0)  # has 5 bits by default

    def axon_bits(self):
        if self.pop_type == 16:
            return 10 - self.atom_bits_extra()
        else:
            return 12

    def axon_cx_base(self, axon_idx):
        if self.axon_cx_bases is None:
            return 0
        cx_base = self.axon_cx_bases[axon_idx]
        return cx_base if cx_base > -1024 else None

    def axon_populations(self, axon_idx):
        weight_idx = self.axon_weight_idx(axon_idx)
        return self.weights[weight_idx].shape[0]

    def axon_weight_idx(self, axon_idx):
        return (self.axon_to_weight_map[axon_idx]
                if self.axon_to_weight_map is not None else axon_idx)

    def axon_weights_indices(self, axon_idx, atom=0):
        weight_idx = self.axon_weight_idx(axon_idx)
        w = self.weights[weight_idx]
        i = self.indices[weight_idx]
        return w[atom, :], i[atom, :]

    def bits(self):
        return sum(self.synapse_fmt.bits_per_axon(w.size)
                   for w in self.weights)

    def format(self, **kwargs):
        if self.synapse_fmt is None:
            self.synapse_fmt = SynapseFmt()
        self.synapse_fmt.set(**kwargs)

    def idx_bits(self):
        idxBits = int(np.ceil(np.log2(self.max_ind() + 1)))
        assert idxBits <= SynapseFmt.INDEX_BITS_MAP[-1], (
            "idxBits out of range, ensemble too large?")
        idxBits = next(i for i, v in enumerate(SynapseFmt.INDEX_BITS_MAP)
                       if v >= idxBits)
        return idxBits

    def idxs_per_synapse(self):
        return 2 if self.learning else 1

    def max_abs_weight(self):
        return max(np.abs(w).max() if w.size > 0 else -np.inf
                   for w in self.weights)

    def max_ind(self):
        return max(i.max() if len(i) > 0 else -1 for i in self.indices)

    def _set_weights_indices(self, weights, indices=None):
        weights = [np.array(w, copy=False, dtype=np.float32, ndmin=2)
                   for w in weights]
        assert all(w.ndim == 2 for w in weights), (
            "Weights must be shape (n_axons,) (n_populations, n_compartments)")
        assert all(w.shape[0] == weights[0].shape[0] for w in weights), (
            "All axon weights must have the same number of populations")
        self.weights = weights

        if indices is None:
            indices = [np.zeros((w.shape[0], 1), dtype=np.int32)
                       + np.arange(w.shape[1], dtype=np.int32)
                       for w in self.weights]
        indices = [np.array(i, copy=False, dtype=np.int32, ndmin=2)
                   for i in indices]
        assert all(i.ndim == 2 for i in indices), (
            "Indices must be shape (n_axons,) (n_populations, n_compartments)")
        assert all(i.shape == w.shape for i, w in zip(indices, weights)), (
            "Indices shapes must match weights shapes")
        assert len(weights) == len(indices)
        self.indices = indices

    def set_diagonal_weights(self, diag):
        weights = diag.ravel()
        indices = list(range(len(weights)))
        self._set_weights_indices(weights, indices)
        assert len(self.weights) == self.n_axons

        idxBits = self.idx_bits()
        self.format(compression=3, idxBits=idxBits, fanoutType=1,
                    numSynapses=63, wgtBits=7)

    def set_full_weights(self, weights):
        self._set_weights_indices(weights)
        assert len(self.weights) == self.n_axons, (
            "Full weights must have different weights for each axon")

        idxBits = self.idx_bits()
        self.format(compression=3, idxBits=idxBits, fanoutType=1,
                    numSynapses=63, wgtBits=7)

    def set_learning(
            self, learning_rate=1., tracing_tau=2, tracing_mag=1.0, wgt_exp=4):
        assert tracing_tau == int(tracing_tau), "tracing_tau must be integer"

        self.learning = True
        self.tracing_tau = int(tracing_tau)
        self.tracing_mag = tracing_mag
        self.format(learningCfg=1, stdpProfile=0)
        # ^ stdpProfile hard-coded for now (see hardware.builder)

        self.train_epoch = 2
        self.learn_epoch_k = 1
        self.learn_epoch = self.train_epoch * 2**self.learn_epoch_k

        self.learning_rate = learning_rate * self.learn_epoch
        self.learning_wgt_exp = wgt_exp

    def set_population_weights(
            self,
            weights,
            indices,
            axon_to_weight_map,
            cx_bases,
            pop_type=None
    ):
        self._set_weights_indices(weights, indices)
        self.axon_to_weight_map = axon_to_weight_map
        self.axon_cx_bases = cx_bases
        self.pop_type = 16 if pop_type is None else pop_type

        idxBits = self.idx_bits()
        self.format(compression=0,
                    idxBits=idxBits,
                    fanoutType=1,
                    numSynapses=63,
                    wgtBits=7)

    def size(self):
        return sum(w.size for w in self.weights)


class Probe(object):
    _slice = slice

    def __init__(self, target=None, key=None, slice=None, weights=None,
                 synapse=None):
        self.target = target
        self.key = key
        self.slice = slice if slice is not None else self._slice(None)
        self.weights = weights
        self.synapse = synapse
        self.use_snip = False
        self.snip_info = None
