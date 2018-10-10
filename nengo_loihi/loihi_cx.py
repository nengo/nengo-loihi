from __future__ import division

import collections
import logging
import warnings

import numpy as np
import nengo
from nengo.exceptions import BuildError, SimulationError
from nengo.utils.compat import is_integer, is_iterable

from nengo_loihi.loihi_api import (
    BIAS_MAX,
    bias_to_manexp,
    overflow_signed,
    SynapseFmt,
    tracing_mag_int_frac,
    Q_BITS, U_BITS,
    VTH_MAX,
    vth_to_manexp,
)

logger = logging.getLogger(__name__)


class CxGroup(object):
    """Class holding Loihi objects that can be placed on the chip or Lakemont.

    Typically an ensemble or node, can be a special decoding ensemble. Once
    implemented, SNIPS might use this as well.

    Before ``discretize`` has been called, most parameters in this class are
    floating-point values. Calling ``discretize`` converts them to integer
    values inplace, for use on Loihi.

    Attributes
    ----------
    n : int
        The number of compartments in the group.
    label : string
        A label for the group (for debugging purposes).
    decayU : (n,) ndarray
        Input (synapse) decay constant for each compartment.
    decayV : (n,) ndarray
        Voltage decay constant for each compartment.
    tau_s : float or None
        Time constant used to set decayU. None if decayU has not been set.
    scaleU : bool
        Scale input (U) by decayU so that the integral of U is
        the same before and after filtering.
    scaleV : bool
        Scale voltage (V) by decayV so that the integral of V is
        the same before and after filtering.
    refractDelay : (n,) ndarray
        Compartment refractory delays, in time steps.
    vth : (n,) ndarray
        Compartment voltage thresholds.
    bias : (n,) ndarray
        Compartment biases.
    enableNoise : (n,) ndarray
        Whether to enable noise for each compartment.
    vmin : float or int (range [-2**23 + 1, 0])
        Minimum voltage for all compartments, in Loihi voltage units.
    vmax : float or int (range [2**9 - 1, 2**23 - 1])
        Maximum voltage for all compartments, in Loihi voltage units.
    noiseMantOffset0 : float or int
        Offset for noise generation.
    noiseExp0 : float or int
        Exponent for noise generation. Floating point values are base 10
        in units of current or voltage. Integer values are in base 2.
    noiseAtDenOrVm : {0, 1}
        Inject noise into current (0) or voltage (1).
    synapses : list of CxSynapse
        CxSynapse objects projecting to these compartments.
    named_synapses : dict
        Dictionary mapping names to CxSynapse objects.
    axons : list of CxAxon
        CxAxon objects outputting from these compartments.
    named_axons : dict
        Dictionary mapping names to CxAxon objects.
    probes : list of CxProbe
        CxProbes recording information from these compartments.
    location : {"core", "cpu"}
        Whether these compartments are on a Loihi core
        or handled by the Loihi x86 processor (CPU).
    """
    # threshold at which U/V scaling is allowed
    DECAY_SCALE_TH = 0.5 / 2**12  # half of one decay scaling unit

    def __init__(self, n, label=None, location='core'):
        self.n = n
        self.label = label

        self.decayU = np.ones(n, dtype=np.float32)  # default to no filter
        self.decayV = np.zeros(n, dtype=np.float32)  # default to integration
        self.tau_s = None
        self.scaleU = True
        self.scaleV = False

        self.refractDelay = np.zeros(n, dtype=np.int32)
        self.vth = np.zeros(n, dtype=np.float32)
        self.bias = np.zeros(n, dtype=np.float32)
        self.enableNoise = np.zeros(n, dtype=bool)

        # parameters common to core
        self.vmin = 0
        self.vmax = np.inf
        self.noiseMantOffset0 = 0
        self.noiseExp0 = 0
        self.noiseAtDendOrVm = 0

        self.synapses = []
        self.named_synapses = {}
        self.axons = []
        self.named_axons = {}
        self.probes = []

        assert location in ('core', 'cpu')
        self.location = location

    def __str__(self):
        return "%s(%s)" % (
            type(self).__name__, self.label if self.label else '')

    def add_synapses(self, synapses, name=None):
        """Add a CxSynapses object to ensemble."""

        assert synapses.group is None
        synapses.group = self
        self.synapses.append(synapses)
        if name is not None:
            assert name not in self.named_synapses
            self.named_synapses[name] = synapses

    def add_axons(self, axons, name=None):
        """Add a CxAxons object to ensemble."""

        assert axons.group is None
        axons.group = self
        self.axons.append(axons)
        if name is not None:
            assert name not in self.named_axons
            self.named_axons[name] = axons

    def add_probe(self, probe):
        """Add a CxProbe object to ensemble."""
        if probe.target is None:
            probe.target = self
        assert probe.target is self
        self.probes.append(probe)

    def configure_default_filter(self, tau_s, dt=0.001):
        """Set the default Lowpass synaptic input filter for Cx.

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
        """Set Lowpass synaptic input filter for Cx to time constant tau_s.

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

    def configure_lif(self, tau_rc=0.02, tau_ref=0.001, vth=1, dt=0.001):
        self.decayV[:] = -np.expm1(-dt/np.asarray(tau_rc))
        self.refractDelay[:] = np.round(tau_ref / dt) + 1
        self.vth[:] = vth
        self.vmin = 0
        self.vmax = np.inf
        self.scaleV = np.all(self.decayV > self.DECAY_SCALE_TH)
        if not self.scaleV:
            raise BuildError(
                "Voltage (V) scaling is required with LIF neurons. Perhaps "
                "the neuron tau_rc time constant is too large.")

    def configure_relu(self, tau_ref=0.0, vth=1, dt=0.001):
        self.decayV[:] = 0.
        self.refractDelay[:] = np.round(tau_ref / dt) + 1
        self.vth[:] = vth
        self.vmin = 0
        self.vmax = np.inf
        self.scaleV = False

    def configure_nonspiking(self, tau_ref=0.0, vth=1, dt=0.001):
        self.decayV[:] = 1.
        self.refractDelay[:] = 1
        self.vth[:] = vth
        self.vmin = 0
        self.vmax = np.inf
        self.scaleV = False

    def discretize(self):  # noqa C901
        def discretize(target, value):
            assert target.dtype == np.float32
            # new = np.round(target * scale).astype(np.int32)
            new = np.round(value).astype(np.int32)
            target.dtype = np.int32
            target[:] = new

        # --- discretize decayU and decayV
        u_infactor = (
            self.decayU.copy() if self.scaleU else np.ones_like(self.decayU))
        v_infactor = (
            self.decayV.copy() if self.scaleV else np.ones_like(self.decayV))
        discretize(self.decayU, self.decayU * (2**12 - 1))
        discretize(self.decayV, self.decayV * (2**12 - 1))
        self.scaleU = False
        self.scaleV = False

        # --- vmin and vmax
        vmine = np.clip(np.round(np.log2(-self.vmin + 1)), 0, 2**5-1)
        self.vmin = -2**vmine + 1
        vmaxe = np.clip(np.round((np.log2(self.vmax + 1) - 9)*0.5), 0, 2**3-1)
        self.vmax = 2**(9 + 2*vmaxe) - 1

        # --- discretize weights and vth
        w_maxs = [s.max_abs_weight() for s in self.synapses]
        w_max = max(w_maxs) if len(w_maxs) > 0 else 0
        b_max = np.abs(self.bias).max()
        wgtExp = -7

        if w_max > 1e-8:
            w_scale = (255. / w_max)
            s_scale = 1. / (u_infactor * v_infactor)

            for wgtExp in range(0, -8, -1):
                v_scale = s_scale * w_scale * SynapseFmt.get_scale(wgtExp)
                b_scale = v_scale * v_infactor
                vth = np.round(self.vth * v_scale)
                bias = np.round(self.bias * b_scale)
                if (vth <= VTH_MAX).all() and (np.abs(bias) <= BIAS_MAX).all():
                    break
            else:
                raise BuildError("Could not find appropriate wgtExp")
        elif b_max > 1e-8:
            b_scale = BIAS_MAX / b_max
            while b_scale*b_max > 1:
                v_scale = b_scale / v_infactor
                w_scale = b_scale * u_infactor / SynapseFmt.get_scale(wgtExp)
                vth = np.round(self.vth * v_scale)
                bias = np.round(self.bias * b_scale)
                if np.all(vth <= VTH_MAX):
                    break

                b_scale /= 2.
            else:
                raise BuildError("Could not find appropriate bias scaling")
        else:
            v_scale = np.array([VTH_MAX / (self.vth.max() + 1)])
            vth = np.round(self.vth * v_scale)
            b_scale = v_scale * v_infactor
            bias = np.round(self.bias * b_scale)
            w_scale = (v_scale * v_infactor * u_infactor
                       / SynapseFmt.get_scale(wgtExp))

        vth_man, vth_exp = vth_to_manexp(vth)
        discretize(self.vth, vth_man * 2**vth_exp)

        bias_man, bias_exp = bias_to_manexp(bias)
        discretize(self.bias, bias_man * 2**bias_exp)

        for i, synapse in enumerate(self.synapses):
            if w_maxs[i] > 1e-16:
                dWgtExp = int(np.floor(np.log2(w_max / w_maxs[i])))
                assert dWgtExp >= 0
                wgtExp2 = max(wgtExp - dWgtExp, -6)
            else:
                wgtExp2 = -6
                dWgtExp = wgtExp - wgtExp2
            synapse.format(wgtExp=wgtExp2)
            for w, idxs in zip(synapse.weights, synapse.indices):
                ws = w_scale[idxs] if is_iterable(w_scale) else w_scale
                discretize(w, synapse.synapse_fmt.discretize_weights(
                    w * ws * 2**dWgtExp))
            # TODO: scale this properly, hardcoded for now
            if synapse.tracing:
                synapse.synapse_fmt.wgtExp = 4

        # --- noise
        assert (v_scale[0] == v_scale).all()
        noiseExp0 = np.round(np.log2(10.**self.noiseExp0 * v_scale[0]))
        if noiseExp0 < 0:
            warnings.warn("Noise amplitude falls below lower limit")
        if noiseExp0 > 23:
            warnings.warn(
                "Noise amplitude exceeds upper limit (%d > 23)" % (noiseExp0,))
        self.noiseExp0 = int(np.clip(noiseExp0, 0, 23))
        self.noiseMantOffset0 = int(np.round(2*self.noiseMantOffset0))

        for p in self.probes:
            if p.key == 'v' and p.weights is not None:
                p.weights /= v_scale[0]

    def validate(self):
        if self.location == 'cpu':
            return  # none of these checks currently apply to Lakemont

        N_CX_MAX = 1024
        if self.n > N_CX_MAX:
            raise BuildError("Number of compartments (%d) exceeded max (%d)" %
                             (self.n, N_CX_MAX))

        IN_AXONS_MAX = 4096
        n_axons = sum(s.n_axons for s in self.synapses)
        if n_axons > IN_AXONS_MAX:
            raise BuildError("Input axons (%d) exceeded max (%d)" % (
                n_axons, IN_AXONS_MAX))

        MAX_SYNAPSE_BITS = 16384*64
        synapse_bits = sum(s.bits() for s in self.synapses)
        if synapse_bits > MAX_SYNAPSE_BITS:
            raise BuildError("Total synapse bits (%d) exceeded max (%d)" % (
                synapse_bits, MAX_SYNAPSE_BITS))

        OUT_AXONS_MAX = 4096
        n_axons = sum(a.axon_slots() for a in self.axons)
        if n_axons > OUT_AXONS_MAX:
            raise BuildError("Output axons (%d) exceeded max (%d)" % (
                n_axons, OUT_AXONS_MAX))

        for synapses in self.synapses:
            synapses.validate()

        for axons in self.axons:
            axons.validate()

        for probe in self.probes:
            probe.validate()


class CxSynapses(object):
    """A group of Loihi synapses that share some properties.

    Attributes
    ----------
    n_axons : int
        Number of input axons to this group of synapses.
    group : CxGroup
        The CxGroup (compartments) that these synapses input into.
    synapse_fmt : SynapseFmt
        The synapse format object for these synapses.
    weights : (n_axons,) list of (n_populations, n_compartments) ndarray
        The synapse weights. Organized as a list of arrays so each axon
        can have a different number of target compartments.
    indices : (population, axon, compartment) ndarray
        The synapse indices.
    tracing : bool
        Whether synaptic tracing is enabled for these synapses.
    tracing_tau : float
        The tracing time constant.
    tracing_mag : float
        The tracing increment magnitude.
    """
    def __init__(self, n_axons, label=None):
        self.n_axons = n_axons
        self.label = label
        self.group = None
        self.synapse_fmt = None
        self.weights = None
        self.indices = None
        self.axon_cx_bases = None
        self.axon_to_weight_map = None
        self.tracing = False
        self.tracing_tau = None
        self.tracing_mag = None
        self.pop_type = 0  # one of (0, 16, 32) for discrete, pop16, pop32

    def __str__(self):
        return "%s(%s)" % (
            type(self).__name__, self.label if self.label else '')

    def size(self):
        return sum(w.size for w in self.weights)

    def bits(self):
        return sum(self.synapse_fmt.bits_per_axon(w.size)
                   for w in self.weights)

    def max_abs_weight(self):
        return max(np.abs(w).max() if w.size > 0 else -np.inf
                   for w in self.weights)

    def max_ind(self):
        return max(i.max() if len(i) > 0 else -1 for i in self.indices)

    def idx_bits(self):
        idxBits = int(np.ceil(np.log2(self.max_ind() + 1)))
        assert idxBits <= SynapseFmt.INDEX_BITS_MAP[-1], (
            "idxBits out of range, ensemble too large?")
        idxBits = next(i for i, v in enumerate(SynapseFmt.INDEX_BITS_MAP)
                       if v >= idxBits)
        return idxBits

    def idxs_per_synapse(self):
        return 2 if self.tracing else 1

    def atom_bits_extra(self):
        atom_bits = self.atom_bits()
        assert atom_bits <= 9, "Cannot have more than 9 atom bits"
        return max(atom_bits - 5, 0)  # has 5 bits by default

    def atom_bits(self):
        max_populations = max(w.shape[0] for w in self.weights)
        return int(np.ceil(np.log2(max_populations)))

    def axon_bits(self):
        if self.pop_type == 16:
            return 10 - self.atom_bits_extra()
        else:
            return 12

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

    def axon_cx_base(self, axon_idx):
        if self.axon_cx_bases is None:
            return 0
        cx_base = self.axon_cx_bases[axon_idx]
        return cx_base if cx_base > -1024 else None

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

    def set_full_weights(self, weights):
        self._set_weights_indices(weights)
        assert len(self.weights) == self.n_axons, (
            "Full weights must have different weights for each axon")

        idxBits = self.idx_bits()
        self.format(compression=3, idxBits=idxBits, fanoutType=1,
                    numSynapses=63, wgtBits=7)

    def set_diagonal_weights(self, diag):
        weights = diag.ravel()
        indices = list(range(len(weights)))
        self._set_weights_indices(weights, indices)
        assert len(self.weights) == self.n_axons

        idxBits = self.idx_bits()
        self.format(compression=3, idxBits=idxBits, fanoutType=1,
                    numSynapses=63, wgtBits=7)

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

    def set_learning(self, tracing_tau=2, tracing_mag=1.0):
        assert tracing_tau == int(tracing_tau), "tracing_tau must be integer"
        self.tracing = True
        self.tracing_tau = int(tracing_tau)
        self.tracing_mag = tracing_mag
        self.format(learningCfg=1, stdpProfile=0)
        # ^ stdpProfile hard-coded for now (see loihi_interface)

        mag_int, _ = tracing_mag_int_frac(self)
        assert int(mag_int) < 2**7

    def format(self, **kwargs):
        if self.synapse_fmt is None:
            self.synapse_fmt = SynapseFmt()
        self.synapse_fmt.set(**kwargs)

    def validate(self):
        if self.axon_cx_bases is not None:
            assert np.all(self.axon_cx_bases < 256), "CxBase cannot be > 256"
        if self.pop_type == 16:
            if self.axon_cx_bases is not None:
                assert np.all(self.axon_cx_bases % 4 == 0)


class CxAxons(object):
    """A group of axons, targeting a specific CxSynapses object.

    Attributes
    ----------
    cx_atoms : list of length ``group.n``
        Atom (weight index) associated with each group compartment.
    cx_to_axon_map : list of length ``group.n``
        Index of the axon in `target` targeted by each group compartment.
    group : CxGroup
        Parent CxGroup for this object (set in `CxGroup.add_axons`).
    n_axons : int
        The number of outgoing axons.
    target : CxSynapses
        Target synapses for these axons.
    """

    class Spike(object):
        """A spike, targeting a particular axon within a CxSynapses object.

        The CxSynapses target is implicit, given by the CxAxons object that
        creates this Spike.

        Parameters
        ----------
        axon_id : int
            The index of the axon within the targeted CxSynapses object.
        atom : int, optional (Default: 0)
            An index into the target CxSynapses weights. This allows spikes
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
        self.group = None

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

    def set_axon_map(self, cx_to_axon_map, cx_atoms=None):
        self.cx_to_axon_map = cx_to_axon_map
        self.cx_atoms = cx_atoms

    def map_cx_axons(self, cx_idxs):
        return (self.cx_to_axon_map[cx_idxs]
                if self.cx_to_axon_map is not None else cx_idxs)

    def map_cx_atoms(self, cx_idxs):
        return (self.cx_atoms[cx_idxs] if self.cx_atoms is not None else
                [0 for _ in cx_idxs])

    def map_cx_spikes(self, cx_idxs):
        axon_ids = self.map_cx_axons(cx_idxs)
        atoms = self.map_cx_atoms(cx_idxs)
        return [self.Spike(axon_id, atom=atom) if axon_id >= 0 else None
                for axon_id, atom in zip(axon_ids, atoms)]

    def validate(self):
        if isinstance(self.target, CxSynapses):
            if self.cx_atoms is not None:
                cx_idxs = np.arange(len(self.cx_atoms))
                axon_ids = self.map_cx_axons(cx_idxs)
                for atom, axon_id in zip(self.cx_atoms, axon_ids):
                    n_populations = self.target.axon_populations(axon_id)
                    assert 0 <= atom < n_populations


class CxProbe(object):
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

    def validate(self):
        pass


class CxSpikeInput(object):
    def __init__(self, n):
        self.n = n
        self.spikes = {}  # map sim timestep index to list of spike inds
        self.axons = []
        self.probes = []

    def add_axons(self, axons):
        assert axons.group is None
        axons.group = self
        self.axons.append(axons)

    def add_probe(self, probe):
        if probe.target is None:
            probe.target = self
        assert probe.target is self
        self.probes.append(probe)

    def add_spikes(self, ti, spike_idxs):
        assert is_integer(ti)
        ti = int(ti)
        assert ti > 0, "Spike times must be >= 1 (got %d)" % ti
        assert ti not in self.spikes
        self.spikes[ti] = spike_idxs

    def clear_spikes(self):
        self.spikes.clear()

    def spike_times(self):
        return sorted(self.spikes)

    def spike_idxs(self, ti):
        return self.spikes.get(ti, [])


class CxModel(object):

    def __init__(self, dt=0.001, label=None):
        self.dt = dt
        self.label = label

        self.cx_inputs = collections.OrderedDict()
        self.cx_groups = collections.OrderedDict()

    def add_input(self, input):
        assert isinstance(input, CxSpikeInput)
        assert input not in self.cx_inputs
        self.cx_inputs[input] = len(self.cx_inputs)

    def add_group(self, group):
        assert isinstance(group, CxGroup)
        assert group not in self.cx_groups
        self.cx_groups[group] = len(self.cx_groups)

    def discretize(self):
        for group in self.cx_groups:
            group.discretize()

    def validate(self):
        if len(self.cx_groups) == 0:
            raise BuildError("No neurons marked for execution on-chip. "
                             "Please mark some ensembles as on-chip.")

        for group in self.cx_groups:
            group.validate()


class CxSimulator(object):
    """Software emulator for Loihi chip.

    Parameters
    ----------
    model : Model
        Model specification that will be simulated.
    seed : int, optional (Default: None)
        A seed for all stochastic operations done in this simulator.
    """

    strict = False

    def __init__(self, model, seed=None):
        self.closed = False

        self.build(model, seed=seed)

        self._chip2host_sent_steps = 0
        self._probe_filters = {}
        self._probe_filter_pos = {}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    @classmethod
    def error(cls, msg):
        if cls.strict:
            raise SimulationError(msg)
        else:
            warnings.warn(msg)

    def build(self, model, seed=None):  # noqa: C901
        """Set up NumPy arrays to emulate chip memory and I/O."""
        model.validate()

        if seed is None:
            seed = np.random.randint(2**31 - 1)

        logger.debug("CxSimulator seed: %d", seed)
        self.seed = seed
        self.rng = np.random.RandomState(seed)

        self.t = 0

        self.model = model
        self.inputs = list(self.model.cx_inputs)
        self.groups = sorted(self.model.cx_groups,
                             key=lambda g: g.location == 'cpu')
        self.probe_outputs = {}
        for obj in self.inputs + self.groups:
            for probe in obj.probes:
                self.probe_outputs[probe] = []

        self.n_cx = sum(group.n for group in self.groups)
        self.group_cxs = {}
        cx_slice = None
        i0, i1 = 0, 0
        for group in self.groups:
            if group.location == 'cpu' and cx_slice is None:
                cx_slice = slice(0, i0)

            i1 = i0 + group.n
            self.group_cxs[group] = slice(i0, i1)
            i0 = i1

        self.cx_slice = slice(0, i0) if cx_slice is None else cx_slice
        self.cpu_slice = slice(self.cx_slice.stop, i1)

        # --- allocate group memory
        group_dtype = self.groups[0].vth.dtype
        assert group_dtype in (np.float32, np.int32)
        for group in self.groups:
            assert group.vth.dtype == group_dtype
            assert group.bias.dtype == group_dtype

        logger.debug("CxSimulator dtype: %s", group_dtype)

        MAX_DELAY = 1  # don't do delay yet
        self.q = np.zeros((MAX_DELAY, self.n_cx), dtype=group_dtype)
        self.u = np.zeros(self.n_cx, dtype=group_dtype)
        self.v = np.zeros(self.n_cx, dtype=group_dtype)
        self.s = np.zeros(self.n_cx, dtype=bool)  # spiked
        self.c = np.zeros(self.n_cx, dtype=np.int32)  # spike counter
        self.w = np.zeros(self.n_cx, dtype=np.int32)  # ref period counter

        # --- allocate group parameters
        self.decayU = np.hstack([group.decayU for group in self.groups])
        self.decayV = np.hstack([group.decayV for group in self.groups])
        self.scaleU = np.hstack([
            group.decayU if group.scaleU else np.ones_like(group.decayU)
            for group in self.groups])
        self.scaleV = np.hstack([
            group.decayV if group.scaleV else np.ones_like(group.decayV)
            for group in self.groups])

        def decay_float(x, u, d, s):
            return (1 - d)*x + s*u

        def decay_int(x, u, d, s, a=12, b=0):
            r = (2**a - b - np.asarray(d)).astype(np.int64)
            x = np.sign(x) * np.right_shift(np.abs(x) * r, a)  # round to zero
            return x + u  # no scaling on u

        if group_dtype == np.int32:
            assert (self.scaleU == 1).all()
            assert (self.scaleV == 1).all()
            self.decayU_fn = lambda x, u: decay_int(
                x, u, d=self.decayU, s=self.scaleU, b=1)
            self.decayV_fn = lambda x, u: decay_int(
                x, u, d=self.decayV, s=self.scaleV)

            def overflow(x, bits, name=None):
                _, o = overflow_signed(x, bits=bits, out=x)
                if np.any(o):
                    self.error("Overflow" + (" in %s" % name if name else ""))
        elif group_dtype == np.float32:
            self.decayU_fn = lambda x, u: decay_float(
                x, u, d=self.decayU, s=self.scaleU)
            self.decayV_fn = lambda x, u: decay_float(
                x, u, d=self.decayV, s=self.scaleV)

            def overflow(x, bits, name=None):
                pass  # do not do overflow in floating point

        self.overflow = overflow

        ones = lambda n: np.ones(n, dtype=group_dtype)
        self.vth = np.hstack([group.vth for group in self.groups])
        self.vmin = np.hstack([
            group.vmin*ones(group.n) for group in self.groups])
        self.vmax = np.hstack([
            group.vmax*ones(group.n) for group in self.groups])

        self.bias = np.hstack([group.bias for group in self.groups])
        self.ref = np.hstack([group.refractDelay for group in self.groups])

        # --- allocate synapse memory
        self.axons_in = {synapses: [] for group in self.groups
                         for synapses in group.synapses}
        self.z = {synapses: np.zeros(synapses.n_axons, dtype=np.float64)
                  for group in self.groups for synapses in group.synapses
                  if synapses.tracing}  # synapse traces

        # --- noise
        enableNoise = np.hstack([
            group.enableNoise*ones(group.n) for group in self.groups])
        noiseExp0 = np.hstack([
            group.noiseExp0*ones(group.n) for group in self.groups])
        noiseMantOffset0 = np.hstack([
            group.noiseMantOffset0*ones(group.n) for group in self.groups])
        noiseTarget = np.hstack([
            group.noiseAtDendOrVm*ones(group.n) for group in self.groups])
        if group_dtype == np.int32:
            if np.any(noiseExp0 < 7):
                warnings.warn("Noise amplitude falls below lower limit")
            noiseExp0[noiseExp0 < 7] = 7
            noiseMult = np.where(enableNoise, 2**(noiseExp0 - 7), 0)

            def noiseGen(n=self.n_cx, rng=self.rng):
                x = rng.randint(-128, 128, size=n)
                return (x + 64*noiseMantOffset0) * noiseMult
        elif group_dtype == np.float32:
            noiseMult = np.where(enableNoise, 10.**noiseExp0, 0)

            def noiseGen(n=self.n_cx, rng=self.rng):
                x = rng.uniform(-1, 1, size=n)
                return (x + noiseMantOffset0) * noiseMult

        self.noiseGen = noiseGen
        self.noiseTarget = noiseTarget

    def clear(self):
        """Clear all signals set in `build` (to free up memory)"""
        self.q = None
        self.u = None
        self.v = None
        self.s = None
        self.c = None
        self.w = None

        self.vth = None
        self.vmin = None
        self.vmax = None

        self.bias = None
        self.ref = None
        self.a_in = None
        self.z = None

        self.noiseGen = None
        self.noiseTarget = None

    def close(self):
        self.closed = True
        self.clear()

    def chip2host(self, probes_receivers=None):
        if probes_receivers is None:
            probes_receivers = {}

        increment = None
        for cx_probe, receiver in probes_receivers.items():
            # extract the probe data from the simulator
            x = self.probe_outputs[cx_probe][self._chip2host_sent_steps:]
            if len(x) > 0:
                if increment is None:
                    increment = len(x)
                else:
                    assert increment == len(x)
                if cx_probe.weights is not None:
                    x = np.dot(x, cx_probe.weights)
                for j in range(len(x)):
                    receiver.receive(
                        self.model.dt * (self._chip2host_sent_steps + j + 2),
                        x[j]
                    )
        if increment is not None:
            self._chip2host_sent_steps += increment

    def host2chip(self, spikes, errors):
        for cx_spike_input, t, spike_idxs in spikes:
            cx_spike_input.add_spikes(t, spike_idxs)

        learning_rate = 50  # This is set to match hardware
        for synapses, t, e in errors:
            z = self.z[synapses]
            x = np.hstack([-e, e])

            delta_w = np.outer(z, x) * learning_rate

            for i, w in enumerate(synapses.weights):
                w += delta_w[i].astype('int32')

    def step(self):  # noqa: C901
        """Advance the simulation by 1 step (``dt`` seconds)."""
        self.t += 1

        # --- connections
        self.q[:-1] = self.q[1:]  # advance delays
        self.q[-1] = 0

        # --- clear spikes going in to each synapse
        for axons_in_spikes in self.axons_in.values():
            axons_in_spikes.clear()

        # --- inputs pass spikes to synapses
        if self.t >= 2:  # input spikes take one time-step to arrive
            for input in self.inputs:
                cx_idxs = input.spike_idxs(self.t - 1)
                for axons in input.axons:
                    spikes = axons.map_cx_spikes(cx_idxs)
                    self.axons_in[axons.target].extend(spikes)

        # --- axons pass spikes to synapses
        for group in self.groups:
            cx_idxs = self.s[self.group_cxs[group]].nonzero()[0]
            for axons in group.axons:
                spikes = axons.map_cx_spikes(cx_idxs)
                self.axons_in[axons.target].extend(spikes)

        # --- synapse spikes use weights to modify compartment input
        for group in self.groups:
            for synapses in group.synapses:
                b_slice = self.group_cxs[synapses.group]
                qb = self.q[:, b_slice]
                # delays = np.zeros(qb.shape[1], dtype=np.int32)

                for spike in self.axons_in[synapses]:
                    # qb[0, indices[spike.axon_id]] += weights[spike.axon_id]
                    cx_base = synapses.axon_cx_base(spike.axon_id)
                    if cx_base is None:
                        continue

                    weights, indices = synapses.axon_weights_indices(
                        spike.axon_id, atom=spike.atom)
                    qb[0, cx_base + indices] += weights

                if synapses.tracing:
                    z = self.z[synapses]
                    tau = synapses.tracing_tau
                    mag = synapses.tracing_mag

                    decay = np.exp(-1.0 / tau)
                    z *= decay

                    for spike in self.axons_in[synapses]:
                        z[spike.axon_id] += mag

        # --- updates
        q0 = self.q[0, :]

        noise = self.noiseGen()
        q0[self.noiseTarget == 0] += noise[self.noiseTarget == 0]
        self.overflow(q0, bits=Q_BITS, name="q0")

        self.u[:] = self.decayU_fn(self.u[:], q0)
        self.overflow(self.u, bits=U_BITS, name="U")
        u2 = self.u + self.bias
        u2[self.noiseTarget == 1] += noise[self.noiseTarget == 1]
        self.overflow(u2, bits=U_BITS, name="u2")

        self.v[:] = self.decayV_fn(self.v, u2)
        # We have not been able to create V overflow on the chip, so we do
        # not include it here. See github.com/nengo/nengo-loihi/issues/130
        # self.overflow(self.v, bits=V_BIT, name="V")

        np.clip(self.v, self.vmin, self.vmax, out=self.v)
        self.v[self.w > 0] = 0
        # TODO^: don't zero voltage in case neuron is saving overshoot

        self.s[:] = (self.v > self.vth)

        cx = self.cx_slice
        cpu = self.cpu_slice
        self.v[cx][self.s[cx]] = 0
        self.v[cpu][self.s[cpu]] -= self.vth[cpu][self.s[cpu]]

        self.w[self.s] = self.ref[self.s]
        np.clip(self.w - 1, 0, None, out=self.w)  # decrement w

        self.c[self.s] += 1

        # --- probes
        for input in self.inputs:
            for probe in input.probes:
                assert probe.key == 's'
                p_slice = probe.slice
                x = input.spikes[self.t][p_slice].copy()
                self.probe_outputs[probe].append(x)

        for group in self.groups:
            for probe in group.probes:
                x_slice = self.group_cxs[probe.target]
                p_slice = probe.slice
                assert hasattr(self, probe.key), "probe key not found"
                x = getattr(self, probe.key)[x_slice][p_slice].copy()
                self.probe_outputs[probe].append(x)

    def run_steps(self, steps):
        """Simulate for the given number of ``dt`` steps.

        Parameters
        ----------
        steps : int
            Number of steps to run the simulation for.
        """
        for _ in range(steps):
            self.step()

    def _filter_probe(self, cx_probe, data):
        dt = self.model.dt
        i = self._probe_filter_pos.get(cx_probe, 0)
        if i == 0:
            shape = data[0].shape
            synapse = cx_probe.synapse
            rng = None
            step = (synapse.make_step(shape, shape, dt, rng, dtype=data.dtype)
                    if synapse is not None else None)
            self._probe_filters[cx_probe] = step
        else:
            step = self._probe_filters[cx_probe]

        if step is None:
            self._probe_filter_pos[cx_probe] = i + len(data)
            return data
        else:
            filt_data = np.zeros_like(data)
            for k, x in enumerate(data):
                filt_data[k] = step((i + k) * dt, x)

            self._probe_filter_pos[cx_probe] = i + k
            return filt_data

    def get_probe_output(self, cx_probe):
        assert isinstance(cx_probe, CxProbe)
        x = np.asarray(self.probe_outputs[cx_probe], dtype=np.float32)
        x = x if cx_probe.weights is None else np.dot(x, cx_probe.weights)
        return self._filter_probe(cx_probe, x)


class PESModulatoryTarget(object):
    def __init__(self, target):
        self.target = target
        self.errors = collections.OrderedDict()

    def clear(self):
        self.errors.clear()

    def receive(self, t, x):
        assert len(self.errors) == 0 or t >= next(reversed(self.errors))
        if t in self.errors:
            self.errors[t] += x
        else:
            self.errors[t] = np.array(x)

    def collect_errors(self):
        for t, x in self.errors.items():
            yield (self.target, t, x)


class HostSendNode(nengo.Node):
    """For sending host->chip messages"""

    def __init__(self, dimensions):
        self.queue = []
        super(HostSendNode, self).__init__(self.update,
                                           size_in=dimensions, size_out=0)

    def update(self, t, x):
        assert len(self.queue) == 0 or t > self.queue[-1][0]
        self.queue.append((t, x))


class HostReceiveNode(nengo.Node):
    """For receiving chip->host messages"""

    def __init__(self, dimensions):
        self.queue = [(0, np.zeros(dimensions))]
        self.queue_index = 0
        super(HostReceiveNode, self).__init__(self.update,
                                              size_in=0, size_out=dimensions)

    def update(self, t):
        while (len(self.queue) > self.queue_index + 1
               and self.queue[self.queue_index][0] < t):
            self.queue_index += 1
        return self.queue[self.queue_index][1]

    def receive(self, t, x):
        self.queue.append((t, x))


class ChipReceiveNode(nengo.Node):
    """For receiving host->chip messages"""

    def __init__(self, dimensions, size_out):
        self.raw_dimensions = dimensions
        self.spikes = []
        self.cx_spike_input = None  # set by builder
        super(ChipReceiveNode, self).__init__(
            self.update, size_in=0, size_out=size_out)

    def clear(self):
        self.spikes.clear()

    def receive(self, t, x):
        assert len(self.spikes) == 0 or t > self.spikes[-1][0]
        assert x.ndim == 1
        self.spikes.append((t, x.nonzero()[0]))

    def update(self, t):
        raise SimulationError("ChipReceiveNodes should not be run")

    def collect_spikes(self):
        assert self.cx_spike_input is not None
        for t, x in self.spikes:
            yield (self.cx_spike_input, t, x)


class ChipReceiveNeurons(ChipReceiveNode):
    """Passes spikes directly (no on-off neuron encoding)"""
    def __init__(self, dimensions, neuron_type=None):
        self.neuron_type = neuron_type
        super(ChipReceiveNeurons, self).__init__(dimensions, dimensions)
