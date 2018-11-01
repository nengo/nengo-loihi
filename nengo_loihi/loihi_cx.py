from __future__ import division

import collections
import logging
import warnings

import numpy as np
from nengo.exceptions import BuildError, SimulationError
from nengo.utils.compat import is_iterable, range

from nengo_loihi.loihi_api import (
    BIAS_MAX,
    bias_to_manexp,
    LEARN_FRAC,
    learn_overflow_bits,
    overflow_signed,
    shift,
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

        AXONS_MAX = 4096
        MAX_SYNAPSE_BITS = 16384*64
        n_axons = sum(s.n_axons for s in self.synapses)
        if n_axons > AXONS_MAX:
            raise BuildError("Total axons (%d) exceeded max (%d)" % (
                n_axons, AXONS_MAX))

        synapse_bits = sum(s.bits() for s in self.synapses)
        if synapse_bits > MAX_SYNAPSE_BITS:
            raise BuildError("Total synapse bits (%d) exceeded max (%d)" % (
                synapse_bits, MAX_SYNAPSE_BITS))

    def add_axons(self, axons, name=None):
        """Add a CxAxons object to ensemble."""

        assert axons.group is None
        axons.group = self
        self.axons.append(axons)
        if name is not None:
            assert name not in self.named_axons
            self.named_axons[name] = axons

        assert axons.n_axons == self.n, "Axons currently only one-to-one"

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
        wgtExp = 0

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
            if synapse.tracing:
                # TODO: scale this properly, hardcoded for now
                wgtExp2 = 4
                dWgtExp = wgtExp - wgtExp2
            elif w_maxs[i] > 1e-16:
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

            # discretize learning
            if synapse.tracing:
                synapse.tracing_tau = int(np.round(synapse.tracing_tau))

                if is_iterable(w_scale):
                    assert np.all(w_scale == w_scale[0])
                w_scale_i = w_scale[0] if is_iterable(w_scale) else w_scale

                ws = w_scale_i * 2**dWgtExp
                synapse.learning_rate *= ws
                synapse.learning_rate *= 2**learn_overflow_bits(2)

                # TODO: Currently, Loihi learning rate fixed at 2**-7, but
                # by using microcode generation it can be adjusted.
                lscale = 2**-7 / synapse.learning_rate
                synapse.learning_rate *= lscale
                synapse.tracing_mag /= lscale

                lr_exp = int(np.floor(np.log2(synapse.learning_rate)))
                lr_int = int(np.round(synapse.learning_rate * 2**(-lr_exp)))
                synapse.learning_rate = lr_int * 2**lr_exp
                synapse._lr_int = lr_int
                synapse._lr_exp = lr_exp
                assert lr_exp >= -7

                mag_int, mag_frac = tracing_mag_int_frac(synapse.tracing_mag)
                if mag_int > 127:
                    mag_int = 127
                    mag_frac = 127
                synapse.tracing_mag = mag_int + mag_frac / 128.

                print("Learning rate: %0.3e, mag: %0.3e" % (
                    synapse.learning_rate, synapse.tracing_mag))

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


class CxSynapses(object):
    def __init__(self, n_axons, label=None):
        self.n_axons = n_axons
        self.label = label
        self.group = None
        self.synapse_fmt = None
        self.weights = None
        self.indices = None

        self.learning_rate = 1.
        self.tracing = False
        self.tracing_tau = None
        self.tracing_mag = None

    def __str__(self):
        return "%s(%s)" % (
            type(self).__name__, self.label if self.label else '')

    def size(self):
        return sum(len(w) for w in self.weights)

    def bits(self):
        return sum(self.synapse_fmt.bits_per_axon(len(w))
                   for w in self.weights)

    def max_abs_weight(self):
        return max(np.abs(w).max() if len(w) > 0 else -np.inf
                   for w in self.weights)

    def max_ind(self):
        return max(i.max() if len(i) > 0 else -1 for i in self.indices)

    def set_full_weights(self, weights):
        self.weights = [w.astype(np.float32) for w in weights]
        self.indices = [np.arange(w.size) for w in weights]
        assert weights.shape[0] == self.n_axons

        idxBits = int(np.ceil(np.log2(self.max_ind() + 1)))
        assert idxBits <= SynapseFmt.INDEX_BITS_MAP[-1]
        idxBits = next(i for i, v in enumerate(SynapseFmt.INDEX_BITS_MAP)
                       if v >= idxBits)
        self.format(compression=3, idxBits=idxBits, fanoutType=1,
                    numSynapses=63, wgtBits=7)

    def set_diagonal_weights(self, diag):
        diag = diag.ravel()
        self.weights = [d.reshape(1).astype(np.float32) for d in diag]
        self.indices = [np.array([i]) for i in range(len(diag))]
        assert len(self.weights) == self.n_axons

        idxBits = int(np.ceil(np.log2(self.max_ind() + 1)))
        assert idxBits <= SynapseFmt.INDEX_BITS_MAP[-1]
        idxBits = next(i for i, v in enumerate(SynapseFmt.INDEX_BITS_MAP)
                       if v >= idxBits)
        self.format(compression=3, idxBits=idxBits, fanoutType=1,
                    numSynapses=63, wgtBits=7)

    def set_learning(self, learning_rate=1., tracing_tau=2, tracing_mag=1.0):
        from nengo_loihi.splitter import PESModulatoryTarget

        assert tracing_tau == int(tracing_tau), "tracing_tau must be integer"

        self.learning_rate = learning_rate

        self.tracing = True
        self.tracing_tau = int(tracing_tau)
        self.tracing_mag = tracing_mag
        self.format(learningCfg=1, stdpProfile=0)
        # ^ stdpProfile hard-coded for now (see loihi_interface)

        self.train_epoch = 2
        self.learn_epoch_k = 1
        self.learn_epoch = self.train_epoch * 2**self.learn_epoch_k

        self.learning_rate = (self.learning_rate * self.learn_epoch
                              / PESModulatoryTarget.ERROR_SCALE)

    def format(self, **kwargs):
        if self.synapse_fmt is None:
            self.synapse_fmt = SynapseFmt()
        self.synapse_fmt.set(**kwargs)


class CxAxons(object):
    def __init__(self, n_axons, label=None):
        self.n_axons = n_axons
        self.label = label
        self.group = None

        self.target = None
        self.target_inds = slice(None)  # which synapse inputs are targeted
        # ^ TODO: this does not allow multiple pre-cx per axon, loihi does

    def __str__(self):
        return "%s(%s)" % (
            type(self).__name__, self.label if self.label else '')


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

    def get_loihi(self, seed=None):
        from nengo_loihi.loihi_interface import LoihiSimulator
        return LoihiSimulator(self, seed=seed)

    def get_simulator(self, seed=None):
        return CxSimulator(self, seed=seed)

    def validate(self):
        if len(self.cx_groups) == 0:
            raise BuildError("No neurons marked for execution on-chip. "
                             "Please mark some ensembles as on-chip.")


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
        self.build(model, seed=seed)

        self._probe_filters = {}
        self._probe_filter_pos = {}

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
        self.probe_outputs = collections.defaultdict(list)

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

        # --- allocate synapse/learning memory
        learning_synapses = [
            synapses for group in self.groups
            for synapses in group.synapses if synapses.tracing]
        self.a_in = {synapses: np.zeros(synapses.n_axons, dtype=np.int32)
                     for group in self.groups for synapses in group.synapses}
        self.z = {synapses: np.zeros(synapses.n_axons, dtype=group_dtype)
                  for synapses in learning_synapses}
        self.z_spikes = {synapses: np.zeros(synapses.n_axons, dtype=bool)
                         for synapses in learning_synapses}
        self.pes_errors = {synapses: np.zeros(group.n//2, dtype=group_dtype)
                           for synapses in learning_synapses}
        # ^ Currently, PES learning only happens on Nodes, where we have
        # pairs of on/off neurons. Therefore, the number of error dimensions
        # is half the number of neurons.

        if group_dtype == np.int32:
            def stochastic_round(x, dtype=group_dtype, rng=self.rng,
                                 clip=None, name="values"):
                x_sign = np.sign(x).astype(dtype)
                x_frac, x_int = np.modf(np.abs(x))
                p = rng.rand(*x.shape)
                y = x_int.astype(dtype) + (x_frac > p)
                if clip is not None:
                    q = y > clip
                    if np.any(q):
                        warnings.warn("Clipping %s" % name)
                return x_sign * y

            def trace_round(x, dtype=group_dtype, rng=self.rng):
                return stochastic_round(x, dtype=dtype, rng=rng,
                                        clip=127, name="synapse trace")

            def weight_update(synapses, delta_ws):
                synapse_fmt = synapses.synapse_fmt
                wgt_exp = synapse_fmt.realWgtExp
                shift_bits = synapse_fmt.shift_bits
                overflow = learn_overflow_bits(n_factors=2)
                for w, delta_w in zip(synapses.weights, delta_ws):
                    product = shift(
                        delta_w * synapses._lr_int,
                        LEARN_FRAC + synapses._lr_exp - overflow)
                    learn_w = shift(w, LEARN_FRAC - wgt_exp) + product
                    learn_w[:] = stochastic_round(
                        learn_w * 2**(-LEARN_FRAC - shift_bits),
                        clip=2**(8 - shift_bits) - 1, name="weights")
                    w[:] = np.left_shift(learn_w, wgt_exp + shift_bits)

        elif group_dtype == np.float32:
            def trace_round(x, dtype=group_dtype):
                return x  # no rounding

            def weight_update(synapses, delta_ws):
                for w, delta_w in zip(synapses.weights, delta_ws):
                    w[:] += delta_w

        self.trace_round = trace_round
        self.weight_update = weight_update

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

    def step(self):  # noqa: C901
        """Advance the simulation by 1 step (``dt`` seconds)."""

        # --- connections
        self.q[:-1] = self.q[1:]  # advance delays
        self.q[-1] = 0

        for a_in in self.a_in.values():
            a_in[:] = 0

        for input in self.inputs:
            for axons in input.axons:
                synapses = axons.target
                assert axons.target_inds == slice(None)
                self.a_in[synapses] += input.spikes[self.t]

        for group in self.groups:
            for axons in group.axons:
                synapses = axons.target
                s_in = self.a_in[synapses]

                a_slice = self.group_cxs[axons.group]
                sa = self.s[a_slice]
                np.add.at(s_in, axons.target_inds, sa)  # allows repeat inds

        for group in self.groups:
            for synapses in group.synapses:
                s_in = self.a_in[synapses]

                b_slice = self.group_cxs[synapses.group]
                weights = synapses.weights
                indices = synapses.indices
                qb = self.q[:, b_slice]
                # delays = np.zeros(qb.shape[1], dtype=np.int32)

                for i in s_in.nonzero()[0]:
                    for _ in range(s_in[i]):  # faster than mult since likely 1
                        qb[0, indices[i]] += weights[i]
                    # qb[delays[indices[i]], indices[i]] += weights[i]

                # --- learning trace
                z_spikes = self.z_spikes.get(synapses, None)
                if z_spikes is not None:
                    if np.any((s_in + z_spikes) > 1):
                        self.error("Synaptic trace spikes lost")
                    z_spikes |= s_in > 0

                z = self.z.get(synapses, None)
                if z is not None and self.t % synapses.train_epoch == 0:
                    tau = synapses.tracing_tau
                    mag = synapses.tracing_mag
                    decay = np.exp(-synapses.train_epoch / tau)
                    zi = decay*z + mag*z_spikes
                    z[:] = self.trace_round(zi)
                    z_spikes[:] = 0

                # --- learning update
                pes_e = self.pes_errors.get(synapses, None)
                if pes_e is not None and self.t % synapses.learn_epoch == 0:
                    assert z is not None
                    x = np.hstack([-pes_e, pes_e])
                    delta_w = np.outer(z, x)
                    self.weight_update(synapses, delta_w)

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
                assert hasattr(self, probe.key)
                x = getattr(self, probe.key)[x_slice][p_slice].copy()
                self.probe_outputs[probe].append(x)

        self.t += 1

    def clear_pes_errors(self):
        for synapses in self.pes_errors:
            self.pes_errors[synapses][:] = 0

    def add_pes_errors(self, synapses, errors):
        assert synapses.tracing
        target_errors = self.pes_errors[synapses]
        assert target_errors.shape == errors.shape
        target_errors[:] += errors

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

    def get_probe_output(self, probe):
        cx_probe = self.model.objs[probe]['out']
        assert isinstance(cx_probe, CxProbe)
        x = np.asarray(self.probe_outputs[cx_probe], dtype=np.float32)
        x = x if cx_probe.weights is None else np.dot(x, cx_probe.weights)
        return self._filter_probe(cx_probe, x)
