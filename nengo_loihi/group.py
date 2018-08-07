import warnings

from nengo.utils.compat import is_iterable
import numpy as np

from nengo_loihi.discretize import (
    BIAS_MAX,
    bias_to_manexp,
    discretize,
    VTH_MAX,
    vth_to_manexp
)
from nengo_loihi.synapses import SynapseFmt


class CompartmentGroup(object):
    """A group of compartments."""
    DEFAULT_WEIGHT_EXP = -7

    def __init__(self, n_compartments):
        self.n_compartments = n_compartments

        self.decayU = np.zeros(n_compartments, dtype=np.float32)
        self.decayV = np.zeros(n_compartments, dtype=np.float32)
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

        # determined in `discretize`
        self.weight_exp = None
        self.v_scale = None
        self.w_scale = None

    def set_decay_U(self, tau_s, dt):
        self.decayU[:] = 1 if tau_s == 0 else -np.expm1(-dt/np.asarray(tau_s))

    def configure_filter(self, tau_s, dt=0.001):
        """Synaptic input filter for Cx."""
        self.set_decay_U(tau_s, dt)

    def configure_lif(
            self, tau_s=0.005, tau_rc=0.02, tau_ref=0.001, vth=1, dt=0.001):
        self.set_decay_U(tau_s, dt)
        self.decayV[:] = -np.expm1(-dt/np.asarray(tau_rc))
        self.refractDelay[:] = np.round(tau_ref / dt) + 1
        self.vth[:] = vth
        self.vmin = 0
        self.vmax = np.inf
        self.scaleU = True
        self.scaleV = np.all(self.decayV > 1e-15)

    def configure_relu(self, tau_s=0.0, tau_ref=0.0, vth=1, dt=0.001):
        self.set_decay_U(tau_s, dt)
        self.decayV[:] = 0.
        self.refractDelay[:] = np.round(tau_ref / dt) + 1
        self.vth[:] = vth
        self.vmin = 0
        self.vmax = np.inf
        self.scaleU = True
        self.scaleV = False

    def configure_nonspiking(self, tau_s=0.0, tau_ref=0.0, vth=1, dt=0.001):
        self.set_decay_U(tau_s, dt)
        self.decayV[:] = 1.
        self.refractDelay[:] = 1
        self.vth[:] = vth
        self.vmin = 0
        self.vmax = np.inf
        self.scaleU = True
        self.scaleV = False

    def discretize(self, w_max):
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
        weight_exp = self.DEFAULT_WEIGHT_EXP
        b_max = np.abs(self.bias).max()

        if w_max > 1e-8:
            w_scale = (255. / w_max)
            s_scale = 1. / (u_infactor * v_infactor)

            # Determine a better weight_exp
            for weight_exp in range(7, -8, -1):
                v_scale = s_scale * w_scale * SynapseFmt.get_scale(weight_exp)
                b_scale = v_scale * v_infactor
                vth = np.round(self.vth * v_scale)
                bias = np.round(self.bias * b_scale)
                if (vth <= VTH_MAX).all() and (np.abs(bias) <= BIAS_MAX).all():
                    break
            else:
                raise ValueError("Could not find appropriate weight_exp")
        elif b_max > 1e-8:
            b_scale = BIAS_MAX / b_max
            while b_scale*b_max > 1:
                v_scale = b_scale / v_infactor
                w_scale = (
                    b_scale * u_infactor / SynapseFmt.get_scale(weight_exp))
                vth = np.round(self.vth * v_scale)
                bias = np.round(self.bias * b_scale)
                if np.all(vth <= VTH_MAX):
                    break

                b_scale /= 2.
            else:
                raise ValueError("Could not find appropriate bias scaling")
        else:
            v_scale = np.array([VTH_MAX / (self.vth.max() + 1)])
            vth = np.round(self.vth * v_scale)
            b_scale = v_scale * v_infactor
            bias = np.round(self.bias * b_scale)
            w_scale = (v_scale * v_infactor * u_infactor
                       / SynapseFmt.get_scale(weight_exp))

        vth_man, vth_exp = vth_to_manexp(vth)
        discretize(self.vth, vth_man * 2**vth_exp)

        bias_man, bias_exp = bias_to_manexp(bias)
        discretize(self.bias, bias_man * 2**bias_exp)

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

        self.weight_exp = weight_exp
        self.v_scale = v_scale[0]
        self.w_scale = w_scale


class SynapseGroup(object):
    def __init__(self, n_synapses):
        self.n_synapses = n_synapses

        self.synapses = []
        self.named_synapses = {}

    def add(self, synapses, name=None):
        """Add a CxSynapses object to this group."""

        self.synapses.append(synapses)
        if name is not None:
            assert name not in self.named_synapses
            self.named_synapses[name] = synapses

        AXONS_MAX = 4096
        MAX_SYNAPSE_BITS = 16384*64
        n_axons = sum(s.n_axons for s in self.synapses)
        if n_axons > AXONS_MAX:
            raise ValueError("Total axons (%d) exceeded max (%d)" % (
                n_axons, AXONS_MAX))

        synapse_bits = sum(s.n_bits for s in self.synapses)
        if synapse_bits > MAX_SYNAPSE_BITS:
            raise ValueError("Total synapse bits (%d) exceeded max (%d)" % (
                synapse_bits, MAX_SYNAPSE_BITS))

    def max_weight(self):
        w_maxs = [s.max_abs_weight for s in self.synapses]
        return max(w_maxs) if len(w_maxs) > 0 else 0

    def discretize(self, w_scale, weight_exp):
        max_weight = self.max_weight()

        for i, synapse in enumerate(self.synapses):
            s_max_weight = synapse.max_abs_weight
            if s_max_weight > 1e-16:
                d_weight_exp = int(
                    np.floor(np.log2(max_weight / s_max_weight))
                )
                assert d_weight_exp >= 0
                weight_exp2 = max(weight_exp - d_weight_exp, -6)
            else:
                weight_exp2 = -6
                d_weight_exp = weight_exp - weight_exp2
            synapse.fmt.wgtExp = weight_exp2
            for w, idxs in zip(synapse.weights, synapse.indices):
                ws = w_scale[idxs] if is_iterable(w_scale) else w_scale
                discretize(w, synapse.fmt.discretize_weights(
                    w * ws * 2**d_weight_exp))
            # TODO: scale this properly, hardcoded for now
            if synapse.learning:
                synapse.fmt.weight_exp = 4


class AxonGroup(object):
    def __init__(self, n_axons):
        self.n_axons = n_axons
        self.axons = []
        self.named_axons = {}

    def add(self, axons, name=None):
        """Add a CxAxons object to this group."""
        self.axons.append(axons)
        if name is not None:
            assert name not in self.named_axons
            self.named_axons[name] = axons


class ProbeGroup(object):
    def __init__(self):
        self.probes = []

    def add(self, probe):
        """Add a CxProbe object to ensemble."""
        # if probe.target is None:
        #     probe.target = self
        # assert probe.target is self
        self.probes.append(probe)

    def discretize(self, v_scale):
        for p in self.probes:
            if p.key == 'v' and p.weights is not None:
                p.weights /= v_scale


class CoreGroup(object):
    """A group of Loihi objects to be placed on a Loihi neuron core.

    Consists of a `.CompartmentGroup`, `.SynapseGroup` `.AxonGroup`
    and `.ProbeGroup`.
    """

    def __init__(self, n_compartments, label=None):
        self.label = label

        self.axons = AxonGroup(n_axons=n_compartments)
        self.compartments = CompartmentGroup(n_compartments=n_compartments)
        self.synapses = SynapseGroup(n_synapses=n_compartments)
        self.probes = ProbeGroup()

    @property
    def n_compartments(self):
        return self.compartments.n_compartments

    def __str__(self):
        return "%s(%s)" % (
            type(self).__name__, self.label if self.label else '')

    def discretize(self):
        w_max = self.synapses.max_weight()

        self.compartments.discretize(w_max)
        self.synapses.discretize(
            self.compartments.w_scale, self.compartments.weight_exp)
        self.probes.discretize(self.compartments.v_scale)
