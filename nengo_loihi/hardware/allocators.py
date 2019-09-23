import logging

from nengo.exceptions import ValidationError
import numpy as np

from nengo_loihi.discretize import tracing_mag_int_frac, vth_to_manexp
from nengo_loihi.hardware.nxsdk_objects import (
    Board,
    CompartmentConfig,
    TraceConfig,
    VthConfig,
)
from nengo_loihi.nxsdk_obfuscation import d

logger = logging.getLogger(__name__)


def compute_cfgs(core, list_cfgs):
    cfg_lists = []
    for block in core.blocks:
        cfg_lists.append(list_cfgs(block))

    cfgs = list(set(p for plist in cfg_lists for p in plist))
    cfg_idxs = {}
    for block, plist in zip(core.blocks, cfg_lists):
        cfg_idxs[block] = np.zeros(len(plist), dtype=int)
        for k, cfg in enumerate(cfgs):
            cfg_idxs[block][[p == cfg for p in plist]] = k

    return cfgs, cfg_idxs


def core_compartment_cfgs(core):
    """Compute all compartment_cfgs needed for a core"""

    def list_compartment_cfgs(block):
        cfgs = []
        for i in range(block.compartment.n_compartments):
            cfgs.append(
                CompartmentConfig(
                    decay_u=block.compartment.decay_u[i],
                    decay_v=block.compartment.decay_v[i],
                    refract_delay=block.compartment.refract_delay[i],
                    enable_noise=block.compartment.enable_noise[i],
                )
            )

        return cfgs

    return compute_cfgs(core, list_compartment_cfgs)


def core_vth_cfgs(core):
    """Compute all vth_cfgs needed for a core"""

    def list_vth_cfgs(block):
        cfgs = []
        vth, _ = vth_to_manexp(block.compartment.vth)
        for i in range(block.compartment.n_compartments):
            cfgs.append(VthConfig(vth=vth[i]))

        return cfgs

    return compute_cfgs(core, list_vth_cfgs)


def core_stdp_pre_cfgs(core):
    cfgs = []
    cfg_idxs = {}
    for synapse in core.synapses:
        if synapse.learning:
            mag_int, mag_frac = tracing_mag_int_frac(synapse.tracing_mag)
            tracecfg = TraceConfig(
                tau=synapse.tracing_tau, spike_int=mag_int, spike_frac=mag_frac
            )

            if tracecfg in cfgs:
                cfg_idxs[synapse] = cfgs.index(tracecfg)
            else:
                cfg_idxs[synapse] = len(cfgs)
                cfgs.append(tracecfg)
        else:
            cfg_idxs[synapse] = None

    return cfgs, cfg_idxs


class Allocator:
    """Responsible for allocating the board's devices to models."""

    def __call__(self, model):
        """Returns a Board object corresponding to the given model."""
        raise NotImplementedError()


class OneToOne(Allocator):
    """Assigns each block and input to distinct cores on the same chip."""

    def block_to_chip(self, block, chip):
        if block.compartment.n_compartments > d(b"MTAyNA==", int):
            raise ValidationError("Segment does not fit on one core", "n_neurons")

        core = chip.new_core()
        core.add_block(block)

        compartment_cfgs, compartment_cfg_idxs = core_compartment_cfgs(core)
        [core.add_compartment_cfg(cfg) for cfg in compartment_cfgs]
        core.compartment_cfg_idxs = compartment_cfg_idxs

        vth_cfgs, vth_cfg_idxs = core_vth_cfgs(core)
        [core.add_vth_cfg(cfg) for cfg in vth_cfgs]
        core.vth_cfg_idxs = vth_cfg_idxs

        for synapse in block.synapses:
            core.add_synapse(synapse)

        stdp_pre_cfgs, stdp_pre_cfg_idxs = core_stdp_pre_cfgs(core)
        [core.add_stdp_pre_cfg(stdp_pre_cfg) for stdp_pre_cfg in stdp_pre_cfgs]
        core.stdp_pre_cfg_idxs = stdp_pre_cfg_idxs

        core.stdp_pre_cfg_idx = None  # hardware.builder will set
        core.stdp_cfg_idx = None  # hardware.builder will set

    def input_to_chip(self, input, chip):
        # TODO: how to allocate inputs?
        core = chip.new_core()
        core.add_input(input)

    def __call__(self, model):
        board = Board()
        chip = board.new_chip()

        for block in model.blocks:
            self.block_to_chip(block, chip)

        for input in model.inputs:
            self.input_to_chip(input, chip)

        board.probes.extend(model.probes)

        return board


class RoundRobin(OneToOne):
    """Assigns each block and input to the next chip in round-robin order."""

    def __init__(self, n_chips):
        self.n_chips = n_chips

    def __call__(self, model):
        board = Board()

        # We must dynamically allocate the chips
        # as needed because nxsdk==0.8.0 hits
        # an assertion if any chips contain 0 cores
        def get_chip(i):
            if i < self.n_chips:
                board.new_chip()
            return board.chips[i % self.n_chips]

        i = 0
        for block in model.blocks:
            self.block_to_chip(block, get_chip(i))
            i += 1

        for input in model.inputs:
            self.input_to_chip(input, get_chip(i))
            i += 1

        logger.info("Round-robin allocation across %d chips", board.n_chips)

        board.probes.extend(model.probes)

        return board
