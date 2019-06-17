import collections
import logging

from nengo.exceptions import ValidationError
import numpy as np

from nengo_loihi.block import LoihiBlock, Synapse, Axon
from nengo_loihi.discretize import (
    tracing_mag_int_frac,
    vth_to_manexp,
)
from nengo_loihi.hardware.nxsdk_objects import (
    Board,
    CompartmentConfig,
    TraceConfig,
    VthConfig,
)
from nengo_loihi.inputs import DVSInput, SpikeInput
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
            cfgs.append(CompartmentConfig(
                decay_u=block.compartment.decay_u[i],
                decay_v=block.compartment.decay_v[i],
                refract_delay=block.compartment.refract_delay[i],
                enable_noise=block.compartment.enable_noise[i],
                bap_action=block.compartment.bap_action[i],
            ))

        return cfgs

    return compute_cfgs(core, list_compartment_cfgs)


def core_vth_cfgs(core):
    """Compute all vth_cfgs needed for a core"""
    def list_vth_cfgs(block):
        cfgs = []
        vth, _ = vth_to_manexp(block.compartment.vth)
        for i in range(block.compartment.n_compartments):
            cfgs.append(VthConfig(
                vth=vth[i],
            ))

        return cfgs

    return compute_cfgs(core, list_vth_cfgs)


def core_stdp_pre_cfgs(core):
    cfgs = []
    cfg_idxs = {}
    for synapse in core.synapses:
        if synapse.learning:
            mag_int, mag_frac = tracing_mag_int_frac(synapse.tracing_mag)
            tracecfg = TraceConfig(
                tau=synapse.tracing_tau,
                spike_int=mag_int,
                spike_frac=mag_frac,
            )

            if tracecfg in cfgs:
                cfg_idxs[synapse] = cfgs.index(tracecfg)
            else:
                cfg_idxs[synapse] = len(cfgs)
                cfgs.append(tracecfg)
        else:
            cfg_idxs[synapse] = None

    return cfgs, cfg_idxs


def allocate_core(core):
    assert len(core.blocks) == 1, "Currently only implemented for one block"

    compartment_cfgs, compartment_cfg_idxs = core_compartment_cfgs(core)
    [core.add_compartment_cfg(cfg) for cfg in compartment_cfgs]
    core.compartment_cfg_idxs = compartment_cfg_idxs

    vth_cfgs, vth_cfg_idxs = core_vth_cfgs(core)
    [core.add_vth_cfg(cfg) for cfg in vth_cfgs]
    core.vth_cfg_idxs = vth_cfg_idxs

    for block in core.blocks:
        for synapse in block.synapses:
            core.add_synapse(synapse)

    stdp_pre_cfgs, stdp_pre_cfg_idxs = core_stdp_pre_cfgs(core)
    [core.add_stdp_pre_cfg(stdp_pre_cfg) for stdp_pre_cfg in stdp_pre_cfgs]
    core.stdp_pre_cfg_idxs = stdp_pre_cfg_idxs

    core.stdp_pre_cfg_idx = None  # hardware.builder will set
    core.stdp_cfg_idx = None  # hardware.builder will set


class Allocator:
    """Responsible for allocating the board's devices to models."""

    def __call__(self, model):
        """Returns a Board object corresponding to the given model."""
        raise NotImplementedError()


class OneToOne(Allocator):
    """Assigns each block and input to distinct cores on the same chip."""

    def block_to_chip(self, block, chip):
        if block.compartment.n_compartments > d(b'MTAyNA==', int):
            raise ValidationError("Segment does not fit on one chip",
                                  "n_neurons")

        core = chip.new_core()
        core.add_block(block)
        allocate_core(core)

    def input_to_chip(self, input, chip):
        chip.add_input(input)

    def dvs_to_chip(self, inputs, chip):  # noqa: C901
        dvs_inputs = [input for input in inputs if isinstance(input, DVSInput)]
        assert len(dvs_inputs) <= 1, "There can be only one"
        assert len(chip.cores) == 0
        if len(dvs_inputs) == 0:
            return inputs

        dvs_input = dvs_inputs[0]
        inputs = [input for input in inputs if input is not dvs_input]
        cores = [chip.new_core() for i in range(dvs_input.N_CORES)]

        if dvs_input.file_node is not None:
            e_t, e_idx = dvs_input.file_node.read_events(
                stride_xyp=(2*180, 2, 1), pool_xy=(1, 1))

            n_pins = 180*240*2
            spike_input = SpikeInput(n_pins)
            inputs.append(spike_input)

            dt = 0.001
            dt_us = dt * 1e6  # dt in us
            t = dvs_input.file_node.t_start * 1e6  # time in us
            ti = 0  # image index
            k = 0  # event index

            while t <= e_t[-1]:
                t += dt_us
                ti += 1
                k1 = k + np.searchsorted(e_t[k:], t)
                spike_input.add_spikes(ti, e_idx[k:k1])
                k = k1

            for i, core in enumerate(cores):
                n_neurons = min(n_pins - i*1024, 1024)
                block = LoihiBlock(n_neurons)
                block.compartment.decay_u[:] = 2**12 - 2
                block.compartment.decay_v[:] = 0
                block.compartment.scale_u = False
                block.compartment.scale_v = False
                block.compartment.vth[:] = 128
                block.compartment.vmin = 0
                block.compartment.vmax = 2**(9 + 2*7) - 1
                block.compartment.refract_delay[:] = 1
                core.add_block(block)

                synapse = Synapse(n_neurons)
                synapse._set_weights_indices(
                    weights=[255 for _ in range(n_neurons)],
                    indices=list(range(n_neurons)),
                )
                idxBits = synapse.idx_bits()
                synapse.format(compression=3, idxBits=idxBits, fanoutType=1,
                               numSynapses=63, wgtBits=7, wgtExp=0)
                block.add_synapse(synapse)

                axon = Axon(n_pins, label='DVS core %d' % i)
                axon.target = synapse
                target_axons = -np.ones(n_pins, dtype=int)
                target_axons[i*1024:i*1024 + n_neurons] = np.arange(n_neurons)
                axon.set_compartment_axon_map(target_axons)
                spike_input.add_axon(axon)

                # --- allocate block on core
                allocate_core(core)

        # --- set up axons from compartments to desired targets
        # cx_map maps logical (pooled, proper channel position) axons to
        # target synmaps. We need to figure out the actual mapping for pins
        stride_y = 2
        stride_x = 2 * 180
        stride_p = 1
        pool_y, pool_x = dvs_input.pool

        pin_map = collections.defaultdict(list)
        pin_atom = {}

        for axon in dvs_input.axons:
            cx_map = (np.arange(axon.n_axons)
                      if axon.compartment_map is None
                      else axon.compartment_map)
            cx_atoms = (np.zeros(axon.n_axons, dtype=int)
                        if axon.compartment_atoms is None
                        else axon.compartment_atoms)

            for cx, (taxon_idx, atom) in enumerate(zip(cx_map, cx_atoms)):
                if taxon_idx < 0:
                    continue

                # x, y, p coordinates in logical (pooled) image
                if dvs_input.channels_last:
                    p = cx % dvs_input.polarity
                    x = (cx // dvs_input.polarity) % dvs_input.width
                    y = (cx // (dvs_input.polarity * dvs_input.width))
                    assert y < dvs_input.height
                else:
                    x = cx % dvs_input.width
                    y = (cx // dvs_input.width) % dvs_input.height
                    p = (cx // (dvs_input.width * dvs_input.height))
                    assert p < dvs_input.polarity

                for yy in range(y*pool_y, (y+1)*pool_y):
                    for xx in range(x*pool_x, (x+1)*pool_x):
                        pin = yy*stride_y + xx*stride_x + p*stride_p
                        pin_map[pin].append((axon.target, taxon_idx))
                        assert pin_atom.setdefault(pin, atom) == atom

        for i, core in enumerate(cores):
            assert len(core.blocks) == 1
            block = core.blocks[0]

            pin0 = i * 1024
            targets = set()
            for pin in range(pin0, pin0 + block.n_neurons):
                for target, taxon_idx in pin_map[pin]:
                    targets.add(target)

            axon_idxs = {target: -1 * np.ones(block.n_neurons, dtype=int)
                         for target in targets}
            atoms = {target: np.zeros(block.n_neurons, dtype=int)
                     for target in targets}

            for k, pin in enumerate(range(pin0, pin0 + block.n_neurons)):
                for target, taxon_idx in pin_map[pin]:
                    axon_idxs[target][k] = taxon_idx
                    atoms[target][k] = pin_atom[pin]

            for target in targets:
                axon = Axon(block.n_neurons)
                axon.target = target
                axon.set_compartment_axon_map(axon_idxs[target],
                                              atoms=atoms[target])
                block.add_axon(axon)

        return inputs

    def __call__(self, model):
        board = Board()
        chip = board.new_chip()

        inputs = self.dvs_to_chip(model.inputs, chip)

        for input in inputs:
            self.input_to_chip(input, chip)

        for block in model.blocks:
            self.block_to_chip(block, chip)

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
            if len(board.chips) <= i < self.n_chips:
                board.new_chip()
            return board.chips[i % self.n_chips]

        assert not any(isinstance(input, DVSInput) for input in model.inputs)

        # TODO: inputs should go on chips based on what they're inputting to
        for input in model.inputs:
            self.input_to_chip(input, get_chip(0))

        i = 0
        for block in model.blocks:
            self.block_to_chip(block, get_chip(i))
            i += 1

        logger.info("Round-robin allocation across %d chips", board.n_chips)

        return board


class GreedyChip(OneToOne):
    def __init__(self, n_chips):
        self.n_chips = n_chips
        self.cores_per_chip = 128

    def __call__(self, model):
        board = Board()
        board.new_chip()

        def get_chip():
            chip = board.chips[-1]
            assert len(chip.cores) <= self.cores_per_chip
            if len(chip.cores) == self.cores_per_chip:
                assert len(board.chips) < self.n_chips
                chip = board.new_chip()

            return chip

        inputs = self.dvs_to_chip(model.inputs, get_chip())

        # TODO: inputs should go on chips based on what they're inputting to
        for input in inputs:
            self.input_to_chip(input, get_chip())

        for block in model.blocks:
            self.block_to_chip(block, get_chip())

        logger.info("Greedy allocation across %d chips", len(board.chips))

        return board
