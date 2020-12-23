import collections
import logging

import numpy as np
from nengo.exceptions import ValidationError

from nengo_loihi.block import Axon, LoihiBlock, Synapse
from nengo_loihi.builder.discretize import tracing_mag_int_frac, vth_to_manexp
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
            cfgs.append(
                CompartmentConfig(
                    decay_u=block.compartment.decay_u[i],
                    decay_v=block.compartment.decay_v[i],
                    refract_delay=block.compartment.refract_delay[i],
                    enable_noise=block.compartment.enable_noise[i],
                    bap_action=block.compartment.bap_action[i],
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

    def block_to_new_core(self, block, chip):
        """Assign a block to a new core on the chip.

        Parameters
        ----------
        block : LoihiBlock
            The block to allocate.
        chip : Chip
            The chip from which to obtain a new core.
        """
        if block.compartment.n_compartments > d(b"MTAyNA==", int):
            raise ValidationError("Segment does not fit on one core", "n_neurons")

        core = chip.new_core()
        core.add_block(block)
        allocate_core(core)

    def input_to_board(self, input, board):
        """Assign an input to a board.

        Parameters
        ----------
        input : LoihiInput
            The input to allocate.
        board : Board
            The board on which to place the input.
        """
        board.add_input(input)

    def __call__(self, model, n_chips):
        """Returns a Board object corresponding to the given model."""
        raise NotImplementedError()

    def dvs_to_chip(self, inputs, chip):  # noqa: C901
        dvs_inputs = [input for input in inputs if isinstance(input, DVSInput)]
        assert len(dvs_inputs) <= 1, "There can be only one"
        assert len(chip.cores) == 0
        if len(dvs_inputs) == 0:
            return inputs

        dvs_input = dvs_inputs[0]
        inputs = [input for input in inputs if input is not dvs_input]
        cores = [chip.new_core() for i in range(dvs_input.N_CORES)]

        # these strides are based on how NxSDK maps live DVS to cores
        stride_y = 2
        stride_x = 2 * 180
        stride_p = 1

        if dvs_input.file_node is not None:
            e_t, e_idx = dvs_input.file_node._read_events(
                stride_yxp=(stride_y, stride_x, stride_p), pool_yx=(1, 1)
            )

            n_pins = 180 * 240 * 2
            spike_input = SpikeInput(n_pins)
            inputs.append(spike_input)

            dt = 0.001  # TODO: get simulator dt
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
                n_neurons = min(n_pins - i * 1024, 1024)
                block = LoihiBlock(n_neurons)
                block.compartment.decay_u[:] = 2 ** 12 - 2
                block.compartment.decay_v[:] = 0
                block.compartment.scale_u = False
                block.compartment.scale_v = False
                block.compartment.vth[:] = 128
                block.compartment.vmin = 0
                block.compartment.vmax = 2 ** (9 + 2 * 7) - 1
                block.compartment.refract_delay[:] = 1
                core.add_block(block)

                synapse = Synapse(n_neurons)
                synapse._set_weights_indices(
                    weights=[255 for _ in range(n_neurons)],
                    indices=list(range(n_neurons)),
                )
                idx_bits = synapse.idx_bits()
                synapse.format(
                    compression=3,
                    idx_bits=idx_bits,
                    fanout_type=1,
                    n_synapses=63,
                    weight_bits=7,
                    weight_exp=0,
                )
                block.add_synapse(synapse)

                axon = Axon(n_pins, label="DVS core %d" % i)
                axon.target = synapse
                target_axons = -np.ones(n_pins, dtype=int)
                target_axons[i * 1024 : i * 1024 + n_neurons] = np.arange(n_neurons)
                axon.set_compartment_axon_map(target_axons)
                spike_input.add_axon(axon)

                # --- allocate block on core
                allocate_core(core)
        else:
            chip.add_input(dvs_input)

            # Add empty blocks to reserve them for DVS. The parameters will be
            # loaded by the HardwareInterface using loadNeuroCores.
            n_pins = 180 * 240 * 2
            for i, core in enumerate(cores):
                core.build_axons_only = True

                n_neurons = min(n_pins - i * 1024, 1024)
                block = LoihiBlock(n_neurons)
                core.add_block(block)

        # --- set up axons from compartments to desired targets
        # `dvs_input.axons` map logical (pooled, proper channel position) axons to
        # target synmaps. `pin_map` will contain the actual mapping for pins.
        pool_y, pool_x = dvs_input.pool

        pin_map = collections.defaultdict(list)
        pin_atom = {}

        for axon in dvs_input.axons:
            # logical compartment indices
            compartment_idxs = np.arange(dvs_input.size)
            target_axon_idxs = axon.map_axon(compartment_idxs)
            target_atoms = axon.map_atoms(compartment_idxs)

            # x, y, p coordinates in logical (pooled) image
            if dvs_input.channels_last:
                comp_y, comp_x, comp_p = np.unravel_index(
                    compartment_idxs,
                    (dvs_input.height, dvs_input.width, dvs_input.polarity)
                )
            else:
                comp_p, comp_y, comp_x = np.unravel_index(
                    compartment_idxs,
                    (dvs_input.polarity, dvs_input.height, dvs_input.width)
                )

            for x, y, p, taxon_idx, atom in zip(
                comp_x, comp_y, comp_p, target_axon_idxs, target_atoms
            ):
                if taxon_idx < 0:
                    continue

                X, Y = np.meshgrid(
                    np.arange(x * pool_x, (x + 1) * pool_x),
                    np.arange(y * pool_y, (y + 1) * pool_y),
                )
                pins = X.ravel() * stride_x + Y.ravel() * stride_y + p * stride_p
                for pin in pins:
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

            axon_idxs = {
                target: -1 * np.ones(block.n_neurons, dtype=int) for target in targets
            }
            atoms = {target: np.zeros(block.n_neurons, dtype=int) for target in targets}

            for k, pin in enumerate(range(pin0, pin0 + block.n_neurons)):
                for target, taxon_idx in pin_map[pin]:
                    axon_idxs[target][k] = taxon_idx
                    atoms[target][k] = pin_atom[pin]

            for target in targets:
                axon = Axon(block.n_neurons)
                axon.target = target
                axon.set_compartment_axon_map(axon_idxs[target], atoms=atoms[target])
                block.add_axon(axon)

        return inputs


class Greedy(Allocator):
    """Assigns each block to distinct cores on as few chips as possible.

    Parameters
    ----------
    cores_per_chip : int, optional (Default: 128)
        Number of cores to use on each chip.
    """

    def __init__(self, cores_per_chip=128):
        if cores_per_chip > 128:
            raise ValueError("Chips cannot have more than 128 cores")

        self.cores_per_chip = cores_per_chip

    def __call__(self, model, n_chips):
        board = Board()
        board.new_chip()

        def get_chip():
            chip = board.chips[-1]
            assert len(chip.cores) <= self.cores_per_chip
            if len(chip.cores) == self.cores_per_chip:
                assert len(board.chips) < n_chips, (
                    "The network needs more chips than requested (%d)" % n_chips,
                )
                chip = board.new_chip()

            return chip

        dvs_chip = get_chip()
        inputs = self.dvs_to_chip(model.inputs, dvs_chip)

        for input in inputs:
            self.input_to_board(input, board)

        for block in model.blocks:
            self.block_to_new_core(block, get_chip())

        board.probes.extend(model.probes)

        logger.info("Greedy allocation across %d chips", board.n_chips)

        return board


class RoundRobin(Allocator):
    """Assigns each block to distinct cores on as many chips as possible.

    Each chip is used in round-robin order.
    """

    def __call__(self, model, n_chips):
        board = Board()

        # We must dynamically allocate the chips
        # as needed because nxsdk==0.8.0 hits
        # an assertion if any chips contain 0 cores
        def get_chip(i):
            if len(board.chips) <= i < n_chips:
                board.new_chip()
            return board.chips[i % n_chips]

        assert not any(isinstance(input, DVSInput) for input in model.inputs)

        for input in model.inputs:
            self.input_to_board(input, board)

        i = 0
        for block in model.blocks:
            self.block_to_new_core(block, get_chip(i))
            i += 1

        board.probes.extend(model.probes)

        logger.info("Round-robin allocation across %d chips", board.n_chips)

        return board
