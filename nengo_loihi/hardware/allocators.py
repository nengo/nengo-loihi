import logging

from nengo.exceptions import ValidationError
import numpy as np

from nengo_loihi.builder.discretize import tracing_mag_int_frac, vth_to_manexp
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

        def get_chip(i):
            chip = board.chips[-1]
            assert len(chip.cores) <= self.cores_per_chip
            if len(chip.cores) == self.cores_per_chip:
                assert len(board.chips) < n_chips, (
                    "The network needs more chips than requested (%d)" % n_chips,
                )
                chip = board.new_chip()

            return chip

        i = 0
        for input in model.inputs:
            self.input_to_board(input, board)
            i += 1

        for block in model.blocks:
            self.block_to_new_core(block, get_chip(i))
            i += 1

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

        for input in model.inputs:
            self.input_to_board(input, board)

        i = 0
        for block in model.blocks:
            self.block_to_new_core(block, get_chip(i))
            i += 1

        board.probes.extend(model.probes)

        logger.info("Round-robin allocation across %d chips", board.n_chips)

        return board


def ens_to_block_rates(model, ens_rates):
    block_rates = {}
    for ens, rates in ens_rates.items():
        if ens not in model.objs:
            if ens in model.host_pre.sig or ens in model.host.sig:
                continue  # this ensemble is not on chip, so skip it
            else:
                raise ValueError("Ensemble %s does not appear in the model" % (ens,))

        assert len(rates) == ens.n_neurons
        blocks = model.objs[ens]["out"]
        blocks = blocks if isinstance(blocks, (list, tuple)) else [blocks]

        for block in blocks:
            comp_idxs = model.block_comp_map.get(block, None)
            if comp_idxs is None:
                assert len(blocks) == 1
                assert block.compartment.n_compartments == ens.n_neurons
                block_rates[block] = rates
            else:
                block_rates[block] = rates[comp_idxs]

    return block_rates


def compute_block_conns(block_map, block_rates=None, conns_in=False):
    # --- store number of axons from block i to block j
    block_conns = {k: {} for k in block_map}
    if conns_in:
        block_conns_in = {k: {} for k in block_map}

    synapse_block_map = {}
    for i, block_i in block_map.items():
        for synapse in block_i.synapses:
            assert id(synapse) not in synapse_block_map
            synapse_block_map[id(synapse)] = i

    for i, block_i in block_map.items():
        for axon in block_i.axons:
            j = synapse_block_map[id(axon.target)]

            if i == j:
                # don't care about self connections
                continue

            # use non-zero value as default, so that even if all rates are zero, this
            # still gets recognized as a connection from i to j
            block_conns[i].setdefault(j, 1e-16)
            if conns_in:
                block_conns_in[j].setdefault(i, 1e-16)

            if block_rates is None:
                val = axon.n_axons
            elif block_i not in block_rates:
                raise KeyError("block %s not in block_rates" % (block_i,))
            else:
                rates = block_rates[block_i]
                comp_idxs = np.arange(block_i.compartment.n_compartments)
                axon_ids = axon.map_axon(comp_idxs)
                assert axon_ids.size == rates.size
                val = rates[axon_ids >= 0].sum()

            block_conns[i][j] += val
            if conns_in:
                block_conns_in[j][i] += val

    return (block_conns, block_conns_in) if conns_in else block_conns


def measure_interchip_conns(board, block_rates=None):
    i = 0
    block_map = {}
    block_chip = {}
    for chip in board.chips:
        chip_idx = board.chip_idxs[chip]
        for core in chip.cores:
            # core_idx = chip.core_idxs[core]
            for block in core.blocks:
                block_map[i] = block
                block_chip[i] = chip_idx
                i += 1

    block_conns = compute_block_conns(block_map, block_rates=block_rates)

    stats = {"interchip": 0, "intrachip": 0}
    stats["interchip_pairs"] = []
    stats["intrachip_pairs"] = []
    for i, block in block_map.items():
        chip_idx_i = block_chip[i]
        for j, weight in block_conns[i].items():
            if i == j:
                continue

            chip_idx_j = block_chip[j]
            key = "intrachip" if chip_idx_i == chip_idx_j else "interchip"
            stats[key] += weight
            stats["%s_pairs" % (key,)].append((block_map[i], block_map[j]))

    return stats


class GreedyComms(Greedy):
    """Assigns each block to a core, using as few chips as possible, minimizing comms.

    A variant of the `.Greedy` allocator that also minimizes inter-chip communication.

    Starts by arbitrarily assigning a block to a chip. Then adds the block that has the
    most communication with the first block to that same chip. Continue adding blocks
    with the most communication to already placed blocks, until the chip is full. Then
    start a new chip using the block with the least communication.
    """

    def __init__(self, cores_per_chip=128, ensemble_rates=None):
        super().__init__(cores_per_chip=cores_per_chip)
        self.ensemble_rates = ensemble_rates

    def __call__(self, model, n_chips):
        block_map = {k: block for k, block in enumerate(model.blocks)}
        block_rates = (
            ens_to_block_rates(model, self.ensemble_rates)
            if self.ensemble_rates is not None
            else None
        )
        block_conns_out, block_conns_in = compute_block_conns(
            block_map, block_rates=block_rates, conns_in=True
        )

        # find blocks with no pre block
        no_pre_blocks = []
        for i in block_map:
            if sum(v for v in block_conns_in[i].values()) == 0:
                no_pre_blocks.append(i)

        # --- create board
        board = Board()

        # add inputs to board
        for input in model.inputs:
            self.input_to_board(input, board)

        # --- add blocks to chips
        chip = None
        unallocated_blocks = set(block_map)

        while len(unallocated_blocks) > 0:
            if chip is None or len(chip.cores) == self.cores_per_chip:
                assert len(board.chips) < n_chips, (
                    "The network needs more chips than requested (%d)" % n_chips,
                )

                # start a new chip
                chip = board.new_chip()

                # choose a no-pre block, if possible
                for block_idx in no_pre_blocks:
                    if block_idx in unallocated_blocks:
                        break
                else:
                    block_idx = next(iter(unallocated_blocks))

                chip_blocks = set()
            else:
                # choose the block with the largest connection to blocks on this chip
                block_idx = -1
                max_conn = 0
                for i in chip_blocks:
                    for j in unallocated_blocks.intersection(block_conns_out[i]):
                        ij = block_conns_out[i][j]
                        if ij > max_conn:
                            max_conn = ij
                            block_idx = j

                    for j in unallocated_blocks.intersection(block_conns_in[i]):
                        ij = block_conns_in[i][j]
                        if ij > max_conn:
                            max_conn = ij
                            block_idx = j

                if block_idx < 0:
                    # none of the remaining blocks connect to blocks on this chip,
                    # so pick a no-pre block if possible, otherwise any block will do.
                    for block_idx in no_pre_blocks:
                        if block_idx in unallocated_blocks:
                            break
                    else:
                        block_idx = next(iter(unallocated_blocks))

            block = block_map[block_idx]
            self.block_to_new_core(block, chip)

            chip_blocks.add(block_idx)
            unallocated_blocks.remove(block_idx)

        # add probes
        board.probes.extend(model.probes)

        logger.info("Greedy allocation across %d chips", board.n_chips)

        return board


class PartitionComms(Allocator):
    """Uses METIS partitioner to spread blocks across all chips, minimizing comms.

    Spreads blocks equally across cores and minimizes inter-chip communication.

    TODO:
    - Potentially allow more blocks on one chip, if it will improve communication.
    - Check that partitioning is always balanced, and no chips will have too many cores.
    """

    def __init__(self, cores_per_chip=128, ensemble_rates=None):
        import networkx
        import nxmetis

        super().__init__()
        # super().__init__(cores_per_chip=cores_per_chip)
        self.ensemble_rates = ensemble_rates
        if ensemble_rates is not None:
            raise NotImplementedError(
                "Rate-based optimization not implemented, since METIS requires "
                "integer weights."
            )

        self.networkx = networkx
        self.nxmetis = nxmetis

    def __call__(self, model, n_chips):
        block_map = {k: block for k, block in enumerate(model.blocks)}
        block_rates = (
            ens_to_block_rates(model, self.ensemble_rates)
            if self.ensemble_rates is not None
            else None
        )
        block_conns = compute_block_conns(block_map, block_rates=block_rates)

        # partition graph
        G = self.networkx.Graph()
        G.add_nodes_from(block_map.keys())

        edge_map = set()
        for i in block_map:
            for j, val in block_conns[i].items():
                if (i, j) in edge_map or (j, i) in edge_map:
                    continue

                val = val + block_conns[j].get(i, 0)
                # G.add_edge(i, j, weight=float(val))
                G.add_edge(i, j, weight=int(round(val)))  # weights must be integers
                edge_map.add((i, j))
                edge_map.add((j, i))

        objval, parts = self.nxmetis.partition(G, nparts=int(n_chips))

        for i, part in enumerate(parts):
            if len(part) > 128:
                raise ValueError(
                    "Partition %d has %d cores, which exceeds the available 128 cores"
                    % (i, len(part))
                )

        # --- create board
        board = Board()

        # add inputs to board
        for input in model.inputs:
            self.input_to_board(input, board)

        # blocks to chips
        for part in parts:
            chip = board.new_chip()
            for block_idx in part:
                block = block_map[block_idx]
                self.block_to_new_core(block, chip)

        # add probes
        board.probes.extend(model.probes)

        logger.info("METIS allocation across %d chips", board.n_chips)

        return board
