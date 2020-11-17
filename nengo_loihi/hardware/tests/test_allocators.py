import nengo
import numpy as np
import pytest
from nengo.exceptions import ValidationError

import nengo_loihi
from nengo_loihi.block import Axon, LoihiBlock, Synapse
from nengo_loihi.builder import Model
from nengo_loihi.builder.discretize import discretize_model
from nengo_loihi.hardware.allocators import (
    HAS_NXMETIS,
    Greedy,
    GreedyInterchip,
    PartitionInterchip,
    RoundRobin,
    core_stdp_pre_cfgs,
    ensemble_to_block_rates,
    estimate_interchip_activity,
)
from nengo_loihi.hardware.nxsdk_objects import Board
from nengo_loihi.inputs import LoihiInput


def ceil_div(a, b):
    return -((-a) // b)


def test_core_stdp_pre_cfgs():
    core = Board().new_chip().new_core()

    def new_syn(tracing_mag=None):
        syn = Synapse(n_axons=1)
        syn.set_weights(np.array([[1]]))
        if tracing_mag is not None:
            syn.set_learning(tracing_mag=tracing_mag)
        core.add_synapse(syn)
        return syn

    profile_idxs = {}
    # Do this one by one to guarantee order of created tracecfgs
    profile_idxs[new_syn(0.1)] = 0
    profile_idxs[new_syn(0.2)] = 1
    profile_idxs[new_syn(0.2)] = 1
    profile_idxs[new_syn(0.3)] = 2
    profile_idxs[new_syn(0.3)] = 2
    profile_idxs[new_syn()] = None

    profiles, ret_idxs = core_stdp_pre_cfgs(core)
    assert len(profiles) == 3
    assert ret_idxs == profile_idxs


def test_big_block_error():
    model = Model()
    model.add_block(LoihiBlock(1050))

    with pytest.raises(ValidationError, match="Segment does not fit"):
        Greedy()(model, n_chips=1)


def _basic_model(n_blocks=2):
    model = Model()

    blocks = []
    for _ in range(n_blocks):
        block = LoihiBlock(1)
        block.compartment.configure_lif()
        model.add_block(block)
        blocks.append(block)

    for i in range(n_blocks - 1):
        axon = Axon(1)
        blocks[i].add_axon(axon)

        synapse = Synapse(1)
        synapse.set_weights([[1]])
        axon.target = synapse
        blocks[i + 1].add_synapse(synapse)

    axon0 = Axon(1)
    input = LoihiInput()
    input.add_axon(axon0)
    model.add_input(input)

    synapse0 = Synapse(1)
    synapse0.set_weights([[1]])
    axon0.target = synapse0
    blocks[0].add_synapse(synapse0)

    discretize_model(model)

    return model


@pytest.mark.parametrize("allocator", [Greedy(), RoundRobin()])
def test_basic(allocator):
    # RoundRobin is equivalent to Greedy when n_chips == 1
    n_blocks = 3
    model = _basic_model(n_blocks=n_blocks)
    board = allocator(model, n_chips=1)

    assert board.n_chips == 1
    assert board.n_cores_per_chip == [n_blocks]
    assert board.n_synapses_per_core == [[1] * n_blocks]
    assert len(board.inputs) == 1

    chip = board.chips[0]
    assert chip.board is board
    assert chip.n_cores == n_blocks

    for i in range(n_blocks):
        assert chip.cores[i].chip is chip
        assert len(chip.cores[i].synapses) == 1
        assert len(chip.cores[i].blocks) == 1


def test_round_robin_allocator_under():
    model = _basic_model(n_blocks=3)

    board = RoundRobin()(model, n_chips=2)

    assert board.n_chips == 2
    assert board.n_cores_per_chip == [2, 1]
    assert board.n_synapses_per_core == [[1, 1], [1]]
    assert len(board.inputs) == 1

    chip = board.chips[0]
    assert chip.board is board
    assert chip.n_cores == 2

    for i in range(2):
        assert chip.cores[i].chip is chip
        assert len(chip.cores[i].synapses) == 1
        assert len(chip.cores[i].blocks) == 1

    chip = board.chips[1]
    assert chip.board is board
    assert chip.n_cores == 1

    assert chip.cores[0].chip is chip
    assert len(chip.cores[0].synapses) == 1
    assert len(chip.cores[0].blocks) == 1


def test_round_robin_allocator_over():
    model = _basic_model(n_blocks=3)

    board = RoundRobin()(model, n_chips=4)

    assert board.n_chips == 3
    assert board.n_cores_per_chip == [1, 1, 1]
    assert board.n_synapses_per_core == [[1], [1], [1]]
    assert len(board.inputs) == 1

    for i in range(3):
        chip = board.chips[i]
        assert chip.board is board
        assert chip.n_cores == 1

        assert chip.cores[0].chip is chip
        assert len(chip.cores[0].synapses) == 1
        assert len(chip.cores[0].blocks) == 1


def test_greedy_chip_allocator_cfg_check():
    model = _basic_model(n_blocks=400)

    with pytest.raises(AssertionError, match="The network needs more chips"):
        Greedy()(model, n_chips=2)

    with pytest.raises(ValueError, match="Chips cannot have more than 128 cores"):
        Greedy(cores_per_chip=130)(model, n_chips=4)


@pytest.mark.parametrize("Allocator", [GreedyInterchip, PartitionInterchip])
def test_interchip_allocators(Allocator, Simulator):
    if Allocator is PartitionInterchip:
        pytest.importorskip("nxmetis")

    rng = np.random.RandomState(1)  # same seed for all allocators, to compare
    with nengo.Network(seed=0) as net:
        n_ensembles = 256
        n_neurons = rng.randint(64, 256, size=n_ensembles)
        ensembles = [nengo.Ensemble(n, dimensions=1) for n in n_neurons]

        conn_pairs = rng.randint(0, n_ensembles, size=(2 * n_ensembles, 2))
        for i, j in conn_pairs:
            ei, ej = ensembles[i].neurons, ensembles[j].neurons
            nengo.Connection(
                ei,
                ej,
                transform=rng.uniform(-0.1, 0.1, size=(ej.size_in, ei.size_out)),
            )

    ens_rates = {
        ensemble: rng.uniform(1, 100, size=1)
        * rng.uniform(0.9, 1, size=ensemble.n_neurons)
        for ensemble in ensembles
    }

    with Simulator(net, target="sim") as sim:
        model = sim.model
        n_chips = 3
        block_rates = ensemble_to_block_rates(model, ens_rates)
        sim.timers.start("norates")
        board_norates = Allocator()(model, n_chips=n_chips)
        sim.timers.stop("norates")
        sim.timers.start("rates")
        board_rates = Allocator(ensemble_rates=ens_rates)(model, n_chips=n_chips)
        sim.timers.stop("rates")

    norates_axons = estimate_interchip_activity(board_norates)
    norates_spikes = estimate_interchip_activity(board_norates, block_rates=block_rates)
    rates_axons = estimate_interchip_activity(board_rates)
    rates_spikes = estimate_interchip_activity(board_rates, block_rates=block_rates)

    print(f"Using allocator {Allocator.__name__}")
    print(
        f"No rates: {norates_axons['interchip']} axons, "
        f"{norates_spikes['interchip']:.2f} spikes, "
        f"took {sim.timers['norates']:.3f} seconds"
    )
    print(
        f"Rates: {rates_axons['interchip']} axons, "
        f"{rates_spikes['interchip']:.2f} spikes, "
        f"took {sim.timers['rates']:.3f} seconds"
    )
    assert norates_axons["interchip"] < rates_axons["interchip"]
    assert rates_spikes["interchip"] < norates_spikes["interchip"]


def test_interchip_helpers(Simulator, rng):
    """Test other cases for helper functions used by Interchip allocators."""

    with nengo.Network() as net:
        nengo_loihi.add_params(net)
        a = nengo.Ensemble(2000, 1)  # this ensemble will be split (2 blocks)
        b = nengo.Ensemble(1000, 1)  # this ensemble will be one block
        c = nengo.Ensemble(1000, 1)  # this ensemble will be off-chip
        net.config[c].on_chip = False
        nengo.Connection(a, b)

    with nengo.Network():
        d = nengo.Ensemble(10, 1)  # this ensemble is in a different network (errors)

    ens_rates = {
        a: rng.uniform(size=a.n_neurons),
        b: rng.uniform(size=b.n_neurons),
        c: rng.uniform(size=c.n_neurons),
    }

    with Simulator(net) as sim:
        # --- test ensemble_to_block_rates
        block_rates = ensemble_to_block_rates(sim.model, ens_rates)

        a_blocks = sim.model.objs[a]["out"]
        b_blocks = sim.model.objs[b]["out"]
        assert set(block_rates) == (set(a_blocks) | set(b_blocks))

        i = 0
        for block in a_blocks:
            assert np.array_equal(
                ens_rates[a][i : i + block.n_neurons], block_rates[block]
            )
            i += block.n_neurons

        assert len(b_blocks) == 1 and np.array_equal(
            ens_rates[b], block_rates[b_blocks[0]]
        )

        # test ValueError if ensemble not in model
        ens_rates[d] = rng.uniform(size=d.n_neurons)
        with pytest.raises(ValueError, match="Ensemble.*does not appear in the model"):
            ensemble_to_block_rates(sim.model, ens_rates)

        # --- test estimate_interblock_activity error
        partial_ens_rates = {a: ens_rates[a]}
        with pytest.raises(KeyError, match="block.*not in block_rates"):
            GreedyInterchip(ensemble_rates=partial_ens_rates)(sim.model, n_chips=2)


@pytest.mark.target_loihi
# Unclear why this requires nahuku32 or poihiki, as the basic multichip functionality
# in this test should run on nahuku08, but currently it's hanging there.
@pytest.mark.requires_multichip_snips
def test_allocator_integration_consistency(Simulator, seed, allclose):
    # test that we get the same simulations results across allocators.
    # the determinism of the allocation itself is covered by other unit tests.
    n_neurons = 64
    n_ensembles = 8
    probe_tau = 0.01
    sim_t = 0.1

    with nengo.Network(seed=seed) as model:
        prev = nengo.Node(output=1)

        p = []
        for _ in range(n_ensembles):
            ens = nengo.Ensemble(n_neurons, 1)
            nengo.Connection(prev, ens)
            p.append(nengo.Probe(ens, synapse=probe_tau))
            prev = ens

    with Simulator(model, target="sim") as sim_ref:
        sim_ref.run(sim_t)

    # one block each for ensemble, connection, probe, minus no final connection
    n_blocks = n_ensembles * 3 - 1
    allocation = [
        (1, 1, RoundRobin()),
        (7, 7, RoundRobin()),
        (6, ceil_div(n_blocks, 4), Greedy(cores_per_chip=4)),
        (8, ceil_div(n_blocks, 5), Greedy(cores_per_chip=5)),
        (6, ceil_div(n_blocks, 4), GreedyInterchip(cores_per_chip=4)),
    ]
    if HAS_NXMETIS:
        allocation.append((6, 6, PartitionInterchip()))

    for n_chips, n_chips_used, allocator in allocation:
        with Simulator(
            model,
            precompute=True,
            hardware_options={"n_chips": n_chips, "allocator": allocator},
        ) as sim_loihi:
            sim_loihi.run(sim_t)

        assert len(sim_loihi.model.blocks) == n_blocks
        assert n_chips_used == sim_loihi.sims["loihi"].board.n_chips
        for p_i in p:
            assert allclose(sim_loihi.data[p_i], sim_ref.data[p_i]), allocator
