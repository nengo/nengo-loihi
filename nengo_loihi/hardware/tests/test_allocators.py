import nengo
import numpy as np
import pytest
from nengo.exceptions import ValidationError
from nengo.utils.numpy import rms

from nengo_loihi.block import Axon, LoihiBlock, Synapse
from nengo_loihi.builder import Model
from nengo_loihi.builder.discretize import discretize_model
from nengo_loihi.hardware.allocators import Greedy, RoundRobin, core_stdp_pre_cfgs
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


@pytest.mark.slow
@pytest.mark.target_loihi
def test_deterministic_network_allocation(Simulator, seed):
    # test that we get the same simulations results across allocators.
    # the determinism of the allocation itself is covered by other unit tests.
    n_neurons = 64
    n_ensembles = 8
    tau = 0.1
    sim_t = 1.0

    with nengo.Network(seed=seed) as model:
        prev = nengo.Node(output=1)

        p = []
        for i in range(n_ensembles):
            ens = nengo.Ensemble(n_neurons, 1)
            nengo.Connection(prev, ens, synapse=tau)
            p.append(nengo.Probe(ens, synapse=tau))
            prev = ens

    with nengo.Simulator(model) as sim_ref:
        sim_ref.run(sim_t)

    # one block each for ensemble, connection, probe, minus no final connection
    n_blocks = n_ensembles * 3 - 1
    allocation = [
        (1, 1, RoundRobin()),
        (3, 3, RoundRobin()),
        (8, 8, RoundRobin()),
        (6, ceil_div(n_blocks, 4), Greedy(cores_per_chip=4)),
        (8, ceil_div(n_blocks, 5), Greedy(cores_per_chip=5)),
    ]

    sim_prev = None
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
            assert rms(sim_loihi.data[p_i] - sim_ref.data[p_i]) < 0.05
            if sim_prev is not None:
                assert np.allclose(sim_prev.data[p_i], sim_loihi.data[p_i])
        sim_prev = sim_loihi
