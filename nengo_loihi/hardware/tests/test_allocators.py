import nengo
from nengo.exceptions import BuildError, ValidationError
from nengo.utils.numpy import rms
import numpy as np
import pytest

from nengo_loihi.block import LoihiBlock, Synapse, Axon
from nengo_loihi.builder import Model
from nengo_loihi.discretize import discretize_model
from nengo_loihi.hardware.allocators import core_stdp_pre_cfgs, OneToOne, RoundRobin
from nengo_loihi.hardware.nxsdk_objects import Board
from nengo_loihi.inputs import LoihiInput


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


def test_block_size(Simulator):
    with nengo.Network() as net:
        nengo.Ensemble(1024, 1)

    # n_neurons within limit, no problem
    with Simulator(net) as sim:
        sim.run_steps(5)

    with nengo.Network() as net:
        nengo.Ensemble(1025, 1)
    with pytest.raises(BuildError, match="Number of compartments"):
        with Simulator(net):
            pass


def test_one_to_one_allocator_big_block_error():
    model = Model()
    model.add_block(LoihiBlock(1050))

    with pytest.raises(ValidationError, match="Segment does not fit"):
        OneToOne()(model)


def _basic_model():
    model = Model()

    block0 = LoihiBlock(1)
    block0.compartment.configure_lif()
    model.add_block(block0)

    block1 = LoihiBlock(1)
    block1.compartment.configure_lif()
    model.add_block(block1)

    axon1 = Axon(1)
    block0.add_axon(axon1)

    synapse1 = Synapse(1)
    synapse1.set_weights([[1]])
    axon1.target = synapse1
    block1.add_synapse(synapse1)

    axon0 = Axon(1)
    input = LoihiInput()
    input.add_axon(axon0)
    model.add_input(input)

    synapse0 = Synapse(1)
    synapse0.set_weights([[1]])
    axon0.target = synapse0
    block0.add_synapse(synapse0)

    discretize_model(model)

    return model


@pytest.mark.parametrize("allocator", [OneToOne(), RoundRobin(n_chips=1)])
def test_one_to_one_allocator(allocator):
    # RoundRobin(n_chips=1) is equivalent to OneToOne()
    model = _basic_model()
    board = allocator(model)

    assert board.n_chips == 1
    assert board.n_cores_per_chip == [3]
    assert board.n_synapses_per_core == [[1, 1, 0]]

    chip = board.chips[0]
    assert chip.board is board
    assert chip.n_cores == 3

    assert chip.cores[0].chip is chip
    assert len(chip.cores[0].synapses) == 1
    assert len(chip.cores[0].blocks) == 1
    assert len(chip.cores[0].inputs) == 0

    assert chip.cores[1].chip is chip
    assert len(chip.cores[1].synapses) == 1
    assert len(chip.cores[1].blocks) == 1
    assert len(chip.cores[1].inputs) == 0

    assert chip.cores[2].chip is chip
    assert len(chip.cores[2].synapses) == 0
    assert len(chip.cores[2].blocks) == 0
    assert len(chip.cores[2].inputs) == 1


def test_round_robin_allocator_under():
    model = _basic_model()

    board = RoundRobin(n_chips=2)(model)

    assert board.n_chips == 2
    assert board.n_cores_per_chip == [2, 1]
    assert board.n_synapses_per_core == [[1, 0], [1]]

    chip0 = board.chips[0]
    assert chip0.board is board
    assert chip0.n_cores == 2

    assert chip0.cores[0].chip is chip0
    assert len(chip0.cores[0].synapses) == 1
    assert len(chip0.cores[0].blocks) == 1
    assert len(chip0.cores[0].inputs) == 0

    assert chip0.cores[1].chip is chip0
    assert len(chip0.cores[1].synapses) == 0
    assert len(chip0.cores[1].blocks) == 0
    assert len(chip0.cores[1].inputs) == 1

    chip1 = board.chips[1]
    assert chip1.board is board
    assert chip1.n_cores == 1

    assert chip1.cores[0].chip is chip1
    assert len(chip1.cores[0].synapses) == 1
    assert len(chip1.cores[0].blocks) == 1
    assert len(chip1.cores[0].inputs) == 0


def test_round_robin_allocator_over():
    model = _basic_model()

    board = RoundRobin(n_chips=4)(model)

    assert board.n_chips == 3
    assert board.n_cores_per_chip == [1, 1, 1]
    assert board.n_synapses_per_core == [[1], [1], [0]]

    chip0 = board.chips[0]
    assert chip0.board is board
    assert chip0.n_cores == 1

    assert chip0.cores[0].chip is chip0
    assert len(chip0.cores[0].synapses) == 1
    assert len(chip0.cores[0].blocks) == 1
    assert len(chip0.cores[0].inputs) == 0

    chip1 = board.chips[1]
    assert chip1.board is board
    assert chip1.n_cores == 1

    assert chip1.cores[0].chip is chip1
    assert len(chip1.cores[0].synapses) == 1
    assert len(chip1.cores[0].blocks) == 1
    assert len(chip1.cores[0].inputs) == 0

    chip2 = board.chips[2]
    assert chip2.board is board
    assert chip2.n_cores == 1

    assert chip2.cores[0].chip is chip2
    assert len(chip2.cores[0].synapses) == 0
    assert len(chip2.cores[0].blocks) == 0
    assert len(chip2.cores[0].inputs) == 1


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

    allocation = [
        (1, OneToOne()),
        (1, RoundRobin(n_chips=1)),
        (3, RoundRobin(n_chips=3)),
        (8, RoundRobin(n_chips=8)),
    ]

    sim_prev = None
    for n_chips_used, allocator in allocation:
        with Simulator(
            model, precompute=True, hardware_options={"allocator": allocator}
        ) as sim_loihi:
            sim_loihi.run(sim_t)

        assert n_chips_used == sim_loihi.sims["loihi"].board.n_chips
        for p_i in p:
            assert rms(sim_loihi.data[p_i] - sim_ref.data[p_i]) < 0.05
            if sim_prev is not None:
                assert np.allclose(sim_prev.data[p_i], sim_loihi.data[p_i])
        sim_prev = sim_loihi
