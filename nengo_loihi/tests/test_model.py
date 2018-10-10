import numpy as np

from nengo_loihi.block import Axon, LoihiBlock, Probe, Synapse
from nengo_loihi.builder import Model
from nengo_loihi.discretize import discretize_model
from nengo_loihi.emulator import EmulatorInterface
from nengo_loihi.hardware import HardwareInterface
from nengo_loihi.inputs import SpikeInput


def test_simulator_noise(request, plt, seed):
    target = request.config.getoption("--target")
    n_cx = 10

    model = Model()
    block = LoihiBlock(n_cx)
    block.compartment.configure_relu()

    block.compartment.bias[:] = np.linspace(0, 0.01, n_cx)

    block.compartment.enableNoise[:] = 1
    block.compartment.noiseExp0 = -2
    block.compartment.noiseMantOffset0 = 0
    block.compartment.noiseAtDendOrVm = 1

    probe = Probe(target=block, key='voltage')
    block.add_probe(probe)
    model.add_block(block)

    discretize_model(model)

    if target == 'loihi':
        with HardwareInterface(model, use_snips=False, seed=seed) as sim:
            sim.run_steps(1000)
            y = sim.get_probe_output(probe)
    else:
        with EmulatorInterface(model, seed=seed) as sim:
            sim.run_steps(1000)
            y = sim.get_probe_output(probe)

    plt.plot(y)
    plt.yticks(())


def test_population_input(request, allclose):
    target = request.config.getoption("--target")
    dt = 0.001

    n_inputs = 3
    n_axons = 1
    n_cx = 2

    steps = 6
    spike_times_inds = [(1, [0]),
                        (3, [1]),
                        (5, [2])]

    model = Model()

    input = SpikeInput(n_inputs)
    model.add_input(input)
    spikes = [(input, ti, inds) for ti, inds in spike_times_inds]

    input_axon = Axon(n_axons)
    axon_map = np.zeros(n_inputs, dtype=int)
    atoms = np.arange(n_inputs)
    input_axon.set_axon_map(axon_map, atoms)
    input.add_axon(input_axon)

    block = LoihiBlock(n_cx)
    block.compartment.configure_lif(tau_rc=0., tau_ref=0., dt=dt)
    block.compartment.configure_filter(0, dt=dt)
    model.add_block(block)

    synapse = Synapse(n_axons)
    weights = 0.1 * np.array([[[1, 2], [2, 3], [4, 5]]], dtype=float)
    indices = np.array([[[0, 1], [0, 1], [0, 1]]], dtype=int)
    axon_to_weight_map = np.zeros(n_axons, dtype=int)
    cx_bases = np.zeros(n_axons, dtype=int)
    synapse.set_population_weights(
        weights, indices, axon_to_weight_map, cx_bases, pop_type=32)
    block.add_synapse(synapse)
    input_axon.target = synapse

    probe = Probe(target=block, key='voltage')
    block.add_probe(probe)

    discretize_model(model)

    if target == 'loihi':
        with HardwareInterface(model, use_snips=True) as sim:
            sim.run_steps(steps, blocking=False)
            for ti in range(1, steps+1):
                spikes_i = [spike for spike in spikes if spike[1] == ti]
                sim.host2chip(spikes_i, [])
                sim.chip2host()

            y = sim.get_probe_output(probe)
    else:
        for inp, ti, inds in spikes:
            inp.add_spikes(ti, inds)

        with EmulatorInterface(model) as sim:
            sim.run_steps(steps)
            y = sim.get_probe_output(probe)

    vth = block.compartment.vth[0]
    assert (block.compartment.vth == vth).all()
    z = y / vth
    assert allclose(z[[1, 3, 5]], weights[0], atol=4e-2, rtol=0)
