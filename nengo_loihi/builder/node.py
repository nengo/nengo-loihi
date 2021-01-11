import nengo
import numpy as np

from nengo_loihi.builder.builder import Builder
from nengo_loihi.builder.inputs import ChipReceiveNode
from nengo_loihi.dvs import DVSFileChipProcess
from nengo_loihi.inputs import ChipProcess, SpikeInput


@Builder.register(ChipReceiveNode)
def build_chip_receive_node(model, node):
    spike_input = SpikeInput(node.raw_dimensions, label=node.label)
    model.add_input(spike_input)
    model.objs[node]["out"] = spike_input
    node.spike_target = spike_input


@Builder.register(nengo.Node)
def build_node(model, node):
    assert isinstance(
        node.output, ChipProcess
    ), "Only ChipProcess nodes should be placed on the Loihi chip (see splitter.py)"
    model.objs[node]["out"] = model.build(node.output, node=node)


@Builder.register(DVSFileChipProcess)
def build_dvs_file_chip_process(model, process, node=None):
    e_t, e_idx = process._read_events()

    label = node.label if node is not None else "DVSFileChipProcess"
    spike_input = SpikeInput(process.size, label=label)

    dt_us = model.dt * 1e6
    t = process.t_start * 1e6  # time in us
    image_idx = 0
    event_idx = 0

    while t <= e_t[-1]:
        t += dt_us
        image_idx += 1
        new_event_idx = event_idx + np.searchsorted(e_t[event_idx:], t)
        spike_input.add_spikes(image_idx, e_idx[event_idx:new_event_idx])
        event_idx = new_event_idx

    model.add_input(spike_input)
    return spike_input
