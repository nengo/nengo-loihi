import numpy as np

from nengo_loihi.builder.builder import Builder
from nengo_loihi.builder.inputs import ChipReceiveNode, DVSFileChipNode
from nengo_loihi.inputs import SpikeInput


@Builder.register(ChipReceiveNode)
def build_chip_receive_node(model, node):
    spike_input = SpikeInput(node.raw_dimensions, label=node.label)
    model.add_input(spike_input)
    model.objs[node]["out"] = spike_input
    node.spike_target = spike_input


@Builder.register(DVSFileChipNode)
def build_dvs_file_chip_node(model, node):
    e_t, e_idx = node._read_events()

    spike_input = SpikeInput(node.raw_dimensions, label=node.label)

    dt_us = model.dt * 1e6
    t = node.t_start * 1e6  # time in us
    ti = 0  # image index
    k = 0  # event index

    while t <= e_t[-1]:
        t += dt_us
        ti += 1
        k1 = k + np.searchsorted(e_t[k:], t)
        spike_input.add_spikes(ti, e_idx[k:k1])
        k = k1

    model.add_input(spike_input)
    model.objs[node]["out"] = spike_input
