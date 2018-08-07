import nengo

from nengo_loihi.builder import Builder
from nengo_loihi.splitter import ChipReceiveNode


@Builder.register(nengo.Node)
def build_node(model, node):
    if isinstance(node, ChipReceiveNode):
        model.add_input(node.spike_input)
        model.objs[node]['out'] = node.spike_input
        return
    else:
        raise NotImplementedError()
