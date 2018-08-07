import nengo

from nengo_loihi.builder import Builder
from nengo_loihi.splitter import ChipReceiveNode


@Builder.register(nengo.Node)
def build_node(model, node):
    if isinstance(node, ChipReceiveNode):
        cx_spiker = node.cx_spike_input
        model.add_input(cx_spiker)
        model.objs[node]['out'] = cx_spiker
        return
    else:
        raise NotImplementedError()
