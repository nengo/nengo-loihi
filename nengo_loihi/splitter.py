import nengo
import numpy as np

from . import loihi_cx
from .neurons import NIF


def is_on_chip(obj):
    """Determine if a component should be placed on the chip or host"""

    if isinstance(obj, nengo.Ensemble):
        if isinstance(obj.neuron_type, nengo.Direct):
            return False
        else:
            return True
    elif isinstance(obj, nengo.Node):
        return False
    elif isinstance(obj, nengo.ensemble.Neurons):
        return is_on_chip(obj.ensemble)
    raise Exception('Unhandled object type: %s' % obj)


class HostSendNode(nengo.Node):
    """For sending host->chip messages"""

    def __init__(self, dimensions):
        self.queue = []
        super(HostSendNode, self).__init__(self.update,
                                           size_in=dimensions, size_out=0)

    def update(self, t, x):
        assert len(self.queue) == 0 or t > self.queue[-1][0]
        self.queue.append((t, x))


class HostReceiveNode(nengo.Node):
    """For receiving chip->host messages"""

    def __init__(self, dimensions):
        self.queue = [(0, np.zeros(dimensions))]
        self.queue_index = 0
        super(HostReceiveNode, self).__init__(self.update,
                                              size_in=0, size_out=dimensions)

    def update(self, t):
        while (len(self.queue) > self.queue_index + 1 and
               self.queue[self.queue_index][0] < t):
            self.queue_index += 1
        return self.queue[self.queue_index][1]

    def receive(self, t, x):
        self.queue.append((t, x))


class ChipReceiveNode(nengo.Node):
    """For receiving host->chip messages"""

    def __init__(self, dimensions):
        self.cx_spike_input = loihi_cx.CxSpikeInput(
            np.zeros((0, dimensions * 2), dtype=bool))
        self.last_time = None
        super(ChipReceiveNode, self).__init__(self.update,
                                              size_in=0, size_out=dimensions)

    def update(self, t):
        raise Exception('ChipReceiveNodes should not acutally be run')

    def receive(self, t, x):
        assert self.last_time is None or t > self.last_time
        # TODO: make this stacking efficient
        self.cx_spike_input.spikes = np.vstack([self.cx_spike_input.spikes,
                                                [x > 0]])
        self.last_time = t


def split(model, inter_rate, inter_n):  # noqa: C901
    """Split a model into code running on the host and on-chip"""

    host = nengo.Network(seed=model.seed)
    chip = nengo.Network(seed=model.seed)
    chip2host_params = {}
    chip2host_receivers = {}
    host2chip_senders = {}

    for ens in model.all_ensembles:
        if is_on_chip(ens):
            chip.ensembles.append(ens)
        else:
            host.ensembles.append(ens)

    for node in model.all_nodes:
        if is_on_chip(node):
            chip.nodes.append(node)
        else:
            host.nodes.append(node)

    for probe in model.all_probes:
        target = probe.target
        if isinstance(target, nengo.base.ObjView):
            target = target.obj
        if isinstance(target, nengo.ensemble.Neurons):
            target = target.ensemble
        if is_on_chip(target):
            chip.probes.append(probe)
        else:
            host.probes.append(probe)

    for c in model.all_connections:
        pre_onchip = is_on_chip(c.pre_obj)
        post_onchip = is_on_chip(c.post_obj)
        if pre_onchip and post_onchip:
            chip.connections.append(c)
        elif not pre_onchip and not post_onchip:
            host.connections.append(c)
        elif post_onchip and not pre_onchip:
            dim = c.size_out
            with chip:
                receive = ChipReceiveNode(dim)
                nengo.Connection(receive, c.post, synapse=c.synapse)
            with host:
                max_rate = inter_rate * inter_n
                assert max_rate <= 1000

                ens = nengo.Ensemble(
                    2 * dim, dim, neuron_type=NIF(tau_ref=0.0),
                    encoders=np.vstack([np.eye(dim), -np.eye(dim)]),
                    max_rates=[max_rate] * dim + [max_rate] * dim,
                    intercepts=[-1] * dim + [-1] * dim)

                send = HostSendNode(dim * 2)
                nengo.Connection(c.pre, ens,
                                 function=c.function,
                                 solver=c.solver,
                                 eval_points=c.eval_points,
                                 scale_eval_points=c.scale_eval_points,
                                 synapse=None,
                                 transform=c.transform)
                nengo.Connection(ens.neurons, send, synapse=None)
                host2chip_senders[send] = receive
        elif pre_onchip and not post_onchip:
            dim = c.size_out
            with host:
                receive = HostReceiveNode(dim)
                nengo.Connection(receive, c.post, synapse=c.synapse)
            with chip:
                probe = nengo.Probe(c.pre, synapse=None, solver=c.solver)
                chip2host_params[probe] = dict(
                    function=c.function,
                    eval_points=c.eval_points,
                    scale_eval_points=c.scale_eval_points,
                    transform=c.transform)
                chip2host_receivers[probe] = receive
        else:
            raise Exception('Unhandled Connection %s' % c)
    return host, chip, host2chip_senders, chip2host_params, chip2host_receivers
