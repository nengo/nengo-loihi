import logging

import nengo
from nengo.exceptions import BuildError
import numpy as np

from nengo_loihi import loihi_cx
from nengo_loihi.neurons import NIF

logger = logging.getLogger(__name__)


def is_on_chip(obj, config):
    """Determine if a component should be placed on the chip or host"""

    if isinstance(obj, nengo.Ensemble):
        if isinstance(obj.neuron_type, nengo.Direct):
            return False
        else:
            return config[obj].on_chip
    elif isinstance(obj, nengo.Node):
        return False
    elif isinstance(obj, nengo.ensemble.Neurons):
        return is_on_chip(obj.ensemble, config)
    elif isinstance(obj, nengo.connection.LearningRule):
        return is_on_chip(obj.connection.pre_obj, config)
    raise Exception('Unhandled object type: %s' % obj)


class PESModulatoryTarget(object):
    def __init__(self, target):
        self.target = target


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

    def __init__(self, dimensions, size_out):
        self.raw_dimensions = dimensions
        self.cx_spike_input = loihi_cx.CxSpikeInput(
            np.zeros((0, dimensions), dtype=bool))
        self.last_time = None
        super(ChipReceiveNode, self).__init__(self.update,
                                              size_in=0, size_out=size_out)

    def update(self, t):
        raise Exception('ChipReceiveNodes should not actually be run')

    def receive(self, t, x):
        assert self.last_time is None or t > self.last_time
        # TODO: make this stacking efficient
        self.cx_spike_input.spikes = np.vstack([self.cx_spike_input.spikes,
                                                [x > 0]])
        self.last_time = t


class ChipReceiveNeurons(ChipReceiveNode):
    """Passes spikes directly (no on-off neuron encoding)"""
    def __init__(self, dimensions, neuron_type=None):
        self.neuron_type = neuron_type
        super(ChipReceiveNeurons, self).__init__(dimensions, dimensions)


def split(model, inter_rate, inter_n, dt):  # noqa: C901
    """Split a model into code running on the host and on-chip"""

    logger.info("Splitting model into host and chip parts")
    host = nengo.Network(seed=model.seed)
    chip = nengo.Network(seed=model.seed)
    chip2host_params = {}
    chip2host_receivers = {}
    host2chip_senders = {}

    for ens in model.all_ensembles:
        if is_on_chip(ens, model.config):
            logger.debug("Adding %s to chip", ens)
            chip.ensembles.append(ens)
        else:
            logger.debug("Adding %s to host", ens)
            host.ensembles.append(ens)

    for node in model.all_nodes:
        if is_on_chip(node, model.config):
            logger.debug("Adding %s to chip", node)
            chip.nodes.append(node)
        else:
            logger.debug("Adding %s to host", node)
            host.nodes.append(node)

    for probe in model.all_probes:
        target = probe.target
        if isinstance(target, nengo.base.ObjView):
            target = target.obj
        if isinstance(target, nengo.ensemble.Neurons):
            target = target.ensemble
        if is_on_chip(target, model.config):
            logger.debug("Adding %s to chip", probe)
            chip.probes.append(probe)
        else:
            logger.debug("Adding %s to host", probe)
            host.probes.append(probe)

    modulatory_nodes = {}
    modulated_conns = {}
    for c in model.all_connections:
        pre_onchip = is_on_chip(c.pre_obj, model.config)
        post_onchip = is_on_chip(c.post_obj, model.config)
        if pre_onchip and post_onchip:
            assert c.learning_rule_type is None
            logger.debug("Adding %s to chip", c)
            chip.connections.append(c)
        elif not pre_onchip and not post_onchip:
            logger.debug("Adding %s to host", c)
            host.connections.append(c)
        elif post_onchip and not pre_onchip:
            if isinstance(c.pre_obj, nengo.ensemble.Neurons):
                # send spikes over and do the rest of the connection on-chip
                assert not isinstance(c.post, nengo.connection.LearningRule)
                dim = c.size_in
                with chip:
                    logger.debug("Creating ChipReceiveNeurons for %s", c)
                    receive = ChipReceiveNeurons(
                        dim, neuron_type=c.pre_obj.ensemble.neuron_type)
                    nengo.Connection(receive, c.post,
                                     transform=c.transform,
                                     synapse=c.synapse)
                with host:
                    logger.debug("Creating HostSendNode for %s", c)
                    send = HostSendNode(dim)
                    nengo.Connection(c.pre, send, synapse=None)
                host2chip_senders[send] = receive

            elif isinstance(c.post, nengo.connection.LearningRule):
                dim = c.size_out
                with host:
                    send = HostSendNode(dim)
                    logger.debug("Creating HostSendNode for %s", c)
                    modulatory_nodes[c] = send
                    nengo.Connection(c.pre, send,
                                     function=c.function,
                                     solver=c.solver,
                                     eval_points=c.eval_points,
                                     scale_eval_points=c.scale_eval_points,
                                     synapse=c.synapse,
                                     transform=c.transform)
            else:
                dim = c.size_out
                with chip:
                    logger.debug("Creating ChipReceiveNode for %s", c)
                    receive = ChipReceiveNode(dim * 2, size_out=dim)
                    nengo.Connection(receive, c.post, synapse=c.synapse)
                with host:
                    max_rate = inter_rate * inter_n
                    rtol = 1e-8  # allow for floating point inaccuracies
                    if max_rate > (1. / dt) * (1 + rtol):
                        raise BuildError(
                            "Simulator `dt` must be <= %s (got %s)" % (
                                1. / max_rate, dt))

                    logger.debug("Creating NIF ensemble for %s", c)
                    ens = nengo.Ensemble(
                        2 * dim, dim, neuron_type=NIF(tau_ref=0.0),
                        encoders=np.vstack([np.eye(dim), -np.eye(dim)]),
                        max_rates=[max_rate] * dim + [max_rate] * dim,
                        intercepts=[-1] * dim + [-1] * dim)

                    # scale the input spikes based on the radius of the
                    # target ensemble
                    seed = model.seed if c.seed is None else c.seed
                    transform = nengo.dists.get_samples(
                        c.transform,
                        n=c.size_out,
                        d=c.size_mid,
                        rng=np.random.RandomState(seed=seed))
                    if isinstance(c.post_obj, nengo.Ensemble):
                        transform = transform / c.post_obj.radius

                    logger.debug("Creating HostSendNode for %s", c)
                    send = HostSendNode(dim * 2)
                    nengo.Connection(c.pre, ens,
                                     function=c.function,
                                     solver=c.solver,
                                     eval_points=c.eval_points,
                                     scale_eval_points=c.scale_eval_points,
                                     synapse=None,
                                     transform=transform)
                    nengo.Connection(ens.neurons, send, synapse=None)
                host2chip_senders[send] = receive
        elif pre_onchip and not post_onchip:
            dim = c.size_out
            with host:
                logger.debug("Creating HostReceiveNode for %s", c)
                receive = HostReceiveNode(dim)
                nengo.Connection(receive, c.post, synapse=c.synapse)
            with chip:
                logger.debug("Creating Probe for %s", c)
                seed = model.seed if c.seed is None else c.seed
                transform = nengo.dists.get_samples(
                    c.transform,
                    n=c.size_out,
                    d=c.size_mid,
                    rng=np.random.RandomState(seed=seed))

                if (isinstance(c.pre, nengo.ensemble.Neurons) and
                        transform.ndim == 2):
                    # decoders manually specified in the transform
                    # should be handled like a normal decoder
                    probe = nengo.Probe(
                        c.pre.ensemble,
                        synapse=None,
                        solver=nengo.solvers.NoSolver(transform.T))
                    dims = transform.shape[0]
                    chip2host_params[probe] = dict(
                        learning_rule_type=c.learning_rule_type,
                        function=lambda x, dims=dims: np.zeros(dims),
                        transform=np.array(1),
                    )
                else:
                    probe = nengo.Probe(c.pre, synapse=None, solver=c.solver)
                    chip2host_params[probe] = dict(
                        learning_rule_type=c.learning_rule_type,
                        function=c.function,
                        eval_points=c.eval_points,
                        scale_eval_points=c.scale_eval_points,
                        transform=transform
                    )
                chip2host_receivers[probe] = receive
                if c.learning_rule_type is not None:
                    modulated_conns[c] = probe
        else:
            raise Exception('Unhandled Connection %s' % c)

    for conn, send in modulatory_nodes.items():
        for conn2, probe in modulated_conns.items():
            if conn.post_obj == conn2.learning_rule:
                host2chip_senders[send] = PESModulatoryTarget(probe)
                break
        else:
            raise Exception('Could not find Connection for %s' % conn)

    return host, chip, host2chip_senders, chip2host_params, chip2host_receivers


def base_obj(obj):
    if isinstance(obj, nengo.ensemble.Neurons):
        return obj.ensemble
    elif isinstance(obj, nengo.base.ObjView):
        return obj.obj
    assert isinstance(obj, (nengo.Node, nengo.Ensemble))
    return obj


def split_pre_from_host(host_model):  # noqa: C901
    assert len(host_model.networks) == 0
    logger.info("Splitting pre model from host")

    pre = nengo.Network()
    inputs = {}
    outputs = {}
    probes = {}
    queue = []
    for n in host_model.nodes[:]:
        inputs[n] = []
        outputs[n] = []
        probes[n] = []
        if isinstance(n, HostSendNode):
            host_model.nodes.remove(n)
            pre.nodes.append(n)
            queue.append(n)
    for e in host_model.ensembles:
        inputs[e] = []
        outputs[e] = []
        probes[e] = []
    for c in host_model.connections:
        inputs[base_obj(c.post_obj)].append(c)
        outputs[base_obj(c.pre_obj)].append(c)

    for p in host_model.probes:
        probes[base_obj(p.target)].append(p)

    while len(queue) > 0:
        n = queue.pop()
        for c in inputs[n]:
            if c not in host_model.connections:
                continue
            host_model.connections.remove(c)
            pre.connections.append(c)
            pre_obj = base_obj(c.pre_obj)
            if pre_obj in host_model.nodes:
                if isinstance(pre_obj, HostReceiveNode):
                    raise Exception('Cannot precompute input, '
                                    'as it is dependent on output')
                host_model.nodes.remove(pre_obj)
                pre.nodes.append(pre_obj)
                queue.append(pre_obj)
            elif pre_obj in host_model.ensembles:
                host_model.ensembles.remove(pre_obj)
                pre.ensembles.append(pre_obj)
                queue.append(pre_obj)
        for c in outputs[n]:
            if c not in host_model.connections:
                continue
            host_model.connections.remove(c)
            pre.connections.append(c)
            post_obj = base_obj(c.post_obj)
            if post_obj in host_model.nodes:
                if isinstance(post_obj, HostReceiveNode):
                    raise Exception('Cannot precompute input, '
                                    'as it is dependent on output')
                host_model.nodes.remove(post_obj)
                pre.nodes.append(post_obj)
                queue.append(post_obj)
            if post_obj in host_model.ensembles:
                host_model.ensembles.remove(post_obj)
                pre.ensembles.append(post_obj)
                queue.append(post_obj)

    for obj in pre.ensembles + pre.nodes:
        for p in probes[obj]:
            host_model.probes.remove(p)
            pre.probes.append(p)

    return pre
