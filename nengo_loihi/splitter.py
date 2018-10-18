from collections import defaultdict
import logging

import nengo
from nengo.exceptions import BuildError, SimulationError
import numpy as np

from nengo_loihi import loihi_cx
from nengo_loihi.neurons import NIF

logger = logging.getLogger(__name__)


def base_obj(obj):
    if isinstance(obj, nengo.ensemble.Neurons):
        return obj.ensemble
    elif isinstance(obj, nengo.base.ObjView):
        return obj.obj
    elif isinstance(obj, nengo.connection.LearningRule):
        return obj.connection
    return obj


class SplitNetworks(object):
    def __init__(self, original, max_rate=1000, inter_tau=0.005):
        self.original = original
        self.max_rate = max_rate
        self.inter_tau = inter_tau

        self.host = nengo.Network(seed=original.seed)
        self.chip = nengo.Network(seed=original.seed)
        self.host_pre = nengo.Network(seed=original.seed)

        self.targets = ("host", "chip", "host_pre")

        # Interactions between rules
        self.needs_sender = {}

        # Used later in the build process
        self.chip2host_params = {}
        self.chip2host_receivers = {}
        self.host2chip_senders = {}

        self.adds = {}
        self.moves = {}
        self.removes = []

    def __contains__(self, obj):
        obj = base_obj(obj)
        return (obj in self.moves
                or obj in self.adds
                or obj in self.removes)

    def add(self, obj, target):
        assert target in self.targets, "invalid target"
        obj = base_obj(obj)
        assert obj not in self, "obj already moved"
        self.adds[obj] = target

    def finalize(self):
        def _add(obj, net):
            for cls in type(obj).__mro__:
                if cls in net.objects:
                    net.objects[cls].append(obj)
                    break
            else:
                assert False, "cannot handle type %r" % (type(obj).__name__,)

        # Ensure that all objects have been dealt with
        for obj in self.original.all_objects:
            if not isinstance(obj, nengo.Network):
                assert obj in self, (
                    "%s not moved or explicitly removed" % (obj,))

        # Process moves and adds
        for obj, target in self.moves.items():
            _add(obj, getattr(self, target))
        for obj, target in self.adds.items():
            _add(obj, getattr(self, target))

    def location(self, obj, default=None):
        obj = base_obj(obj)
        return self.moves.get(obj, self.adds.get(obj, default))

    def move(self, obj, target, force=False):
        obj = base_obj(obj)
        if not force:
            assert obj not in self, "already moved"
        assert target in self.targets, "invalid target"
        logger.debug("Moving %s to %s", obj, target)
        if obj in self.adds:
            self.adds[obj] = target
        else:
            self.moves[obj] = target

    def remove(self, obj):
        obj = base_obj(obj)
        logger.debug("Removing %s", obj)
        self.removes.append(obj)


def split(net, precompute, max_rate, inter_tau):
    logger.info("Splitting model into host and chip parts")
    networks = SplitNetworks(net, max_rate=max_rate, inter_tau=inter_tau)

    # --- Step 1: place ensembles and nodes
    place_nodes(networks)
    place_ensembles(networks)

    # --- Step 2: place simple connections
    place_internetwork_connections(networks)

    # --- Step 3: split complex connections
    split_host_to_chip_connections(networks)
    split_chip_to_host_connections(networks)
    split_host_to_learning_rules(networks)

    # --- Step 4: place precomputable parts of host
    if precompute:
        split_pre_from_host(networks)

    # --- Step 5: place probes
    place_probes(networks)

    # Commit to the moves marked in the previous steps
    networks.finalize()
    if precompute:
        assert len(networks.host_pre.all_objects) > 0, (
            "No precomputable objects")
    else:
        assert len(networks.host_pre.all_objects) == 0, (
            "Object erroneously added to host_pre")

    return networks


def place_nodes(networks):
    # Only ChipReceiveNodes can be run on chip
    for node in networks.original.all_nodes:
        if isinstance(node, ChipReceiveNode):
            networks.move(node, "chip")
        else:
            networks.move(node, "host")


def place_ensembles(networks):
    config = networks.original.config

    for ens in networks.original.all_ensembles:
        # User-specified config takes precedence
        if config[ens].on_chip is not None:
            networks.move(ens, "chip" if config[ens].on_chip else "host")
        # Direct mode ensembles must be off chip
        elif isinstance(ens.neuron_type, nengo.Direct):
            networks.move(ens, "host")

    for conn in networks.original.all_connections:
        # `post` of learning rules must be off chip
        if (conn.learning_rule_type is not None
                and isinstance(base_obj(conn.post_obj), nengo.Ensemble)
                and conn.post_obj not in networks):
            networks.move(conn.post_obj, "host")
        # `error` of learning rules must be off chip
        elif (isinstance(conn.post_obj, nengo.connection.LearningRule)
              and isinstance(base_obj(conn.pre_obj), nengo.Ensemble)
              and conn.pre_obj not in networks):
            networks.move(conn.pre_obj, "host")

    # All other ensembles are placed on chip
    for ens in networks.original.all_ensembles:
        if ens not in networks:
            networks.move(ens, "chip")


def place_internetwork_connections(networks):
    """Connections from two objects placed in the same location go there.

    That is, connections from two objects on the host are done on the host,
    and connections from two objects on the chip are done on the chip.
    """
    for conn in networks.original.all_connections:
        pre_loc = networks.location(conn.pre_obj)
        post_loc = networks.location(conn.post_obj)
        if pre_loc == post_loc:
            if pre_loc == "chip":
                assert conn.learning_rule_type is None
            networks.move(conn, pre_loc)


def split_host_to_chip_connections(networks):
    for conn in networks.original.all_connections:
        if conn in networks:
            # Already processed
            continue

        pre_loc = networks.location(conn.pre_obj)
        post_loc = networks.location(conn.post_obj)
        if pre_loc == "host" and post_loc == "chip":
            if isinstance(conn.pre_obj, nengo.ensemble.Neurons):
                split_host_neurons_to_chip(networks, conn)
            else:
                split_host_to_chip(networks, conn)
            assert conn in networks


def split_host_neurons_to_chip(networks, conn):
    """Send spikes over and do the rest of the connection on-chip"""

    assert not isinstance(conn.post, nengo.connection.LearningRule)
    dim = conn.size_in

    logger.debug("Creating ChipReceiveNeurons for %s", conn)
    receive = ChipReceiveNeurons(
        dim,
        neuron_type=conn.pre_obj.ensemble.neuron_type,
        add_to_container=False,
    )
    networks.add(receive, "chip")
    receive2post = nengo.Connection(
        receive, conn.post,
        transform=conn.transform,
        synapse=conn.synapse,
        add_to_container=False,
    )
    networks.add(receive2post, "chip")

    logger.debug("Creating HostSendNode for %s", conn)
    send = HostSendNode(dim, add_to_container=False)
    networks.add(send, "host")
    pre2send = nengo.Connection(
        conn.pre, send, synapse=None, add_to_container=False)
    networks.add(pre2send, "host")

    networks.host2chip_senders[send] = receive
    networks.remove(conn)


def split_host_to_chip(networks, conn):
    dim = conn.size_out
    logger.debug("Creating ChipReceiveNode for %s", conn)
    receive = ChipReceiveNode(
        dim * 2, size_out=dim, add_to_container=False)
    networks.add(receive, "chip")
    receive2post = nengo.Connection(receive, conn.post,
                                    synapse=networks.inter_tau,
                                    add_to_container=False)
    networks.add(receive2post, "chip")

    logger.debug("Creating NIF ensemble for %s", conn)
    ens = nengo.Ensemble(
        2 * dim, dim,
        neuron_type=NIF(tau_ref=0.0),
        encoders=np.vstack([np.eye(dim), -np.eye(dim)]),
        max_rates=np.ones(dim * 2) * networks.max_rate,
        intercepts=np.ones(dim * 2) * -1,
        add_to_container=False)
    networks.add(ens, "host")

    # scale the input spikes based on the radius of the
    # target ensemble
    seed = networks.original.seed if conn.seed is None else conn.seed
    transform = nengo.dists.get_samples(
        conn.transform,
        n=conn.size_out,
        d=conn.size_mid,
        rng=np.random.RandomState(seed=seed))
    if isinstance(conn.post_obj, nengo.Ensemble):
        transform = transform / conn.post_obj.radius
    pre2ens = nengo.Connection(conn.pre, ens,
                               function=conn.function,
                               solver=conn.solver,
                               eval_points=conn.eval_points,
                               scale_eval_points=conn.scale_eval_points,
                               synapse=conn.synapse,
                               transform=transform,
                               add_to_container=False)
    networks.add(pre2ens, "host")

    logger.debug("Creating HostSendNode for %s", conn)
    send = HostSendNode(dim * 2, add_to_container=False)
    networks.add(send, "host")
    ensneurons2send = nengo.Connection(
        ens.neurons, send, synapse=None, add_to_container=False)
    networks.add(ensneurons2send, "host")
    networks.remove(conn)

    networks.host2chip_senders[send] = receive


def split_chip_to_host_connections(networks):
    for conn in networks.original.all_connections:
        if conn in networks:
            # Already processed
            continue

        pre_loc = networks.location(conn.pre_obj)
        post_loc = networks.location(conn.post_obj)
        # All other connections should be processed by this point
        if pre_loc == "chip" and post_loc == "host":
            split_chip_to_host(networks, conn)
            assert conn in networks


def split_chip_to_host(networks, conn):
    dim = conn.size_out

    logger.debug("Creating HostReceiveNode for %s", conn)
    receive = HostReceiveNode(dim, add_to_container=False)
    networks.add(receive, "host")
    receive2post = nengo.Connection(
        receive, conn.post, synapse=conn.synapse, add_to_container=False)
    networks.add(receive2post, "host")

    logger.debug("Creating Probe for %s", conn)
    seed = networks.original.seed if conn.seed is None else conn.seed
    transform = nengo.dists.get_samples(
        conn.transform,
        n=conn.size_out,
        d=conn.size_mid,
        rng=np.random.RandomState(seed=seed))

    probe = nengo.Probe(
        conn.pre,
        synapse=None,
        solver=conn.solver,
        add_to_container=False,
    )
    networks.chip2host_params[probe] = dict(
        learning_rule_type=conn.learning_rule_type,
        function=conn.function,
        eval_points=conn.eval_points,
        scale_eval_points=conn.scale_eval_points,
        transform=transform
    )
    networks.add(probe, "chip")
    networks.chip2host_receivers[probe] = receive

    if conn.learning_rule_type is not None:
        if not isinstance(conn.pre_obj, nengo.Ensemble):
            raise NotImplementedError(
                "Learning rule presynaptic object must be an Ensemble "
                "(got %r)" % type(conn.pre_obj).__name__)
        networks.needs_sender[conn.learning_rule] = PESModulatoryTarget(probe)
    networks.remove(conn)


def split_host_to_learning_rules(networks):
    for conn in networks.original.all_connections:
        if conn in networks:
            # Already processed
            continue

        pre_loc = networks.location(conn.pre_obj)
        if (pre_loc == "host"
                and isinstance(conn.post_obj, nengo.connection.LearningRule)):
            split_host_to_learning_rule(networks, conn)
            assert conn in networks


def split_host_to_learning_rule(networks, conn):
    dim = conn.size_out
    logger.debug("Creating HostSendNode for %s", conn)
    send = HostSendNode(dim, add_to_container=False)
    networks.add(send, "host")

    pre2send = nengo.Connection(conn.pre, send,
                                function=conn.function,
                                solver=conn.solver,
                                eval_points=conn.eval_points,
                                scale_eval_points=conn.scale_eval_points,
                                synapse=conn.synapse,
                                transform=conn.transform,
                                add_to_container=False)
    networks.add(pre2send, "host")
    pes_target = networks.needs_sender[conn.post_obj]
    networks.host2chip_senders[send] = pes_target
    networks.remove(conn)


def place_probes(networks):
    for probe in networks.original.all_probes:
        target = base_obj(probe.target)
        networks.move(probe, networks.location(target))


def split_pre_from_host(networks):  # noqa: C901
    logger.info("Splitting pre model from host")

    inputs = defaultdict(list)
    outputs = defaultdict(list)
    queue = []

    for d in [networks.moves, networks.adds]:
        for obj in d:
            if isinstance(obj, nengo.Connection):
                inputs[base_obj(obj.post_obj)].append(obj)
                outputs[base_obj(obj.pre_obj)].append(obj)
            elif isinstance(obj, HostSendNode):
                networks.move(obj, "host_pre", force=True)
                queue.append(obj)

    while len(queue) > 0:
        node_or_ens = queue.pop()

        for conn in inputs[node_or_ens] + outputs[node_or_ens]:
            if networks.location(conn) != "host":
                continue
            networks.move(conn, "host_pre", force=True)

            if conn in inputs[node_or_ens]:
                obj = base_obj(conn.pre_obj)
            elif conn in outputs[node_or_ens]:
                obj = base_obj(conn.post_obj)

            if (isinstance(obj, (nengo.Node, nengo.Ensemble))
                    and networks.location(obj) == "host"):
                if isinstance(obj, HostReceiveNode):
                    raise BuildError("Cannot precompute input, "
                                     "as it is dependent on output")
                networks.move(obj, "host_pre", force=True)
                queue.append(obj)


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
        while (len(self.queue) > self.queue_index + 1
               and self.queue[self.queue_index][0] < t):
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
        raise SimulationError("ChipReceiveNodes should not be run")

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
