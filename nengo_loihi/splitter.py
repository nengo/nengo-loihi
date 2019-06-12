from collections import defaultdict

from nengo import Direct, Ensemble, Node, Probe
from nengo.connection import LearningRule
from nengo.exceptions import BuildError

from nengo_loihi.builder.inputs import ChipReceiveNode
from nengo_loihi.config import add_params
from nengo_loihi.inputs import ChipProcess
from nengo_loihi.passthrough import PassthroughSplit, base_obj


class PrecomputableSplit:
    """Find all objects that send data to chip, directly or indirectly.

    To find these objects, we compute the transitive closure on all host objects
    that send data to chip, looking for objects that receive output from the chip,
    using a breadth-first search.

    Parameters
    ----------
    network : Network
        The original Network supplied to the builder.
    hostchip: HostChipSplit
        Tracks whether objects will be on the host or chip.
    passthrough : PassthroughSplit
        Tracks passthrough nodes and connections that will be added/removed.
    strict : bool
        If True, a BuildError will be raised if there are no precomputable objects.
    """

    def __init__(self, network, hostchip, passthrough, strict):
        self.hostchip = hostchip
        self.passthrough = passthrough

        self.objs = set()
        self._precomputable = True
        self._conns = (
            set(network.all_connections) | self.passthrough.to_add
        ) - self.passthrough.to_remove

        # Learning rules are not supported with precompute=True because
        # this would require a hybrid simulation where some parts of the
        # model interact with the host while other parts are precomputed
        # ahead of time. The simulator assumes that precompute=True does not
        # require any interaction between host and chip between time-steps.
        # Also see issue #214.
        has_learning = any(conn.learning_rule is not None for conn in self._conns)

        if not has_learning:
            self._find_precomputable_objs()
        else:
            self._precomputable = False
            if strict and has_learning:
                raise BuildError(
                    "precompute=True not supported when using learning rules"
                )

        if strict and not self._precomputable:
            raise BuildError("Cannot precompute input, as it is dependent on output")

    def _find_precomputable_objs(self):
        # Set up the breadth-first search
        queue = []
        head = 0

        # Forwards and backwards adjacency lists
        pre_to_conn = defaultdict(list)
        post_to_conn = defaultdict(list)

        # Initialize queue with the pre objects on host->chip connections.
        # We assume that all `conn.pre` objects are precomputable, and then
        # set a flag to mark the host model as not precomputable if one of them
        # turns out to rely on chip output.
        for conn in self._conns:
            pre, post = base_obj(conn.pre), base_obj(conn.post)
            pre_to_conn[pre].append(conn)
            post_to_conn[post].append(conn)
            assert pre not in self.passthrough.to_remove
            assert post not in self.passthrough.to_remove

            if self.hostchip.on_chip(post) and not self.hostchip.on_chip(pre):
                self.mark_precomputable(pre, queue)

        # Traverse all connected objects breadth-first
        while head < len(queue):
            node_or_ens = queue[head]
            head += 1

            # Handle forwards adjacencies
            for conn in pre_to_conn[node_or_ens]:
                assert base_obj(conn.pre) is node_or_ens
                post = base_obj(conn.post)
                if not self.hostchip.on_chip(post):
                    self.mark_precomputable(post, queue)

            # Handle backwards adjacencies
            for conn in post_to_conn[node_or_ens]:
                assert base_obj(conn.post) is node_or_ens
                pre = base_obj(conn.pre)
                if not self.hostchip.on_chip(pre):
                    self.mark_precomputable(pre, queue)
                else:
                    # Found an input to the chip that relies on an output from the chip.
                    # The model is not precomputable.
                    self.objs.clear()
                    self._precomputable = False
                    return

    def mark_precomputable(self, obj, queue):
        assert isinstance(obj, (Node, Ensemble))
        if obj not in self.objs:
            self.objs.add(obj)
            queue.append(obj)

    def precomputable(self, obj=None):
        if obj is None:
            return self._precomputable

        if isinstance(obj, Probe):
            obj = base_obj(obj.target)

        return obj in self.objs


class HostChipSplit:
    """Place all objects in a network on host or chip."""

    def __init__(self, network):
        # We call this in case it hasn't been called before, as we expect the
        # on_chip configuration option to be defined for these objects.
        # It is safe to call it twice.
        add_params(network)

        # Objects split to the host.
        self.host_objs = set()

        # Objects split to the chip.
        self.chip_objs = set()

        # Place objects on host or chip
        self._place_nodes(network)
        self._place_ensembles(network)
        self._place_probes(network)

    def _place_nodes(self, network):
        """Place nodes.

        Nodes go on the host, unless they are `.ChipReceiveNode` or have a
        `.ChipProcess` output.
        """

        for node in network.all_nodes:
            if isinstance(node, ChipReceiveNode) or isinstance(
                node.output, ChipProcess
            ):
                self.chip_objs.add(node)
            else:
                self.host_objs.add(node)

    def _place_ensembles(self, network):
        """Place ensembles.

        Ensembles should go on the chip, unless:

        1. The user has specified they should not
        2. The ensemble is running in direct mode
        3. They are the ``post`` in a learned connection.
        4. They are the ``pre`` in a connection to a LearningRule
           (i.e., they provide the error signal for a learned connection).
        """

        # Enforce rules 1 and 2
        for ens in network.all_ensembles:
            if network.config[ens].on_chip is False or isinstance(
                ens.neuron_type, Direct
            ):
                self.host_objs.add(ens)
            else:
                self.chip_objs.add(ens)

        for conn in network.all_connections:
            pre, post = base_obj(conn.pre), base_obj(conn.post)

            # Enforce rule 3
            if (
                conn.learning_rule_type is not None
                and isinstance(post, Ensemble)
                and post in self.chip_objs
            ):
                if network.config[post].on_chip:
                    raise BuildError(
                        "Post ensemble (%r) of learned connection (%r) must not be "
                        "configured as on_chip." % (post, conn)
                    )
                self.host_objs.add(post)
                self.chip_objs.remove(post)

            # Enforce rule 4
            elif (
                isinstance(post, LearningRule)
                and isinstance(pre, Ensemble)
                and pre in self.chip_objs
            ):
                if network.config[pre].on_chip:
                    raise BuildError(
                        "Pre ensemble (%r) of error connection (%r) must not be "
                        "configured as on_chip." % (pre, conn)
                    )
                self.host_objs.add(pre)
                self.chip_objs.remove(pre)

    def _place_probes(self, network):
        """Place probes. Probes go where their probed object is."""

        for probe in network.all_probes:
            obj = base_obj(probe.target)
            if obj in self.host_objs:
                self.host_objs.add(probe)
            elif obj in self.chip_objs:
                self.chip_objs.add(probe)
            else:
                raise BuildError("Object (%r) is not a part of the network" % (obj,))

    def on_chip(self, obj):
        if not isinstance(obj, (Ensemble, Node, Probe)):
            raise BuildError(
                "Locations are only established for ensembles ",
                "nodes, and probes -- not for %r" % (obj,),
            )
        if obj in self.chip_objs:
            return True
        elif obj in self.host_objs:
            return False
        raise BuildError("Object (%r) is not a part of the network" % (obj,))


class Split:
    """Creates a set of directives to guide the builder.

    Parameters
    ----------
    network : Network
        The original Network supplied to the builder.
    precompute : bool, optional (Default: None)
        Whether model inputs should be precomputed to speed up simulation.
        The splitter will always determine precomputable objects, but if precompute
        is set to True then an error will be raised if the model cannot be precomputed.
    remove_passthrough : bool, optional (Default: True)
        Whether we should mark passthrough nodes that can be removed.
        Connections to replace the passthrough nodes will also be determined.
    """

    def __init__(self, network, precompute=None, remove_passthrough=True):
        self.network = network

        # Place objects on host or chip
        self.hostchip = HostChipSplit(network)

        # Determine how passthrough nodes will be handled
        if remove_passthrough:
            self.passthrough = PassthroughSplit(self.network, self.hostchip)
        else:
            self.passthrough = PassthroughSplit(None, None)

        # Determine which host objects are precomputable
        self._precomputable = PrecomputableSplit(
            network, self.hostchip, self.passthrough, strict=precompute is True
        )
        self.precompute = precompute
        if self.precompute is None:
            self.precompute = self.precomputable()

    def precomputable(self, obj=None):
        return self._precomputable.precomputable(obj=obj)

    def on_chip(self, obj):
        return self.hostchip.on_chip(obj)
