from collections import defaultdict

from nengo import Direct, Ensemble, Node, Probe
from nengo.exceptions import BuildError
from nengo.connection import LearningRule

from nengo_loihi.passthrough import base_obj, is_passthrough, PassthroughSplit


class Split:
    """Creates a set of directives to guide the builder."""

    def __init__(self, network, precompute=False, remove_passthrough=True):
        self.network = network

        # subset of network: only nodes and ensembles;
        # probes are handled dynamically
        self._seen_objects = set()

        # subset of seen, marking which are run on the hardware;
        # those running on the host are "seen - chip"
        self._chip_objects = set()

        # Step 1. Place nodes on host
        self._seen_objects.update(network.all_nodes)

        # Step 2. Place all possible ensembles on chip
        # Note: assumes add_params already called by the simulator
        for ens in network.all_ensembles:
            if (network.config[ens].on_chip in (None, True)
                    and not isinstance(ens.neuron_type, Direct)):
                self._chip_objects.add(ens)
            self._seen_objects.add(ens)

        # Step 3. Move learning ensembles (post and error) to host
        for conn in network.all_connections:
            pre = base_obj(conn.pre)
            post = base_obj(conn.post)
            if (conn.learning_rule_type is not None
                    and isinstance(post, Ensemble)
                    and post in self._chip_objects):
                if network.config[post].on_chip:
                    raise BuildError("Post ensemble (%r) of learned "
                                     "connection (%r) must not be configured "
                                     "as on_chip." % (post, conn))
                self._chip_objects.remove(post)
            elif (isinstance(post, LearningRule)
                  and isinstance(pre, Ensemble)
                  and pre in self._chip_objects):
                if network.config[pre].on_chip:
                    raise BuildError("Pre ensemble (%r) of error "
                                     "connection (%r) must not be configured "
                                     "as on_chip." % (pre, conn))
                self._chip_objects.remove(pre)

        # Step 4. Mark passthrough nodes for removal
        if remove_passthrough:
            passthroughs = set(
                obj for obj in network.all_nodes if is_passthrough(obj))
            ignore = self._seen_objects - self._chip_objects - passthroughs
            self.passthrough = PassthroughSplit(network, ignore)
        else:
            self.passthrough = PassthroughSplit(None)

        # Step 5. Split precomputable parts of host
        # This is a subset of host, marking which are precomputable
        if precompute:
            self._host_precomputable_objects = self._preclosure()
        else:
            self._host_precomputable_objects = set()

    def _preclosure(self):  # noqa: C901
        """Returns all objects that [in]directly send data to chip."""
        # performs a "transitive closure" on all host objects that
        # send data to the chip. if any of these objects receive
        # output from the chip, then a BuildError is raised
        precomputable = set()

        # forwards and backwards adjacency lists
        pre_to_conn = defaultdict(list)
        post_to_conn = defaultdict(list)

        # data-structure for breadth-first search
        queue = []
        head = 0

        def mark_precomputable(obj):
            assert isinstance(obj, (Node, Ensemble))
            if obj not in precomputable:
                precomputable.add(obj)
                queue.append(obj)

        # determine which connections will actually be built
        conns = ((set(self.network.all_connections)
                  | self.passthrough.to_add) - self.passthrough.to_remove)

        # Initialize queue with the pre objects on host->chip connections.
        # We assume that all `conn.pre` objects are pre-computable, and then
        # raise an error later if one of them turns out to rely on chip output.
        # Learning rules are not supported with precompute=True because
        # this would require a hybrid simulation where some parts of the
        # model interact with the host while other parts are precomputed
        # ahead of time. The simulator assumes that precompute=True does not
        # require any interaction between host and chip between time-steps.
        # Also see issue #214.
        for conn in conns:
            pre, post = base_obj(conn.pre), base_obj(conn.post)
            pre_to_conn[pre].append(conn)
            post_to_conn[post].append(conn)
            assert pre not in self.passthrough.to_remove
            assert post not in self.passthrough.to_remove

            if (isinstance(post, LearningRule)
                    or conn.learning_rule is not None):
                raise BuildError("precompute=True not supported when using "
                                 "learning rules")

            if self.on_chip(post) and not self.on_chip(pre):
                mark_precomputable(pre)

        # traverse all connected objects breadth-first
        while head < len(queue):
            node_or_ens = queue[head]
            head += 1

            # handle forwards adjacencies
            for conn in pre_to_conn[node_or_ens]:
                assert base_obj(conn.pre) is node_or_ens
                post = base_obj(conn.post)
                if not self.on_chip(post):
                    mark_precomputable(post)

            # handle backwards adjacencies
            for conn in post_to_conn[node_or_ens]:
                assert base_obj(conn.post) is node_or_ens
                pre = base_obj(conn.pre)
                if self.on_chip(pre):
                    raise BuildError("Cannot precompute input, "
                                     "as it is dependent on output")
                mark_precomputable(pre)

        return precomputable

    def is_precomputable(self, obj):
        if isinstance(obj, Probe):
            obj = base_obj(obj.target)
        return (not self.on_chip(obj)
                and obj in self._host_precomputable_objects)

    def on_chip(self, obj):
        if isinstance(obj, Probe):
            obj = base_obj(obj.target)
        if not isinstance(obj, (Ensemble, Node)):
            raise BuildError("Locations are only established for ensembles ",
                             "nodes, and probes -- not for %r" % (obj,))
        if obj not in self._seen_objects:
            raise BuildError("Object (%r) is not a part of the network"
                             % (obj,))
        return obj in self._chip_objects
