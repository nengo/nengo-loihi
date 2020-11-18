import warnings
from collections import OrderedDict

import numpy as np
from nengo import Connection, Dense, Lowpass, Node, Probe
from nengo.base import ObjView
from nengo.connection import LearningRule
from nengo.ensemble import Neurons
from nengo.exceptions import BuildError, NengoException

from nengo_loihi.compat import is_transform_type


def is_passthrough(obj):
    return isinstance(obj, Node) and obj.output is None


def base_obj(obj):
    """Returns the object underlying some view or neurons."""
    if isinstance(obj, ObjView):
        obj = obj.obj
    if isinstance(obj, Neurons):
        return obj.ensemble
    return obj


class ClusterError(NengoException):
    pass


class Cluster:
    """A collection of passthrough Nodes directly connected to each other.

    When removing passthrough Nodes, we often have large chains of Nodes that
    should be removed together. That is, we tend to have a directed graph,
    starting with inputs coming from Ensembles and non-passthrough Nodes, and
    ending in Ensembles and non-passthrough Nodes. This Cluster object
    represents this collection of passthrough Nodes and allows us to remove
    them all at once.
    """

    def __init__(self, obj):
        self.objs = set([obj])  # the Nodes in the cluster
        self.conns_in = set()  # Connections into the cluster
        self.conns_out = set()  # Connections out of the cluster
        self.conns_mid = set()  # Connections within the cluster
        self.probed_objs = set()  # Nodes that have Probes on them

    def merge_with(self, other):
        """Combine this Cluster with another Cluster"""
        self.objs.update(other.objs)
        self.conns_in.update(other.conns_in)
        self.conns_out.update(other.conns_out)
        self.conns_mid.update(other.conns_mid)
        self.probed_objs.update(other.probed_objs)

    def merge_transforms(self, node, sizes, transforms, slices):
        """Return an equivalent transform to the two provided transforms.

        This is for finding a transform that converts this::

            a = nengo.Node(size1)
            b = nengo.Node(size2)
            nengo.Connection(a, node[slice1], transform=trans1)
            nengo.Connection(node[slice2], b, transform=trans2)

        Into this::

            a = nengo.Node(size1)
            b = nengo.Node(size2)
            nengo.Connection(a, b, transform=t)

        """

        def format_transform(size, transform):
            if is_transform_type(transform, "NoTransform"):
                transform = np.array(1.0)
            elif is_transform_type(transform, "Dense"):
                transform = transform.init
            else:
                raise NotImplementedError(
                    "Mergeable transforms must be Dense; "
                    "set remove_passthrough=False"
                )

            if not isinstance(transform, np.ndarray):
                raise NotImplementedError(
                    "Mergeable transforms must be specified as Numpy arrays, "
                    "not distributions. Set `remove_passthrough=False`."
                )

            if transform.ndim == 0:  # scalar
                transform = np.eye(size) * transform
            elif transform.ndim != 2:
                raise BuildError("Unhandled transform shape: %s" % (transform.shape,))

            return transform

        assert (
            len(sizes) == len(transforms) == len(slices) == 2
        ), "Only merging two transforms is currently supported"
        mid_t = np.eye(node.size_in)[slices[1], slices[0]]
        transform = np.dot(
            format_transform(sizes[1], transforms[1]),
            np.dot(mid_t, format_transform(sizes[0], transforms[0])),
        )

        return Dense(transform.shape, init=transform)

    def merge_synapses(self, syn1, syn2):
        """Return an equivalent synapse for the two provided synapses."""
        if syn1 is None:
            return syn2
        elif syn2 is None:
            return syn1
        else:
            assert isinstance(syn1, Lowpass) and isinstance(syn2, Lowpass)
            warnings.warn(
                "Combining two Lowpass synapses, this may change the "
                "behaviour of the network (set `remove_passthrough=False` "
                "to avoid this)."
            )
            return Lowpass(syn1.tau + syn2.tau)

    def generate_from(self, obj, outputs, previous=None):
        """Generates all direct Connections from obj out of the Cluster.

        This is a recursive process, starting at this obj (a Node within the
        Cluster) and iterating to find all outputs and all probed Nodes
        within the Cluster. The transform and synapse values needed are
        computed while iterating through the graph.

        Return values can be used to make equivalent Connection objects::

            nengo.Connection(
                obj[pre_slice], post, transform=trans, synapse=syn)

        """
        previous = [] if previous is None else previous
        if obj not in outputs:
            return

        if obj in self.probed_objs:
            # this Node has a Probe, so we need to keep it around and create
            # a new Connection that goes to it, as the original Connections
            # will get removed
            trans1 = Dense((obj.size_out, obj.size_out), init=1.0)
            yield (slice(None), trans1, None, obj)

        for c in outputs[obj]:
            # should not be possible to have learning on connection from node
            assert c.learning_rule_type is None
            # should not be possible to have post_obj be LearningRule due to special
            # case rule in PassthroughSplit._on_chip
            assert not isinstance(c.post_obj, LearningRule)

            if c.post_obj in previous:
                # cycles of passthrough Nodes are possible in Nengo, but
                # cannot be compiled away
                raise ClusterError("no loops allowed")

            if c in self.conns_out:
                # this is an output from the Cluster, so stop iterating
                yield c.pre_slice, c.transform, c.synapse, c.post
            else:
                # this Connection goes to another passthrough Node in this
                # Cluster, so iterate into that Node and continue
                for pre_slice, transform, synapse, post in self.generate_from(
                    c.post_obj, outputs, previous=previous + [obj]
                ):

                    syn = self.merge_synapses(c.synapse, synapse)
                    trans = self.merge_transforms(
                        c.post_obj,
                        [c.pre.size_out, post.size_in],
                        [c.transform, transform],
                        [c.post_slice, pre_slice],
                    )

                    yield c.pre_slice, trans, syn, post

    def generate_conns(self):
        """Generate the set of direct Connections replacing this Cluster."""
        outputs = {}
        for c in self.conns_in | self.conns_mid | self.conns_out:
            pre = c.pre_obj
            if pre not in outputs:
                outputs[pre] = set([c])
            else:
                outputs[pre].add(c)

        for c in self.conns_in:
            assert c.post_obj in self.objs
            for k, (pre_slice, transform, synapse, post) in enumerate(
                self.generate_from(c.post_obj, outputs)
            ):
                syn = self.merge_synapses(c.synapse, synapse)
                trans = self.merge_transforms(
                    c.post_obj,
                    [c.size_mid, post.size_in],
                    [c.transform, transform],
                    [c.post_slice, pre_slice],
                )

                if not np.allclose(trans.init, 0):
                    yield Connection(
                        pre=c.pre,
                        post=post,
                        function=c.function,
                        eval_points=c.eval_points,
                        scale_eval_points=c.scale_eval_points,
                        synapse=syn,
                        transform=trans,
                        add_to_container=False,
                        label=(None if c.label is None else "%s_%d" % (c.label, k)),
                    )


class PassthroughSplit:
    """Create a set of Connections that could replace the passthrough Nodes.

    This does not actually modify the Network, but instead returns the
    set of passthrough Nodes to be removed, the Connections to be removed,
    and the Connections that should be added to replace the Nodes and
    Connections.
    """

    def __init__(self, network, hostchip):
        self.network = network
        self.hostchip = hostchip

        self.to_remove = set()
        self.to_add = set()

        if self.network is not None:
            self.clusters = self._find_clusters()
            self._already_split = set()
            for cluster in self.clusters.values():
                if cluster not in self._already_split:
                    self._split_cluster(cluster)

    def _find_clusters(self):
        """Find Clusters for the given Network."""

        # find which objects have Probes, as we need to make sure to keep them
        probed_objs = set(base_obj(p.target) for p in self.network.all_probes)

        clusters = OrderedDict()  # mapping from object to its Cluster
        for c in self.network.all_connections:
            # We assume that neither pre nor post can be a probe which
            # simplifies things slightly because we don't need to be
            # concerned with any underlying target.
            base_pre = base_obj(c.pre)
            base_post = base_obj(c.post)
            assert not isinstance(base_pre, Probe)
            assert not isinstance(base_post, Probe)

            pass_pre = is_passthrough(c.pre_obj)
            pass_post = is_passthrough(c.post_obj)

            if pass_pre and c.pre_obj not in clusters:
                # add new objects to their own initial Cluster
                clusters[c.pre_obj] = Cluster(c.pre_obj)
                if c.pre_obj in probed_objs:
                    clusters[c.pre_obj].probed_objs.add(c.pre_obj)

            if pass_post and c.post_obj not in clusters:
                # add new objects to their own initial Cluster
                clusters[c.post_obj] = Cluster(c.post_obj)
                if c.post_obj in probed_objs:
                    clusters[c.post_obj].probed_objs.add(c.post_obj)

            if pass_pre and pass_post:
                # both pre and post are passthrough, so merge the two
                # clusters into one cluster
                cluster = clusters[base_pre]
                cluster.merge_with(clusters[base_post])
                for obj in cluster.objs:
                    clusters[obj] = cluster
                cluster.conns_mid.add(c)
            elif pass_pre:
                # pre is passthrough but post is not, so this is an output
                cluster = clusters[base_pre]
                cluster.conns_out.add(c)
            elif pass_post:
                # pre is not a passthrough but post is, so this is an input
                cluster = clusters[base_post]
                cluster.conns_in.add(c)
        return clusters

    def _on_chip(self, obj):
        if isinstance(obj, LearningRule):
            return False
        return self.hostchip.on_chip(obj)

    def _split_cluster(self, cluster):
        """Split a Cluster."""
        assert cluster not in self._already_split
        self._already_split.add(cluster)

        onchip_input = any(self._on_chip(base_obj(c.pre)) for c in cluster.conns_in)
        onchip_output = any(self._on_chip(base_obj(c.post)) for c in cluster.conns_out)

        has_input = len(cluster.conns_in) > 0
        no_output = len(cluster.conns_out) + len(cluster.probed_objs) == 0

        if has_input and ((onchip_input and onchip_output) or no_output):
            try:
                new_conns = list(cluster.generate_conns())
            except ClusterError:
                # this Cluster has an issue, so don't remove it
                return

            self.to_remove.update(cluster.objs - cluster.probed_objs)
            self.to_remove.update(
                cluster.conns_in | cluster.conns_mid | cluster.conns_out
            )
            self.to_add.update(new_conns)
