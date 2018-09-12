import warnings

import nengo
from nengo.exceptions import BuildError, NengoException
import numpy as np


def is_passthrough(obj):
    return isinstance(obj, nengo.Node) and obj.output is None


def base_obj(obj):
    """Returns the Ensemble or Node underlying an object"""
    if isinstance(obj, nengo.ensemble.Neurons):
        return obj.ensemble
    return obj


class ClusterException(NengoException):
    pass


class Cluster(object):
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

    def merge_transforms(self, size1, trans1, slice1,
                         node, slice2, trans2, size2):
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
        if trans1.ndim == 0:  # scalar
            trans1 = np.eye(size1) * trans1
        elif trans1.ndim != 2:
            raise BuildError("Unhandled transform shape: %s" % (trans1.shape,))

        if trans2.ndim == 0:  # scalar
            trans2 = np.eye(size2) * trans2
        elif trans2.ndim != 2:
            raise BuildError("Unhandled transform shape: %s" % (trans2.shape,))

        mid_t = np.eye(node.size_in)[slice2, slice1]
        return np.dot(trans2, np.dot(mid_t, trans1))

    def merge_synapses(self, syn1, syn2):
        """Return an equivalent synapse for the two provided synapses."""
        if syn1 is None:
            return syn2
        elif syn2 is None:
            return syn1
        else:
            assert isinstance(syn1, nengo.Lowpass) and isinstance(
                syn2, nengo.Lowpass)
            warnings.warn(
                "Combining two Lowpass synapses, this may change the "
                "behaviour of the network (set `remove_passthrough=False` "
                "to avoid this).")
            return nengo.Lowpass(syn1.tau + syn2.tau)

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
            yield slice(None), np.array(1.0), None, obj

        for c in outputs[obj]:
            if c.learning_rule_type is not None:
                raise ClusterException("no learning allowed")
            elif isinstance(c.post_obj, nengo.connection.LearningRule):
                raise ClusterException("no error signals allowed")
            elif c.post_obj in previous:
                # cycles of passthrough Nodes are possible in Nengo, but
                # cannot be compiled away
                raise ClusterException("no loops allowed")

            if c in self.conns_out:
                # this is an output from the Cluster, so stop iterating
                yield c.pre_slice, c.transform, c.synapse, c.post
            else:
                # this Connection goes to another passthrough Node in this
                # Cluster, so iterate into that Node and continue
                for pre_slice, transform, synapse, post in self.generate_from(
                        c.post_obj, outputs, previous=previous+[obj]):

                    syn = self.merge_synapses(c.synapse, synapse)
                    trans = self.merge_transforms(c.pre.size_out,
                                                  c.transform,
                                                  c.post_slice,
                                                  c.post_obj,
                                                  pre_slice,
                                                  transform,
                                                  post.size_in)

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
            for pre_slice, transform, synapse, post in self.generate_from(
                    c.post_obj, outputs):
                syn = self.merge_synapses(c.synapse, synapse)
                trans = self.merge_transforms(c.size_mid,
                                              c.transform,
                                              c.post_slice,
                                              c.post_obj,
                                              pre_slice,
                                              transform,
                                              post.size_in)

                if not np.allclose(trans, np.zeros_like(trans)):
                    yield nengo.Connection(
                        pre=c.pre,
                        post=post,
                        function=c.function,
                        eval_points=c.eval_points,
                        scale_eval_points=c.scale_eval_points,
                        synapse=syn,
                        transform=trans,
                        add_to_container=False)


def find_clusters(net, offchip):
    """Create the Clusters for a given nengo Network."""

    # find which objects have Probes, as we need to make sure to keep them
    probed_objs = set(base_obj(p.target) for p in net.all_probes)

    clusters = {}   # mapping from object to its Cluster
    for c in net.all_connections:
        base_pre = base_obj(c.pre_obj)
        base_post = base_obj(c.post_obj)

        pass_pre = is_passthrough(c.pre_obj) and c.pre_obj not in offchip
        if pass_pre and c.pre_obj not in clusters:
            # add new objects to their own initial Cluster
            clusters[c.pre_obj] = Cluster(c.pre_obj)
            if c.pre_obj in probed_objs:
                clusters[c.pre_obj].probed_objs.add(c.pre_obj)

        pass_post = is_passthrough(c.post_obj) and c.post_obj not in offchip
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


def convert_passthroughs(network, offchip):
    """Create a set of Connections that could replace the passthrough Nodes.

    This does not actually modify the Network, but instead returns the
    set of passthrough Nodes to be removed, the Connections to be removed,
    and the Connections that should be added to replace the Nodes and
    Connections.

    The parameter offchip provides a list of objects that should be considered
    to be offchip. The system will only remove passthrough Nodes that go
    between two onchip objects.
    """
    clusters = find_clusters(network, offchip=offchip)

    removed_passthroughs = set()
    removed_connections = set()
    added_connections = set()
    handled_clusters = set()
    for cluster in clusters.values():
        if cluster not in handled_clusters:
            handled_clusters.add(cluster)
            onchip_input = False
            onchip_output = False
            for c in cluster.conns_in:
                if base_obj(c.pre_obj) not in offchip:
                    onchip_input = True
                    break
            for c in cluster.conns_out:
                if base_obj(c.post_obj) not in offchip:
                    onchip_output = True
                    break
            has_input = len(cluster.conns_in) > 0
            no_output = len(cluster.conns_out) + len(cluster.probed_objs) == 0

            if has_input and ((onchip_input and onchip_output) or no_output):
                try:
                    new_conns = list(cluster.generate_conns())
                except ClusterException:
                    # this Cluster has an issue, so don't remove it
                    continue

                removed_passthroughs.update(cluster.objs - cluster.probed_objs)
                removed_connections.update(cluster.conns_in
                                           | cluster.conns_mid
                                           | cluster.conns_out)
                added_connections.update(new_conns)
    return removed_passthroughs, removed_connections, added_connections
