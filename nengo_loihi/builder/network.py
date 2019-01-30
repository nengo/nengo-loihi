import logging

from nengo import Network, Ensemble, Connection, Node, Probe
import nengo.utils.numpy as npext
import numpy as np

from nengo_loihi.builder.builder import Builder

logger = logging.getLogger(__name__)


@Builder.register(Network)
def build_network(model, network):
    def get_seed(obj, rng):
        return (rng.randint(npext.maxint)
                if not hasattr(obj, 'seed') or obj.seed is None else obj.seed)

    if network not in model.seeds:
        model.seeded[network] = getattr(network, 'seed', None) is not None
        model.seeds[network] = get_seed(network, np.random)

    # # Set config
    # old_config = model.config
    # model.config = network.config

    # assign seeds to children
    rng = np.random.RandomState(model.seeds[network])
    # Put probes last so that they don't influence other seeds
    sorted_types = (Connection, Ensemble, Network, Node, Probe)
    assert all(tp in sorted_types for tp in network.objects)
    for obj_type in sorted_types:
        for obj in network.objects[obj_type]:
            model.seeded[obj] = (model.seeded[network]
                                 or getattr(obj, 'seed', None) is not None)
            model.seeds[obj] = get_seed(obj, rng)

    logger.debug("Network step 1: Building ensembles and nodes")
    for obj in network.ensembles + network.nodes:
        model.build(obj)

    logger.debug("Network step 2: Building subnetworks")
    for subnetwork in network.networks:
        model.build(subnetwork)

    logger.debug("Network step 3: Building connections")
    for conn in network.connections:
        model.build(conn)

    logger.debug("Network step 4: Building probes")
    for probe in network.probes:
        model.build(probe)

    # # Unset config
    # model.config = old_config
    model.params[network] = None
