from nengo.builder.network import build_network as nengo_build_network
from nengo.builder.network import seed_network
from nengo.network import Network

from nengo_loihi.builder.builder import Builder
from nengo_loihi.builder.discretize import discretize_model
from nengo_loihi.builder.split_blocks import split_model
from nengo_loihi.builder.validate import validate_model
from nengo_loihi.splitter import Split


@Builder.register(Network)
def build_network(
    model, network, precompute=None, remove_passthrough=True, discretize=True
):

    if model.toplevel is None:
        # We don't set model.toplevel to network because `nengo_build_network`
        # will do that and relies on it being `None` initially.

        # Ensure seeds are identical to Nengo
        # Note: This does nothing for nengo<=2.8.0, seeds will always be different
        seed_network(network, seeds=model.seeds, seeded=model.seeded)

        # Determine how to split the host into one, two or three models
        model.split = Split(
            network, precompute=precompute, remove_passthrough=remove_passthrough
        )

    # Delegate most of the network building to Nengo
    nengo_build_network(model, network, progress=None)

    if network is model.toplevel:
        # Build the extra passthrough connections into the model
        passthrough = model.split.passthrough
        for conn in passthrough.to_add:
            # Note: connections added by the passthrough splitter do not have seeds
            model.seeds[conn] = None
            model.seeded[conn] = False
            model.build(conn)

        # Split blocks into blocks that will fit on cores
        split_model(model)

        if discretize:
            discretize_model(model)

        validate_model(model)
