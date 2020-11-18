import nengo
import numpy as np
import pytest
from nengo.exceptions import BuildError
from packaging.version import parse as parse_version


@pytest.mark.skipif(
    parse_version(nengo.__version__) <= parse_version("2.8.0"),
    reason="requires more recent Nengo version",
)
def test_split_conv2d_transform_error(Simulator):
    with nengo.Network() as net:
        node_offchip = nengo.Node([1])
        ens_onchip = nengo.Ensemble(10, 1)
        conv2d = nengo.Convolution(
            n_filters=1, input_shape=(1, 1, 1), kernel_size=(1, 1)
        )
        nengo.Connection(node_offchip, ens_onchip, transform=conv2d)

    with pytest.raises(BuildError, match="Conv2D"):
        with Simulator(net):
            pass


@pytest.mark.parametrize("pre_dims", [1, 3])
@pytest.mark.parametrize("post_dims", [1, 3])
@pytest.mark.parametrize("learn", [True, False])
@pytest.mark.parametrize("use_solver", [True, False])
def test_manual_decoders(seed, Simulator, pre_dims, post_dims, learn, use_solver):

    with nengo.Network(seed=seed) as model:
        pre = nengo.Ensemble(
            50, dimensions=pre_dims, gain=np.ones(50), bias=np.ones(50) * 5
        )
        post = nengo.Node(size_in=post_dims)

        learning_rule_type = nengo.PES() if learn else None
        weights = np.zeros((post_dims, 50))
        if use_solver:
            conn = nengo.Connection(
                pre,
                post,
                function=lambda x: np.zeros(post_dims),
                learning_rule_type=learning_rule_type,
                solver=nengo.solvers.NoSolver(weights.T),
            )
        else:
            conn = nengo.Connection(
                pre.neurons,
                post,
                learning_rule_type=learning_rule_type,
                transform=weights,
            )

        if learn:
            error = nengo.Node(np.zeros(post_dims))
            nengo.Connection(error, conn.learning_rule)

        pre_probe = nengo.Probe(pre.neurons, synapse=None)
        post_probe = nengo.Probe(post, synapse=None)

    if not use_solver and learn:
        with pytest.raises(
            NotImplementedError, match="presynaptic object must be an Ensemble"
        ):
            with Simulator(model) as sim:
                pass
    else:
        with Simulator(model) as sim:
            sim.run(0.1)

        # Ensure pre population has a lot of activity
        assert np.mean(sim.data[pre_probe]) > 100
        # But that post has no activity due to the zero weights
        assert np.all(sim.data[post_probe] == 0)
