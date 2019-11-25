from distutils.version import LooseVersion
import logging

import nengo
import numpy as np
import scipy.sparse

logger = logging.getLogger(__name__)


if LooseVersion(nengo.__version__) > LooseVersion("2.8.0"):  # noqa: C901
    from nengo.builder.network import seed_network
    from nengo.builder.transforms import multiply
    from nengo.simulator import SimulationData as NengoSimulationData
    import nengo.transforms as nengo_transforms
    from nengo.utils.numpy import is_array, is_integer, is_iterable, is_number
    from nengo.utils.testing import signals_allclose

    def conn_solver(solver, activities, targets, rng):
        return solver(activities, targets, rng=rng)

    def transform_array(transform):
        return transform.init

    def sample_transform(conn, rng=np.random):
        transform = conn.transform.sample(rng=rng)

        # convert SparseMatrix to scipy.sparse
        if isinstance(transform, nengo_transforms.SparseMatrix):
            transform = scipy.sparse.csr_matrix(
                (transform.data, transform.indices.T), shape=transform.shape
            )
        return transform

    def make_process_step(process, shape_in, shape_out, dt, rng, dtype=None):
        state = process.make_state(shape_in, shape_out, dt, dtype=dtype)
        return process.make_step(shape_in, shape_out, dt, rng, state)


else:  # pragma: no cover
    from nengo.builder.connection import multiply
    from nengo.simulator import ProbeDict as NengoSimulationData
    from nengo.utils.compat import is_array, is_integer, is_iterable, is_number
    from nengo.utils.testing import allclose as signals_allclose

    nengo_transforms = None
    from nengo.dists import get_samples as _get_samples

    nengo.solvers.Solver.compositional = True
    nengo.solvers.LstsqDrop.compositional = False
    nengo.solvers.LstsqL1.compositional = False
    nengo.solvers.Nnls.compositional = False

    def conn_solver(solver, activities, targets, rng):
        # pass E=1 because solver.weights requires E to not be None, but we've
        # already multiplied targets by encoders, so multiply by 1 does nothing
        return solver(activities, targets, rng=rng, E=1 if solver.weights else None)

    def transform_array(transform):
        return transform

    def sample_transform(conn, rng=np.random):
        return _get_samples(conn.transform, conn.size_out, d=conn.size_mid, rng=rng)

    def seed_network(*args, **kwargs):
        # nengo <= 2.8.0 will overwrite any seeds set on the model, so no
        # point doing anything in this function
        pass

    def make_process_step(process, shape_in, shape_out, dt, rng, dtype=None):
        assert isinstance(process, nengo.synapses.Synapse)
        return process.make_step(shape_in, shape_out, dt, rng, dtype=dtype)


try:
    import tensorflow as tf

    HAS_TF = True
except ImportError as err:  # pragma: no cover
    tf = None
    HAS_TF = False
    logger.debug("Error importing TensorFlow:\n%s", err)

try:
    import nengo_dl

    HAS_DL = True
    assert HAS_TF, "NengoDL installed without Tensorflow"
except ImportError as err:  # pragma: no cover
    nengo_dl = None
    HAS_DL = False
    logger.debug("Error importing Nengo DL:\n%s", err)
