import logging

import nengo
import numpy as np
import scipy.sparse
from packaging.version import parse as parse_version

logger = logging.getLogger(__name__)

try:
    import tensorflow as tf

    HAS_TF = True
except Exception as err:  # pragma: no cover
    tf = None
    HAS_TF = False
    logger.debug("Error importing TensorFlow:\n%s", err)

try:
    import nengo_dl

    HAS_DL = True
    assert HAS_TF, "NengoDL installed without Tensorflow"
except Exception as err:  # pragma: no cover
    nengo_dl = None
    HAS_DL = False
    logger.debug("Error importing NengoDL:\n%s", err)


def is_transform_type(transform, types):
    types = (types,) if isinstance(types, str) else types
    types = tuple(
        getattr(nengo.transforms, t) for t in types if hasattr(nengo.transforms, t)
    )
    return isinstance(transform, types)


def sample_transform(conn, rng=np.random):
    if is_transform_type(conn.transform, "NoTransform"):
        return np.array(1.0)

    transform = conn.transform.sample(rng=rng)

    # convert SparseMatrix to scipy.sparse
    if isinstance(transform, nengo.transforms.SparseMatrix):
        transform = scipy.sparse.csr_matrix(
            (transform.data, transform.indices.T), shape=transform.shape
        )
    return transform


def make_process_step(process, shape_in, shape_out, dt, rng, dtype=None):
    state = process.make_state(shape_in, shape_out, dt, dtype=dtype)
    return process.make_step(shape_in, shape_out, dt, rng, state)
