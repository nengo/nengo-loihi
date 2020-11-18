import numpy as np
import pytest
import scipy.sparse

from nengo_loihi.builder.sparse_matrix import (
    expand_matrix,
    scale_matrix,
    stack_matrices,
)

matrix_types = [np.ndarray, scipy.sparse.spmatrix]


def toarray(x):
    return x.toarray() if isinstance(x, scipy.sparse.spmatrix) else x


def make_matrix(matrix_type, shape):
    n = max(shape)
    assert n >= 3
    dense = np.zeros((n, n))
    i = np.arange(n)
    dense[i[::-1], i] = -np.arange(1, n + 1)
    dense[i, i] = np.arange(1, n + 1)
    dense = dense[: shape[0], : shape[1]]

    indices = dense.nonzero()
    data = dense[indices]

    if matrix_type == scipy.sparse.spmatrix:
        matrix = scipy.sparse.csr_matrix((data, indices), shape=shape)
    else:
        matrix = dense

    return dense, matrix


@pytest.mark.parametrize("matrix_type", matrix_types)
def test_expand_matrix(matrix_type, monkeypatch):
    # 0-d input
    y = expand_matrix(np.array(2.0), (3, 3))
    assert isinstance(y, scipy.sparse.spmatrix)
    assert np.allclose(toarray(y), np.eye(3) * 2)

    # 1-d input
    y = expand_matrix(np.arange(3), (3, 3))
    assert isinstance(y, scipy.sparse.spmatrix)
    assert np.allclose(toarray(y), np.diag(np.arange(3)))

    # 2-d input
    shape = (3, 4)
    dense_x, sparse_x = make_matrix(matrix_type, shape)
    y = expand_matrix(sparse_x, shape)
    assert isinstance(y, matrix_type)
    assert np.allclose(toarray(y), dense_x)


@pytest.mark.parametrize("matrix_type", matrix_types)
def test_scale_matrix(matrix_type):
    shape = (5, 4)
    dense_x, sparse_x = make_matrix(matrix_type, shape)

    # scalar scale
    scale = 2.5
    y = scale_matrix(sparse_x, scale)
    assert isinstance(y, matrix_type)
    assert np.allclose(toarray(y), scale * dense_x)

    # vector scale
    scale = np.arange(4)
    y = scale_matrix(sparse_x, scale)
    assert isinstance(y, matrix_type)
    assert np.allclose(toarray(y), dense_x * scale)


@pytest.mark.parametrize("matrix_type", matrix_types)
def test_stack_matrices(matrix_type):
    # horizontal
    xd1, xs1 = make_matrix(matrix_type, (5, 4))
    xd2, xs2 = make_matrix(matrix_type, (5, 3))
    y = stack_matrices([xs1, xs2], order="h")
    assert isinstance(y, matrix_type)
    assert np.allclose(toarray(y), np.hstack([xd1, xd2]))

    # vertical
    xd1, xs1 = make_matrix(matrix_type, (5, 4))
    xd2, xs2 = make_matrix(matrix_type, (3, 4))
    y = stack_matrices([xs1, xs2], order="v")
    assert isinstance(y, matrix_type)
    assert np.allclose(toarray(y), np.vstack([xd1, xd2]))
