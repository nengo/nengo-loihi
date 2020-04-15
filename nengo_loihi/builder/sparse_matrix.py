import numpy as np
import scipy.sparse


def expand_matrix(matrix, shape):
    """Ensure matrix is 2-D of the given shape, make it sparse diagonal if not.

    If the matrix is 0-D or 1-D, ensure it's square and make it a diagonal
    sparse matrix. Otherwise, just check the shape.
    """
    if matrix.ndim < 2:
        assert shape[0] == shape[1]
        data = matrix * np.ones(shape[0]) if matrix.ndim == 0 else matrix
        assert data.size == shape[0]
        matrix = scipy.sparse.dia_matrix((data, 0), shape=(shape[0], shape[0]))
    else:
        assert matrix.ndim == 2

    assert matrix.shape == shape
    return matrix


def scale_matrix(matrix, scale):
    scale = np.asarray(scale)

    if isinstance(matrix, scipy.sparse.spmatrix) and scale.size > 1:
        assert scale.ndim < 2
        assert scale.size == matrix.shape[1]
        diag = scipy.sparse.dia_matrix((scale, 0), shape=(scale.size, scale.size))
        return matrix.dot(diag)
    elif scale.size == 1:
        # avoid bug where a sparse n x 1 matrix times a 1-vector gives a vector
        return matrix * scale.item()
    else:
        return matrix * scale


def stack_matrices(matrices, order="v"):
    assert order in ("h", "v")

    if all(isinstance(m, scipy.sparse.spmatrix) for m in matrices):
        return (
            scipy.sparse.vstack(matrices)
            if order == "v"
            else scipy.sparse.hstack(matrices)
        )
    elif all(isinstance(m, np.ndarray) for m in matrices):
        return np.vstack(matrices) if order == "v" else np.hstack(matrices)
    else:
        raise NotImplementedError(
            "All matrices must be the same type, one of: "
            "np.ndarray, scipy.sparse.spmatrix"
        )
