from nengo.exceptions import ValidationError
import pytest

from nengo_loihi.config import BlockShape


def test_block_shape_errors():
    with pytest.raises(ValidationError, match="[Mm]ust be a tuple"):
        BlockShape([5], [15])

    with pytest.raises(ValidationError, match="[Mm]ust be an int"):
        BlockShape((5,), (15.0,))

    with pytest.raises(ValidationError, match="[Mm]ust be the same length"):
        BlockShape((2, 2), (8,))


def test_block_shape_1d():
    block_shape = BlockShape((5,), (15,))

    assert block_shape.shape == (5,)
    assert block_shape.ensemble_shape == (15,)
    assert block_shape.n_splits == 3
    assert block_shape.block_size == 5
    assert block_shape.ensemble_size == 15

    assert list(block_shape.zip_dimensions()) == [(15, 5)]

    with pytest.warns(UserWarning, match="uses the same number of blocks as"):
        block_shape = BlockShape((4,), (9,))
        assert block_shape.shape == (3,)
        assert block_shape.n_splits == 3


def test_block_shape_3d():
    block_shape = BlockShape((2, 3, 2), (4, 6, 7))

    assert block_shape.shape == (2, 3, 2)
    assert block_shape.ensemble_shape == (4, 6, 7)
    assert block_shape.n_splits == 2 * 2 * 4
    assert block_shape.block_size == 2 * 3 * 2
    assert block_shape.ensemble_size == 4 * 6 * 7

    assert list(block_shape.zip_dimensions()) == [(4, 2), (6, 3), (7, 2)]

    with pytest.warns(UserWarning, match="uses the same number of blocks as"):
        block_shape = BlockShape((2, 3, 5), (4, 6, 7))
        assert block_shape.shape == (2, 3, 4)
        assert block_shape.n_splits == 2 * 2 * 2
