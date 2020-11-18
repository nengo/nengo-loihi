import nengo
import pytest
from nengo.exceptions import ValidationError

from nengo_loihi.config import BlockShape, add_params


def test_block_shape_errors():
    with pytest.raises(ValidationError, match="[Mm]ust be a tuple"):
        BlockShape([5], [15])

    with pytest.raises(ValidationError, match="[Mm]ust be an int"):
        BlockShape((5,), (15.0,))

    with pytest.raises(ValidationError, match="[Mm]ust be the same length"):
        BlockShape((2, 2), (8,))

    with nengo.Network() as net:
        add_params(net)
        a = nengo.Ensemble(10, 1)

        with pytest.raises(ValidationError, match="Block shape ensemble size"):
            net.config[a].block_shape = BlockShape((3, 2), (6, 2))


def test_block_shape_1d():
    block_shape = BlockShape((5,), (15,))

    assert block_shape.shape == (5,)
    assert block_shape.ensemble_shape == (15,)
    assert block_shape.n_splits == 3
    assert block_shape.block_size == 5
    assert block_shape.ensemble_size == 15

    assert list(block_shape.zip_dimensions()) == [(15, 5)]


def test_block_shape_3d():
    block_shape = BlockShape((2, 3, 2), (4, 6, 7))

    assert block_shape.shape == (2, 3, 2)
    assert block_shape.ensemble_shape == (4, 6, 7)
    assert block_shape.n_splits == 2 * 2 * 4
    assert block_shape.block_size == 2 * 3 * 2
    assert block_shape.ensemble_size == 4 * 6 * 7

    assert list(block_shape.zip_dimensions()) == [(4, 2), (6, 3), (7, 2)]
