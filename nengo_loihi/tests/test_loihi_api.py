import numpy as np
import pytest

from nengo_loihi.loihi_api import overflow_signed


@pytest.mark.parametrize("b", (8, 16, 17, 23))
def test_overflow_signed(b):
    x = np.arange(-2**(b+2), 2**(b+2), dtype=np.int32)

    # compute check values
    b2 = 2**b
    z = x % b2
    zmask = np.right_shift(x, b) % 2  # sign bit, the b-th bit

    # if the sign bit is set, subtract 2**b to make it negative
    z -= np.left_shift(zmask, b)

    # compute whether it's overflowed
    q = (x < -b2) | (x >= b2)

    y, o = overflow_signed(x, bits=b)
    assert np.array_equal(y, z)
    assert np.array_equal(o, q)
