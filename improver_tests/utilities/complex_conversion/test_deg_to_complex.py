import numpy as np
from numpy.testing import assert_array_almost_equal

# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
from improver.utilities.complex_conversion import deg_to_complex

from . import COMPLEX_ANGLES


def test_converts_single():
    """Tests that degree angle value is converted to complex."""
    expected_out = 0.707106781187 + 0.707106781187j
    result = deg_to_complex(45.0)
    assert_array_almost_equal(result, expected_out)


def test_handles_angle_wrap():
    """Test that code correctly handles 360 and 0 degrees."""
    expected_out = 1 + 0j
    result = deg_to_complex(0)
    assert_array_almost_equal(result, expected_out)

    expected_out = 1 - 0j
    result = deg_to_complex(360)
    assert_array_almost_equal(result, expected_out)


def test_converts_array():
    """Tests that array of floats is converted to complex array."""
    result = deg_to_complex(np.arange(0.0, 360, 10))
    assert isinstance(result, np.ndarray)
    assert_array_almost_equal(result, COMPLEX_ANGLES)
