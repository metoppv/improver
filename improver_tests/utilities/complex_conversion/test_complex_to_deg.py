# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

from improver.utilities.complex_conversion import complex_to_deg

from . import COMPLEX_ANGLES


def test_fails_if_data_is_not_array():
    """Test code raises a Type Error if input data not an array."""
    input_data = 0 - 1j
    msg = "Input data is not a numpy array, but {}".format(type(input_data))
    with pytest.raises(TypeError, match=msg):
        complex_to_deg(input_data)


def test_handles_angle_wrap():
    """Test that code correctly handles 360 and 0 degrees."""
    # Input is complex for 0 and 360 deg - both should return 0.0.
    input_data = np.array([1 + 0j, 1 - 0j])
    result = complex_to_deg(input_data)
    assert (result == 0.0).all()


def test_converts_array():
    """Tests that array of complex values are converted to degrees."""
    result = complex_to_deg(COMPLEX_ANGLES)
    assert isinstance(result, np.ndarray)
    assert_array_almost_equal(result, np.arange(0.0, 360, 10))
