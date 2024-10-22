# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
import numpy as np
from numpy.testing import assert_array_almost_equal

from improver.utilities.complex_conversion import complex_to_deg, deg_to_complex

from . import COMPLEX_ANGLES


def test_roundtrip_complex_deg_complex():
    """Tests that array of values are converted to degrees and back."""
    tmp_degrees = complex_to_deg(COMPLEX_ANGLES)
    result = deg_to_complex(tmp_degrees)
    assert_array_almost_equal(result, COMPLEX_ANGLES)


def test_roundtrip_deg_complex_deg():
    """Tests that array of values are converted to complex and back."""
    src_degrees = np.arange(0.0, 360, 10)
    tmp_complex = deg_to_complex(src_degrees)
    result = complex_to_deg(tmp_complex)
    assert_array_almost_equal(result, src_degrees)
