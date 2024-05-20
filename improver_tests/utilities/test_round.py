# (C) Crown copyright, Met Office. All rights reserved.
#
# This file is part of improver and is released under a BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the "round" module"""


import numpy as np
import pytest

from improver.utilities.round import round_close


def test_round_close():
    """Test output when input is nearly an integer"""
    result = round_close(29.99999)
    assert result == 30
    assert isinstance(result, np.int64)


def test_dtype():
    """Test near-integer output with specific dtype"""
    result = round_close(29.99999, dtype=np.int32)
    assert result == 30
    assert isinstance(result, np.int32)


def test_round_close_array():
    """Test near-integer output from array input"""
    expected = np.array([30, 4], dtype=int)
    result = round_close(np.array([29.999999, 4.0000001]))
    np.testing.assert_array_equal(result, expected)


def test_error_not_close():
    """Test error when output would require significant rounding"""
    with pytest.raises(ValueError):
        round_close(29.9)
