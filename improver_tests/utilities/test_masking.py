# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
import numpy as np
import pytest
from iris.cube import Cube

from improver.utilities.masking import as_masked_array


@pytest.fixture(params=[np.nan, np.inf, -np.inf])
def data_array(request):
    bad_value = request.param
    return np.array([[1.0, 2.0], [bad_value, 4.0], [5.0, bad_value]])


@pytest.fixture()
def mask():
    """Fixture to create a sample mask for the data array"""
    return np.array([[False, False], [True, False], [False, True]])


def test_as_masked_array_from_array(data_array, mask):
    """Test the as_masked_array utility function with a numpy ndarray"""
    expected_mask = mask
    result = as_masked_array(data_array)
    assert isinstance(result, np.ma.MaskedArray)
    np.testing.assert_array_equal(result.mask, expected_mask)


def test_as_masked_array_from_cube(data_array, mask):
    """Test the as_masked_array utility function with an Iris Cube"""
    expected_mask = mask
    cube = Cube(data_array)
    result = as_masked_array(cube)
    assert isinstance(result, np.ma.MaskedArray)
    np.testing.assert_array_equal(result.mask, expected_mask)


def test_as_masked_array_from_masked_array(data_array, mask):
    """Test the as_masked_array utility function with a numpy MaskedArray"""
    expected_mask = mask.copy()
    data_array = np.ma.array(data_array, mask=mask)
    result = as_masked_array(data_array)
    assert result is data_array  # Should return the same object
    np.testing.assert_array_equal(
        result.mask, expected_mask
    )  # Mask should be unchanged
    np.testing.assert_array_equal(
        result.data, data_array.data
    )  # Data should be unchanged


def test_as_masked_array_from_incorrectly_masked_array(data_array, mask):
    """Test the as_masked_array utility function with a numpy MaskedArray that doesn't mask all invalid values"""
    expected_mask = mask.copy()
    mask[2, 0] = False  # Intentionally leave one invalid value unmasked
    data_array = np.ma.array(data_array, mask=mask)
    result = as_masked_array(data_array)
    assert result is data_array  # Should return the same object
    np.testing.assert_array_equal(
        result.mask, expected_mask
    )  # Mask should be unchanged
    np.testing.assert_array_equal(
        result.data, data_array.data
    )  # Data should be unchanged


def test_as_masked_array_no_invalids():
    """Test the as_masked_array utility function with data that has no invalid values"""
    data_array = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    result = as_masked_array(data_array)
    assert isinstance(result, np.ma.MaskedArray)
    assert not np.ma.is_masked(result)
    np.testing.assert_array_equal(result.data, data_array)


def test_as_masked_array_all_invalids():
    """Test the as_masked_array utility function with data that is all invalid values"""
    data_array = np.array([[np.nan, np.inf], [-np.inf, np.nan]])
    expected_mask = np.array([[True, True], [True, True]])
    result = as_masked_array(data_array)
    assert isinstance(result, np.ma.MaskedArray)
    np.testing.assert_array_equal(result.mask, expected_mask)
