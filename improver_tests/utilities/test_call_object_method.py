# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Tests for the call_object_method utility function."""

import iris.analysis
import numpy as np
import pytest
from iris.cube import Cube

from improver.synthetic_data.set_up_test_cubes import set_up_variable_cube
from improver.utilities.call_object_method import call_object_method


@pytest.fixture(name="cube")
def cube_fixture() -> Cube:
    """Set up a cube of data"""
    data = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.float32)
    cube = set_up_variable_cube(
        data,
        name="test_variable",
        units="m/s",
        vertical_levels=[1000, 2000],
        height=True,
    )
    return cube


def test_call_object_method(cube):
    """Test that call_object_method correctly calls a method on an object."""
    result = call_object_method(
        cube, "collapsed", coords="height", aggregator=iris.analysis.SUM
    )
    expected_data = np.array([[6.0, 8.0], [10.0, 12.0]], dtype=np.float32)
    assert np.array_equal(result.data, expected_data)
    assert result.name() == "test_variable"
    assert result.units == "m/s"


def test_call_object_method_invalid_method(cube):
    """Test that call_object_method raises an AttributeError for an invalid method."""
    with pytest.raises(AttributeError):
        call_object_method(cube, "non_existent_method")


def test_call_object_method_invalid_args(cube):
    """Test that call_object_method raises a TypeError for invalid arguments."""
    with pytest.raises(TypeError):
        call_object_method(cube, "collapsed", non_existent_arg=True)


def test_call_object_method_no_args(cube):
    """Test that call_object_method works with no additional arguments."""
    result = call_object_method(cube, "copy")
    assert result == cube
    assert result is not cube  # Ensure it's a copy, not the same object
