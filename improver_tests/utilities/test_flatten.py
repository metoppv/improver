# (C) Crown copyright, Met Office. All rights reserved.
#
# This file is part of improver and is released under a BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for flattening an arbitrarily nested iterable."""

import numpy as np
import pytest
from iris.cube import Cube, CubeList

from improver.synthetic_data.set_up_test_cubes import set_up_variable_cube
from improver.utilities.flatten import flatten


@pytest.fixture
def cube() -> Cube:
    """Sets up a cube for testing"""
    return set_up_variable_cube(np.zeros((2, 2), dtype=np.float32),)


@pytest.mark.parametrize(
    "nested,expected",
    (
        ([0, 1, 2], [0, 1, 2]),
        ([0, 1, [2, 3]], [0, 1, 2, 3]),
        (["a", "b", "c"], ["a", "b", "c"]),
        (["a", "b", ["c", "d"]], ["a", "b", "c", "d"]),
        (
            [np.array([0]), np.array([1]), [np.array([2]), np.array([2])]],
            [np.array([0]), np.array([1]), np.array([2]), np.array([2])],
        ),
        ([cube, cube, [cube, cube]], [cube, cube, cube, cube]),
        (CubeList([cube, cube, CubeList([cube, cube])]), [cube, cube, cube, cube]),
        ([0, [1, [2, [3]]]], [0, 1, 2, 3]),
        ([0, [1, 2], [3]], [0, 1, 2, 3]),
        (["cat"], ["cat"]),
        ((0, 1, (2, 3)), [0, 1, 2, 3]),
    ),
)
def test_basic(nested, expected):
    """Test flattening an arbitrarily nested iterable."""
    result = flatten(nested)
    assert result == expected
    assert isinstance(result, list)


@pytest.mark.parametrize(
    "nested", ((0), ("cat"), ({0: {1: "cat"}, 1: {2: "dog"}}),),
)
def test_exception(nested):
    """Test an exception is raised if inappropriate types
    are provided for flattening."""
    msg = "Expected object of type list or tuple"
    with pytest.raises(ValueError, match=msg):
        flatten(nested)
