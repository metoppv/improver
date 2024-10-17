# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the as_cube function."""
from unittest.mock import patch, sentinel

import pytest
from iris.coords import DimCoord
from iris.cube import Cube, CubeList

from improver.utilities.common_input_handle import as_cube


class HaltExecution(Exception):
    pass


@patch("improver.utilities.common_input_handle.as_cubelist")
def test_as_cubelist_called(mock_as_cubelist):
    """Check that we pass our input arguments directly to as_cubelist."""
    mock_as_cubelist.side_effect = HaltExecution
    try:
        as_cube(sentinel.cube, sentinel.cubelist)
    except HaltExecution:
        pass
    mock_as_cubelist.assert_called_once_with(sentinel.cube, sentinel.cubelist)


def test_cubelist_as_cube():
    """Test that a Cube is returned when a CubeList is provided."""
    cube = Cube([0])
    cubes = CubeList([cube])
    res = as_cube(cubes)
    assert id(res) == id(cube)


def test_multiple_cube_return():
    """Test that an error is raised in the case where otherwise multiple cubes
    would be returned."""
    cubes = CubeList([Cube([0]), Cube([1])])
    msg = "Unable to return a single cube."
    with pytest.raises(ValueError, match=msg):
        as_cube(cubes)


def test_multiple_cube_input_single_return():
    """Test that a single cube is returned where possible where more than 1
    input cube is provided."""
    cube0 = Cube([0], aux_coords_and_dims=((DimCoord(0, long_name="dim0"), None),))
    cube1 = Cube([0], aux_coords_and_dims=((DimCoord(1, long_name="dim0"), None),))
    target = CubeList([cube0, cube1]).merge_cube()
    res = as_cube(CubeList([cube0, cube1]))
    assert res == target
