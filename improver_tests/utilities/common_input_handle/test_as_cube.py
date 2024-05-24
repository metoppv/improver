# (C) Crown copyright, Met Office. All rights reserved.
#
# This file is part of IMPROVER and is released under a BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
import pytest
from iris.cube import Cube, CubeList

from improver.utilities.common_input_handle import as_cube


def test_cubelist_as_cube():
    """Test that a Cube is returned when a CubeList is provided."""
    cube = Cube([0])
    cubes = CubeList([cube])
    res = as_cube(cubes)
    assert id(res) == id(cube)


def test_cube_as_cube():
    """Test that a Cube is returned when a Cube is provided."""
    cube = Cube([0])
    res = as_cube(cube)
    assert id(res) == id(cube)


def test_no_argument_provided():
    """Test that an error is raised when no cube is provided."""
    msg = "A cube should be provided."
    with pytest.raises(ValueError, match=msg):
        as_cube(None)


def test_non_cube_provided():
    """Test that an error is raised when a non-cube is provided."""
    msg = "A cube should be provided."
    with pytest.raises(TypeError, match=msg):
        as_cube(CubeList(["cube"]))


def test_multiple_cubes_provided():
    """Test that an error is raised when a CubeList containing multiple cubes are provided."""
    cubes = CubeList([Cube([0]), Cube([1])])
    msg = "A single cube should be provided."
    with pytest.raises(ValueError, match=msg):
        as_cube(cubes)


def test_cubelist_containing_non_cube():
    cubes = CubeList(["not_a_cube"])
    msg = "A cube should be provided."
    with pytest.raises(TypeError, match=msg):
        as_cube(cubes)
