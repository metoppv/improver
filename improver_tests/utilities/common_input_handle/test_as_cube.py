# (C) Crown copyright, Met Office. All rights reserved.
#
# This file is part of improver and is released under a BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
from iris.cube import Cube, CubeList
import pytest

from improver.utilities.common_input_handle import as_cube


def test_cubelist_as_cube():
    cube = Cube([0])
    cubes = CubeList([cube])
    res = as_cube(cubes)
    assert id(res) == id(cube)


def test_cube_as_cube():
    cube = Cube([0])
    res = as_cube(cube)
    assert id(res) == id(cube)


def test_no_cube_provided():
    msg = "A cube should be provided."
    with pytest.raises(ValueError, match=msg):
        as_cube(None)


def test_non_cube_provided():
    msg = "A cube should be provided."
    with pytest.raises(TypeError, match=msg):
        as_cube(["cube"])


def test_multiple_cubes_provided():
    cubes = CubeList([Cube([0]), Cube([1])])
    msg = "A single cube should be provided."
    with pytest.raises(ValueError, match=msg):
        as_cube(cubes)
