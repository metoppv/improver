# (C) Crown copyright, Met Office. All rights reserved.
#
# This file is part of improver and is released under a BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
from iris.cube import Cube, CubeList
import pytest

from improver.utilities.common_input_handle import as_cubelist


def test_cubelist_as_cubelist():
    cube = Cube([0])
    res = as_cubelist(cube)
    assert isinstance(res, CubeList)
    assert id(res[0]) == id(cube)


def test_cube_as_cubelist():
    cube = Cube([0])
    res = as_cubelist(cube)
    assert isinstance(res, CubeList)
    assert id(res[0]) == id(cube)


def test_cube_cubelist_mixture_as_cubelist():
    cube = Cube([0])
    cubes = CubeList([cube])
    res = as_cubelist(cube, cubes)
    assert isinstance(res, CubeList)
    assert id(res[0]) == id(cube)
    assert id(res[1]) == id(cube)


def test_argument_provided():
    msg = "One or more cube should be provided."
    with pytest.raises(ValueError, match=msg):
        as_cubelist(None)


def test_no_cube_provided():
    msg = "One or more cube should be provided."
    with pytest.raises(ValueError, match=msg):
        as_cubelist([])
