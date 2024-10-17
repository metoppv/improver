# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the as_cubelist function."""
from itertools import permutations

import pytest
from iris.cube import Cube, CubeList

from improver.utilities.common_input_handle import as_cubelist


def test_cubelist_as_cubelist():
    """Test that a CubeList is returned when a CubeList is provided."""
    cube = Cube([0])
    cubes = CubeList([cube])
    res = as_cubelist(cubes)
    assert isinstance(res, CubeList)
    assert id(res[0]) == id(cube)


def test_iterable_as_cubelist():
    """Test that a CubeList is returned when a list is provided."""
    cube = Cube([0])
    cubes = [cube]
    res = as_cubelist(cubes)
    assert isinstance(res, CubeList)
    assert id(res[0]) == id(cube)


def test_cube_as_cubelist():
    """Test that a CubeList is returned when a Cube is provided."""
    cube = Cube([0])
    res = as_cubelist([cube])
    assert isinstance(res, CubeList)
    assert id(res[0]) == id(cube)


def test_cube_cubelist_mixture_as_cubelist():
    """
    Test that a CubeList is returned when a mixture of Cubes and CubeLists
    are provided.  Additionally demonstrate that the order of the input
    is preserved in the output, with each order permutation verified.
    """
    cube = Cube(0, long_name="1")
    cube2 = Cube(1, long_name="2")
    cube3 = Cube(2, long_name="3")
    cubes = CubeList([cube2])

    inputs = [cube, cubes, cube3]
    returns = [cube, cube2, cube3]

    for pindx in permutations(range(len(inputs))):  # permutation indices
        res = as_cubelist(*[inputs[ind] for ind in pindx])
        assert isinstance(res, CubeList)
        for ind, pind in enumerate(pindx):
            assert id(res[ind]) == id(returns[pind])


def test_no_argument_provided():
    """Test when no argument has been provided."""
    msg = "One or more cubes should be provided."
    with pytest.raises(ValueError, match=msg):
        as_cubelist()


def test_empty_list_provided():
    """Test when an empty list is provided."""
    msg = "One or more cubes should be provided."
    with pytest.raises(ValueError, match=msg):
        as_cubelist([])


def test_non_cube_cubelist_provided():
    """Test when a CubeList containing a non cube is provided."""
    msg = "A non iris Cube object has been provided."
    with pytest.raises(TypeError, match=msg):
        as_cubelist(CubeList(["not_a_cube"]))
