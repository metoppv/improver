# (C) Crown copyright, Met Office. All rights reserved.
#
# This file is part of IMPROVER and is released under a BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""
Unit tests for the function "cube_manipulation.height_of_maximum".
"""

import numpy as np
import pytest
from iris.cube import Cube
from numpy.testing import assert_allclose

from improver.synthetic_data.set_up_test_cubes import set_up_variable_cube
from improver.utilities.cube_manipulation import height_of_maximum


@pytest.fixture(name="input_cube")
def input_cube() -> Cube:
    """Test cube of vertical velocity  on height levels"""
    data = np.array(
        [[[2, 4, 9], [3, 4, 8]], [[5, 3, 3], [4, 2, 7]], [[9, 5, 1], [2, 5, 8]]]
    )
    cube = set_up_variable_cube(
        data=data, name="vertical_velocity", height_levels=[5, 75, 300]
    )
    return cube


@pytest.fixture(name="max_cube")
def max_cube() -> Cube:
    """Test cube of maximum vertical velocities over the height levels"""
    data = np.array([[9, 5, 9], [4, 5, 8]])
    cube = set_up_variable_cube(data=data, name="vertical_velocity", height_levels=[1])
    return cube


@pytest.fixture(name="high_cube")
def high_cube() -> Cube:
    """Test cube when we want the highest maximum"""
    data_high = np.array([[300, 300, 5], [75, 300, 300]])
    cube = set_up_variable_cube(
        data=data_high, name="vertical_velocity", height_levels=[1]
    )
    return cube


@pytest.fixture(name="low_cube")
def low_cube() -> Cube:
    """Test cube when we want the lowest maximum"""
    data_low = np.array([[300, 300, 5], [75, 300, 5]])
    cube = set_up_variable_cube(
        data=data_low, name="vertical_velocity", height_levels=[1]
    )
    return cube

@pytest.mark.parametrize("new_name", [None, "height_of_maximum"])
@pytest.mark.parametrize("find_lowest", ["True", "False"])
def test_basic(input_cube, max_cube,new_name,find_lowest,high_cube,low_cube):
    """Tests that the name of the cube will be correctly updated. Test that
    if find_lowest is true the lowest maximum height will be found"""

    expected_name=new_name if new_name else input_cube.name()
    expected_cube=high_cube if find_lowest else low_cube

    output_cube = height_of_maximum(input_cube, max_cube, new_name=new_name, find_lowest=find_lowest)
    
    assert expected_name == output_cube.name()
    assert_allclose(output_cube.data, expected_cube.data)


def test_one_height(input_cube):
    one_height = input_cube[0]
    msg = "More than 1 height level is required."
    with pytest.raises(ValueError, match=msg):
        height_of_maximum(one_height, one_height)
