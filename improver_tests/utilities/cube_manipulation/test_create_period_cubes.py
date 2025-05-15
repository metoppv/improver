# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""
Unit tests for the function create_period_cubes.
"""

from datetime import datetime

import iris
import numpy as np
import pytest
from iris.cube import Cube
from numpy.testing import assert_allclose

from improver.synthetic_data.set_up_test_cubes import (
    set_up_percentile_cube,
    set_up_variable_cube,
)
from improver.utilities.cube_manipulation import create_period_cubes


@pytest.fixture(name="input_cube")
def input_cube() -> Cube:
    """Test cube of input wind speeds with percentile and time coordinates"""
    data1 = np.array([[1, 2, 3, 4]], dtype=np.float32).reshape([2, 2])
    data2 = np.array([[1, 2, 3, 5]], dtype=np.float32).reshape([2, 2])
    data3 = np.array([[1, 2, 3, 5]], dtype=np.float32).reshape([2, 2])
    data4 = np.array([[1, 2, 3, 7]], dtype=np.float32).reshape([2, 2])
    cube1 = set_up_variable_cube(data1, time=datetime(2017, 11, 10, 0, 0))
    cube2 = set_up_variable_cube(data2, time=datetime(2017, 11, 10, 1, 0))
    cube3 = set_up_variable_cube(data3, time=datetime(2017, 11, 10, 2, 0))
    cube4 = set_up_variable_cube(data4, time=datetime(2017, 11, 10, 3, 0))
    cube_list = iris.cube.CubeList([cube1, cube2, cube3, cube4])
    cube = cube_list.merge()[0]
    return cube


@pytest.fixture(name="result_cube")
def result_cube() -> Cube:
    """Test cube of maximum vertical velocities over the height levels"""
    data = np.array([[1, 2, 3, 5]], dtype=np.float32).reshape([1, 2, 2])
    cube = set_up_percentile_cube(
        data, time=datetime(2017, 11, 10, 3, 0), percentiles=[50]
    )
    return cube


def test_basic(input_cube, result_cube):
    """Test that create_period_cubes combines data over time periods correctly"""
    lower = datetime(2017, 11, 10, 0, 0)
    upper = datetime(2017, 11, 10, 3, 0)
    three_hour = create_period_cubes(
        input_cube,
        period=3,
        coords=["time"],
        method_kwargs={"percent": [50], "fast_percentile_method": True},
    )
    assert_allclose(three_hour.data, result_cube.data)
    assert lower == three_hour.coord("time").cell(0).bound[0]
    assert upper == three_hour.coord("time").cell(0).bound[1]

def test_other_coords(input_cube):
    """Test that there is an error if other coords are used."""
    msg = "Time or time&realization need to be chosen as coords."
    with pytest.raises(ValueError, match=msg):
        create_period_cubes(input_cube, period=3, coords=["time", "latitude"])
