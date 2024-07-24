# (C) Crown copyright, Met Office. All rights reserved.
#
# This file is part of IMPROVER and is released under a BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Tests of read_from_table utilities"""

import numpy as np
import pytest

from improver.synthetic_data.set_up_test_cubes import set_up_variable_cube
from improver.utilities.extract_from_table import ExtractValueFromTable


@pytest.fixture
def table():
    """Set up a dictionary representing a table of data"""

    # table_dict={1:{0:0.95, 10.19:0.9, 15.43:0.85}, 2:{0:0.9, 10.19:0.85, 15.43:0.8}}
    table_dict = {
        0: {1: 0.95, 2: 0.9},
        10.19: {1: 0.9, 2: 0.85},
        15.43: {1: 0.85, 2: 0.8},
    }
    return table_dict


@pytest.fixture
def lapse_class():
    """Set up a cube containing lapse class"""
    data = np.array([[1, 2], [2, 2]])
    return set_up_variable_cube(data=data, name="lapse_class")


@pytest.fixture
def wind_gust_900m():
    """Set up cube containing 900m wind gust data"""
    data = np.array([[10.19, 12], [17, 7]], dtype="float32")
    return set_up_variable_cube(data=data, name="900m_wind_gust")


@pytest.mark.parametrize("new_name", [None, "adjusted_wind_speed_of_gust"])
@pytest.mark.parametrize(
    "lapse_class_value, wind_gust_value, expected",
    [
        [1, -4, 0.95],
        [1, 5, 0.95],
        [2, 10.19, 0.85],
        [2, 14, 0.85],
        [2, 15.43, 0.8],
        [3, 20, 0.8],
    ],
)
def test_read_table(
    table,
    lapse_class,
    wind_gust_900m,
    lapse_class_value,
    wind_gust_value,
    expected,
    new_name,
):
    """Test plugin to extract values from table"""
    wind_gust_900m.data.fill(wind_gust_value)
    lapse_class.data.fill(lapse_class_value)
    result = ExtractValueFromTable(row_name="lapse_class", new_name=new_name)(
        wind_gust_900m, lapse_class, table=table
    )

    expected_data = np.array([[expected, expected], [expected, expected]])
    expected_cube = lapse_class.copy(data=expected_data)

    if new_name:
        expected_cube.rename(new_name)
    print(result.data)
    np.testing.assert_array_almost_equal(result.data, expected_cube.data)
    assert result == expected_cube


def test_too_many_cubes(table, lapse_class, wind_gust_900m):
    """Test that an error is raised if the number of cubes is not equal to 2"""
    with pytest.raises(ValueError, match="Exactly 2 cubes should be provided"):
        ExtractValueFromTable(row_name="lapse_class")(
            wind_gust_900m, lapse_class, lapse_class, table=table
        )


def test_cubes_different_shapes(table, lapse_class, wind_gust_900m):
    """Test that an error is raised if the cubes do not have the same shape"""
    lapse_class = lapse_class[0]
    with pytest.raises(ValueError, match="Shapes of cubes do not match"):
        ExtractValueFromTable(row_name="lapse_class")(
            wind_gust_900m, lapse_class, table=table
        )
