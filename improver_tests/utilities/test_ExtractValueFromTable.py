# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Tests of read_from_table utilities"""

import numpy as np
import pytest

from improver.synthetic_data.set_up_test_cubes import set_up_variable_cube
from improver.utilities.extract_from_table import ExtractValueFromTable


@pytest.fixture
def table_2D():
    """Set up a dictionary representing a table of data"""

    table_dict = {
        "data": {
            0: {1: 0.95, 2: 0.9},
            10.19: {1: 0.9, 2: 0.85},
            15.43: {1: 0.85, 2: 0.8},
        },
        "metadata": {"name": "Gust Factor", "units": "1"},
    }
    return table_dict


@pytest.fixture
def table_2D_random_order():
    """Set up a dictionary representing a table of data but with rows and columns
    in a random order"""

    table_dict = {
        "data": {
            10.19: {2: 0.85, 1: 0.9},
            0: {1: 0.95, 2: 0.9},
            15.43: {2: 0.8, 1: 0.85},
        },
        "metadata": {"name": "Gust Factor", "units": "1"},
    }
    return table_dict


@pytest.fixture
def lapse_class():
    """Set up a cube containing lapse class"""
    data = np.empty((2, 2), dtype=np.float32)  # Values will be overwritten in the test
    return set_up_variable_cube(data=data, name="lapse_class", units="1")


@pytest.fixture
def wind_gust_900m():
    """Set up cube containing 900m wind gust data"""
    data = np.empty((2, 2), dtype=np.float32)  # Values will be overwritten in the test
    return set_up_variable_cube(data=data, name="900m_wind_gust", units="m/s")


@pytest.fixture
def table_1D():
    """Set up a dictionary representing a table of 1D data"""

    table_dict = {
        "data": {1: {5.5: 1, 5.4: 2, 4: 3}},
        "metadata": {"name": "Lapse Class", "units": "1"},
    }
    return table_dict


@pytest.fixture
def lapse_rate():
    """Set up cube containing lapse rate data"""
    data = np.empty((2, 2), dtype=np.float32)  # Values will be overwritten in the test
    return set_up_variable_cube(data=data, name="lapse_rate", units="K/m")


@pytest.mark.parametrize("table_name", ["table_2D", "table_2D_random_order"])
@pytest.mark.parametrize("new_name", [None, "adjusted_wind_speed_of_gust"])
@pytest.mark.parametrize(
    "lapse_class_value, wind_gust_value, expected",
    [
        [1, -4, 0.95],
        [1, 5, 0.95],
        [1, 10.18999992, 0.9],
        [2, 10.19, 0.85],
        [2, 14, 0.85],
        [2, 15.43, 0.8],
        [3, 20, 0.8],
    ],
)
def test_read_table(
    table_name,
    lapse_class,
    wind_gust_900m,
    lapse_class_value,
    wind_gust_value,
    expected,
    new_name,
    request,
):
    """Test plugin to extract values from table"""
    table = request.getfixturevalue(table_name)

    wind_gust_900m.data.fill(wind_gust_value)
    lapse_class.data.fill(lapse_class_value)
    result = ExtractValueFromTable(
        row_name="lapse_class", new_name=new_name, table=table
    )(wind_gust_900m, lapse_class)
    expected_data = np.full_like(
        lapse_class.data, fill_value=expected, dtype=np.float32
    )
    expected_cube = lapse_class.copy(data=expected_data)
    expected_cube.units = "1"
    if new_name:
        expected_cube.rename(new_name)
    np.testing.assert_array_almost_equal(result.data, expected_cube.data)
    assert result == expected_cube


@pytest.mark.parametrize(
    "lapse_rate_value, expected",
    [[7, 1], [5.2, 3], [5.45, 2], [2, 3], [np.nan, np.nan]],
)
def test_read_table_1D(
    table_1D, lapse_rate, wind_gust_900m, lapse_rate_value, expected
):
    """Test plugin to extract values from table"""
    lapse_rate.data.fill(lapse_rate_value)

    result = ExtractValueFromTable(row_name="lapse_rate", table=table_1D)(
        lapse_rate, wind_gust_900m
    )

    expected_data = np.full_like(lapse_rate.data, fill_value=expected, dtype=np.float32)
    expected_cube = lapse_rate.copy(data=expected_data)
    expected_cube.units = "1"

    np.testing.assert_array_almost_equal(result.data, expected_cube.data)
    if not np.isnan(expected):  # Nan values are not equal
        assert result == expected_cube


def test_too_many_cubes(table_2D, lapse_class, wind_gust_900m):
    """Test that an error is raised if the number of cubes is not equal to 2"""
    with pytest.raises(ValueError, match="Exactly 2 cubes should be provided"):
        ExtractValueFromTable(row_name="lapse_class", table=table_2D)(
            wind_gust_900m, lapse_class, lapse_class
        )


def test_cubes_different_shapes(table_2D, lapse_class, wind_gust_900m):
    """Test that an error is raised if the cubes do not have the same shape"""
    lapse_class = lapse_class[0]
    with pytest.raises(ValueError, match="Shapes of cubes do not match"):
        ExtractValueFromTable(row_name="lapse_class", table=table_2D)(
            wind_gust_900m, lapse_class
        )
