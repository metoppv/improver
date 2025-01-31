# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""
Unit tests for the function "cube_manipulation.maximum_in_height".
"""

import numpy as np
import pytest
from iris.cube import Cube

from improver.synthetic_data.set_up_test_cubes import set_up_variable_cube
from improver.utilities.cube_manipulation import maximum_in_height


@pytest.fixture()
def wet_bulb_temperature() -> Cube:
    "Generate a cube of wet bulb temperature on vertical levels"
    data = np.array(
        [
            [[100, 200, 100], [100, 200, 100]],
            [[300, 400, 100], [300, 400, 100]],
            [[200, 300, 300], [200, 300, 300]],
        ]
    )
    cube = set_up_variable_cube(
        data=data,
        name="wet_bulb_temperature",
        vertical_levels=[100, 200, 300],
        height=True,
    )
    return cube


@pytest.mark.parametrize("new_name", (None, "max_wet_bulb_temperature"))
@pytest.mark.parametrize(
    "lower_bound,upper_bound,expected",
    (
        (None, None, [300, 400, 300]),
        (None, 200, [300, 400, 100]),
        (250, None, [200, 300, 300]),
        (50, 1000, [300, 400, 300]),
    ),
)
def test_maximum_in_height(
    lower_bound, upper_bound, expected, wet_bulb_temperature, new_name
):
    """Test that the maximum over the height coordinate is correctly calculated for
    different combinations of upper and lower bounds. Also checks the name of the
    cube is correctly updated."""

    expected_name = "wet_bulb_temperature"
    if new_name:
        expected_name = new_name

    result = maximum_in_height(wet_bulb_temperature, lower_bound, upper_bound, new_name)

    assert np.allclose(result.data, [expected] * 2)
    assert expected_name == result.name()


def test_height_bounds_error(wet_bulb_temperature):
    """Test an error is raised if the input cube doesn't have any vertical levels
    between the height bounds."""

    with pytest.raises(
        ValueError, match="any vertical levels between the provided bounds"
    ):
        maximum_in_height(
            wet_bulb_temperature, lower_height_bound=50, upper_height_bound=75
        )
