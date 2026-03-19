# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
from typing import List

import pytest
from iris.cube import Cube

from improver.utilities.cube_checker import assert_time_coords_valid
from improver_tests.utilities.test_cube_checker import (
    inconsistent_frt,
    inconsistent_time_bounds,
    inconsistent_time_point,
    only_one_cube,
    remove_one_time_bounds,
    remove_two_time_bounds,
)


def swap_frt_for_blend_time(cubes: List[Cube]):
    """Renames the forecast_reference_time coord on each cube to blend_time"""
    for cube in cubes:
        cube.coord("forecast_reference_time").rename("blend_time")


@pytest.mark.parametrize("blend_time", (True, False))
@pytest.mark.parametrize("time_bounds", (True, False))
@pytest.mark.parametrize("input_count", (2, 3))
def test_time_coords_valid(
    cubes: List[Cube], input_count: int, time_bounds: bool, blend_time: bool
):
    """Test that no exceptions are raised when the required conditions are met
    for either 2 or 3 cubes, with or without time bounds, with or without blend_time"""
    if blend_time:
        swap_frt_for_blend_time(cubes)
    assert_time_coords_valid(cubes[:input_count], time_bounds=time_bounds)


@pytest.mark.parametrize(
    "modifier, time_bounds, error_match",
    (
        (inconsistent_time_bounds, True, "^time coordinates do not match."),
        (inconsistent_time_bounds, False, "^air_temperature must not have time bounds"),
        (inconsistent_time_point, True, "^time coordinates do not match."),
        (inconsistent_time_point, False, "^time coordinates do not match."),
        (inconsistent_frt, True, "^forecast_reference_time coordinates do not match."),
        (inconsistent_frt, False, "^forecast_reference_time coordinates do not match."),
        (remove_one_time_bounds, True, "^air_temperature must have time bounds"),
        (
            remove_two_time_bounds,
            True,
            "^air_temperature and air_pressure must have time bounds",
        ),
        (only_one_cube, False, "^Need at least 2 cubes to check. Found 1"),
    ),
)
def test_time_coord_exceptions(
    cubes, modifier: callable, time_bounds: bool, error_match: str
):
    """Checks that assert_time_coords_valid raises useful errors
    when the required conditions are not met."""
    modifier(cubes)
    with pytest.raises(ValueError, match=error_match):
        assert_time_coords_valid(cubes, time_bounds=time_bounds)
