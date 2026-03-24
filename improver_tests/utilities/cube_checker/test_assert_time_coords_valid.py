# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
from datetime import datetime
from typing import List

import numpy as np
import pytest
from iris.cube import Cube

from improver.synthetic_data.set_up_test_cubes import set_up_variable_cube
from improver.utilities.cube_checker import assert_time_coords_valid


@pytest.fixture(name="cubes")
def cubes_fixture(time_bounds) -> List[Cube]:
    """Set up matching r, y, x cubes matching Plugin requirements, with or without time
    bounds"""
    cubes = []
    data = np.ones((2, 3, 4), dtype=np.float32)
    kwargs = {}
    if time_bounds:
        kwargs["time_bounds"] = (
            datetime(2017, 11, 10, 3, 0),
            datetime(2017, 11, 10, 4, 0),
        )
    cube = set_up_variable_cube(data, **kwargs)
    for descriptor in (
        {"name": "air_temperature", "units": "K"},
        {"name": "air_pressure", "units": "Pa"},
        {"name": "relative_humidity", "units": "kg kg-1"},
    ):
        cube = cube.copy()
        cube.rename(descriptor["name"])
        cube.units = descriptor["units"]
        cubes.append(cube)
    return cubes


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


def inconsistent_time_bounds(cubes: List[Cube]):
    """Adds time bounds only to the first cube"""
    time_point = cubes[0].coord("time").points[0]
    cubes[0].coord("time").bounds = (time_point - 10800, time_point)


def inconsistent_time_point(cubes: List[Cube]):
    """Moves time point of first cube back by one hour"""
    cubes[0].coord("time").points = cubes[0].coord("time").points - 3600


def inconsistent_frt(cubes: List[Cube]):
    """Moves forecast_reference_time point of first cube back by one hour"""
    cubes[0].coord("forecast_reference_time").points = (
        cubes[0].coord("forecast_reference_time").points - 3600
    )


def remove_one_time_bounds(cubes: List[Cube]):
    """Removes time bounds from first cube"""
    cubes[0].coord("time").bounds = None


def remove_two_time_bounds(cubes: List[Cube]):
    """Removes time bounds from first two cubes"""
    cubes[0].coord("time").bounds = None
    cubes[1].coord("time").bounds = None


def only_one_cube(cubes: List[Cube]):
    """Removes second and third cubes"""
    cubes.pop(2)
    cubes.pop(1)


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
