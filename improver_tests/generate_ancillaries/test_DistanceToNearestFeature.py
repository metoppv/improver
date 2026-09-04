# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the DistanceToNearestFeature plugin."""

import numpy as np
import pytest
from iris.cube import CubeList
from numpy.testing import assert_array_equal

from improver.generate_ancillaries.generate_distance_to_feature import (
    DistanceToNearestFeature,
)
from improver.spotdata.build_spotdata_cube import build_spotdata_cube


@pytest.fixture()
def distance_cube_template():
    """Set up a site cube containing data at multiple sites."""

    latitude = np.array([48, 49, 50, 51])
    longitude = np.array([-1, 0, -1, 0])

    altitude = np.array(
        [-99999, -99999, -99999, -99999]
    )  # These values are not used but are required for cube creation.
    data = np.array(
        [-99999, -99999, -99999, -99999]
    )  # These values are not used but are required for cube creation.
    wmo_id = [
        "00000",
        "00001",
        "00002",
        "00003",
    ]  # These values are not used but are required for cube creation.

    prob_cube = build_spotdata_cube(
        data,
        name="distance_to_something",
        units="m",
        altitude=altitude,
        wmo_id=wmo_id,
        latitude=latitude,
        longitude=longitude,
    )
    return prob_cube


def test_distance_to_nearest_feature(distance_cube_template):
    """Test the DistanceToNearestFeature class calculates the distance to closest
    feature correctly."""

    river_cube = distance_cube_template.copy()
    river_cube.data = np.array([100, 200, 300, 400])
    lake_cube = distance_cube_template.copy()
    lake_cube.data = np.array([400, 300, 200, 100])
    ocean_cube = distance_cube_template.copy()
    ocean_cube.data = np.array([200, 200, 200, 10])
    water_cubes = CubeList([river_cube, lake_cube, ocean_cube])

    plugin = DistanceToNearestFeature()
    output_cube = plugin.process(water_cubes, "distance_to_water")

    assert output_cube.name() == "distance_to_water"
    assert output_cube.units == "m"
    assert_array_equal(output_cube.data, [100, 200, 200, 10])


# Test test cases and errors match
@pytest.mark.parametrize(
    "test_case, error_msg_part",
    [
        (
            "empty cubelist",
            "The input CubeList distance_to_features cannot be empty.",
        ),
        ("unmatched_coords", "Input cube coordinates"),
    ],
)
def test_errors_raised(test_case, error_msg_part, distance_cube_template):
    """Test the DistanceToNearestFeature class raises errors correctly."""

    if test_case == "empty cubelist":
        water_cubes = CubeList([])
    elif test_case == "unmatched_coords":
        river_cube = distance_cube_template.copy()
        river_cube.data = np.array([100, 200, 300, 400])
        ocean_cube = distance_cube_template.copy()
        lake_cube = distance_cube_template.copy()
        lake_cube.data = np.array([400, 300, 200, 100])
        ocean_cube.data = np.array([200, 200, 200, 10])
        # Change one of the site indices to create a mismatch in the site coordinates
        ocean_cube.coord("wmo_id").points[0] = "20202020"
        water_cubes = CubeList([river_cube, lake_cube, ocean_cube])

    plugin = DistanceToNearestFeature()
    with pytest.raises(ValueError, match=error_msg_part):
        plugin.process(water_cubes, "distance_to_water")
