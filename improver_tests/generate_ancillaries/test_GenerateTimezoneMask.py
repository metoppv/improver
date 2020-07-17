# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown Copyright 2017-2020 Met Office.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
"""Unit tests for the GenerateTimezoneMask plugin."""

import pytest
from datetime import datetime
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal

import iris
from iris.cube import Cube, CubeList

from improver.generate_ancillaries.generate_timezone_mask import GenerateTimezoneMask
from ..set_up_test_cubes import set_up_variable_cube


@pytest.fixture
def global_grid() -> Cube:
    """Global grid template"""

    attributes = {
        "title": "MOGREPS-G Model Forecast on Global 20 km Standard Grid",
        "source": "Met Office Unified Model",
        "institution": "Met Office",
    }

    data = np.zeros((19, 37), dtype=np.float32)
    cube = set_up_variable_cube(
        data,
        name="template",
        grid_spacing=10,
        domain_corner=(-90, -180),
        attributes=attributes,
    )
    return cube


@pytest.fixture
def uk_grid() -> Cube:
    """UK grid template"""

    attributes = {
        "title": "MOGREPS-UK Model Forecast on UK 2 km Standard Grid",
        "source": "Met Office Unified Model",
        "institution": "Met Office",
    }

    data = np.zeros((21, 22), dtype=np.float32)
    cube = set_up_variable_cube(
        data,
        name="template",
        spatial_grid="equalarea",
        grid_spacing=96900.0,
        domain_corner=(-1036000.0, -1158000.0),
        attributes=attributes,
    )
    return cube


@pytest.fixture
def timezone_mask() -> CubeList:
    """A timezone mask cubelist"""

    data = np.zeros((19, 37), dtype=np.float32)
    cube = set_up_variable_cube(
        data, name="template", grid_spacing=10, domain_corner=(-90, -180)
    )

    cubelist = CubeList()
    for offset in range(0, 4):
        mask = cube.copy()
        utc_offset_coord = iris.coords.AuxCoord([offset], long_name="UTC_offset")
        mask.add_aux_coord(utc_offset_coord)
        mask = iris.util.new_axis(mask, "UTC_offset")
        cubelist.append(mask)
    return cubelist


def test__set_time(uk_grid):
    """Test time is set correctly from either the cube or user."""

    # Set by the cube time coordinate
    expected = datetime(2017, 11, 10, 4)
    plugin = GenerateTimezoneMask()
    plugin(uk_grid)
    assert plugin.time == expected

    # Set by the user provided argument
    expected = datetime(2020, 7, 16, 15)
    plugin = GenerateTimezoneMask(time="20200716T1500")
    plugin(uk_grid)
    assert plugin.time == expected


@pytest.mark.parametrize("grid_fixture", ["global_grid", "uk_grid"])
def test__get_coordinate_pairs(request, grid_fixture):
    """Test that a selection of elements are as expected in what is returned
    by the _get_coordinate_pairs function. Tests are for both a native lat-long
    grid and for an equal areas grid that must be transformed."""

    sample_points = [0, 10, -1]
    expected_data = {
        "global_grid": [[-90.0, -180.0], [-90.0, -80.0], [90.0, 180.0]],
        "uk_grid": [[44.517, -17.117], [45.548, -4.913], [62.026, 14.410]],
    }

    grid = request.getfixturevalue(grid_fixture)
    result = GenerateTimezoneMask()._get_coordinate_pairs(grid)

    assert isinstance(result, np.ndarray)
    assert result.shape == (2, np.product(grid.shape))
    for i, ii in enumerate(sample_points):
        assert_array_almost_equal(
            result[:, ii], expected_data[grid_fixture][i], decimal=3
        )


def test__get_coordinate_pairs_exception(global_grid):
    """Test that an exception is raised if longitudes are found outside the
    range -180 to 180."""
    global_grid.coord("longitude").points = global_grid.coord("longitude").points + 360

    with pytest.raises(ValueError, match=r"TimezoneFinder requires .*"):
        GenerateTimezoneMask()._get_coordinate_pairs(global_grid)


def test__calculate_tz_offsets():
    """

    These test also cover the functionality of _calculate_offset.
    """
    pytest.importorskip("timezonefinder")

    # New York, London, and Melbourne
    coordinate_pairs = np.array([[41, 51.5, -37.9], [-74, 0, 145]])

    # Test ignoring daylights savings, so the result should be consistent
    # regardless of the date.
    expected = [-5 * 3600, 0, 10 * 3600]

    # Northern hemisphere winter
    time = datetime(2020, 1, 1, 12)
    plugin = GenerateTimezoneMask(time=time)
    result = plugin._calculate_tz_offsets(coordinate_pairs)
    assert_array_equal(result, expected)

    # Check return type information as well
    assert result.ndim == 1
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.int32

    # Southern hemisphere winter
    time = datetime(2020, 7, 1, 12)
    plugin = GenerateTimezoneMask(time=time)
    result = plugin._calculate_tz_offsets(coordinate_pairs)
    assert_array_equal(result, expected)

    # Test including daylights savings, so the result should change as the
    # date is changed.

    # Northern hemisphere winter
    expected = [-5 * 3600, 0, 11 * 3600]
    time = datetime(2020, 1, 1, 12)
    plugin = GenerateTimezoneMask(ignore_dst=False, time=time)
    result = plugin._calculate_tz_offsets(coordinate_pairs)
    assert_array_equal(result, expected)

    # Southern hemisphere winter
    expected = [-4 * 3600, 1 * 3600, 10 * 3600]
    time = datetime(2020, 7, 1, 12)
    plugin = GenerateTimezoneMask(ignore_dst=False, time=time)
    result = plugin._calculate_tz_offsets(coordinate_pairs)
    assert_array_equal(result, expected)


def test__group_timezones(timezone_mask):
    """
    Group the 4 UTC offset masks into groups.
    """
    # Split the 4 offsets first in equal sized groups, then test unequally
    # sized groups
    groupings = [{0: [0, 1], 1: [2, 3]}, {0: [0, 2], 1: [3]}]

    for groups in groupings:
        plugin = GenerateTimezoneMask(groupings=groups)
        result = plugin._group_timezones(timezone_mask)

        assert len(result) == len(groups)
        for group, cube in zip(groups.values(), result):
            assert cube.coord("UTC_offset").points[0] == group[-1]
            # A single UTC_offset point has no bounds, hence this try-except
            try:
                assert_array_equal(cube.coord("UTC_offset").bounds[0], group)
            except TypeError:
                pass


@pytest.mark.parametrize("grid_fixture", ["global_grid", "uk_grid"])
def test_process(request, grid_fixture):
    """Test that the process method returns cubes that take the expected form.
    Note that the time set by the user is effectively local time at every
    grid point, but the cube stores the time information in UTC.

    For UK users this means that despite the time test below setting a time of
    1500, as it falls within British summer time (BST), the cube time is shown
    as 1400.

    EEEEEEEKKKKKKKK timezone hell. I need the test to work regardless of the
    local timezone. Come back to this with a clearer mind.

    """

    expected = {
        "global_grid": {"shape": (27, 19, 37), "min": -12, "max": 14},
        "uk_grid": {"shape": (4, 21, 22), "min": -2, "max": 1},
    }
    expected_times = [1510286400, 1594908000]
    times = [None, "20200716T1500"]

    grid = request.getfixturevalue(grid_fixture)

    for time, expected_time in zip(times, expected_times):
        result = GenerateTimezoneMask(time=time)(grid)

        print(result)
        assert result.name() == "timezone_mask"
        assert result.units == 1
        assert result.coord('time').points[0] == expected_time
        assert result.shape == expected[grid_fixture]["shape"]
        assert result.coord("UTC_offset").points.min() == expected[grid_fixture]["min"]
        assert result.coord("UTC_offset").points.max() == expected[grid_fixture]["max"]
