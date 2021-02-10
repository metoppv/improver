# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown Copyright 2017-2021 Met Office.
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

from datetime import datetime

import iris
import numpy as np
import pytest
import pytz
from iris.cube import Cube, CubeList
from numpy.testing import assert_array_almost_equal, assert_array_equal

from improver.generate_ancillaries.generate_timezone_mask import GenerateTimezoneMask
from improver.metadata.constants.time_types import TIME_COORDS
from improver.synthetic_data.set_up_test_cubes import set_up_variable_cube

pytest.importorskip("timezonefinder")
pytest.importorskip("numba")

GLOBAL_ATTRIBUTES = {
    "title": "MOGREPS-G Model Forecast on Global 20 km Standard Grid",
    "source": "Met Office Unified Model",
    "institution": "Met Office",
}

UK_ATTRIBUTES = {
    "title": "MOGREPS-UK Model Forecast on UK 2 km Standard Grid",
    "source": "Met Office Unified Model",
    "institution": "Met Office",
}


@pytest.fixture(name="global_grid")
def global_grid_fixture() -> Cube:
    """Global grid template"""

    data = np.zeros((19, 37), dtype=np.float32)
    cube = set_up_variable_cube(
        data,
        name="template",
        grid_spacing=10,
        domain_corner=(-90, -180),
        attributes=GLOBAL_ATTRIBUTES,
    )
    return cube


@pytest.fixture(name="global_grid_360")
def global_grid_fixture_360() -> Cube:
    """Global grid template with longitude running 0 to 360"""

    data = np.zeros((19, 37), dtype=np.float32)
    cube = set_up_variable_cube(
        data,
        name="template",
        grid_spacing=10,
        domain_corner=(-90, 0),
        attributes=GLOBAL_ATTRIBUTES,
    )
    return cube


@pytest.fixture(name="uk_grid")
def uk_grid_fixture() -> Cube:
    """UK grid template"""

    data = np.zeros((21, 22), dtype=np.float32)
    cube = set_up_variable_cube(
        data,
        name="template",
        spatial_grid="equalarea",
        grid_spacing=96900.0,
        domain_corner=(-1036000.0, -1158000.0),
        attributes=UK_ATTRIBUTES,
    )
    return cube


@pytest.fixture(name="timezone_mask")
def timezone_mask_fixture() -> CubeList:
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
    expected = datetime(2017, 11, 10, 4, tzinfo=pytz.utc)
    plugin = GenerateTimezoneMask()
    plugin._set_time(uk_grid)
    assert plugin.time == expected

    # Set by the user provided argument
    expected = datetime(2020, 7, 16, 15, tzinfo=pytz.utc)
    plugin = GenerateTimezoneMask(time="20200716T1500Z")
    plugin._set_time(uk_grid)
    assert plugin.time == expected

    # Check an exception is raised if no time information is provided
    uk_grid.remove_coord("time")
    plugin = GenerateTimezoneMask()
    msg = (
        "The input cube does not contain a 'time' coordinate. "
        "As such a time must be provided by the user."
    )
    with pytest.raises(ValueError, match=msg):
        plugin._set_time(uk_grid)


@pytest.mark.parametrize("grid_fixture", ["global_grid", "global_grid_360", "uk_grid"])
def test__get_coordinate_pairs(request, grid_fixture):
    """Test that a selection of the points returned by _get_coordinate_pairs
    have the expected values. Tests are for both native lat-long grids and for
    an equal areas grid that must be transformed."""

    sample_points = [0, 10, -1]
    expected_data = {
        "global_grid": [[-90.0, -180.0], [-90.0, -80.0], [90.0, 180.0]],
        "global_grid_360": [[-90.0, -180.0], [-90.0, -80.0], [90.0, 180.0]],
        "uk_grid": [[44.517, -17.117], [45.548, -4.913], [62.026, 14.410]],
    }

    grid = request.getfixturevalue(grid_fixture)
    result = GenerateTimezoneMask()._get_coordinate_pairs(grid)

    assert isinstance(result, np.ndarray)
    assert result.shape == (np.product(grid.shape), 2)
    for i, ii in enumerate(sample_points):
        assert_array_almost_equal(
            result[ii, :], expected_data[grid_fixture][i], decimal=3
        )


def test__get_coordinate_pairs_exception(global_grid):
    """Test that an exception is raised if longitudes are found outside the
    range -180 to 180."""

    global_grid.coord("longitude").points = global_grid.coord("longitude").points + 360

    with pytest.raises(ValueError, match=r"TimezoneFinder requires .*"):
        GenerateTimezoneMask()._get_coordinate_pairs(global_grid)


def test__calculate_tz_offsets():
    """
    Test that the expected offsets are returned for several timezones, with and
    without daylight savings.

    These test also cover the functionality of _calculate_offset.
    """

    # New York, London, and Melbourne
    coordinate_pairs = np.array([[41, -74], [51.5, 0], [-37.9, 145]])

    # Test ignoring daylight savings, so the result should be consistent
    # regardless of the date.
    expected = [-5 * 3600, 0, 10 * 3600]

    # Northern hemisphere winter
    time = datetime(2020, 1, 1, 12, tzinfo=pytz.utc)
    plugin = GenerateTimezoneMask(time=time)
    result = plugin._calculate_tz_offsets(coordinate_pairs)
    assert_array_equal(result, expected)

    # Check return type information as well
    assert result.ndim == 1
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.int32

    # Southern hemisphere winter
    time = datetime(2020, 7, 1, 12, tzinfo=pytz.utc)
    plugin = GenerateTimezoneMask(time=time)
    result = plugin._calculate_tz_offsets(coordinate_pairs)
    assert_array_equal(result, expected)

    # Test including daylight savings, so the result should change as the
    # date is changed.

    # Northern hemisphere winter
    expected = [-5 * 3600, 0, 11 * 3600]
    time = datetime(2020, 1, 1, 12, tzinfo=pytz.utc)
    plugin = GenerateTimezoneMask(include_dst=True, time=time)
    result = plugin._calculate_tz_offsets(coordinate_pairs)
    assert_array_equal(result, expected)

    # Southern hemisphere winter
    expected = [-4 * 3600, 1 * 3600, 10 * 3600]
    time = datetime(2020, 7, 1, 12, tzinfo=pytz.utc)
    plugin = GenerateTimezoneMask(include_dst=True, time=time)
    result = plugin._calculate_tz_offsets(coordinate_pairs)
    assert_array_equal(result, expected)


@pytest.mark.parametrize("grid_fixture", ["global_grid", "uk_grid"])
@pytest.mark.parametrize("include_dst", [False, True])
def test__create_template_cube(request, grid_fixture, include_dst):
    """Test the construction of a template cube slice, checking the shape
    data types, and attributes."""

    grid = request.getfixturevalue(grid_fixture)
    time = datetime(2020, 1, 1, 12, tzinfo=pytz.utc)

    expected = {
        "global_grid": {"shape": (19, 37), "attributes": GLOBAL_ATTRIBUTES},
        "uk_grid": {"shape": (21, 22), "attributes": UK_ATTRIBUTES},
    }

    # Set expected includes_daylight_savings attribute
    expected[grid_fixture]["attributes"]["includes_daylight_savings"] = str(include_dst)

    plugin = GenerateTimezoneMask(include_dst=include_dst, time=time)
    result = plugin._create_template_cube(grid)

    assert result.name() == "timezone_mask"
    assert result.units == 1
    assert result.coord("time").points[0] == time.timestamp()
    assert result.coord("time").dtype == np.int64
    assert result.shape == expected[grid_fixture]["shape"]
    assert result.dtype == np.int8
    assert result.attributes == expected[grid_fixture]["attributes"]


@pytest.mark.parametrize(
    "groups", ({0: [0, 1], 3: [2, 3]}, {0: [0, 2], 3: [3]}, {1: [0, 2], 4: [3, 4]})
)
def test__group_timezones(timezone_mask, groups):
    """Test the grouping of different UTC offsets into larger groups using a
    user provided specification. The input cube list contains cubes corresponding
    to 4 UTC offsets. Three tests are run, first grouping these into equal sized
    groups, then into unequally sized groups. Finally the timezones are grouped
    around a point beyond the end of the timezone range found in the timezone
    mask.

    This final test replicates what we want to achieve when grouping data
    to extract from data available at non-hourly intervals. For example we
    want to round UTC+14 data to UTC+15 as that is nearest available 3-hourly
    data interval, though there is no timezone at UTC+15."""

    plugin = GenerateTimezoneMask(groupings=groups)
    result = plugin._group_timezones(timezone_mask)

    assert len(result) == len(groups)
    for (offset, group), cube in zip(groups.items(), result):
        assert cube.coord("UTC_offset").points[0] == offset
        assert cube.coord("UTC_offset").bounds is not None
        if len(group) > 1:
            assert_array_equal(cube.coord("UTC_offset").bounds[0], group)
        else:
            assert cube.coord("UTC_offset").bounds[0][0] == group[0]
            assert cube.coord("UTC_offset").bounds[0][-1] == group[0]


def test__group_timezones_empty_group(timezone_mask):
    """Test the grouping of different UTC offsets into larger groups in a case
    for which a specified group contains no data."""

    groups = {0: [0, 1], 3: [2, 3], 6: [4, 10]}

    plugin = GenerateTimezoneMask(groupings=groups)
    result = plugin._group_timezones(timezone_mask)

    assert len(result) == 2
    for (offset, group), cube in zip(list(groups.items())[:-1], result):
        assert cube.coord("UTC_offset").points[0] == offset
        assert_array_equal(cube.coord("UTC_offset").bounds[0], group)


# Expected data for process tests

# ungrouped
GLOBAL_GRID = {"shape": (27, 19, 37), "min": -12 * 3600, "max": 14 * 3600}
# grouped
GLOBAL_GRID_GR = {"shape": (2, 19, 37), "min": -6 * 3600, "max": 6 * 3600}
UK_GRID_GR = {"shape": (2, 21, 22), "min": -6 * 3600, "max": 6 * 3600}

EXPECTED_TIME = {None: 1510286400, "20200716T1500Z": 1594911600}
EXPECTED = {
    "ungrouped": {
        "global": {
            None: {**GLOBAL_GRID, "data": np.array([1, 1, 1, 1, 1, 0, 1, 1, 1, 1])},
            "20200716T1500Z": {**GLOBAL_GRID, "data": np.ones([10]),},
            "indices": (12, 2),
        },
        "uk": {
            None: {
                "shape": (4, 21, 22),
                "min": -2 * 3600,
                "max": 1 * 3600,
                "data": np.array([1, 1, 0, 0, 0, 1]),
            },
            "20200716T1500Z": {
                "shape": (5, 21, 22),
                "min": -2 * 3600,
                "max": 2 * 3600,
                "data": np.array([1, 1, 1, 1, 0, 1]),
            },
            "indices": (2, 10),
        },
    },
    "grouped": {
        "global": {
            None: {**GLOBAL_GRID_GR, "data": np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1])},
            "20200716T1500Z": {
                **GLOBAL_GRID_GR,
                "data": np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1]),
            },
            "indices": (0, 2),
        },
        "uk": {
            None: {**UK_GRID_GR, "data": np.array([0, 0, 0, 0, 0, 1])},
            "20200716T1500Z": {**UK_GRID_GR, "data": np.array([0, 0, 1, 1, 0, 1])},
            "indices": (0, 9),
        },
    },
}


@pytest.mark.parametrize("grouping", ["ungrouped", "grouped"])
@pytest.mark.parametrize("time", [None, "20200716T1500Z"])
@pytest.mark.parametrize("grid_fixture", ["global_grid", "global_grid_360", "uk_grid"])
def test_process(request, grid_fixture, time, grouping):
    """Test that the process method returns cubes that take the expected form
    for different grids and different dates.

    The output data is primarily checked in the acceptance tests as a reasonably
    large number of data points are required to reliably check it. Here we check
    only a small sample."""

    domain = grid_fixture.split("_")[0]
    groupings = None
    if grouping == "grouped":
        groupings = {-6: [-12, 0], 6: [1, 14]}

    expected = EXPECTED[grouping][domain][time]
    expected_time = EXPECTED_TIME[time]
    index = EXPECTED[grouping][domain]["indices"]

    grid = request.getfixturevalue(grid_fixture)

    result = GenerateTimezoneMask(time=time, include_dst=True, groupings=groupings)(
        grid
    )

    assert result.coord("time").points[0] == expected_time
    assert result.shape == expected["shape"]
    assert result.coord("UTC_offset").points.min() == expected["min"]
    assert result.coord("UTC_offset").points.max() == expected["max"]
    assert result.coord("UTC_offset").points.dtype == TIME_COORDS["UTC_offset"].dtype
    if grouping == "grouped":
        assert (
            result.coord("UTC_offset").bounds.dtype == TIME_COORDS["UTC_offset"].dtype
        )

    assert_array_equal(result.data[index][::4], expected["data"])
