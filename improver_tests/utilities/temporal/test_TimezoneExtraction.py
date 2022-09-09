# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown copyright. The Met Office.
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
"""Unit tests for TimezoneExtraction plugin."""

from datetime import datetime, timedelta

import iris
import numpy as np
import pytest
from cf_units import Unit
from iris.coords import CellMethod
from iris.cube import Cube

from improver.metadata.check_datatypes import check_mandatory_standards
from improver.metadata.constants.time_types import TIME_COORDS
from improver.synthetic_data.set_up_test_cubes import (
    add_coordinate,
    set_up_variable_cube,
)
from improver.utilities.temporal import TimezoneExtraction


def make_input_cube(data_shape, time_bounds=True):
    """Makes a 3D cube (time, y, x) of the described shape, filled with zeroes for use
    in testing."""
    cube = set_up_variable_cube(
        np.zeros(data_shape).astype(np.float32),
        standard_grid_metadata="gl_ens",
        attributes={
            "institution": "unknown",
            "source": "IMPROVER",
            "title": "Unit test",
        },
    )
    cube = add_coordinate(
        cube,
        [datetime(2017, 11, 10, 4, 0) + timedelta(hours=h) for h in range(2)],
        "time",
        coord_units=TIME_COORDS["time"].units,
        dtype=TIME_COORDS["time"].dtype,
        is_datetime=True,
    )
    if time_bounds:
        cube.coord("time").bounds = np.array(
            [
                [
                    np.around(
                        Unit(TIME_COORDS["time"].units).date2num(
                            datetime(2017, 11, 10, 4, 0) + timedelta(hours=h + b)
                        )
                    )
                    for b in [-1, 0]
                ]
                for h in range(2)
            ],
            dtype=TIME_COORDS["time"].dtype,
        )
    return cube


def make_percentile_cube(data_shape_2d, time_bounds=True):
    """Adds a percentile coordinate to make_input_cube"""
    cube = make_input_cube(data_shape_2d, time_bounds=time_bounds)
    cube = add_coordinate(cube, (25, 50, 75), "percentile", "%")
    return cube


def make_timezone_cube():
    """Makes a timezone cube to use in tests. data=0 where points fall in that
    time-zone. data=1 where they don't."""
    cube = set_up_variable_cube(
        np.zeros((3, 4)).astype(np.float32),
        name="timezone_mask",
        units="1",
        standard_grid_metadata="gl_ens",
        attributes={
            "institution": "unknown",
            "source": "IMPROVER",
            "title": "Unit test",
        },
    )
    cube = add_coordinate(
        cube,
        [0, 3600],
        "UTC_offset",
        coord_units="seconds",
        dtype=TIME_COORDS["forecast_period"].dtype,
    )
    true_row = [1, 1, 1, 1]
    false_row = [0, 0, 0, 0]
    cube.data = np.array(
        [[true_row, true_row, false_row], [false_row, false_row, true_row]],
        dtype=np.int8,
    )
    return cube


def assert_metadata_ok(output_cube):
    """Checks that the meta-data of output_cube are as expected"""
    assert isinstance(output_cube, Cube)
    assert output_cube.dtype == np.float32
    assert list(output_cube.coord_dims("time")) == [
        n for n, in [output_cube.coord_dims(c) for c in ["latitude", "longitude"]]
    ]
    assert output_cube.coord("time").dtype == np.int64
    check_mandatory_standards(output_cube)


@pytest.mark.parametrize("with_cell_method", (True, False))
def test_create_output_cube(with_cell_method):
    """Tests that the create_output_cube method builds a cube with appropriate
    meta-data. The Time coord is tested in test_process as it depends on multiple
    methods."""
    data_shape = [3, 4]
    cube = make_input_cube(data_shape)
    if with_cell_method:
        cell_method = CellMethod("minimum", coords="time")
        cube.add_cell_method(cell_method)
    local_time = datetime(2017, 11, 9, 12, 0)
    plugin = TimezoneExtraction()
    plugin.output_data = np.zeros(data_shape, dtype=np.float32)
    plugin.time_points = np.full(
        data_shape,
        fill_value=Unit(TIME_COORDS["time"].units).date2num(
            datetime(2017, 11, 10, 4, 0)
        ),
        dtype=np.int64,
    )
    plugin.time_bounds = None
    result = plugin.create_output_cube(cube, local_time)
    assert_metadata_ok(result)
    assert result.name() == cube.name()
    assert result.units == cube.units
    result_local_time = result.coord("time_in_local_timezone")
    assert [cell.point for cell in result_local_time.cells()] == [local_time]
    expected_shape = data_shape
    assert result.shape == tuple(expected_shape)
    assert result.attributes == cube.attributes
    if with_cell_method:
        assert result.cell_methods == tuple([cell_method])


@pytest.mark.parametrize("include_time_coord", (True, False))
def test_check_input_cube_dims(include_time_coord):
    """Checks that check_input_cube_dims can differentiate between an input cube
    with time, y, x coords and one where time is missing. Also checks that timezone_cube
    has been reordered correctly."""
    cube = make_input_cube([3, 4])
    timezone_cube = make_timezone_cube()
    plugin = TimezoneExtraction()
    if include_time_coord:
        plugin.check_input_cube_dims(cube, timezone_cube)
        assert plugin.timezone_cube.coord_dims("UTC_offset") == tuple(
            [plugin.timezone_cube.ndim - 1]
        )
    else:
        cube.remove_coord("time")
        with pytest.raises(
            ValueError, match=r"Expected coords on input_cube: time, y, x "
        ):
            plugin.check_input_cube_dims(cube, timezone_cube)


def test_check_aux_time_coord():
    """Checks that check_input_cube_dims can work with a auxiliary time
    coordinate. This occurs when partial periods are allowed. In these
    cases the lower time bound may be the same between different time
    points, preventing the creation of a dimension coordinate that
    required monotonically increasing bounds. In this case a anonymous
    coordinate is created by iris that can be named and used for
    reordering.

    A temporary name should be assigned to the time dimension for
    reordering. This should not be present on the output."""
    cube = make_input_cube([3, 4])
    timezone_cube = make_timezone_cube()
    iris.util.demote_dim_coord_to_aux_coord(cube, "time")

    plugin = TimezoneExtraction()
    plugin.check_input_cube_dims(cube, timezone_cube)
    assert plugin.timezone_cube.coord_dims("UTC_offset") == tuple(
        [plugin.timezone_cube.ndim - 1]
    )
    assert "time_points" not in [crd.name() for crd in cube.coords()]


@pytest.mark.parametrize(
    "local_time, expect_success",
    ((datetime(2017, 11, 10, 5, 0), True), (datetime(2017, 11, 10, 6, 0), False)),
)
def test_check_input_cube_time(local_time, expect_success):
    """Checks that check_input_cube_time can differentiate between arguments that match
    expected times and arguments that don't."""
    cube = make_input_cube([3, 4])
    timezone_cube = make_timezone_cube()
    plugin = TimezoneExtraction()
    plugin.check_input_cube_dims(cube, timezone_cube)
    if expect_success:
        plugin.check_input_cube_time(cube, local_time)
    else:
        with pytest.raises(
            ValueError, match=r"Time coord on input cube does not match required times."
        ):
            plugin.check_input_cube_time(cube, local_time)


def test_check_timezones_are_unique_pass():
    """Checks that check_timezones_are_unique allows our test cube"""
    timezone_cube = make_timezone_cube()
    TimezoneExtraction().check_timezones_are_unique(timezone_cube)


@pytest.mark.parametrize("offset", (1, -1))
def test_check_timezones_are_unique_fail(offset):
    """Checks that check_timezones_are_unique fails if we break our test cube"""
    timezone_cube = make_timezone_cube()
    timezone_cube.data[0, 0, 0] += offset
    with pytest.raises(
        ValueError,
        match=r"Timezone cube does not map exactly one time zone to each spatial point",
    ):
        TimezoneExtraction().check_timezones_are_unique(timezone_cube)


@pytest.mark.parametrize("with_percentiles", (True, False))
@pytest.mark.parametrize("input_as_cube", (True, False))
@pytest.mark.parametrize("input_has_time_bounds", (True, False))
def test_process(with_percentiles, input_as_cube, input_has_time_bounds):
    """Checks that the plugin process method returns a cube with expected data and
    time coord for our test data"""
    data_shape = [3, 4]
    data = np.array(
        [np.zeros(data_shape, dtype=np.float32), np.ones(data_shape, dtype=np.float32)]
    )
    expected_data = [[0, 0, 0, 0], [0, 0, 0, 0], [1, 1, 1, 1]]
    if with_percentiles:
        cube = make_percentile_cube(data_shape, time_bounds=input_has_time_bounds)
        data = np.array([data, data, data])
        expected_data = np.array([expected_data, expected_data, expected_data])
    else:
        cube = make_input_cube(data_shape, time_bounds=input_has_time_bounds)
    cube.data = data
    local_time = datetime(2017, 11, 10, 5, 0)
    timezone_cube = make_timezone_cube()
    row1 = [cube.coord("time").units.date2num(datetime(2017, 11, 10, 4, 0))] * 4
    row3 = [cube.coord("time").units.date2num(datetime(2017, 11, 10, 5, 0))] * 4
    expected_times = [row1, row1, row3]
    expected_bounds = np.array(expected_times).reshape((3, 4, 1)) + [[[-3600, 0]]]

    if not input_as_cube:
        # Split cube into a list of cubes
        cube = [c for c in cube.slices_over("time")]
    result = TimezoneExtraction()(cube, timezone_cube, local_time)

    assert_metadata_ok(result)
    assert np.array_equal(result.data, expected_data)
    assert np.array_equal(result.coord("time").points, expected_times)
    if input_has_time_bounds:
        assert np.array_equal(result.coord("time").bounds, expected_bounds)
    else:
        assert result.coord("time").bounds is None


def test_partial_period():
    """Checks that the plugin process method returns a cube with expected data and
    time coord for a case of partial time periods. In this case the time bounds
    are not monotonic, rather the first two times share lower bounds. This requires
    that the time coordinate be an Auxiliary coordinate. The variable time-bounds
    should be reflected in the output time coordinate."""
    data_shape = [3, 4]
    data = np.array(
        [np.zeros(data_shape, dtype=np.float32), np.ones(data_shape, dtype=np.float32)]
    )
    expected_data = [[0, 0, 0, 0], [0, 0, 0, 0], [1, 1, 1, 1]]

    cube = make_input_cube(data_shape, time_bounds=True)
    cube.data = data

    # Make aux time coord with overlapping lower bounds.
    tcrd = iris.coords.AuxCoord.from_coord(cube.coord("time"))
    tbounds = tcrd.bounds.copy()
    tbounds[0, 0] = tbounds[1, 0]
    tcrd.bounds = tbounds
    cube.remove_coord("time")
    cube.add_aux_coord(tcrd, 0)

    local_time = datetime(2017, 11, 10, 5, 0)
    timezone_cube = make_timezone_cube()
    row1 = [cube.coord("time").units.date2num(datetime(2017, 11, 10, 4, 0))] * 4
    row3 = [cube.coord("time").units.date2num(datetime(2017, 11, 10, 5, 0))] * 4
    expected_times = [row1, row1, row3]
    expected_bounds = np.array(expected_times).reshape((3, 4, 1)) + [[[-3600, 0]]]
    # The overlapping bounds we expect to see in the output.
    expected_bounds[:2, :, 0] += 3600

    result = TimezoneExtraction()(cube, timezone_cube, local_time)

    assert_metadata_ok(result)
    assert np.array_equal(result.data, expected_data)
    assert np.array_equal(result.coord("time").points, expected_times)
    assert np.array_equal(result.coord("time").bounds, expected_bounds)
    # Demonstrate there are different length bounds indicating partial periods.
    assert len(np.unique(np.diff(result.coord("time").bounds))) > 1


def test_bad_dtype():
    """Checks that the plugin raises a useful error if the output are float64"""
    cube = make_input_cube([3, 4])
    local_time = datetime(2017, 11, 10, 5, 0)
    timezone_cube = make_timezone_cube()
    timezone_cube.data = timezone_cube.data.astype(np.int32)
    with pytest.raises(
        TypeError,
        match=r"Operation multiply on types \{dtype\(\'.*32\'\), dtype\(\'.*32\'\)\} results in",
    ):
        TimezoneExtraction()(cube, timezone_cube, local_time)


def test_bad_spatial_coords():
    """Checks that the plugin raises a useful error if the longitude coord is shifted by
    180 degrees"""
    cube = make_input_cube([3, 4])
    local_time = datetime(2017, 11, 10, 5, 0)
    timezone_cube = make_timezone_cube()
    timezone_cube.data = timezone_cube.data.astype(np.int32)
    longitude_coord = timezone_cube.coord("longitude")
    timezone_cube.replace_coord(longitude_coord.copy(longitude_coord.points + 180))
    with pytest.raises(
        ValueError,
        match=r"Spatial coordinates on input_cube and timezone_cube do not match.",
    ):
        TimezoneExtraction()(cube, timezone_cube, local_time)
