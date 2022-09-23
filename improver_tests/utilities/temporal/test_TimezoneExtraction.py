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

from datetime import datetime

import iris
import numpy as np
import pytest
from cf_units import Unit
from iris.coords import CellMethod
from iris.cube import Cube
from iris.exceptions import CoordinateNotFoundError

from improver.metadata.check_datatypes import check_mandatory_standards
from improver.metadata.constants.time_types import TIME_COORDS
from improver.utilities.temporal import TimezoneExtraction


def assert_metadata_ok(output_cube):
    """Checks that the meta-data of output_cube are as expected"""
    assert isinstance(output_cube, Cube)
    assert output_cube.dtype == np.float32
    assert list(output_cube.coord_dims("time")) == [
        n for n, in [output_cube.coord_dims(c) for c in ["latitude", "longitude"]]
    ]
    assert output_cube.coord("time").dtype == np.int64
    check_mandatory_standards(output_cube)


@pytest.mark.parametrize("time_bounds", (True, False))
@pytest.mark.parametrize("percentiles", (True, False))
@pytest.mark.parametrize("with_cell_method", (True, False))
def test_create_output_cube(input_cube, data_shape, with_cell_method):
    """Tests that the create_output_cube method builds a cube with appropriate
    meta-data. The Time coord is tested in test_process as it depends on multiple
    methods."""
    if with_cell_method:
        cell_method = CellMethod("minimum", coords="time")
        input_cube.add_cell_method(cell_method)
    local_time = datetime(2017, 11, 9, 12, 0)
    plugin = TimezoneExtraction()
    plugin.output_data = np.zeros(data_shape, dtype=np.float32)
    plugin.time_points = np.full(
        data_shape[-2:],
        fill_value=Unit(TIME_COORDS["time"].units).date2num(
            datetime(2017, 11, 10, 4, 0)
        ),
        dtype=np.int64,
    )
    plugin.time_bounds = None
    result = plugin.create_output_cube(input_cube, local_time)
    assert_metadata_ok(result)
    assert result.name() == input_cube.name()
    assert result.units == input_cube.units
    result_local_time = result.coord("time_in_local_timezone")
    assert [cell.point for cell in result_local_time.cells()] == [local_time]
    expected_shape = data_shape
    assert result.shape == tuple(expected_shape)
    assert result.attributes == input_cube.attributes
    if with_cell_method:
        assert result.cell_methods == tuple([cell_method])


@pytest.mark.parametrize("time_bounds", (True, False))
@pytest.mark.parametrize("percentiles", (True, False))
@pytest.mark.parametrize("include_time_coord", (True, False))
def test_check_input_cube_dims(input_cube, timezone_cube, include_time_coord):
    """Checks that check_input_cube_dims can differentiate between an input cube
    with time + other coords and one where time is missing."""
    plugin = TimezoneExtraction()
    plugin.timezone_cube = timezone_cube
    if include_time_coord:
        plugin.check_input_cube_dims(input_cube)
        assert input_cube.coord_dims("time") == tuple([input_cube.ndim - 1])
    else:
        input_cube.remove_coord("time")
        with pytest.raises(
            CoordinateNotFoundError, match=r"Expected coords on input_cube:"
        ):
            plugin.check_input_cube_dims(input_cube)


@pytest.mark.parametrize("time_bounds", (True, False))
@pytest.mark.parametrize("percentiles", (True, False))
def test_check_aux_time_coord(input_cube, timezone_cube):
    """Checks that check_input_cube_dims can work with an auxiliary time
    coordinate. This occurs when partial periods are allowed. In these
    cases the lower time bound may be the same between different time
    points, preventing the creation of a dimension coordinate that
    requires monotonically increasing bounds. In this case an anonymous
    coordinate is created by iris that can be named and used for
    reordering.

    A temporary name should be assigned to the time dimension for
    reordering. This should not be present on the output, but the time
    coordinate should have been moved to the last dimension."""

    iris.util.demote_dim_coord_to_aux_coord(input_cube, "time")
    plugin = TimezoneExtraction()
    plugin.timezone_cube = timezone_cube
    plugin.check_input_cube_dims(input_cube)
    assert input_cube.coord_dims("time") == tuple([input_cube.ndim - 1])
    assert len(input_cube.dim_coords) == len(input_cube.shape) - 1
    assert "time_points" not in [crd.name() for crd in input_cube.coords()]


@pytest.mark.parametrize("time_bounds", (True, False))
@pytest.mark.parametrize("percentiles", (True, False))
@pytest.mark.parametrize(
    "local_time, expect_success",
    ((datetime(2017, 11, 10, 5, 0), True), (datetime(2017, 11, 10, 6, 0), False)),
)
def test_check_input_cube_time(input_cube, timezone_cube, local_time, expect_success):
    """Checks that check_input_cube_time can differentiate between arguments that match
    expected times and arguments that don't. Also checks that timezone_cube
    has been reordered correctly."""

    plugin = TimezoneExtraction()
    plugin.timezone_cube = timezone_cube
    plugin.check_input_cube_dims(input_cube)
    if expect_success:
        plugin.check_input_cube_time(input_cube, local_time)
        assert plugin.timezone_cube.coord_dims("UTC_offset") == tuple(
            [plugin.timezone_cube.ndim - 1]
        )
    else:
        with pytest.raises(
            ValueError, match=r"Time coord on input cube does not match required times."
        ):
            plugin.check_input_cube_time(input_cube, local_time)


def test_check_timezones_are_unique_pass(timezone_cube):
    """Checks that check_timezones_are_unique allows our test cube"""
    plugin = TimezoneExtraction()
    plugin.timezone_cube = timezone_cube
    plugin.check_timezones_are_unique()


@pytest.mark.parametrize("offset", (1, -1))
def test_check_timezones_are_unique_fail(timezone_cube, offset):
    """Checks that check_timezones_are_unique fails if we break our test cube"""
    timezone_cube.data[0, 0, 0] += offset
    plugin = TimezoneExtraction()
    plugin.timezone_cube = timezone_cube
    with pytest.raises(
        ValueError,
        match=r"Timezone cube does not map exactly one time zone to each spatial point",
    ):
        plugin.check_timezones_are_unique()


@pytest.mark.parametrize("time_bounds", (True, False))
@pytest.mark.parametrize("percentiles", (True, False))
@pytest.mark.parametrize("input_as_cube", (True, False))
def test_process(input_cube, input_as_cube, timezone_cube):
    """Checks that the plugin process method returns a cube with expected data and
    time coord for our test data"""

    data_shape = [3, 4]
    time_bounds = input_cube.coord("time").has_bounds()
    percentiles = "percentile" in [crd.name() for crd in input_cube.coords()]

    data = np.array(
        [np.zeros(data_shape, dtype=np.float32), np.ones(data_shape, dtype=np.float32)]
    )

    expected_data = [[0, 0, 0, 0], [0, 0, 0, 0], [1, 1, 1, 1]]
    if percentiles:
        data = np.array([data, data, data])
        expected_data = np.array([expected_data, expected_data, expected_data])

    input_cube.data = data
    local_time = datetime(2017, 11, 10, 5, 0)
    row1 = [input_cube.coord("time").units.date2num(datetime(2017, 11, 10, 4, 0))] * 4
    row3 = [input_cube.coord("time").units.date2num(datetime(2017, 11, 10, 5, 0))] * 4
    expected_times = [row1, row1, row3]
    expected_bounds = np.array(expected_times).reshape((3, 4, 1)) + [[[-3600, 0]]]

    if not input_as_cube:
        # Split cube into a list of cubes
        input_cube = [c for c in input_cube.slices_over("time")]
    result = TimezoneExtraction()(input_cube, timezone_cube, local_time)

    assert_metadata_ok(result)
    assert np.array_equal(result.data, expected_data)
    assert np.array_equal(result.coord("time").points, expected_times)
    if time_bounds:
        assert np.array_equal(result.coord("time").bounds, expected_bounds)
    else:
        assert result.coord("time").bounds is None


@pytest.mark.parametrize("time_bounds", [True])
@pytest.mark.parametrize("percentiles", [False])
def test_partial_period_update(input_cube, data_shape, timezone_cube):
    """Checks that the plugin process method returns a cube with expected data and
    time coord for a case of partial time periods inidicative of a same day update.
    In this case the time bounds are not monotonic, rather the first two times
    share lower bounds. This requires that the time coordinate be an Auxiliary
    coordinate. The variable time-bounds should be reflected in the output time
    coordinate."""
    data = np.array(
        [np.zeros(data_shape, dtype=np.float32), np.ones(data_shape, dtype=np.float32)]
    )
    expected_data = [[0, 0, 0, 0], [0, 0, 0, 0], [1, 1, 1, 1]]
    input_cube.data = data

    # Make aux time coord with overlapping lower bounds.
    tcrd = iris.coords.AuxCoord.from_coord(input_cube.coord("time"))
    tbounds = tcrd.bounds.copy()
    tbounds[0, 0] = tbounds[1, 0]
    tcrd.bounds = tbounds
    input_cube.remove_coord("time")
    input_cube.add_aux_coord(tcrd, 0)

    local_time = datetime(2017, 11, 10, 5, 0)
    row1 = [input_cube.coord("time").units.date2num(datetime(2017, 11, 10, 4, 0))] * 4
    row3 = [input_cube.coord("time").units.date2num(datetime(2017, 11, 10, 5, 0))] * 4
    expected_times = [row1, row1, row3]
    expected_bounds = np.array(expected_times).reshape((3, 4, 1)) + [[[-3600, 0]]]
    # The overlapping bounds we expect to see in the output.
    expected_bounds[:2, :, 0] += 3600

    result = TimezoneExtraction()(input_cube, timezone_cube, local_time)

    assert_metadata_ok(result)
    assert np.array_equal(result.data, expected_data)
    assert np.array_equal(result.coord("time").points, expected_times)
    assert np.array_equal(result.coord("time").bounds, expected_bounds)
    # Demonstrate there are different length bounds indicating partial periods.
    assert len(np.unique(np.diff(result.coord("time").bounds))) > 1


@pytest.mark.parametrize("time_bounds", [True])
@pytest.mark.parametrize("percentiles", [False])
def test_incomplete_period(input_cube, timezone_cube):
    """Checks that the plugin process method returns None when an incomplete
    period is provided that indicates the inputs are not a same-day update. This
    is an incomplete period that is missing data at the end of a period rather
    than the beginning, i.e. the upper time bound doesn't match the time target
    time point."""
    data_shape = [3, 4]
    data = np.array(
        [np.zeros(data_shape, dtype=np.float32), np.ones(data_shape, dtype=np.float32)]
    )
    input_cube.data = data

    # Make aux time coord with curtailed final point.
    tcrd = input_cube.coord("time").copy()
    tpoints = tcrd.points.copy()
    tbounds = tcrd.bounds.copy()
    tpoints[-1] = tpoints[-1] - 1800
    tbounds[-1, -1] = tbounds[-1, -1] - 1800

    tcrd.points = tpoints
    tcrd.bounds = tbounds
    input_cube.remove_coord("time")
    input_cube.add_dim_coord(tcrd, 0)

    local_time = datetime(2017, 11, 10, 5, 0)
    result = TimezoneExtraction()(input_cube, timezone_cube, local_time)

    assert result is None


@pytest.mark.parametrize("time_bounds", [True])
@pytest.mark.parametrize("percentiles", [False])
def test_bad_dtype(input_cube, timezone_cube):
    """Checks that the plugin raises a useful error if the output are float64"""
    local_time = datetime(2017, 11, 10, 5, 0)
    timezone_cube.data = timezone_cube.data.astype(np.int32)
    with pytest.raises(
        TypeError,
        match=r"Operation multiply on types \{dtype\(\'.*32\'\), dtype\(\'.*32\'\)\} results in",
    ):
        TimezoneExtraction()(input_cube, timezone_cube, local_time)


@pytest.mark.parametrize("time_bounds", [True])
@pytest.mark.parametrize("percentiles", [False])
def test_bad_spatial_coords(input_cube, timezone_cube):
    """Checks that the plugin raises a useful error if the longitude coord is shifted by
    180 degrees"""
    local_time = datetime(2017, 11, 10, 5, 0)
    timezone_cube.data = timezone_cube.data.astype(np.int32)
    longitude_coord = timezone_cube.coord("longitude")
    timezone_cube.replace_coord(longitude_coord.copy(longitude_coord.points + 180))
    with pytest.raises(
        ValueError,
        match=r"Spatial coordinates on input_cube and timezone_cube do not match.",
    ):
        TimezoneExtraction()(input_cube, timezone_cube, local_time)
