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
"""Unit tests for TimezoneExtraction plugin."""

from datetime import datetime, timedelta

import iris
import numpy as np
import pytest
from cf_units import Unit
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
        [datetime(2017, 11, 10, 4, 0) + timedelta(hours=h) for h in range(3)],
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
                for h in range(3)
            ],
            dtype=TIME_COORDS["time"].dtype,
        )
    return cube


def make_timezone_cube():
    """Makes a timezone cube to use in tests. data=0 where points fall in that
    time-zone. data=1 where they don't."""
    cube = set_up_variable_cube(
        np.zeros((3, 3)).astype(np.float32),
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
        [-3600, 0, 3600],
        "UTC_offset",
        coord_units="seconds",
        dtype=TIME_COORDS["forecast_period"].dtype,
    )
    true_row = [1, 1, 1]
    false_row = [0, 0, 0]
    cube.data = np.array(
        [
            [false_row, true_row, true_row],
            [true_row, false_row, true_row],
            [true_row, true_row, false_row],
        ],
        dtype=np.int8,
    )
    return cube


def assert_metadata_ok(output_cube):
    """Checks that the meta-data of output_cube are as expected"""
    assert isinstance(output_cube, Cube)
    assert output_cube.dtype == np.float32
    assert output_cube.coord_dims("time") == (0, 1)
    assert output_cube.coord("time").dtype == np.int64
    check_mandatory_standards(output_cube)


def test_create_output_cube():
    """Tests that the create_output_cube method builds a cube with appropriate
    meta-data"""
    data_shape = [3, 3]
    cube = make_input_cube(data_shape)
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


@pytest.mark.parametrize(
    "data_shape, expect_success", (([3, 3], True), ([3, 3, 3], False))
)
def test_check_input_cube_dims(data_shape, expect_success):
    """Checks that check_input_cube_dims can differentiate between an input cube
    with time, y, x coords and one with time, realization, y, x coords."""
    cube = make_input_cube(data_shape)
    plugin = TimezoneExtraction()
    if expect_success:
        plugin.check_input_cube_dims(cube)
    else:
        with pytest.raises(
            ValueError, match=r"Expected coords on input_cube: time, y, x "
        ):
            plugin.check_input_cube_dims(cube)


@pytest.mark.parametrize(
    "local_time, expect_success",
    ((datetime(2017, 11, 10, 5, 0), True), (datetime(2017, 11, 10, 6, 0), False)),
)
def test_check_input_cube_time(local_time, expect_success):
    """Checks that check_input_cube_time can differentiate between arguments that match
    expected times and arguments that don't."""
    data_shape = [3, 3]
    cube = make_input_cube(data_shape)
    timezone_cube = make_timezone_cube()
    plugin = TimezoneExtraction()
    if expect_success:
        plugin.check_input_cube_time(cube, timezone_cube, local_time)
    else:
        with pytest.raises(
            ValueError, match=r"Time coord on input cube does not match required times."
        ):
            plugin.check_input_cube_time(cube, timezone_cube, local_time)


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


@pytest.mark.parametrize("input_as_cube", (True, False))
@pytest.mark.parametrize("input_has_time_bounds", (True, False))
def test_process(input_as_cube, input_has_time_bounds):
    """Checks that the plugin process method returns the a cube with expected data and
    time coord for our test data"""
    data_shape = [3, 3]
    cube = make_input_cube(data_shape, time_bounds=input_has_time_bounds)
    data = np.array(
        [
            np.full((3, 3), fill_value=-1, dtype=np.float32),
            np.zeros((3, 3), dtype=np.float32),
            np.ones((3, 3), dtype=np.float32),
        ]
    )
    cube.data = data
    local_time = datetime(2017, 11, 10, 5, 0)
    timezone_cube = make_timezone_cube()
    expected_data = [[-1, -1, -1], [0, 0, 0], [1, 1, 1]]
    row1 = [cube.coord("time").units.date2num(datetime(2017, 11, 10, 4, 0))] * 3
    row2 = [cube.coord("time").units.date2num(datetime(2017, 11, 10, 5, 0))] * 3
    row3 = [cube.coord("time").units.date2num(datetime(2017, 11, 10, 6, 0))] * 3
    expected_times = [row1, row2, row3]
    expected_bounds = np.array(expected_times).reshape((3, 3, 1)) + [[[-3600, 0]]]
    if not input_as_cube:
        cube = [c for c in cube.slices_over("time")]
    result = TimezoneExtraction()(cube, timezone_cube, local_time)
    assert_metadata_ok(result)
    assert np.isclose(result.data, expected_data).all()
    assert np.isclose(result.coord("time").points, expected_times).all()
    if input_has_time_bounds:
        assert np.isclose(result.coord("time").bounds, expected_bounds).all()
    else:
        assert result.coord("time").bounds is None


def test_bad_dtype():
    """Checks that the plugin raises a useful error if the output are float64"""
    data_shape = [3, 3]
    cube = make_input_cube(data_shape)
    local_time = datetime(2017, 11, 10, 5, 0)
    timezone_cube = make_timezone_cube()
    timezone_cube.data = timezone_cube.data.astype(np.int32)
    with pytest.raises(
        TypeError,
        match=r"Operation multiply on types \{dtype\(\'.*32\'\), dtype\(\'.*32\'\)\} results in",
    ):
        TimezoneExtraction()(cube, timezone_cube, local_time)
