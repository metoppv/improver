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

import numpy as np
import pytest
from cf_units import Unit

from improver.metadata.constants.time_types import TIME_COORDS
from improver.spotdata.build_spotdata_cube import build_spotdata_cube
from improver.synthetic_data.set_up_test_cubes import (
    add_coordinate,
    set_up_variable_cube,
)

GRIDDDED_DATA_SHAPE = [3, 4]
SPOT_DATA_SHAPE = [10]


def spot_cube(cube, data=None):
    """Copy elements of a gridded cube to populate metadata on a spot cube for
    use in the tests."""
    time_coord = cube.coord("time").copy()
    fp_coord = cube.coord("forecast_period").copy()
    frt_coord = cube.coord("forecast_reference_time").copy()
    attributes = cube.attributes.copy()
    coord_vals = np.arange(SPOT_DATA_SHAPE[0])

    yx_crds = [cube.coord(axis=dim) for dim in ["y", "x"]]
    cslice = next(cube.slices_over(yx_crds))
    dim_coords = cslice.coords(dim_coords=True)

    if data is None:
        shape = (*cslice.shape, SPOT_DATA_SHAPE[0])
        data = np.zeros(shape).astype(np.float32)

    cube = build_spotdata_cube(
        data,
        name=cube.name(),
        units=cube.units,
        altitude=coord_vals,
        latitude=coord_vals,
        longitude=coord_vals,
        wmo_id=coord_vals,
        scalar_coords=[time_coord, fp_coord, frt_coord],
        additional_dims=dim_coords,
    )
    cube.attributes = attributes
    return cube


def create_cube(time_bounds, gridded):
    """Makes a 3D cube (time, y, x) of the described shape, filled with zeroes for use
    in testing."""

    cube = set_up_variable_cube(
        np.zeros(GRIDDDED_DATA_SHAPE).astype(np.float32),
        standard_grid_metadata="gl_ens",
        attributes={
            "institution": "unknown",
            "source": "IMPROVER",
            "title": "Unit test",
        },
    )
    if not gridded:
        cube = spot_cube(cube)

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


@pytest.fixture
def input_cube(time_bounds, percentiles, gridded):
    if percentiles:
        cube = create_cube(time_bounds, gridded)
        cube = add_coordinate(cube, (25, 50, 75), "percentile", "%")
        return cube
    return create_cube(time_bounds, gridded)


@pytest.fixture
def data_shape(percentiles, gridded):
    data_shape = GRIDDDED_DATA_SHAPE if gridded else SPOT_DATA_SHAPE
    if percentiles:
        return (3, *data_shape)
    return data_shape


@pytest.fixture
def test_data(percentiles, gridded, data_shape):

    if gridded:
        data = np.array(
            [
                np.zeros(data_shape[-2:], dtype=np.float32),
                np.ones(data_shape[-2:], dtype=np.float32),
            ]
        )
        expected_data = [[0, 0, 0, 0], [0, 0, 0, 0], [1, 1, 1, 1]]
        if percentiles:
            data = np.array([data, data, data])
            expected_data = np.array([expected_data, expected_data, expected_data])
    else:
        data = np.array(
            [
                np.full(data_shape[-1], 270, dtype=np.float32),
                np.full(data_shape[-1], 280, dtype=np.float32),
            ]
        )
        expected_data = np.array(
            [280, 280, 270, 270, 280, 280, 270, 270, 280, 280], dtype=np.float32
        )
        if percentiles:
            data = np.array([data, data, data])
            expected_data = np.array([expected_data, expected_data, expected_data])

    return data, expected_data


@pytest.fixture
def expected_times(gridded, input_cube):

    tz1 = input_cube.coord("time").units.date2num(datetime(2017, 11, 10, 4, 0))
    tz2 = input_cube.coord("time").units.date2num(datetime(2017, 11, 10, 5, 0))

    if gridded:
        expected_times = [[tz1] * 4, [tz1] * 4, [tz2] * 4]
        expected_bounds = np.array(expected_times).reshape((3, 4, 1)) + [[[-3600, 0]]]

    else:
        expected_times = [tz2, tz2, tz1, tz1, tz2, tz2, tz1, tz1, tz2, tz2]
        expected_bounds = np.array(expected_times).reshape((10, 1)) + [[-3600, 0]]

    return expected_times, expected_bounds


@pytest.fixture
def timezone_cube(gridded):
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
    if not gridded:
        data = np.array(
            [[0, 0, 1, 1, 0, 0, 1, 1, 0, 0], [1, 1, 0, 0, 1, 1, 0, 0, 1, 1]],
            dtype=np.int8,
        )
        cube = spot_cube(cube, data=data)
    return cube
