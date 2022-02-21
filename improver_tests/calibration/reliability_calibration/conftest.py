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
"""Fixtures for construct and aggregate reliability calibration tests."""

from datetime import datetime

import numpy as np
import pytest
from iris.coords import DimCoord
from iris.cube import CubeList

from improver.spotdata.build_spotdata_cube import build_spotdata_cube
from improver.synthetic_data.set_up_test_cubes import (
    construct_scalar_time_coords,
    set_up_probability_cube,
)
from improver.utilities.cube_manipulation import MergeCubes

"""Create forecast and truth cubes for use in testing the reliability
calibration plugin. Two forecast and two truth cubes are created, each
pair containing the same data but given different forecast reference
times and validity times. These times maintain the same forecast period
for each forecast cube.

The truth data for reliability calibration is thresholded data, giving
fields of zeroes and ones.

Each forecast cube in conjunction with the contemporaneous truth cube
will be used to produce a reliability calibration table. When testing
the process method here we expect the final reliability calibration
table for a given threshold (we are only using 283K in the value
comparisons) to be the sum of two of these identical tables."""

_thresholds = [283, 288]
_dummy_point_locations = np.arange(9).astype(np.float32)
_dummy_string_ids = [f"{i}" for i in range(9)]
_threshold_coord = DimCoord(
    _thresholds,
    standard_name="air_temperature",
    var_name="threshold",
    units="K",
    attributes={"spp__relative_to_threshold": "above"},
)


@pytest.fixture
def expected_table_shape_grid():
    return 3, 5, 3, 3


@pytest.fixture
def expected_table_shape_spot():
    return 3, 5, 9


@pytest.fixture
def expected_attributes():
    return {
        "title": "Reliability calibration data table",
        "source": "IMPROVER",
        "institution": "unknown",
    }


@pytest.fixture
def forecast_grid():
    forecast_data = np.arange(9, dtype=np.float32).reshape(3, 3) / 8.0
    forecast_data_stack = np.stack([forecast_data, forecast_data])
    forecast_1 = set_up_probability_cube(forecast_data_stack, _thresholds)
    forecast_2 = set_up_probability_cube(
        forecast_data_stack,
        _thresholds,
        time=datetime(2017, 11, 11, 4, 0),
        frt=datetime(2017, 11, 11, 0, 0),
    )
    forecasts_grid = MergeCubes()([forecast_1, forecast_2])
    return forecasts_grid


@pytest.fixture
def truth_grid():
    truth_data = np.linspace(281, 285, 9, dtype=np.float32).reshape(3, 3)
    # Threshold the truths, giving fields of zeroes and ones.
    truth_data_a = (truth_data > _thresholds[0]).astype(int)
    truth_data_b = (truth_data > _thresholds[1]).astype(int)
    truth_data_stack = np.stack([truth_data_a, truth_data_b])
    truth_1 = set_up_probability_cube(
        truth_data_stack, _thresholds, frt=datetime(2017, 11, 10, 4, 0)
    )
    truth_2 = set_up_probability_cube(
        truth_data_stack,
        _thresholds,
        time=datetime(2017, 11, 11, 4, 0),
        frt=datetime(2017, 11, 11, 4, 0),
    )
    truths_grid = MergeCubes()([truth_1, truth_2])
    return truths_grid


@pytest.fixture
def forecast_spot(forecast_grid):
    forecast_data = forecast_grid.data[0, ...]
    spot_probabilities = forecast_data.reshape((2, 9))
    forecasts_spot_list = CubeList()
    for day in range(5, 7):
        time_coords = construct_scalar_time_coords(
            datetime(2017, 11, day, 4, 0), None, datetime(2017, 11, day, 0, 0),
        )
        time_coords = [t[0] for t in time_coords]
        forecasts_spot_list.append(
            build_spotdata_cube(
                spot_probabilities,
                name="probability_of_air_temperature_above_threshold",
                units="1",
                altitude=_dummy_point_locations,
                latitude=_dummy_point_locations,
                longitude=_dummy_point_locations,
                wmo_id=_dummy_string_ids,
                additional_dims=[_threshold_coord],
                scalar_coords=time_coords,
            )
        )
    forecasts_spot = forecasts_spot_list.merge_cube()
    return forecasts_spot


@pytest.fixture
def truth_spot(truth_grid):
    print(truth_grid)
    print(truth_grid.data.shape)
    truth_data_spot = truth_grid[0, ...].data.reshape((2, 9))
    print(truth_data_spot.shape)
    print(truth_data_spot)
    truths_spot_list = CubeList()
    for day in range(5, 7):
        time_coords = construct_scalar_time_coords(
            datetime(2017, 11, day, 4, 0), None, datetime(2017, 11, day, 4, 0),
        )
        time_coords = [t[0] for t in time_coords]
        truths_spot_list.append(
            build_spotdata_cube(
                truth_data_spot,
                name="probability_of_air_temperature_above_threshold",
                units="1",
                altitude=_dummy_point_locations,
                latitude=_dummy_point_locations,
                longitude=_dummy_point_locations,
                wmo_id=_dummy_string_ids,
                additional_dims=[_threshold_coord],
                scalar_coords=time_coords,
            )
        )
    truths_spot = truths_spot_list.merge_cube()
    return truths_spot


@pytest.fixture
def masked_truths(truth_grid):
    truth_data_grid = truth_grid.data[0, ...]
    masked_array = np.zeros(truth_data_grid.shape, dtype=bool)
    masked_array[:, 0, :2] = True
    masked_truth_data_1 = np.ma.array(truth_data_grid, mask=masked_array)
    masked_array = np.zeros(truth_data_grid.shape, dtype=bool)
    masked_array[:, :2, 0] = True
    masked_truth_data_2 = np.ma.array(truth_data_grid, mask=masked_array)
    masked_truth_1 = set_up_probability_cube(
        masked_truth_data_1, _thresholds, frt=datetime(2017, 11, 10, 4, 0)
    )
    masked_truth_2 = set_up_probability_cube(
        masked_truth_data_2,
        _thresholds,
        time=datetime(2017, 11, 11, 4, 0),
        frt=datetime(2017, 11, 11, 4, 0),
    )
    merged = MergeCubes()([masked_truth_1, masked_truth_2])
    return merged


@pytest.fixture
def expected_table():
    # Note the structure of the expected_table is non-trivial to interpret
    # due to the dimension ordering.
    return np.array(
        [
            [
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 1.0, 0.0]],
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            ],
            [
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                [[0.0, 0.125, 0.25], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [0.375, 0.5, 0.625], [0.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.75, 0.875, 0.0]],
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            ],
            [
                [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                [[0.0, 1.0, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [0.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 1.0, 0.0]],
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            ],
        ],
        dtype=np.float32,
    )


@pytest.fixture
def expected_table_for_mask():
    return np.array(
        [
            [
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 1.0, 0.0]],
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            ],
            [
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.25], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [0.375, 0.5, 0.625], [0.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.75, 0.875, 0.0]],
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            ],
            [
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                [[0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [0.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 1.0, 0.0]],
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            ],
        ],
        dtype=np.float32,
    )


@pytest.fixture
def reliability_cube(forecast_grid, expected_table):
    from improver.calibration.reliability_calibration import (
        ConstructReliabilityCalibrationTables as CalPlugin,
    )

    rel_cube_format = CalPlugin()._create_reliability_table_cube(
        forecast_grid, forecast_grid.coord(var_name="threshold")
    )
    rel_cube = rel_cube_format.copy(data=expected_table)
    return rel_cube


@pytest.fixture
def different_frt(reliability_cube):
    diff_frt = reliability_cube.copy()
    new_frt = diff_frt.coord("forecast_reference_time")
    new_frt.points = new_frt.points + 48 * 3600
    new_frt.bounds = new_frt.bounds + 48 * 3600
    return diff_frt


@pytest.fixture
def masked_reliability_cube(reliability_cube):
    masked_array = np.zeros(reliability_cube.shape, dtype=bool)
    masked_array[:, :, 0, :2] = True
    reliability_cube.data = np.ma.array(reliability_cube.data, mask=masked_array)
    return reliability_cube


@pytest.fixture
def masked_different_frt(different_frt):
    masked_array = np.zeros(different_frt.shape, dtype=bool)
    masked_array[:, :, :2, 0] = True
    different_frt.data = np.ma.array(different_frt.data, mask=masked_array)
    return different_frt


@pytest.fixture
def overlapping_frt(reliability_cube):
    new_frt = reliability_cube.coord("forecast_reference_time")
    new_frt.points = new_frt.points + 6 * 3600
    new_frt.bounds = new_frt.bounds + 6 * 3600
    return reliability_cube


@pytest.fixture
def lat_lon_collapse():
    return np.array(
        [
            [0.0, 0.0, 1.0, 2.0, 1.0],
            [0.0, 0.375, 1.5, 1.625, 1.0],
            [1.0, 2.0, 3.0, 2.0, 1.0],
        ]
    )
