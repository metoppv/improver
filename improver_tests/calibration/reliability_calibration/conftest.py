# (C) Crown copyright, Met Office. All rights reserved.
#
# This file is part of improver and is released under a BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Fixtures for reliability calibration tests."""

from collections import namedtuple
from datetime import datetime

import iris
import numpy as np
import pytest
from iris.coords import DimCoord
from iris.cube import CubeList

from improver.calibration.reliability_calibration import (
    ConstructReliabilityCalibrationTables as CalPlugin,
)
from improver.spotdata.build_spotdata_cube import build_spotdata_cube
from improver.synthetic_data.set_up_test_cubes import (
    construct_scalar_time_coords,
    set_up_probability_cube,
)
from improver.utilities.cube_manipulation import MergeCubes

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
    truth_data_spot = truth_grid[0, ...].data.reshape((2, 9))
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


RelTableInputs = namedtuple("RelTableInputs", ["forecast", "truth", "expected_shape"])


@pytest.fixture(params=["grid", "spot"])
def create_rel_table_inputs(
    request,
    forecast_grid,
    forecast_spot,
    truth_grid,
    truth_spot,
    expected_table_shape_grid,
    expected_table_shape_spot,
):
    if request.param == "grid":
        return RelTableInputs(
            forecast=forecast_grid,
            truth=truth_grid,
            expected_shape=expected_table_shape_grid,
        )
    return RelTableInputs(
        forecast=forecast_spot,
        truth=truth_spot,
        expected_shape=expected_table_shape_spot,
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


@pytest.fixture
def reliability_data():
    reliability_data = np.array(
        [
            [
                [0, 0, 250, 500, 750],  # Observation count
                [0, 250, 500, 750, 1000],  # Sum of forecast probability
                [1000, 1000, 1000, 1000, 1000],  # Forecast count
            ],
            [
                [250, 500, 750, 1000, 1000],  # Observation count
                [0, 250, 500, 750, 1000],  # Sum of forecast probability
                [1000, 1000, 1000, 1000, 1000],  # Forecast count
            ],
        ],
        dtype=np.float32,
    )
    return reliability_data


@pytest.fixture
def reliability_table_agg(forecast_grid, truth_grid, reliability_data):
    reliability_cube_format = CalPlugin().process(forecast_grid, truth_grid)
    reliability_cube_format = reliability_cube_format.collapsed(
        [
            reliability_cube_format.coord(axis="x"),
            reliability_cube_format.coord(axis="y"),
        ],
        iris.analysis.SUM,
    )
    reliability_table_agg = reliability_cube_format.copy(data=reliability_data)
    return reliability_table_agg


@pytest.fixture
def reliability_table_slice(reliability_table_agg):
    return next(reliability_table_agg.slices_over("air_temperature"))


@pytest.fixture
def reliability_table_point_spot(forecast_spot, truth_spot, reliability_data):
    reliability_cube_format = CalPlugin().process(forecast_spot, truth_spot)
    data = np.stack([reliability_data] * 9, axis=-1)
    reliability_table = reliability_cube_format.copy(data=data)
    return reliability_table


@pytest.fixture
def reliability_table_point_grid(forecast_grid, truth_grid, reliability_data):
    reliability_cube_format = CalPlugin().process(forecast_grid, truth_grid)
    data = np.stack([np.stack([reliability_data] * 3, axis=-1)] * 3, axis=-1)
    reliability_table = reliability_cube_format.copy(data=data)
    return reliability_table


RelTables = namedtuple("RelTables", ["table", "indices0", "indices1", "indices2"])


@pytest.fixture(params=["point_spot", "point_grid"])
def create_rel_tables_point(
    request, reliability_table_point_spot, reliability_table_point_grid,
):
    if request.param == "point_spot":
        return RelTables(
            table=reliability_table_point_spot,
            indices0=(0, slice(0, None), slice(0, None), 0),
            indices1=(0, slice(0, None), slice(0, None), 1),
            indices2=(1, slice(0, None), slice(0, None), 0),
        )
    else:
        return RelTables(
            table=reliability_table_point_grid,
            indices0=(0, slice(0, None), slice(0, None), 0, 0),
            indices1=(0, slice(0, None), slice(0, None), 0, 1),
            indices2=(1, slice(0, None), slice(0, None), 0, 1),
        )


@pytest.fixture
def probability_bin_coord(reliability_table_agg):
    return reliability_table_agg.coord("probability_bin")


@pytest.fixture
def default_obs_counts():
    return np.array([0, 250, 500, 750, 1000], dtype=np.float32)


@pytest.fixture
def default_fcst_counts():
    return np.array([1000, 1000, 1000, 1000, 1000], dtype=np.float32)


@pytest.fixture
def expected_enforced_monotonic():
    return np.array(
        [
            [0, 250, 500, 1750],  # Observation count
            [0, 250, 500, 1750],  # Sum of forecast probability
            [1000, 1000, 1000, 2000],  # Forecast count
        ]
    )
