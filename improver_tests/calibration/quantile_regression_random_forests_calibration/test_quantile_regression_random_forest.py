# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the Quantile Regression Random Forest plugins."""

from datetime import datetime as dt

import numpy as np
import pytest
from iris.coords import AuxCoord
from iris.cube import Cube

from improver.calibration.quantile_regression_random_forest import (
    prep_feature,
)
from improver.metadata.constants.time_types import DT_FORMAT
from improver.synthetic_data.set_up_test_cubes import set_up_spot_variable_cube

ALTITUDE = [10, 20]
LATITUDE = [50, 60]
LONGITUDE = [0, 10]
WMO_ID = ["00001", "00002"]


def _create_forecasts(
    forecast_reference_time: str, validity_time: str, data: list[int]
) -> Cube:
    """Create site forecast cube with realizations.

    Args:
        forecast_reference_time: Timestamp e.g. "20170101T0000Z".
        validity_time: Timestamp e.g. "20170101T0600Z".
        data: Data that will be repeated to create a cube with two sites. The
        length of the data will equal the number of realizations created.

    Returns:
        Forecast cube containing three percentiles and two sites.
    """
    data = np.array(data, dtype=np.float32).repeat(2).reshape(3, 2)
    cube = set_up_spot_variable_cube(
        data,
        realizations=range(len(data)),
        name="wind_speed_at_10m",
        units="m s-1",
        wmo_ids=WMO_ID,
        latitudes=np.array([50, 60], np.float32),
        longitudes=np.array([0, 10], np.float32),
        altitudes=np.array([10, 20], np.float32),
        time=dt.strptime(validity_time, DT_FORMAT),
        frt=dt.strptime(forecast_reference_time, DT_FORMAT),
    )
    day_of_training_period = AuxCoord(
        np.int32([31]), long_name="day_of_training_period", units="1"
    )
    cube.add_aux_coord(day_of_training_period)
    return cube


def _create_ancil_file():
    """Create an ancillary file for testing.

    Returns:
        An ancillary cube with a single value.
    """
    data = np.array([2, 3], dtype=np.float32)
    template_cube = set_up_spot_variable_cube(
        data,
        wmo_ids=WMO_ID,
        name="distance_to_water",
        units="m",
    )
    cube = template_cube.copy()
    for coord in ["time", "forecast_reference_time", "forecast_period"]:
        cube.remove_coord(coord)
    return cube


@pytest.mark.parametrize(
    "feature,expected,expected_dtype",
    [
        ("mean", np.array([6, 6], dtype=np.float32), np.float32),
        ("std", np.array([4, 4], dtype=np.float32), np.float32),
        ("latitude", np.array([50, 60], dtype=np.float32), np.float32),
        ("longitude", np.array([0, 10], dtype=np.float32), np.float32),
        ("altitude", np.array([10, 20], dtype=np.float32), np.float32),
        (
            "day_of_year_sin",
            np.array([0.01716633, 0.01716633], dtype=np.float32),
            np.float32,
        ),
        (
            "day_of_year_cos",
            np.array([0.99985266, 0.99985266], dtype=np.float32),
            np.float32,
        ),
        ("hour_of_day_sin", np.array([0, 0], dtype=np.float32), np.float32),
        ("hour_of_day_cos", np.array([-1, -1], dtype=np.float32), np.float32),
        ("forecast_period", np.array([43200, 43200], dtype=np.int32), np.int32),
        ("day_of_training_period", np.array([31, 31], dtype=np.int32), np.int32),
        ("static", np.array([2, 3], dtype=np.float32), np.float32),
    ],
)
def test_prep_feature_single_time(feature, expected, expected_dtype):
    """Test the prep_feature function."""
    forecast_reference_time = "20170101T0000Z"
    validity_time = "20170101T1200Z"
    data = [2, 6, 10]
    forecast_cube = _create_forecasts(forecast_reference_time, validity_time, data)

    if feature == "static":
        feature_cube = _create_ancil_file()
    else:
        feature_cube = forecast_cube.copy()

    result = prep_feature(forecast_cube, feature_cube, feature)
    assert result.shape == (2,)
    assert result.dtype == expected_dtype
    np.testing.assert_allclose(result, expected, atol=1e-6)
