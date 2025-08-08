# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the Quantile Regression Random Forest plugins."""

from datetime import datetime as dt

import iris
import joblib
import numpy as np
import pandas as pd
import pytest
from iris.coords import AuxCoord
from iris.cube import Cube, CubeList

from improver.calibration.quantile_regression_random_forest import (
    ApplyQuantileRegressionRandomForests,
    TrainQuantileRegressionRandomForests,
    prep_feature,
)
from improver.metadata.constants.time_types import DT_FORMAT
from improver.synthetic_data.set_up_test_cubes import set_up_spot_variable_cube
from improver.utilities.temporal import datetime_to_iris_time

ALTITUDE = [10, 20]
LATITUDE = [50, 60]
LONGITUDE = [0, 10]
WMO_ID = ["00001", "00002"]


def _create_forecasts(
    forecast_reference_time: str,
    validity_time: str,
    data: list[int],
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

    data = np.array(data, dtype=np.float32).repeat(2).reshape(len(data), 2)
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
    cube.remove_coord("time")
    return cube


def _add_day_of_training_period(cube, day_of_training_period, secondary_coord):
    """Add day of training period coordinate to the cube.

    Args:
        cube: Cube to which the day of training period coordinate will be added.
        day_of_training_period: Day of training period to be added.
        secondary_coord: Coordinate to associate the day of training period with.

    Returns:
        Cube with the day of training period coordinate added.
    """
    dims = cube.coord_dims(secondary_coord)
    day_of_training_period_coord = AuxCoord(
        np.array(day_of_training_period, dtype=np.int32),
        long_name="day_of_training_period",
        units="1",
    )
    cube.add_aux_coord(day_of_training_period_coord, data_dims=dims)
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
        latitudes=np.array([50, 60], np.float32),
        longitudes=np.array([0, 10], np.float32),
        altitudes=np.array([10, 20], np.float32),
        name="distance_to_water",
        units="m",
    )
    cube = template_cube.copy()
    for coord in ["time", "forecast_reference_time", "forecast_period"]:
        cube.remove_coord(coord)
    return cube


def _run_train_qrf(
    tmp_path,
    feature_config,
    n_estimators,
    max_depth,
    random_state,
    compression,
    transformation,
    pre_transform_addition,
    extra_kwargs,
    include_static,
    forecast_reference_times=[
        "20170101T0000Z",
        "20170101T0000Z",
        "20170102T0000Z",
        "20170102T0000Z",
    ],
    validity_times=[
        "20170101T1200Z",
        "20170101T1800Z",
        "20170102T1200Z",
        "20170102T1800Z",
    ],
    day_of_training_period=[0, 1],
    realization_data=[2, 6, 10],
    truth_data=[[4.2, 3.8], [5.8, 6], [7, 7.3], [9.1, 9.5]],
):
    realization_data = np.array(realization_data, dtype=np.float32)
    forecast_cubes = CubeList()
    for index, (frt, vt) in enumerate(zip(forecast_reference_times, validity_times)):
        forecast_cube = _create_forecasts(frt, vt, realization_data + index)
        forecast_cubes.append(forecast_cube)
    forecast_cube = forecast_cubes.merge_cube()

    forecast_cube = _add_day_of_training_period(
        forecast_cube, day_of_training_period, "forecast_reference_time"
    )

    truth_cube = next(forecast_cube.slices_over("realization")).copy()
    for coord in [
        "day_of_training_period",
        "realization",
    ]:
        truth_cube.remove_coord(coord)
    truth_slices = CubeList()
    for truth_slice in truth_cube.slices_over(
        ["forecast_period", "forecast_reference_time"]
    ):
        frt_coord = truth_slice.coord("forecast_reference_time")
        fp_coord = truth_slice.coord("forecast_period")
        time_point = datetime_to_iris_time(
            frt_coord.cell(0).point._to_real_datetime()
            + pd.to_timedelta(fp_coord.points[0], unit=str(fp_coord.units))
        )
        time_coord = iris.coords.AuxCoord(
            time_point, standard_name="time", units=frt_coord.units
        )
        truth_slice.add_aux_coord(time_coord)
        truth_slice.remove_coord("forecast_period")
        truth_slice.remove_coord("forecast_reference_time")
        truth_slices.append(truth_slice)

    truth_cube = truth_slices.merge_cube()
    truth_cube.data = np.array(truth_data).astype(np.float32)

    feature_cubes = CubeList([forecast_cube])
    if include_static:
        ancil_cube = _create_ancil_file()
        feature_cubes.append(ancil_cube)
        feature_config[ancil_cube.name()] = ["static"]
    if "air_temperature" in feature_config.keys():
        dynamic_cube = forecast_cube.copy(data=forecast_cube.data + 10)
        dynamic_cube.rename("air_temperature")
        dynamic_cube.units = "Celsius"
        feature_cubes.append(dynamic_cube)

    model_output_dir = tmp_path / "train_qrf"
    model_output_dir.mkdir(parents=True)
    model_output = str(model_output_dir / "qrf_model.pkl")

    plugin = TrainQuantileRegressionRandomForests(
        feature_config,
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        transformation=transformation,
        pre_transform_addition=pre_transform_addition,
        compression=compression,
        model_output=model_output,
        **extra_kwargs,
    )
    plugin.process(forecast_cube, truth_cube, feature_cubes=feature_cubes)
    return model_output


@pytest.mark.parametrize(
    "feature,expected,expected_dtype",
    [
        ("mean", np.array([6, 6], dtype=np.float32), np.float32),
        ("std", np.array([4, 4], dtype=np.float32), np.float32),
        ("latitude", np.array([50, 60], dtype=np.float32), np.float32),
        ("longitude", np.array([0, 10], dtype=np.float32), np.float32),
        ("altitude", np.array([10, 20], dtype=np.float32), np.float32),
        (
            "day_of_year",
            np.array([1, 1], dtype=np.float32),
            np.float32,
        ),
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
        ("hour_of_day", np.array([12, 12], dtype=np.float32), np.float32),
        ("hour_of_day_sin", np.array([0, 0], dtype=np.float32), np.float32),
        ("hour_of_day_cos", np.array([-1, -1], dtype=np.float32), np.float32),
        ("forecast_period", np.array([43200, 43200], dtype=np.int32), np.int32),
        ("day_of_training_period", np.array([0, 0], dtype=np.int32), np.int32),
        ("static", np.array([2, 3], dtype=np.float32), np.float32),
    ],
)
def test_prep_feature_single_time(feature, expected, expected_dtype):
    """Test the prep_feature function for a single time."""
    forecast_reference_time = "20170101T0000Z"
    validity_time = "20170101T1200Z"
    data = [2, 6, 10]
    day_of_training_period = [0]
    forecast_cube = _create_forecasts(forecast_reference_time, validity_time, data)
    forecast_cube = _add_day_of_training_period(
        forecast_cube, day_of_training_period, "forecast_reference_time"
    )

    if feature == "static":
        feature_cube = _create_ancil_file()
    else:
        feature_cube = forecast_cube.copy()

    result = prep_feature(forecast_cube, feature_cube, feature)
    assert result.shape == (2,)
    assert result.dtype == expected_dtype
    np.testing.assert_allclose(result, expected, atol=1e-6)


@pytest.mark.parametrize(
    "feature,expected,expected_dtype",
    [
        ("mean", np.repeat(6, 8).astype(np.float32), np.float32),
        ("std", np.repeat(4, 8).astype(np.float32), np.float32),
        ("latitude", np.tile([50, 60], 4).astype(np.float32), np.float32),
        ("longitude", np.tile([0, 10], 4).astype(np.float32), np.float32),
        ("altitude", np.tile([10, 20], 4).astype(np.float32), np.float32),
        (
            "day_of_year",
            np.repeat([1, 2], 4).astype(np.float32),
            np.float32,
        ),
        (
            "day_of_year_sin",
            np.repeat([0.017166, 0.034328], 4).astype(np.float32),
            np.float32,
        ),
        (
            "day_of_year_cos",
            np.repeat([0.999853, 0.999411], 4).astype(np.float32),
            np.float32,
        ),
        ("hour_of_day", np.repeat([12, 12], 4).astype(np.float32), np.float32),
        ("hour_of_day_sin", np.repeat([0, 0], 4).astype(np.float32), np.float32),
        ("hour_of_day_cos", np.repeat([-1, -1], 4).astype(np.float32), np.float32),
        (
            "forecast_period",
            np.tile([43200, 43200, 21600, 21600], 2).astype(np.int32),
            np.int32,
        ),
        ("day_of_training_period", np.repeat([0, 1], 4).astype(np.int32), np.int32),
        ("static", np.tile([2, 3], 4).astype(np.float32), np.float32),
    ],
)
def test_prep_feature_1d_time_dimension(feature, expected, expected_dtype):
    """Test the prep_feature function for multiple times."""
    forecast_reference_times = [
        "20170101T0000Z",
        "20170101T0600Z",
        "20170102T0000Z",
        "20170102T0600Z",
    ]
    validity_times = [
        "20170101T1200Z",
        "20170101T1200Z",
        "20170102T1200Z",
        "20170102T1200Z",
    ]

    data = [2, 6, 10]

    forecast_cubes = CubeList()
    for frt, vt in zip(forecast_reference_times, validity_times):
        forecast_cubes.append(_create_forecasts(frt, vt, data))
    forecast_cube = forecast_cubes.merge_cube()
    day_of_training_period = [0, 0, 1, 1]
    forecast_cube = _add_day_of_training_period(
        forecast_cube, day_of_training_period, "forecast_reference_time"
    )

    if feature == "static":
        feature_cube = _create_ancil_file()
    else:
        feature_cube = forecast_cube.copy()

    result = prep_feature(forecast_cube, feature_cube, feature)

    assert result.shape == (8,)
    assert result.dtype == expected_dtype
    np.testing.assert_allclose(result, expected, atol=1e-6)


@pytest.mark.parametrize(
    "feature,expected,expected_dtype",
    [
        ("mean", np.repeat(6, 12).astype(np.float32), np.float32),
        ("std", np.repeat(4, 12).astype(np.float32), np.float32),
        ("latitude", np.tile([50, 60], 6).astype(np.float32), np.float32),
        ("longitude", np.tile([0, 10], 6).astype(np.float32), np.float32),
        ("altitude", np.tile([10, 20], 6).astype(np.float32), np.float32),
        (
            "day_of_year",
            np.repeat([1, 2, 3], 4).astype(np.float32),
            np.float32,
        ),
        (
            "day_of_year_sin",
            np.repeat([0.017166, 0.034328, 0.051479], 4).astype(np.float32),
            np.float32,
        ),
        (
            "day_of_year_cos",
            np.repeat([0.999853, 0.999411, 0.998674], 4).astype(np.float32),
            np.float32,
        ),
        ("hour_of_day", np.tile([6, 6, 12, 12], 3).astype(np.float32), np.float32),
        ("hour_of_day_sin", np.tile([1, 1, 0, 0], 3).astype(np.float32), np.float32),
        ("hour_of_day_cos", np.tile([0, 0, -1, -1], 3).astype(np.float32), np.float32),
        (
            "forecast_period",
            np.tile([21600, 21600, 43200, 43200], 3).astype(np.int32),
            np.int32,
        ),
        ("day_of_training_period", np.repeat([0, 1, 2], 4).astype(np.int32), np.int32),
        ("static", np.tile([2, 3], 6).astype(np.float32), np.float32),
    ],
)
def test_prep_feature_2d_time_dimension(feature, expected, expected_dtype):
    """Test the prep_feature function for multiple times."""
    forecast_reference_times = [
        "20170101T0000Z",
        "20170101T0000Z",
        "20170102T0000Z",
        "20170102T0000Z",
        "20170103T0000Z",
        "20170103T0000Z",
    ]
    validity_times = [
        "20170101T0600Z",
        "20170101T1200Z",
        "20170102T0600Z",
        "20170102T1200Z",
        "20170103T0600Z",
        "20170103T1200Z",
    ]

    data = [2, 6, 10]
    forecast_cubes = CubeList()
    for frt, vt in zip(forecast_reference_times, validity_times):
        forecast_cubes.append(_create_forecasts(frt, vt, data))
    forecast_cube = forecast_cubes.merge_cube()
    # Ensure coordinate order is forecast_reference_time, forecast_period,
    # realization, spot_index
    forecast_cube.transpose([1, 0, 2, 3])
    day_of_training_period = [0, 1, 2]
    forecast_cube = _add_day_of_training_period(
        forecast_cube, day_of_training_period, "forecast_reference_time"
    )

    if feature == "static":
        feature_cube = _create_ancil_file()
    else:
        feature_cube = forecast_cube.copy()

    result = prep_feature(forecast_cube, feature_cube, feature)

    assert result.shape == (12,)
    assert result.dtype == expected_dtype
    np.testing.assert_allclose(result, expected, atol=1e-6)


@pytest.mark.parametrize(
    "n_estimators,max_depth,random_state,compression,transformation,pre_transform_addition,extra_kwargs,include_static,expected",
    [
        (2, 2, 55, 5, None, 0, {}, False, 4.15),  # Basic test case
        (2, 2, 54, 5, None, 0, {}, False, 4.2),  # Different random state
        (1, 1, 55, 5, None, 0, {}, False, 4.2),  # Fewer estimators and reduced depth
        (2, 2, 55, 5, "log", 10, {}, False, 2.65),  # Log transformation
        (2, 2, 55, 5, "log10", 10, {}, False, 1.15),  # Log10 transformation
        (2, 2, 55, 5, "sqrt", 10, {}, False, 3.76),  # Square root transformation
        (2, 2, 55, 5, "cbrt", 10, {}, False, 2.42),  # Cube root transformation
        (2, 2, 55, 5, None, 0, {"criterion": "absolute_error"}, False, 4.15),  # noqa # Different criterion
        (2, 5, 55, 5, None, 0, {}, True, 4.15),  # Include static data
    ],
)
def test_train_qrf_single_lead_times(
    tmp_path,
    n_estimators,
    max_depth,
    random_state,
    compression,
    transformation,
    pre_transform_addition,
    extra_kwargs,
    include_static,
    expected,
):
    """Test the TrainQuantileRegressionRandomForests plugin when the forecast cube
    for training contains a single lead time."""

    feature_config = {"wind_speed_at_10m": ["mean", "std", "latitude", "longitude"]}

    model_output = _run_train_qrf(
        tmp_path,
        feature_config,
        n_estimators,
        max_depth,
        random_state,
        compression,
        transformation,
        pre_transform_addition,
        extra_kwargs,
        include_static,
        forecast_reference_times=[
            "20170101T0000Z",
            "20170102T0000Z",
        ],
        validity_times=[
            "20170101T1200Z",
            "20170102T1200Z",
        ],
        day_of_training_period=[0, 1],
        realization_data=[2, 6, 10],
        truth_data=[[4.2, 4.2], [4.1, 4.1]],
    )
    qrf_model = joblib.load(model_output)

    assert qrf_model.n_estimators == n_estimators
    assert qrf_model.max_depth == max_depth
    assert qrf_model.random_state == random_state

    current_forecast = [5, 3, 55, 5]
    if include_static:
        current_forecast.append(2.5)
    result = qrf_model.predict(
        np.expand_dims(np.array(current_forecast), 0), quantiles=[0.5]
    )
    np.testing.assert_almost_equal(result, expected, decimal=2)


@pytest.mark.parametrize(
    "n_estimators,max_depth,random_state,compression,transformation,pre_transform_addition,extra_kwargs,include_static,expected",
    [
        (2, 2, 55, 5, None, 0, {}, False, 4.0),  # Basic test case
        (1, 1, 55, 5, None, 0, {}, False, 3.8),  # Fewer estimators and reduced depth
        (1, 1, 73, 5, None, 0, {}, False, 7),  # Different random state
        (2, 2, 55, 5, "log", 10, {}, False, 2.743),  # Log transformation
        (2, 2, 55, 5, "log10", 10, {}, False, 1.191),  # Log10 transformation
        (2, 2, 55, 5, "sqrt", 10, {}, False, 3.946),  # Square root transformation
        (2, 2, 55, 5, "cbrt", 10, {}, False, 2.496),  # Cube root transformation
        (2, 2, 55, 5, None, 0, {"criterion": "absolute_error"}, False, 4),  # noqa # Different criterion
        (2, 5, 55, 5, None, 0, {}, True, 4),  # Include static data
    ],
)
def test_train_qrf_multiple_lead_times(
    tmp_path,
    n_estimators,
    max_depth,
    random_state,
    compression,
    transformation,
    pre_transform_addition,
    extra_kwargs,
    include_static,
    expected,
):
    """Test the TrainQuantileRegressionRandomForests plugin when multiple lead times
    are provided in the forecast cube for training."""

    feature_config = {"wind_speed_at_10m": ["mean", "std", "latitude", "longitude"]}

    model_output = _run_train_qrf(
        tmp_path,
        feature_config,
        n_estimators,
        max_depth,
        random_state,
        compression,
        transformation,
        pre_transform_addition,
        extra_kwargs,
        include_static,
    )
    qrf_model = joblib.load(model_output)

    assert qrf_model.n_estimators == n_estimators
    assert qrf_model.max_depth == max_depth
    assert qrf_model.random_state == random_state

    current_forecast = [5, 3, 55, 5]
    if include_static:
        current_forecast.append(2.5)
    result = qrf_model.predict(
        np.expand_dims(np.array(current_forecast), 0), quantiles=[0.5]
    )
    np.testing.assert_almost_equal(result, expected, decimal=3)


@pytest.mark.parametrize(
    "feature_config,data,include_static,expected",
    [
        ({"wind_speed_at_10m": ["mean"]}, [5], False, [4]),  # One feature
        ({"wind_speed_at_10m": ["latitude"]}, [55], False, [6.65]),  # noqa Without the target
        ({"wind_speed_at_10m": ["mean"]}, [5], True, [4]),  # With static data
        (
            {"wind_speed_at_10m": ["mean"], "air_temperature": ["mean"]},
            [5],
            False,
            [4],
        ),  # Multiple dynamic features
        (
            {"wind_speed_at_10m": ["mean"], "pressure_at_mean_sea_level": ["mean"]},
            [5],
            False,
            "Feature cube for pressure_at_mean_sea_level",
        ),  # Multiple dynamic features
    ],
)
def test_alternative_feature_configs(
    tmp_path,
    feature_config,
    data,
    include_static,
    expected,
):
    """Test the TrainQuantileRegressionRandomForests plugin for a few different
    configurations of the feature_config dictionary."""
    n_estimators = 2
    max_depth = 5
    random_state = 55
    compression = 5
    extra_kwargs = {}
    transformation = None
    pre_transform_addition = 0

    if "pressure_at_mean_sea_level" in feature_config:
        with pytest.raises(ValueError, match=expected):
            _run_train_qrf(
                tmp_path,
                feature_config,
                n_estimators,
                max_depth,
                random_state,
                compression,
                transformation,
                pre_transform_addition,
                extra_kwargs,
                include_static,
            )
        return

    model_output = _run_train_qrf(
        tmp_path,
        feature_config,
        n_estimators,
        max_depth,
        random_state,
        compression,
        transformation,
        pre_transform_addition,
        extra_kwargs,
        include_static,
    )
    qrf_model = joblib.load(model_output)

    assert qrf_model.n_estimators == n_estimators
    assert qrf_model.max_depth == max_depth
    assert qrf_model.random_state == random_state

    current_forecast = [*data]
    if include_static:
        current_forecast.append(2.5)
    if "air_temperature" in feature_config:
        current_forecast.append(15.0)

    result = qrf_model.predict(
        np.expand_dims(np.array(current_forecast), 0), quantiles=[0.5]
    )
    np.testing.assert_almost_equal(result, expected, decimal=2)


@pytest.mark.parametrize(
    "quantiles,transformation,pre_transform_addition,include_static,expected",
    [
        ([0.5], None, 0, False, [4, 4]),  # One quantile
        # ([0.1, 0.5, 0.9], None, 0, False, [[9.1, 9.14], [9.1, 9.3], [9.1, 9.46]]),  # noqa Multiple quantiles
        # ([0.1, 0.5, 0.9], "log", 10, False, [[4.46, 4.46], [5.54, 5.54], [6.7, 6.7]]),  # noqa Log transformation
        # (
        #     [0.1, 0.5, 0.9],
        #     "log10",
        #     10,
        #     False,
        #     [[4.46, 4.46], [5.54, 5.54], [6.7, 6.7]],
        # ),  # Log10 transformation
        # (
        #     [0.1, 0.5, 0.9],
        #     "sqrt",
        #     10,
        #     False,
        #     [[4.47, 4.47], [5.57, 5.57], [6.71, 6.71]],
        # ),  # Square root transformation
        # (
        #     [0.1, 0.5, 0.9],
        #     "cbrt",
        #     10,
        #     False,
        #     [[4.47, 4.47], [5.56, 5.56], [6.7, 6.7]],
        # ),  # Cube root transformation
        # ([0.1, 0.5, 0.9], None, 0, True, [[9.1, 9.14], [9.1, 9.3], [9.1, 9.46]]),  # noqa Include static data
    ],
)
def test_apply_qrf(
    tmp_path,
    quantiles,
    transformation,
    pre_transform_addition,
    include_static,
    expected,
):
    """Test the ApplyQuantileRegressionRandomForests plugin."""
    feature_config = {"wind_speed_at_10m": ["mean", "std", "latitude", "longitude"]}
    n_estimators = 2
    max_depth = 2
    random_state = 55
    compression = 5
    extra_kwargs = {}

    model_output = _run_train_qrf(
        tmp_path,
        feature_config,
        n_estimators,
        max_depth,
        random_state,
        compression,
        transformation,
        pre_transform_addition,
        extra_kwargs,
        include_static,
    )
    qrf_model = joblib.load(model_output)

    frt = "20170103T0000Z"
    vt = "20170103T1200Z"
    day_of_training_period = 2
    data = np.arange(6, (len(quantiles) * 6) + 1, 6)

    forecast_cube = _create_forecasts(frt, vt, data)
    forecast_cube = _add_day_of_training_period(
        forecast_cube, day_of_training_period, "forecast_reference_time"
    )

    feature_cubes = CubeList([forecast_cube])
    if include_static:
        ancil_cube = _create_ancil_file()
        feature_cubes.append(ancil_cube)

    plugin = ApplyQuantileRegressionRandomForests(
        feature_config, quantiles, transformation, pre_transform_addition
    )
    result = plugin.process(qrf_model, feature_cubes, forecast_cube)

    assert isinstance(result, Cube)
    assert result.shape == (len(quantiles), 2)
    assert result.dtype == np.float32
    expected_cube = forecast_cube.copy(
        data=np.full((len(quantiles), 2), expected, dtype=np.float32)
    )
    assert result.name() == expected_cube.name()
    assert result.units == expected_cube.units
    np.testing.assert_almost_equal(result.data, expected_cube.data, decimal=2)
