# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the Quantile Regression Random Forest plugins."""

import itertools
from datetime import datetime as dt

import iris
import joblib
import numpy as np
import pandas as pd
import pytest
from iris.cube import Cube
from iris.pandas import as_data_frame
from pandas.testing import assert_frame_equal

from improver.calibration.quantile_regression_random_forest import (
    ApplyQuantileRegressionRandomForests,
    TrainQuantileRegressionRandomForests,
    _check_valid_transformation,
    apply_transformation,
    prep_feature,
    prep_features_from_config,
    quantile_forest_package_available,
    sanitise_forecast_dataframe,
)
from improver.metadata.constants.time_types import DT_FORMAT
from improver.synthetic_data.set_up_test_cubes import set_up_spot_variable_cube

pytest.importorskip("quantile_forest")

ALTITUDE = [10, 20]
LATITUDE = [50, 60]
LONGITUDE = [0, 10]
WMO_ID = ["00001", "00002"]

iris.FUTURE.pandas_ndim = True


def _create_forecasts(
    forecast_reference_time: str,
    validity_time: str,
    data: list[int],
    representation: str = "realization",
    return_cube: bool = False,
) -> Cube | pd.DataFrame:
    """Create site forecast cube with realizations.

    Args:
        forecast_reference_time: Timestamp e.g. "20170101T0000Z".
        validity_time: Timestamp e.g. "20170101T0600Z".
        data: Data that will be repeated to create a cube with two sites. The
        length of the data will equal the number of realizations created.

    Returns:
        Forecast cube containing three percentiles and two sites.
    """
    data = np.array([data, data + 2], dtype=np.float32).T
    cube = set_up_spot_variable_cube(
        data,
        realizations=range(len(data)),
        name="wind_speed_at_10m",
        units="m s-1",
        wmo_ids=WMO_ID,
        unique_site_id=WMO_ID,
        unique_site_id_key="station_id",
        latitudes=np.array([50, 60], np.float32),
        longitudes=np.array([0, 10], np.float32),
        altitudes=np.array([10, 20], np.float32),
        time=dt.strptime(validity_time, DT_FORMAT),
        frt=dt.strptime(forecast_reference_time, DT_FORMAT),
    )
    if representation == "percentile":
        increment = (1 / (len(cube.coord("realization").points) + 1)) * 100
        cube.coord("realization").rename("percentile")
        percentiles = np.array(np.arange(increment, 100, increment), dtype=np.float32)
        cube.coord("percentile").points = percentiles

    if return_cube:
        return cube
    df = as_data_frame(
        cube,
        add_aux_coords=[
            "altitude",
            "latitude",
            "longitude",
            "wmo_id",
            "forecast_period",
            "forecast_reference_time",
            "time",
        ],
    ).reset_index()
    df["time"] = df["time"].apply(lambda x: x._to_real_datetime())
    df["forecast_reference_time"] = df["forecast_reference_time"].apply(
        lambda x: x._to_real_datetime()
    )
    return df


def _add_day_of_training_period(df):
    """Add day of training period coordinate to the dataframe.

    Args:
        df: DataFrame to which the day of training period coordinate will be added.
        day_of_training_period: Day of training period to be added.

    Returns:
        Cube with the day of training period coordinate added.
    """
    df["day_of_training_period"] = df["time"].dt.dayofyear - np.min(
        df["time"].dt.dayofyear
    )
    return df


def _create_ancil_file(return_cube=False):
    """Create an ancillary file for testing.

    Returns:
        An ancillary DataFrame without temporal columns.
    """
    data = np.array([2, 3], dtype=np.float32)
    template_cube = set_up_spot_variable_cube(
        data,
        wmo_ids=WMO_ID,
        unique_site_id=WMO_ID,
        unique_site_id_key="station_id",
        latitudes=np.array([50, 60], np.float32),
        longitudes=np.array([0, 10], np.float32),
        altitudes=np.array([10, 20], np.float32),
        name="distance_to_water",
        units="m",
    )
    cube = template_cube.copy()
    for coord in ["time", "forecast_reference_time", "forecast_period"]:
        cube.remove_coord(coord)
    if return_cube:
        return cube
    return as_data_frame(
        cube,
        add_aux_coords=[
            "altitude",
            "latitude",
            "longitude",
            "wmo_id",
        ],
    ).reset_index()


def _run_train_qrf(
    feature_config,
    n_estimators,
    max_depth,
    random_state,
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
    realization_data=[2, 6, 10],
    truth_data=[4.2, 3.8, 5.8, 6, 7, 7.3, 9.1, 9.5],
    tmp_path=None,
    compression=5,
    site_id="wmo_id",
):
    realization_data = np.array(realization_data, dtype=np.float32)
    forecast_dfs = []
    for index, (frt, vt) in enumerate(zip(forecast_reference_times, validity_times)):
        forecast_df = _create_forecasts(frt, vt, realization_data + index)
        forecast_dfs.append(forecast_df)
    forecast_df = pd.concat(forecast_dfs)
    forecast_df = _add_day_of_training_period(forecast_df)

    truth_df = forecast_df.copy()
    truth_df.drop(
        columns=[
            "forecast_period",
            "forecast_reference_time",
            "day_of_training_period",
            "realization",
            "wind_speed_at_10m",
        ],
        inplace=True,
    )
    truth_df.drop_duplicates(inplace=True)
    truth_df["ob_value"] = np.array(truth_data, dtype=np.float32)

    if include_static:
        ancil_df = _create_ancil_file()
        forecast_df = forecast_df.merge(
            ancil_df[[site_id, "distance_to_water"]], on=[site_id], how="left"
        )
        feature_config["distance_to_water"] = ["static"]
    if "air_temperature" in feature_config.keys():
        forecast_df["air_temperature"] = np.array(
            forecast_df["wind_speed_at_10m"] + 10, dtype=np.float32
        )

    plugin = TrainQuantileRegressionRandomForests(
        target_name="wind_speed_at_10m",
        feature_config=feature_config,
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        transformation=transformation,
        pre_transform_addition=pre_transform_addition,
        unique_site_id_keys=site_id,
        **extra_kwargs,
    )
    result = plugin.process(forecast_df, truth_df)
    if tmp_path is not None:
        model_output = tmp_path / "qrf_model.pickle"
        joblib.dump(result, model_output, compress=compression)
        return model_output
    return result


def test_quantile_forest_package_available():
    """Test the quantile_forest_package_available function."""
    result = quantile_forest_package_available()
    try:
        from quantile_forest import RandomForestQuantileRegressor  # noqa

        expected = True
    except ModuleNotFoundError:
        expected = False
    assert result == expected


@pytest.mark.parametrize("representation", ["percentile", "realization"])
@pytest.mark.parametrize(
    "feature_name,expected,expected_dtype",
    [
        ("mean", np.tile([6, 8], 3).astype(np.float32), np.float32),
        ("std", np.repeat(4, 6).astype(np.float32), np.float32),
        ("min", np.tile([2, 4], 3).astype(np.float32), np.float32),
        ("max", np.tile([10, 12], 3).astype(np.float32), np.float32),
        ("percentile_50", np.tile([6, 8], 3).astype(np.float32), np.float32),
        ("members_below_5", np.repeat(1, 6).astype(np.float32), np.float32),
        ("members_above_5", np.repeat(2, 6).astype(np.float32), np.float32),
        ("latitude", np.tile([50, 60], 3).astype(np.float32), np.float32),
        ("longitude", np.tile([0, 10], 3).astype(np.float32), np.float32),
        ("altitude", np.tile([10, 20], 3).astype(np.float32), np.float32),
        (
            "day_of_year",
            np.tile([1, 1], 3).astype(np.int32),
            np.int32,
        ),
        (
            "day_of_year_sin",
            np.tile([0.01716633, 0.01716633], 3).astype(np.float32),
            np.float32,
        ),
        (
            "day_of_year_cos",
            np.tile([0.99985266, 0.99985266], 3).astype(np.float32),
            np.float32,
        ),
        ("hour_of_day", np.repeat(12, 6).astype(np.int32), np.int32),
        ("hour_of_day_sin", np.repeat(0, 6).astype(np.float32), np.float32),
        ("hour_of_day_cos", np.repeat(-1, 6).astype(np.float32), np.float32),
        ("forecast_period", np.repeat(43200, 6).astype(np.int32), np.int32),
        ("day_of_training_period", np.repeat(0, 6).astype(np.int32), np.int32),
        ("static", np.tile([2, 3], 3).astype(np.float32), np.float32),
    ],
)
def test_prep_feature_single_time(
    representation, feature_name, expected, expected_dtype
):
    """Test the prep_feature function for a single time."""
    variable_name = "wind_speed_at_10m"

    forecast_reference_time = "20170101T0000Z"
    validity_time = "20170101T1200Z"
    data = np.array([2, 6, 10])
    forecast_df = _create_forecasts(
        forecast_reference_time, validity_time, data, representation
    )
    forecast_df = _add_day_of_training_period(forecast_df)

    if feature_name == "static":
        feature_df = _create_ancil_file()
        forecast_df = forecast_df.merge(
            feature_df[["wmo_id", "distance_to_water"]], on=["wmo_id"], how="left"
        )
    else:
        forecast_df = forecast_df.copy()

    result = prep_feature(forecast_df, variable_name, feature_name)

    if feature_name in [
        "mean",
        "std",
        "min",
        "max",
        "percentile_50",
        "members_below_5",
        "members_above_5",
    ]:
        assert result.shape == (6, 13)
        variable_name_modified = f"{variable_name}_{feature_name}"
    elif feature_name in [
        "day_of_year",
        "day_of_year_sin",
        "day_of_year_cos",
        "hour_of_day",
        "hour_of_day_sin",
        "hour_of_day_cos",
    ]:
        assert result.shape == (6, 13)
        variable_name_modified = feature_name
    elif feature_name in ["static"]:
        assert result.shape == (6, 13)
        variable_name_modified = "distance_to_water"
    else:
        assert result.shape == (6, 12)
        variable_name_modified = feature_name

    assert result[variable_name_modified].dtype == expected_dtype
    np.testing.assert_allclose(result[variable_name_modified], expected, atol=1e-6)


@pytest.mark.parametrize("scenario", ["uneven_percentiles1", "uneven_percentiles2"])
def test_prep_feature_invalid_percentiles(scenario):
    """Test that an error is raised if invalid percentiles are provided."""
    variable_name = "wind_speed_at_10m"

    forecast_reference_time = "20170101T0000Z"
    validity_time = "20170101T1200Z"
    data = np.array([2, 6, 10])
    forecast_df = _create_forecasts(
        forecast_reference_time, validity_time, data, representation="percentile"
    )
    forecast_df = _add_day_of_training_period(forecast_df)

    if scenario == "uneven_percentiles1":
        forecast_df.replace(to_replace={25: 10, 50: 20, 75: 20}, inplace=True)
    elif scenario == "uneven_percentiles2":
        forecast_df.replace(to_replace={25: 10, 75: 90}, inplace=True)

    with pytest.raises(
        ValueError, match="Forecast percentiles must be equally spaced."
    ):
        prep_feature(forecast_df, variable_name, "mean")


@pytest.mark.parametrize(
    "feature_name,expected,expected_dtype",
    [
        ("mean", np.tile([6, 8], 18).astype(np.float32), np.float32),
        ("std", np.repeat(4, 36).astype(np.float32), np.float32),
        ("min", np.tile([2, 4], 18).astype(np.float32), np.float32),
        ("max", np.tile(np.array([10, 12], dtype=np.float32), 18), np.float32),
        ("percentile_50", np.tile(np.array([6, 8], dtype=np.float32), 18), np.float32),
        ("members_below_5", np.repeat(1, 36).astype(np.float32), np.float32),
        ("members_above_5", np.repeat(2, 36).astype(np.float32), np.float32),
        ("latitude", np.tile([50, 60], 18).astype(np.float32), np.float32),
        ("longitude", np.tile([0, 10], 18).astype(np.float32), np.float32),
        ("altitude", np.tile([10, 20], 18).astype(np.float32), np.float32),
        (
            "day_of_year",
            np.repeat([1, 2, 3], 12).astype(np.int32),
            np.int32,
        ),
        (
            "day_of_year_sin",
            np.repeat([0.017166, 0.034328, 0.051479], 12).astype(np.float32),
            np.float32,
        ),
        (
            "day_of_year_cos",
            np.repeat([0.999853, 0.999411, 0.998674], 12).astype(np.float32),
            np.float32,
        ),
        ("hour_of_day", np.tile(np.repeat([6, 12], 6), 3).astype(np.int32), np.int32),
        (
            "hour_of_day_sin",
            np.tile(np.repeat([1, 0], 6), 3).astype(np.float32),
            np.float32,
        ),
        (
            "hour_of_day_cos",
            np.tile(np.repeat([0, -1], 6), 3).astype(np.float32),
            np.float32,
        ),
        (
            "forecast_period",
            np.tile(np.repeat([21600, 43200], 6), 3).astype(np.int32),
            np.int32,
        ),
        ("day_of_training_period", np.repeat([0, 1, 2], 12).astype(np.int32), np.int32),
        ("static", np.tile([2, 3], 18).astype(np.float32), np.float32),
    ],
)
def test_prep_feature_more_times(feature_name, expected, expected_dtype):
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

    data = np.array([2, 6, 10])

    variable_name = "wind_speed_at_10m"

    data = np.array([2, 6, 10])

    forecast_dfs = []
    for frt, vt in zip(forecast_reference_times, validity_times):
        forecast_dfs.append(_create_forecasts(frt, vt, data))
    forecast_df = pd.concat(forecast_dfs)
    forecast_df = _add_day_of_training_period(forecast_df)

    if feature_name == "static":
        feature_df = _create_ancil_file()
        forecast_df = forecast_df.merge(
            feature_df[["wmo_id", "distance_to_water"]], on=["wmo_id"], how="left"
        )

    else:
        forecast_df = forecast_df.copy()

    result = prep_feature(forecast_df, variable_name, feature_name)

    if feature_name in [
        "mean",
        "std",
        "min",
        "max",
        "percentile_50",
        "members_below_5",
        "members_above_5",
    ]:
        assert result.shape == (36, 13)
        variable_name_modified = f"{variable_name}_{feature_name}"
    elif feature_name in [
        "day_of_year",
        "day_of_year_sin",
        "day_of_year_cos",
        "hour_of_day",
        "hour_of_day_sin",
        "hour_of_day_cos",
    ]:
        assert result.shape == (36, 13)
        variable_name_modified = feature_name
    elif feature_name in ["static"]:
        assert result.shape == (36, 13)
        variable_name_modified = "distance_to_water"
    else:
        assert result.shape == (36, 12)
        variable_name_modified = feature_name

    assert result[variable_name_modified].dtype == expected_dtype
    np.testing.assert_allclose(result[variable_name_modified], expected, atol=1e-6)


@pytest.mark.parametrize(
    "feature_config",
    [
        {"wind_speed_at_10m": ["mean", "std", "latitude", "longitude"]},
        {"wind_speed_at_10m": ["latitude", "longitude"]},
        {
            "wind_speed_at_10m": ["latitude", "longitude"],
            "distance_to_water": ["static"],
        },
        {"wind_speed_at_10m": ["latitude", "longitude", "height"]},
    ],
)
def test_sanitise_forecast_dataframe(feature_config):
    """Test sanitise_forecast_dataframe function."""
    data_dict = {
        "wmo_id": np.tile(WMO_ID, 3),
        "latitude": np.tile(LATITUDE, 3),
        "longitude": np.tile(LONGITUDE, 3),
        "altitude": np.tile(ALTITUDE, 3),
        "wind_speed_at_10m_mean": np.repeat(5, 6),
        "wind_speed_at_10m_std": np.repeat(1, 6),
        "wind_speed_at_10m": np.tile([4, 6], 3),
        "realization": np.tile([1, 2], 3),
        "distance_to_water": np.tile([2.0, 3.0], 3),
    }
    df = pd.DataFrame(data_dict)

    expected = df.copy()
    if (
        "mean" in feature_config["wind_speed_at_10m"]
        or "std" in feature_config["wind_speed_at_10m"]
    ):
        expected.drop(
            columns=[
                "wind_speed_at_10m",
            ],
            inplace=True,
        )
    expected = expected[expected["realization"] == 1]
    expected = expected.drop(columns=["realization"])

    result = sanitise_forecast_dataframe(df, feature_config)
    assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "feature_config",
    [
        {"wind_speed_at_10m": ["mean", "std", "latitude", "longitude"]},
        {"wind_speed_at_10m": ["latitude", "longitude"]},
        {
            "wind_speed_at_10m": ["latitude", "longitude"],
            "distance_to_water": ["static"],
        },
        {"wind_speed_at_10m": ["latitude", "longitude", "height"]},
        {"wind_speed_at_10m": ["mean", "std"], "surface_temperature": ["mean"]},
    ],
)
def test_prep_features_from_config(feature_config):
    """Test the prep_features_from_config function."""
    data_dict = {
        "wmo_id": np.tile(WMO_ID, 3),
        "latitude": np.tile(LATITUDE, 3),
        "longitude": np.tile(LONGITUDE, 3),
        "altitude": np.tile(ALTITUDE, 3),
        "wind_speed_at_10m": np.tile([4, 6], 3),
        "realization": np.tile([1, 2], 3),
        "distance_to_water": np.tile([2.0, 3.0], 3),
        "blend_time": pd.Timestamp("2017-01-02 00:00:00", tz="utc"),
        "forecast_period": 6 * 3.6e12,
        "forecast_reference_time": pd.Timestamp("2017-01-02 00:00:00", tz="utc"),
    }
    df = pd.DataFrame(data_dict)

    expected = list(itertools.chain.from_iterable(feature_config.values()))
    if "mean" in expected:
        expected = ["wind_speed_at_10m_mean" if e == "mean" else e for e in expected]
    if "std" in expected:
        expected = ["wind_speed_at_10m_std" if e == "std" else e for e in expected]
    if "static" in expected:
        expected = ["distance_to_water" if e == "static" else e for e in expected]

    if "height" in expected:
        with pytest.raises(
            ValueError,
            match=(
                "Feature 'height' for variable 'wind_speed_at_10m' is not supported."
            ),
        ):
            prep_features_from_config(df, feature_config)
    elif "surface_temperature" in feature_config.keys():
        with pytest.raises(
            ValueError,
            match=(
                "Feature 'surface_temperature' is not present "
                "in the forecast DataFrame."
            ),
        ):
            prep_features_from_config(df, feature_config)
    else:
        _, result_names = prep_features_from_config(df, feature_config)
        assert result_names == expected


@pytest.mark.parametrize(
    "transformation", ["log", "log10", "sqrt", "cbrt", None, "yeojohnson"]
)
def test_check_valid_transformation(transformation):
    """Test the _check_valid_transformation function."""

    if transformation == "yeojohnson":
        with pytest.raises(
            ValueError, match="Currently the only supported transformations"
        ):
            _check_valid_transformation(transformation)
    else:
        result = _check_valid_transformation(transformation)
        assert result is None


@pytest.mark.parametrize("transformation", ["log", "log10", "sqrt", "cbrt", None])
def test_apply_transformation(transformation, pre_transform_addition=10):
    """Test the apply_transformation function."""
    data = np.array([0, 1, 2], dtype=np.float32)

    if transformation == "log":
        expected = np.log(data + pre_transform_addition)
    elif transformation == "log10":
        expected = np.log10(data + pre_transform_addition)
    elif transformation == "sqrt":
        expected = np.sqrt(data + pre_transform_addition)
    elif transformation == "cbrt":
        expected = np.cbrt(data + pre_transform_addition)
    else:
        expected = data

    result = apply_transformation(data, transformation, pre_transform_addition)
    np.testing.assert_allclose(result, expected, atol=1e-6)


@pytest.mark.parametrize(
    "n_estimators,max_depth,random_state,transformation,pre_transform_addition,extra_kwargs,include_static,expected",
    [
        (2, 2, 55, None, 0, {}, False, 4.2),  # Basic test case
        (2, 2, 54, None, 0, {}, False, 4.15),  # Different random state
        (1, 1, 54, None, 0, {}, False, 4.2),  # Fewer estimators and reduced depth
        (2, 2, 55, "log", 10, {}, False, 2.65),  # Log transformation
        (2, 2, 55, "log10", 10, {}, False, 1.15),  # Log10 transformation
        (2, 2, 55, "sqrt", 10, {}, False, 3.76),  # Square root transformation
        (2, 2, 55, "cbrt", 10, {}, False, 2.42),  # Cube root transformation
        (2, 2, 55, None, 0, {"criterion": "absolute_error"}, False, 4.2),  # noqa # Different criterion
        (2, 5, 55, None, 0, {}, True, 4.2),  # Include static data
    ],
)
def test_train_qrf_single_lead_times(
    n_estimators,
    max_depth,
    random_state,
    transformation,
    pre_transform_addition,
    extra_kwargs,
    include_static,
    expected,
):
    """Test the TrainQuantileRegressionRandomForests plugin when the forecast cube
    for training contains a single lead time."""

    feature_config = {"wind_speed_at_10m": ["mean", "std", "latitude", "longitude"]}

    qrf_model = _run_train_qrf(
        feature_config,
        n_estimators,
        max_depth,
        random_state,
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
        realization_data=[2, 6, 10],
        truth_data=[4.2, 4.1, 4.2, 4.1],
    )

    assert qrf_model.n_estimators == n_estimators
    assert qrf_model.max_depth == max_depth
    assert qrf_model.random_state == random_state

    current_forecast = [7.5, 3, 55, 5]
    if include_static:
        current_forecast.append(2.5)
    result = qrf_model.predict(
        np.expand_dims(np.array(current_forecast), 0), quantiles=[0.5]
    )
    np.testing.assert_almost_equal(result, expected, decimal=2)


@pytest.mark.parametrize(
    "n_estimators,max_depth,random_state,transformation,pre_transform_addition,extra_kwargs,include_static,expected",
    [
        (2, 2, 55, None, 0, {}, False, 5.8),  # Basic test case
        (1, 1, 55, None, 0, {}, False, 3.8),  # Fewer estimators and reduced depth
        (1, 1, 73, None, 0, {}, False, 5.8),  # Different random state
        (2, 2, 55, "log", 10, {}, False, 2.752),  # Log transformation
        (2, 2, 55, "log10", 10, {}, False, 1.195),  # Log10 transformation
        (2, 2, 55, "sqrt", 10, {}, False, 3.964),  # Square root transformation
        (2, 2, 55, "cbrt", 10, {}, False, 2.504),  # Cube root transformation
        (2, 2, 55, None, 0, {"criterion": "absolute_error"}, False, 5.8),  # noqa # Different criterion
        (1, 1, 73, None, 0, {}, True, 3.8),  # Include static data
    ],
)
def test_train_qrf_multiple_lead_times(
    n_estimators,
    max_depth,
    random_state,
    transformation,
    pre_transform_addition,
    extra_kwargs,
    include_static,
    expected,
):
    """Test the TrainQuantileRegressionRandomForests plugin when multiple lead times
    are provided in the forecast cube for training."""

    feature_config = {"wind_speed_at_10m": ["mean", "std", "latitude", "longitude"]}

    qrf_model = _run_train_qrf(
        feature_config,
        n_estimators,
        max_depth,
        random_state,
        transformation,
        pre_transform_addition,
        extra_kwargs,
        include_static,
    )

    assert qrf_model.n_estimators == n_estimators
    assert qrf_model.max_depth == max_depth
    assert qrf_model.random_state == random_state

    current_forecast = [7.5, 3, 55, 5]
    if include_static:
        current_forecast.append(3.2)
    result = qrf_model.predict(
        np.expand_dims(np.array(current_forecast), 0), quantiles=[0.5]
    )
    np.testing.assert_almost_equal(result, expected, decimal=3)


@pytest.mark.parametrize(
    "feature_config,data,include_static,site_id,expected",
    [
        ({"wind_speed_at_10m": ["mean"]}, [5], False, "wmo_id", [5]),  # One feature
        ({"wind_speed_at_10m": ["mean"]}, [5], False, "station_id", [5]),  # One feature
        ({"wind_speed_at_10m": ["latitude"]}, [61], False, "wmo_id", [7.75]),  # noqa Without the target
        ({"wind_speed_at_10m": ["mean"]}, [5], True, "wmo_id", [4]),  # With static data
        (
            {"wind_speed_at_10m": ["mean"], "air_temperature": ["mean"]},
            [5],
            False,
            "wmo_id",
            [5],
        ),  # Multiple dynamic features
        (
            {"wind_speed_at_10m": ["mean"], "pressure_at_mean_sea_level": ["mean"]},
            [5],
            False,
            "wmo_id",
            "Feature 'pressure_at_mean_sea_level' is not present",
        ),  # Multiple dynamic features
    ],
)
def test_alternative_feature_configs(
    feature_config,
    data,
    include_static,
    site_id,
    expected,
):
    """Test the TrainQuantileRegressionRandomForests plugin for a few different
    configurations of the feature_config dictionary."""
    n_estimators = 2
    max_depth = 5
    random_state = 55
    extra_kwargs = {}
    transformation = None
    pre_transform_addition = 0

    if "pressure_at_mean_sea_level" in feature_config:
        with pytest.raises(ValueError, match=expected):
            _run_train_qrf(
                feature_config,
                n_estimators,
                max_depth,
                random_state,
                transformation,
                pre_transform_addition,
                extra_kwargs,
                include_static,
            )
        return

    qrf_model = _run_train_qrf(
        feature_config,
        n_estimators,
        max_depth,
        random_state,
        transformation,
        pre_transform_addition,
        extra_kwargs,
        include_static,
        site_id=site_id,
    )

    assert qrf_model.n_estimators == n_estimators
    assert qrf_model.max_depth == max_depth
    assert qrf_model.random_state == random_state

    current_forecast = [*data]
    if include_static:
        current_forecast.append(3.2)
    if "air_temperature" in feature_config:
        current_forecast.append(15.0)

    result = qrf_model.predict(
        np.expand_dims(np.array(current_forecast), 0), quantiles=[0.5]
    )
    np.testing.assert_almost_equal(result, expected, decimal=2)


@pytest.mark.parametrize(
    "quantiles,transformation,pre_transform_addition,include_static,expected",
    [
        ([0.5], None, 0, False, [5, 4.9]),  # One quantile
        ([0.1, 0.5, 0.9], None, 0, False, [[7.25, 8.25, 9.25], [6.35, 7.75, 9.14]]),  # noqa Multiple quantiles
        ([0.1, 0.5, 0.9], "log", 10, False, [[4.48, 5.67, 6.96], [4.48, 5.67, 6.96]]),  # noqa Log transformation
        (
            [0.1, 0.5, 0.9],
            "log10",
            10,
            False,
            [[4.48, 5.67, 6.96], [4.48, 5.67, 6.96]],
        ),  # Log10 transformation
        (
            [0.1, 0.5, 0.9],
            "sqrt",
            10,
            False,
            [[4.5, 5.71, 6.98], [4.5, 5.71, 6.98]],
        ),  # Square root transformation
        (
            [0.1, 0.5, 0.9],
            "cbrt",
            10,
            False,
            [[4.49, 5.7, 6.97], [4.49, 5.7, 6.97]],
        ),  # Cube root transformation
        ([0.1, 0.5, 0.9], None, 0, True, [[7.25, 8.25, 9.25], [6.35, 7.75, 9.15]]),  # noqa Include static data
    ],
)
def test_apply_qrf(
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
    extra_kwargs = {}

    qrf_model = _run_train_qrf(
        feature_config,
        n_estimators,
        max_depth,
        random_state,
        transformation,
        pre_transform_addition,
        extra_kwargs,
        include_static,
    )

    frt = "20170103T0000Z"
    vt = "20170103T1200Z"
    data = np.arange(6, (len(quantiles) * 6) + 1, 6)

    forecast_df = _create_forecasts(frt, vt, data)
    forecast_df = _add_day_of_training_period(forecast_df)

    if include_static:
        ancil_df = _create_ancil_file()
        forecast_df = forecast_df.merge(
            ancil_df[["wmo_id", "distance_to_water"]], on=["wmo_id"], how="left"
        )

    plugin = ApplyQuantileRegressionRandomForests(
        "air_temperature",
        feature_config,
        quantiles,
        transformation,
        pre_transform_addition,
    )
    result = plugin.process(qrf_model, forecast_df)

    assert isinstance(result, np.ndarray)
    if len(quantiles) == 3:
        assert result.shape == (2, 3)
    else:
        assert result.shape == (2,)
    assert result.dtype == np.float32
    # expected_data = np.full((len(quantiles), 2), expected, dtype=np.float32)
    np.testing.assert_almost_equal(result, expected, decimal=2)
