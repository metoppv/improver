# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the LoadAndTrainQRF plugin."""

import iris
import joblib
import numpy as np
import pandas as pd
import pytest

from improver.calibration import FORECAST_SCHEMA
from improver.calibration.load_and_train_quantile_regression_random_forest import (
    LoadAndTrainQRF,
)
from improver.synthetic_data.set_up_test_cubes import set_up_spot_variable_cube

ALTITUDE = [10, 20]
LATITUDE = [50, 60]
LONGITUDE = [0, 10]
WMO_ID = ["03001", "03002", "03003", "03004", "03005"]


def _create_multi_site_forecast_parquet_file(tmp_path):
    """Create a Parquet file with forecast data."""

    data_dict = {
        "percentile": np.repeat(50, 5),
        "forecast": [281, 272, 287, 280, 290],
        "altitude": [10, 83, 56, 23, 2],
        "blend_time": [pd.Timestamp("2017-01-02 00:00:00")] * 5,
        "forecast_period": [pd.Timedelta(nanoseconds=6 * 3600 * 1e12)] * 5,
        "forecast_reference_time": [pd.Timestamp("2017-01-02 00:00:00")] * 5,
        "latitude": [60.1, 59.9, 59.7, 58, 57],
        "longitude": [1, 2, -1, -2, -3],
        "time": [pd.Timestamp("2017-01-02 06:00:00")] * 5,
        "wmo_id": ["03001", "03002", "03003", "03004", "03005"],
        "station_id": ["03001", "03002", "03003", "03004", "03005"],
        "cf_name": ["air_temperature"] * 5,
        "units": ["K"] * 5,
        "experiment": ["latestblend"] * 5,
        "period": [pd.NaT] * 5,
        "height": [1.5] * 5,
        "diagnostic": ["temperature_at_screen_level"] * 5,
    }
    # Add wind speed to demonstrate filtering.
    wind_speed_dict = data_dict.copy()
    wind_speed_dict["forecast"] = [8, 19, 16, 12, 10]
    wind_speed_dict["cf_name"] = "wind_speed"
    wind_speed_dict["units"] = "m s-1"
    wind_speed_dict["diagnostic"] = "wind_speed_at_10m"
    data_df = pd.DataFrame(data_dict)
    wind_speed_df = pd.DataFrame(wind_speed_dict)
    joined_df = pd.concat([data_df, wind_speed_df], ignore_index=True)

    joined_df["forecast_period"] = joined_df["forecast_period"].astype(
        "timedelta64[ms]"
    )
    output_dir = tmp_path / "forecast_parquet_files"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = str(output_dir / "forecast.parquet")
    joined_df.to_parquet(
        output_path, index=False, engine="pyarrow", schema=FORECAST_SCHEMA
    )
    return output_dir, data_dict["wmo_id"]


def _create_multi_percentile_forecast_parquet_file(tmp_path):
    """Create a Parquet file with forecast data."""

    data_dict = {
        "percentile": [16 + 2 / 3, 33 + 1 / 3, 50, 66 + 2 / 3, 83 + 1 / 3],
        "forecast": [272, 274, 275, 277, 280],
        "altitude": [10, 10, 10, 10, 10],
        "blend_time": [pd.Timestamp("2017-01-02 00:00:00")] * 5,
        "forecast_period": [pd.Timedelta(nanoseconds=6 * 3600 * 1e12)] * 5,
        "forecast_reference_time": [pd.Timestamp("2017-01-02 00:00:00")] * 5,
        "latitude": [60.1, 60.1, 60.1, 60.1, 60.1],
        "longitude": [1, 1, 1, 1, 1],
        "time": [pd.Timestamp("2017-01-02 06:00:00")] * 5,
        "wmo_id": ["03001", "03001", "03001", "03001", "03001"],
        "station_id": ["03001", "03001", "03001", "03001", "03001"],
        "cf_name": ["air_temperature"] * 5,
        "units": ["K"] * 5,
        "experiment": ["latestblend"] * 5,
        "period": [pd.NaT] * 5,
        "height": [1.5] * 5,
        "diagnostic": ["temperature_at_screen_level"] * 5,
    }
    # Add wind speed to demonstrate filtering.
    wind_speed_dict = data_dict.copy()
    wind_speed_dict["forecast"] = [6, 10, 11, 12, 15]
    wind_speed_dict["cf_name"] = "wind_speed"
    wind_speed_dict["units"] = "m s-1"
    wind_speed_dict["diagnostic"] = "wind_speed_at_10m"
    data_df = pd.DataFrame(data_dict)
    wind_speed_df = pd.DataFrame(wind_speed_dict)
    joined_df = pd.concat([data_df, wind_speed_df], ignore_index=True)
    joined_df["forecast_period"] = joined_df["forecast_period"].astype(
        "timedelta64[ms]"
    )

    output_dir = tmp_path / "forecast_parquet_files"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = str(output_dir / "forecast.parquet")
    joined_df.to_parquet(output_path, index=False, engine="pyarrow")
    return output_dir, data_dict["wmo_id"]


def _create_multi_forecast_period_forecast_parquet_file(tmp_path):
    """Create a Parquet file with forecast data."""

    data_dict = {
        "percentile": [50, 50, 50, 50],
        "forecast": [277, 270, 280, 269],
        "altitude": [10, 83, 10, 83],
        "blend_time": [pd.Timestamp("2017-01-02 00:00:00")] * 4,
        "forecast_period": np.repeat(
            [
                pd.Timedelta(nanoseconds=6 * 3600 * 1e12),
                pd.Timedelta(nanoseconds=12 * 3600 * 1e12),
            ],
            2,
        ),
        "forecast_reference_time": [pd.Timestamp("2017-01-02 00:00:00")] * 4,
        "latitude": [60.1, 59.9, 60.1, 59.9],
        "longitude": [1, 2, 1, 2],
        "time": np.repeat(
            [pd.Timestamp("2017-01-02 06:00:00"), pd.Timestamp("2017-01-02 12:00:00")],
            2,
        ),
        "wmo_id": ["03001", "03002", "03001", "03002"],
        "station_id": ["03001", "03002", "03001", "03002"],
        "cf_name": ["air_temperature"] * 4,
        "units": ["K"] * 4,
        "experiment": ["latestblend"] * 4,
        "period": [pd.NaT] * 4,
        "height": [1.5] * 4,
        "diagnostic": ["temperature_at_screen_level"] * 4,
    }
    # Add wind speed to demonstrate filtering.
    wind_speed_dict = data_dict.copy()
    wind_speed_dict["forecast"] = [6, 16, 12, 15]
    wind_speed_dict["cf_name"] = "wind_speed"
    wind_speed_dict["units"] = "m s-1"
    wind_speed_dict["diagnostic"] = "wind_speed_at_10m"
    data_df = pd.DataFrame(data_dict)
    wind_speed_df = pd.DataFrame(wind_speed_dict)
    joined_df = pd.concat([data_df, wind_speed_df], ignore_index=True)
    joined_df["forecast_period"] = joined_df["forecast_period"].astype(
        "timedelta64[ms]"
    )

    output_dir = tmp_path / "forecast_parquet_files"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = str(output_dir / "forecast.parquet")
    joined_df.to_parquet(output_path, index=False, engine="pyarrow")
    return output_dir, data_dict["wmo_id"]


def _create_multi_site_truth_parquet_file(tmp_path):
    data_dict = {
        "diagnostic": ["temperature_at_screen_level"] * 5,
        "latitude": [60.1, 59.9, 59.7, 58, 57],
        "longitude": [1, 2, -1, -2, -3],
        "altitude": [10, 83, 56, 23, 2],
        "time": [pd.Timestamp("2017-01-02 06:00:00")] * 5,
        "wmo_id": ["03001", "03002", "03003", "03004", "03005"],
        "ob_value": [276, 270, 289, 290, 301],
    }
    wind_speed_dict = data_dict.copy()
    wind_speed_dict["ob_value"] = [3, 22, 24, 11, 9]
    wind_speed_dict["diagnostic"] = "wind_speed_at_10m"
    data_df = pd.DataFrame(data_dict)
    wind_speed_df = pd.DataFrame(wind_speed_dict)
    joined_df = pd.concat([data_df, wind_speed_df], ignore_index=True)

    output_dir = tmp_path / "truth_parquet_files"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = str(output_dir / "truth.parquet")
    joined_df.to_parquet(output_path, index=False, engine="pyarrow")
    return output_dir


def _create_multi_percentile_truth_parquet_file(tmp_path):
    data_dict = {
        "diagnostic": ["temperature_at_screen_level"],
        "latitude": [60.1],
        "longitude": [1],
        "altitude": [10],
        "time": [pd.Timestamp("2017-01-02 06:00:00")],
        "wmo_id": ["03001"],
        "ob_value": [276],
    }
    wind_speed_dict = data_dict.copy()
    wind_speed_dict["ob_value"] = [9]
    wind_speed_dict["diagnostic"] = "wind_speed_at_10m"
    data_df = pd.DataFrame(data_dict)
    wind_speed_df = pd.DataFrame(wind_speed_dict)
    joined_df = pd.concat([data_df, wind_speed_df], ignore_index=True)

    output_dir = tmp_path / "truth_parquet_files"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = str(output_dir / "truth.parquet")
    joined_df.to_parquet(output_path, index=False, engine="pyarrow")
    return output_dir


def _create_multi_forecast_period_truth_parquet_file(tmp_path):
    data_dict = {
        "diagnostic": ["temperature_at_screen_level"] * 4,
        "latitude": [60.1, 59.9, 60.1, 59.9],
        "longitude": [1, 2, 1, 2],
        "altitude": [10, 83, 10, 83],
        "time": np.repeat(
            [pd.Timestamp("2017-01-02 06:00:00"), pd.Timestamp("2017-01-02 12:00:00")],
            2,
        ),
        "wmo_id": ["03001", "03002", "03001", "03002"],
        "ob_value": [280, 273, 284, 275],
    }
    wind_speed_dict = data_dict.copy()
    wind_speed_dict["ob_value"] = [2, 11, 10, 14]
    wind_speed_dict["diagnostic"] = "wind_speed_at_10m"
    data_df = pd.DataFrame(data_dict)
    wind_speed_df = pd.DataFrame(wind_speed_dict)
    joined_df = pd.concat([data_df, wind_speed_df], ignore_index=True)

    output_dir = tmp_path / "truth_parquet_files"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = str(output_dir / "truth.parquet")
    joined_df.to_parquet(output_path, index=False, engine="pyarrow")
    return output_dir


def _create_ancil_file(tmp_path, wmo_ids):
    """Create an ancillary file for testing.

    Returns:
        An ancillary cube with a single value.
    """
    data = np.array(range(len(wmo_ids)), dtype=np.float32)
    template_cube = set_up_spot_variable_cube(
        data,
        wmo_ids=wmo_ids,
        name="distance_to_water",
        units="m",
    )
    cube = template_cube.copy()
    for coord in ["time", "forecast_reference_time", "forecast_period"]:
        cube.remove_coord(coord)
    output_dir = tmp_path / "ancil_files"
    output_dir.mkdir(parents=True)
    output_path = str(output_dir / "ancil.nc")
    iris.save(cube, output_path)
    return output_path


@pytest.mark.parametrize(
    "forecast_creation,truth_creation,include_static,expected",
    [
        (
            _create_multi_site_forecast_parquet_file,
            _create_multi_site_truth_parquet_file,
            False,
            5.6,
        ),
        (
            _create_multi_site_forecast_parquet_file,
            _create_multi_site_truth_parquet_file,
            True,
            5.64,
        ),
        (
            _create_multi_percentile_forecast_parquet_file,
            _create_multi_percentile_truth_parquet_file,
            False,
            5.62,
        ),
        (
            _create_multi_percentile_forecast_parquet_file,
            _create_multi_percentile_truth_parquet_file,
            True,
            5.62,
        ),
        (
            _create_multi_forecast_period_forecast_parquet_file,
            _create_multi_forecast_period_truth_parquet_file,
            False,
            5.61,
        ),
        (
            _create_multi_forecast_period_forecast_parquet_file,
            _create_multi_forecast_period_truth_parquet_file,
            True,
            5.64,
        ),
    ],
)
def test_load_and_train_qrf(
    tmp_path, forecast_creation, truth_creation, include_static, expected
):
    feature_config = {"air_temperature": ["mean", "std", "altitude"]}
    n_estimators = 2
    max_depth = 5
    random_state = 46

    forecast_path, wmo_ids = forecast_creation(tmp_path)
    truth_path = truth_creation(tmp_path)
    file_paths = [forecast_path, truth_path]

    model_output_dir = tmp_path / "train_qrf"
    model_output_dir.mkdir(parents=True)
    model_output = str(model_output_dir / "qrf_model.pkl")

    if include_static:
        ancil_path = _create_ancil_file(tmp_path, list(set(wmo_ids)))
        file_paths.append(ancil_path)
        feature_config["distance_to_water"] = ["static"]

    # Create an instance of LoadAndTrainQRF with the required parameters
    plugin = LoadAndTrainQRF(
        experiment="latestblend",
        feature_config=feature_config,
        target_diagnostic_name="temperature_at_screen_level",
        forecast_periods="6:18:6",
        cycletime="20170103T0000Z",
        training_length=2,
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        transformation="log",
        pre_transform_addition=1,
    )
    plugin(file_paths, model_output=model_output)

    qrf_model = joblib.load(model_output)

    assert qrf_model.n_estimators == n_estimators
    assert qrf_model.max_depth == max_depth
    assert qrf_model.random_state == random_state

    current_forecast = [279, 3, 55]
    if include_static:
        current_forecast.append(2.5)
    result = qrf_model.predict(
        np.expand_dims(np.array(current_forecast), 0), quantiles=[0.5]
    )
    np.testing.assert_almost_equal(result, expected, decimal=2)


@pytest.mark.parametrize("make_files", [(False, True)])
def test_load_and_train_qrf_no_paths(tmp_path, make_files):
    feature_config = {"air_temperature": ["mean", "std", "altitude"]}
    n_estimators = 2
    max_depth = 5
    random_state = 46

    file_paths = [
        tmp_path / "partition" / "forecast_table/",
        tmp_path / "partition" / "truth_table/",
    ]
    if make_files:
        for file_path in file_paths:
            (tmp_path / file_path).mkdir(parents=True, exist_ok=True)

    model_output_dir = tmp_path / "train_qrf"
    model_output_dir.mkdir(parents=True)
    model_output = str(model_output_dir / "qrf_model.pkl")

    plugin = LoadAndTrainQRF(
        experiment="latestblend",
        feature_config=feature_config,
        target_diagnostic_name="temperature_at_screen_level",
        forecast_periods="6:12:6",
        cycletime="20170102T0000Z",
        training_length=2,
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        transformation="log",
        pre_transform_addition=1,
    )
    result = plugin(file_paths, model_output=model_output)
    # Expecting None since no valid paths are provided
    assert result is None
    # Check if the model output file is not created
    assert not (model_output_dir / "qrf_model.pkl").exists()
