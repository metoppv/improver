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

from improver.calibration.load_and_train_quantile_regression_random_forest import (
    LoadAndTrainQRF,
)
from improver.calibration.quantile_regression_random_forest import (
    quantile_forest_package_available,
)
from improver.synthetic_data.set_up_test_cubes import set_up_spot_variable_cube

pytest.importorskip("quantile_forest")


ALTITUDE = [10, 20]
LATITUDE = [50, 60]
LONGITUDE = [0, 10]
WMO_ID = ["03001", "03002", "03003", "03004", "03005"]


def _create_multi_site_forecast_parquet_file(tmp_path, representation="percentile"):
    """Create a parquet file with multi-site forecast data."""

    data_dict = {
        "percentile": np.repeat(50, 5),
        "forecast": [281, 272, 287, 280, 290],
        "altitude": [10, 83, 56, 23, 2],
        "blend_time": [pd.Timestamp("2017-01-02 00:00:00")] * 5,
        "forecast_period": [6 * 3.6e12] * 5,
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
    if representation == "realization":
        data_dict["realization"] = list(range(len(data_dict["percentile"])))
        data_dict.pop("percentile")
    elif representation == "kittens":
        data_dict["kittens"] = list(range(len(data_dict["percentile"])))
        data_dict.pop("percentile")

    # Add wind speed to demonstrate filtering.
    wind_speed_dict = data_dict.copy()
    wind_speed_dict["forecast"] = [8, 19, 16, 12, 10]
    wind_speed_dict["cf_name"] = "wind_speed"
    wind_speed_dict["units"] = "m s-1"
    wind_speed_dict["diagnostic"] = "wind_speed_at_10m"
    data_df = pd.DataFrame(data_dict)
    wind_speed_df = pd.DataFrame(wind_speed_dict)
    joined_df = pd.concat([data_df, wind_speed_df], ignore_index=True)

    output_dir = tmp_path / "forecast_parquet_files"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = str(output_dir / "forecast.parquet")
    joined_df.to_parquet(output_path, index=False, engine="pyarrow")
    return output_dir, data_dict["wmo_id"]


def _create_multi_percentile_forecast_parquet_file(tmp_path, representation=None):
    """Create a parquet file with multi-percentile forecast data."""

    data_dict = {
        "percentile": [16 + 2 / 3, 33 + 1 / 3, 50, 66 + 2 / 3, 83 + 1 / 3],
        "forecast": [272, 274, 275, 277, 280],
        "altitude": [10, 10, 10, 10, 10],
        "blend_time": [pd.Timestamp("2017-01-02 00:00:00")] * 5,
        "forecast_period": [6 * 3.6e12] * 5,
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

    output_dir = tmp_path / "forecast_parquet_files"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = str(output_dir / "forecast.parquet")
    joined_df.to_parquet(output_path, index=False, engine="pyarrow")
    return output_dir, data_dict["wmo_id"]


def _create_multi_forecast_period_forecast_parquet_file(tmp_path, representation=None):
    """Create a parquet file with multi-forecast period forecast data."""

    data_dict = {
        "percentile": [50, 50, 50, 50],
        "forecast": [277, 270, 280, 269],
        "altitude": [10, 83, 10, 83],
        "blend_time": [pd.Timestamp("2017-01-02 00:00:00")] * 4,
        "forecast_period": np.repeat(
            [
                [6 * 3.6e12],
                [12 * 3.6e12],
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

    output_dir = tmp_path / "forecast_parquet_files"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = str(output_dir / "forecast.parquet")
    joined_df.to_parquet(output_path, index=False, engine="pyarrow")
    return output_dir, data_dict["wmo_id"]


def _create_multi_site_truth_parquet_file(tmp_path):
    """Create a parquet file with multi-site truth data."""
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
    """Create a parquet file with multi-percentile truth data."""
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
    """Create a parquet file with multi-forecast period truth data."""
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


def _create_multi_site_truth_parquet_file_alt(tmp_path):
    """Create a parquet file with multi-site truth data for wind speed."""
    data_dict = {
        "diagnostic": ["wind_speed_at_10m"] * 5,
        "latitude": [60.1, 59.9, 59.7, 58, 57],
        "longitude": [1, 2, -1, -2, -3],
        "altitude": [10, 83, 56, 23, 2],
        "time": [pd.Timestamp("2017-01-02 06:00:00")] * 5,
        "wmo_id": ["03001", "03002", "03003", "03004", "03005"],
        "ob_value": [10, 25, 4, 3, 11],
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
    "forecast_creation,truth_creation,forecast_periods,include_static,remove_target,representation,expected",
    [
        (
            _create_multi_site_forecast_parquet_file,
            _create_multi_site_truth_parquet_file,
            "6:18:6",
            False,
            False,
            "percentile",
            5.6,
        ),
        (
            _create_multi_site_forecast_parquet_file,
            _create_multi_site_truth_parquet_file,
            "6:18:6",
            True,
            False,
            "percentile",
            5.64,
        ),
        (
            _create_multi_percentile_forecast_parquet_file,
            _create_multi_percentile_truth_parquet_file,
            "6:18:6",
            False,
            False,
            "percentile",
            5.62,
        ),
        (
            _create_multi_percentile_forecast_parquet_file,
            _create_multi_percentile_truth_parquet_file,
            "6:18:6",
            True,
            False,
            "percentile",
            5.62,
        ),
        (
            _create_multi_forecast_period_forecast_parquet_file,
            _create_multi_forecast_period_truth_parquet_file,
            "6:18:6",
            False,
            False,
            "percentile",
            5.61,
        ),
        (
            _create_multi_forecast_period_forecast_parquet_file,
            _create_multi_forecast_period_truth_parquet_file,
            "6:18:6",
            True,
            True,  # Remove target feature
            "percentile",
            5.62,
        ),
        (
            _create_multi_site_forecast_parquet_file,
            _create_multi_site_truth_parquet_file,
            "6:18:6",
            False,
            False,
            "realization",  # Provide realization input
            5.62,
        ),
        (
            _create_multi_forecast_period_forecast_parquet_file,
            _create_multi_forecast_period_truth_parquet_file,
            "12",
            False,
            False,
            "percentile",
            5.64,
        ),
    ],
)
def test_load_and_train_qrf(
    tmp_path,
    forecast_creation,
    truth_creation,
    forecast_periods,
    include_static,
    remove_target,
    representation,
    expected,
):
    """Test the LoadAndTrainQRF plugin."""
    feature_config = {"air_temperature": ["mean", "std", "altitude"]}
    n_estimators = 2
    max_depth = 5
    random_state = 46

    forecast_path, wmo_ids = forecast_creation(tmp_path, representation)
    truth_path = truth_creation(tmp_path)
    file_paths = [forecast_path, truth_path]

    model_output_dir = tmp_path / "train_qrf"
    model_output_dir.mkdir(parents=True)
    model_output = str(model_output_dir / "qrf_model.pkl")

    if include_static:
        ancil_path = _create_ancil_file(tmp_path, sorted(list(set(wmo_ids))))
        file_paths.append(ancil_path)
        feature_config["distance_to_water"] = ["static"]

    if remove_target:
        feature_config.pop("air_temperature")

    # Create an instance of LoadAndTrainQRF with the required parameters
    plugin = LoadAndTrainQRF(
        experiment="latestblend",
        feature_config=feature_config,
        target_diagnostic_name="temperature_at_screen_level",
        target_cf_name="air_temperature",
        forecast_periods=forecast_periods,
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

    if remove_target:
        current_forecast = []
    else:
        current_forecast = [5.64, 3, 55]

    if include_static:
        current_forecast.append(2.5)

    result = qrf_model.predict(
        np.expand_dims(np.array(current_forecast), 0), quantiles=[0.5]
    )
    np.testing.assert_almost_equal(result, expected, decimal=2)


@pytest.mark.parametrize("make_files", [(False, True)])
def test_load_and_train_qrf_no_paths(tmp_path, make_files):
    """Test the LoadAndTrainQRF plugin when the no valid file paths are provided.
    Either the paths do not exist, or the paths exist but the directories are empty."""
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
        target_cf_name="air_temperature",
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


@pytest.mark.parametrize(
    "cycletime,forecast_periods",
    [
        ("20200102T0000Z", "6:12:6"),
        ("20170102T0000Z", "30:36:6"),
    ],
)
def test_load_and_train_qrf_mismatches(tmp_path, cycletime, forecast_periods):
    """Test the LoadAndTrainQRF plugin when the cycletime or forecast_periods
    requested are not present in the provided files."""
    feature_config = {"air_temperature": ["mean", "std", "altitude"]}
    n_estimators = 2
    max_depth = 5
    random_state = 46

    file_paths = [
        tmp_path / "partition" / "forecast_table/",
        tmp_path / "partition" / "truth_table/",
    ]

    model_output_dir = tmp_path / "train_qrf"
    model_output_dir.mkdir(parents=True)
    model_output = str(model_output_dir / "qrf_model.pkl")

    plugin = LoadAndTrainQRF(
        experiment="latestblend",
        feature_config=feature_config,
        target_diagnostic_name="temperature_at_screen_level",
        target_cf_name="air_temperature",
        forecast_periods=forecast_periods,
        cycletime=cycletime,
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


@pytest.mark.parametrize(
    "exception,forecast_creation,truth_creation,forecast_periods,representation",
    [
        (
            "non_matching_truth",
            _create_multi_site_forecast_parquet_file,
            _create_multi_site_truth_parquet_file_alt,
            "6:18:6",
            "percentile",
        ),
        (
            "missing_static_feature",
            _create_multi_site_forecast_parquet_file,
            _create_multi_site_truth_parquet_file,
            "6:18:6",
            "percentile",
        ),
        (
            "missing_dynamic_feature",
            _create_multi_site_forecast_parquet_file,
            _create_multi_site_truth_parquet_file,
            "6:18:6",
            "percentile",
        ),
        (
            "no_percentile_realization",
            _create_multi_site_forecast_parquet_file,
            _create_multi_site_truth_parquet_file,
            "6:18:6",
            "kittens",
        ),
        (
            "alternative_forecast_period",
            _create_multi_site_forecast_parquet_file,
            _create_multi_site_truth_parquet_file,
            "6,12",
            "percentile",
        ),
        (
            "no_quantile_forest_package",
            _create_multi_site_forecast_parquet_file,
            _create_multi_site_truth_parquet_file,
            "6:18:6",
            "percentile",
        ),
    ],
)
def test_unexpected(
    tmp_path,
    exception,
    forecast_creation,
    truth_creation,
    forecast_periods,
    representation,
):
    """Test LoadAndTrainQRF plugin behaviour in atypical situations."""
    feature_config = {"air_temperature": ["mean", "std", "altitude"]}
    n_estimators = 2
    max_depth = 5
    random_state = 46

    forecast_path, _ = forecast_creation(tmp_path, representation)
    truth_path = truth_creation(tmp_path)
    file_paths = [forecast_path, truth_path]

    model_output_dir = tmp_path / "train_qrf"
    model_output_dir.mkdir(parents=True)
    model_output = str(model_output_dir / "qrf_model.pkl")

    # Create an instance of LoadAndTrainQRF with the required parameters
    plugin = LoadAndTrainQRF(
        experiment="latestblend",
        feature_config=feature_config,
        target_diagnostic_name="temperature_at_screen_level",
        target_cf_name="air_temperature",
        forecast_periods=forecast_periods,
        cycletime="20170103T0000Z",
        training_length=2,
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        transformation="log",
        pre_transform_addition=1,
    )

    if exception == "non_matching_truth":
        with pytest.raises(IOError, match="The requested filepath"):
            plugin(file_paths, model_output=model_output)
    elif exception == "missing_static_feature":
        feature_config = {
            "wind_speed_at_10m": ["mean", "std"],
            "distance_to_water": ["static"],
        }
        plugin.feature_config = feature_config
        with pytest.raises(ValueError, match="The number of cubes loaded."):
            plugin.process(file_paths=file_paths)
    elif exception == "missing_dynamic_feature":
        feature_config = {
            "wind_speed_at_10m": ["mean", "std"],
            "air_temperature": ["mean", "std"],
        }
        plugin.feature_config = feature_config
        with pytest.raises(ValueError, match="The number of cubes loaded."):
            plugin.process(file_paths=file_paths)
    elif exception == "no_percentile_realization":
        with pytest.raises(ValueError, match="The forecast parquet file"):
            plugin(file_paths, model_output=model_output)
    elif exception == "alternative_forecast_period":
        with pytest.raises(ValueError, match="The forecast_periods argument"):
            plugin(file_paths, model_output=model_output)
    elif exception == "no_quantile_forest_package":
        plugin.quantile_forest_installed = False
        result = plugin(file_paths, model_output=model_output)
        assert result is None
    else:
        raise ValueError(f"Unknown exception type: {exception}")
