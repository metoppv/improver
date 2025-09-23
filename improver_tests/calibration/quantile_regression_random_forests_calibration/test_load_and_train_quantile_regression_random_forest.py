# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the LoadForTrainQRF and PrepareAndTrain QRF plugins."""

import iris
import numpy as np
import pandas as pd
import pytest

from improver.calibration.load_and_train_quantile_regression_random_forest import (
    LoadForTrainQRF,
    PrepareAndTrainQRF,
)
from improver.synthetic_data.set_up_test_cubes import set_up_spot_variable_cube

pytest.importorskip("quantile_forest")


ALTITUDE = [10, 20]
LATITUDE = [50, 60]
LONGITUDE = [0, 10]
SITE_ID = ["03001", "03002", "03003", "03004", "03005"]


def _create_multi_site_forecast_parquet_file(tmp_path, representation="percentile"):
    """Create a parquet file with multi-site forecast data.

    Args:
        tmp_path: Temporary path to save the parquet file.
        representation: The type of ensemble representation to use. Options are
            "percentile" or "realization".
    """

    data_dict = {
        "percentile": np.repeat(50.0, 5),
        "forecast": [281.0, 272.0, 287.0, 280.0, 290.0],
        "altitude": [10.0, 83.0, 56.0, 23.0, 2.0],
        "blend_time": [pd.Timestamp("2017-01-02 00:00:00", tz="utc")] * 5,
        "forecast_period": [6 * 3.6e12] * 5,
        "forecast_reference_time": [pd.Timestamp("2017-01-02 00:00:00", tz="utc")] * 5,
        "latitude": [60.1, 59.9, 59.7, 58, 57],
        "longitude": [1.0, 2.0, -1.0, -2.0, -3.0],
        "time": [pd.Timestamp("2017-01-02 06:00:00", tz="utc")] * 5,
        "wmo_id": ["03001", "03002", "03003", "03004", "03005"],
        "station_id": ["03001", "03002", "03003", "03004", "03005"],
        "cf_name": ["air_temperature"] * 5,
        "units": ["K"] * 5,
        "experiment": ["latestblend"] * 5,
        "period": [pd.NA] * 5,
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
    wind_dir_dict = data_dict.copy()
    wind_dir_dict["forecast"] = [90, 100, 110, 120, 130]
    wind_dir_dict["cf_name"] = "wind_direction"
    wind_dir_dict["units"] = "degrees"
    wind_dir_dict["diagnostic"] = "wind_from_direction"
    data_df = pd.DataFrame(data_dict)
    wind_speed_df = pd.DataFrame(wind_speed_dict)
    wind_dir_df = pd.DataFrame(wind_dir_dict)
    joined_df = pd.concat([data_df, wind_speed_df, wind_dir_df], ignore_index=True)

    output_dir = tmp_path / "forecast_parquet_files"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "forecast.parquet"
    joined_df.to_parquet(output_path, index=False, engine="pyarrow")

    return output_dir, joined_df, data_dict["wmo_id"]


def _create_multi_percentile_forecast_parquet_file(tmp_path, representation=None):
    """Create a parquet file with multi-percentile forecast data."""

    data_dict = {
        "percentile": [16 + 2 / 3, 33 + 1 / 3, 50, 66 + 2 / 3, 83 + 1 / 3],
        "forecast": [272.0, 274.0, 275.0, 277.0, 280.0],
        "altitude": [10.0, 10.0, 10.0, 10.0, 10.0],
        "blend_time": [pd.Timestamp("2017-01-02 00:00:00", tz="utc")] * 5,
        "forecast_period": [6 * 3.6e12] * 5,
        "forecast_reference_time": [pd.Timestamp("2017-01-02 00:00:00", tz="utc")] * 5,
        "latitude": [60.1, 60.1, 60.1, 60.1, 60.1],
        "longitude": [1, 1, 1, 1, 1],
        "time": [pd.Timestamp("2017-01-02 06:00:00", tz="utc")] * 5,
        "wmo_id": ["03001", "03001", "03001", "03001", "03001"],
        "station_id": ["03001", "03001", "03001", "03001", "03001"],
        "cf_name": ["air_temperature"] * 5,
        "units": ["K"] * 5,
        "experiment": ["latestblend"] * 5,
        "period": [pd.NA] * 5,
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
    wind_speed_dict["forecast"] = [6, 10, 11, 12, 15]
    wind_speed_dict["cf_name"] = "wind_speed"
    wind_speed_dict["units"] = "m s-1"
    wind_speed_dict["diagnostic"] = "wind_speed_at_10m"
    wind_dir_dict = data_dict.copy()
    wind_dir_dict["forecast"] = [90, 100, 110, 120, 130]
    wind_dir_dict["cf_name"] = "wind_direction"
    wind_dir_dict["units"] = "degrees"
    wind_dir_dict["diagnostic"] = "wind_from_direction"
    data_df = pd.DataFrame(data_dict)
    wind_speed_df = pd.DataFrame(wind_speed_dict)
    wind_dir_df = pd.DataFrame(wind_dir_dict)
    joined_df = pd.concat([data_df, wind_speed_df, wind_dir_df], ignore_index=True)

    output_dir = tmp_path / "forecast_parquet_files"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "forecast.parquet"
    joined_df.to_parquet(output_path, index=False, engine="pyarrow")
    return output_dir, joined_df, data_dict["wmo_id"]


def _create_multi_forecast_period_forecast_parquet_file(tmp_path, representation=None):
    """Create a parquet file with multi-forecast period forecast data."""

    data_dict = {
        "percentile": [50.0, 50.0, 50.0, 50.0],
        "forecast": [277.0, 270.0, 280.0, 269.0],
        "altitude": [10.0, 83.0, 10.0, 83.0],
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
        "period": [pd.NA] * 4,
        "height": [1.5] * 4,
        "diagnostic": ["temperature_at_screen_level"] * 4,
    }
    if representation == "realization":
        data_dict["realization"] = [0, 1, 0, 1]
        data_dict.pop("percentile")
    elif representation == "kittens":
        data_dict["kittens"] = [0, 1, 0, 1]
        data_dict.pop("percentile")
    # Add wind speed to demonstrate filtering.
    wind_speed_dict = data_dict.copy()
    wind_speed_dict["forecast"] = [6, 16, 12, 15]
    wind_speed_dict["cf_name"] = "wind_speed"
    wind_speed_dict["units"] = "m s-1"
    wind_speed_dict["diagnostic"] = "wind_speed_at_10m"
    wind_dir_dict = data_dict.copy()
    wind_dir_dict["forecast"] = [180, 190, 200, 210]
    wind_dir_dict["cf_name"] = "wind_from_direction"
    wind_dir_dict["units"] = "degrees"
    wind_dir_dict["diagnostic"] = "wind_direction"
    data_df = pd.DataFrame(data_dict)
    wind_speed_df = pd.DataFrame(wind_speed_dict)
    wind_dir_df = pd.DataFrame(wind_dir_dict)
    joined_df = pd.concat([data_df, wind_speed_df, wind_dir_df], ignore_index=True)

    output_dir = tmp_path / "forecast_parquet_files"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "forecast.parquet"
    joined_df.to_parquet(output_path, index=False, engine="pyarrow")
    return output_dir, joined_df, data_dict["wmo_id"]


def _create_multi_site_truth_parquet_file(tmp_path):
    """Create a parquet file with multi-site truth data."""
    data_dict = {
        "diagnostic": ["temperature_at_screen_level"] * 5,
        "latitude": [60.1, 59.9, 59.7, 58, 57],
        "longitude": [1.0, 2.0, -1.0, -2.0, -3.0],
        "altitude": [10.0, 83.0, 56.0, 23.0, 2.0],
        "time": [pd.Timestamp("2017-01-02 06:00:00")] * 5,
        "wmo_id": ["03001", "03002", "03003", "03004", "03005"],
        "station_id": ["03001", "03002", "03003", "03004", "03005"],
        "ob_value": [276.0, 270.0, 289.0, 290.0, 301.0],
    }
    wind_speed_dict = data_dict.copy()
    wind_speed_dict["ob_value"] = [3, 22, 24, 11, 9]
    wind_speed_dict["diagnostic"] = "wind_speed_at_10m"
    data_df = pd.DataFrame(data_dict)
    wind_speed_df = pd.DataFrame(wind_speed_dict)
    joined_df = pd.concat([data_df, wind_speed_df], ignore_index=True)

    output_dir = tmp_path / "truth_parquet_files"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "truth.parquet"
    joined_df.to_parquet(output_path, index=False, engine="pyarrow")
    return output_dir, joined_df


def _create_multi_percentile_truth_parquet_file(tmp_path):
    """Create a parquet file with multi-percentile truth data."""
    data_dict = {
        "diagnostic": ["temperature_at_screen_level"],
        "latitude": [60.1],
        "longitude": [1.0],
        "altitude": [10.0],
        "time": [pd.Timestamp("2017-01-02 06:00:00")],
        "wmo_id": ["03001"],
        "station_id": ["03001"],
        "ob_value": [276.0],
    }
    wind_speed_dict = data_dict.copy()
    wind_speed_dict["ob_value"] = [9]
    wind_speed_dict["diagnostic"] = "wind_speed_at_10m"
    data_df = pd.DataFrame(data_dict)
    wind_speed_df = pd.DataFrame(wind_speed_dict)
    joined_df = pd.concat([data_df, wind_speed_df], ignore_index=True)

    output_dir = tmp_path / "truth_parquet_files"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "truth.parquet"
    joined_df.to_parquet(output_path, index=False, engine="pyarrow")
    return output_dir, joined_df


def _create_multi_forecast_period_truth_parquet_file(tmp_path):
    """Create a parquet file with multi-forecast period truth data."""
    data_dict = {
        "diagnostic": ["temperature_at_screen_level"] * 4,
        "latitude": [60.1, 59.9, 60.1, 59.9],
        "longitude": [1.0, 2.0, 1.0, 2.0],
        "altitude": [10.0, 83.0, 10.0, 83.0],
        "time": np.repeat(
            [pd.Timestamp("2017-01-02 06:00:00"), pd.Timestamp("2017-01-02 12:00:00")],
            2,
        ),
        "wmo_id": ["03001", "03002", "03001", "03002"],
        "station_id": ["03001", "03002", "03001", "03002"],
        "ob_value": [280, 273, 284, 275],
    }
    wind_speed_dict = data_dict.copy()
    wind_speed_dict["ob_value"] = [2.0, 11.0, 10.0, 14.0]
    wind_speed_dict["diagnostic"] = "wind_speed_at_10m"
    data_df = pd.DataFrame(data_dict)
    wind_speed_df = pd.DataFrame(wind_speed_dict)
    joined_df = pd.concat([data_df, wind_speed_df], ignore_index=True)

    output_dir = tmp_path / "truth_parquet_files"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "truth.parquet"
    joined_df.to_parquet(output_path, index=False, engine="pyarrow")
    return output_dir, joined_df


def _create_multi_site_truth_parquet_file_alt(tmp_path, site_id="wmo_id"):
    """Create a parquet file with multi-site truth data for wind speed."""
    data_dict = {
        "diagnostic": ["wind_speed_at_10m"] * 5,
        "latitude": [60.1, 59.9, 59.7, 58, 57],
        "longitude": [1.0, 2.0, -1.0, -2.0, -3.0],
        "altitude": [10.0, 83.0, 56.0, 23.0, 2.0],
        "time": [pd.Timestamp("2017-01-02 06:00:00")] * 5,
        "wmo_id": ["03001", "03002", "03003", "03004", "03005"],
        "station_id": ["03001", "03002", "03003", "03004", "03005"],
        "ob_value": [10.0, 25.0, 4.0, 3.0, 11.0],
    }
    wind_speed_dict = data_dict.copy()
    wind_speed_dict["ob_value"] = [3.0, 22.0, 24.0, 11.0, 9.0]
    wind_speed_dict["diagnostic"] = "wind_speed_at_10m"
    data_df = pd.DataFrame(data_dict)
    wind_speed_df = pd.DataFrame(wind_speed_dict)
    joined_df = pd.concat([data_df, wind_speed_df], ignore_index=True)

    output_dir = tmp_path / "truth_parquet_files"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "truth.parquet"
    joined_df.to_parquet(output_path, index=False, engine="pyarrow")
    return output_dir, joined_df


def _create_ancil_file(tmp_path, site_ids):
    """Create an ancillary file for testing.

    Returns:
        An ancillary cube with a single value.
    """
    data = np.array(range(len(site_ids)), dtype=np.float32)
    template_cube = set_up_spot_variable_cube(
        data,
        wmo_ids=site_ids,
        unique_site_id=site_ids,
        unique_site_id_key="station_id",
        name="distance_to_water",
        units="m",
    )
    cube = template_cube.copy()
    for coord in ["time", "forecast_reference_time", "forecast_period"]:
        cube.remove_coord(coord)
    output_dir = tmp_path / "ancil_files"
    output_dir.mkdir(parents=True)
    output_path = output_dir / "ancil.nc"
    iris.save(cube, str(output_path))
    return output_path, cube


def filter_forecast_periods(forecast_df, forecast_periods):
    """Filter the forecast DataFrame to only include the requested forecast periods."""
    if ":" in forecast_periods:
        forecast_periods = [
            fp * 3600 for fp in range(*map(int, forecast_periods.split(":")))
        ]
    else:
        forecast_periods = [int(forecast_periods) * 3600]
    return forecast_df[
        forecast_df["forecast_period"].isin(np.array(forecast_periods) * 1e9)
    ].reset_index(drop=True)


def amend_expected_forecast_df(
    forecast_df, forecast_periods, parquet_diagnostic_names, representation, site_id
):
    forecast_df = filter_forecast_periods(forecast_df, forecast_periods)
    for column in ["time", "forecast_reference_time", "blend_time"]:
        forecast_df[column] = pd.to_datetime(forecast_df[column], unit="ns", utc=True)
    for column in ["forecast_period", "period"]:
        forecast_df[column] = pd.to_timedelta(forecast_df[column], unit="ns")

    base_df = forecast_df[forecast_df["diagnostic"] == parquet_diagnostic_names[0]]
    for parquet_diagnostic_name in parquet_diagnostic_names[1:]:
        additional_df = forecast_df[
            forecast_df["diagnostic"] == parquet_diagnostic_name
        ]
        base_df = pd.merge(
            base_df,
            additional_df[
                [
                    *site_id,
                    "forecast_reference_time",
                    "forecast_period",
                    representation,
                    "forecast",
                ]
            ].rename(columns={"forecast": parquet_diagnostic_name}),
            on=[
                *site_id,
                "forecast_reference_time",
                "forecast_period",
                representation,
            ],
            how="left",
        )
    forecast_df = base_df

    forecast_df.rename(columns={"forecast": "air_temperature"}, inplace=True)
    return forecast_df


def amend_expected_truth_df(truth_df, parquet_diagnostic_name):
    truth_df = truth_df[truth_df["diagnostic"] == parquet_diagnostic_name]
    truth_df["time"] = pd.to_datetime(truth_df["time"], unit="ns", utc=True)
    return truth_df


@pytest.mark.parametrize("representation", ["percentile", "realization"])
@pytest.mark.parametrize("include_dynamic", [True, False])
@pytest.mark.parametrize("include_static", [True, False])
@pytest.mark.parametrize("include_noncube_static", [True, False])
@pytest.mark.parametrize("remove_target", [True, False])
@pytest.mark.parametrize(
    "site_id", ["wmo_id", "station_id", ["wmo_id"], ["latitude", "longitude"]]
)
@pytest.mark.parametrize(
    "forecast_creation,truth_creation,forecast_periods",
    [
        (
            _create_multi_site_forecast_parquet_file,
            _create_multi_site_truth_parquet_file,
            "6:18:6",
        ),
        (
            _create_multi_percentile_forecast_parquet_file,
            _create_multi_percentile_truth_parquet_file,
            "6:18:6",
        ),
        (
            _create_multi_forecast_period_forecast_parquet_file,
            _create_multi_forecast_period_truth_parquet_file,
            "6:18:6",
        ),
        (
            _create_multi_forecast_period_forecast_parquet_file,
            _create_multi_forecast_period_truth_parquet_file,
            "12",
        ),
    ],
)
def test_load_for_qrf(
    tmp_path,
    representation,
    include_dynamic,
    include_static,
    include_noncube_static,
    remove_target,
    site_id,
    forecast_creation,
    truth_creation,
    forecast_periods,
):
    """Test the LoadForTrainQRF plugin."""
    feature_config = {"air_temperature": ["mean", "std", "altitude"]}
    parquet_diagnostic_names = ["temperature_at_screen_level"]

    if isinstance(site_id, str):
        site_id = [site_id]

    forecast_path, base_expected_forecast_df, site_ids = forecast_creation(
        tmp_path, representation
    )
    truth_path, expected_truth_df = truth_creation(tmp_path)

    file_paths = [forecast_path, truth_path]

    if include_dynamic:
        feature_config["wind_speed_at_10m"] = ["mean", "std"]
        parquet_diagnostic_names.append("wind_speed_at_10m")

    if include_static:
        ancil_path, expected_cube = _create_ancil_file(
            tmp_path, sorted(list(set(site_ids)))
        )
        file_paths.append(ancil_path)
        feature_config["distance_to_water"] = ["static"]

    if include_noncube_static:
        feature_config["wind_from_direction"] = ["static"]
        parquet_diagnostic_names.append("wind_from_direction")

    if remove_target:
        feature_config.pop("air_temperature")

    expected_forecast_df = amend_expected_forecast_df(
        base_expected_forecast_df.copy(),
        forecast_periods,
        parquet_diagnostic_names,
        representation,
        site_id,
    )
    expected_truth_df = amend_expected_truth_df(
        expected_truth_df, "temperature_at_screen_level"
    )

    # Create an instance of LoadForTrainQRF with the required parameters
    plugin = LoadForTrainQRF(
        experiment="latestblend",
        feature_config=feature_config,
        parquet_diagnostic_names=parquet_diagnostic_names,
        target_cf_name="air_temperature",
        forecast_periods=forecast_periods,
        cycletime="20170103T0000Z",
        training_length=2,
        unique_site_id_keys=site_id,
    )
    forecast_df, truth_df, cube_inputs = plugin(file_paths)

    assert isinstance(forecast_df, pd.DataFrame)
    assert isinstance(truth_df, pd.DataFrame)

    pd.testing.assert_frame_equal(
        forecast_df,
        expected_forecast_df,
        check_dtype=False,
        check_datetimelike_compat=True,
    )
    pd.testing.assert_frame_equal(
        truth_df, expected_truth_df, check_dtype=False, check_datetimelike_compat=True
    )
    if include_static:
        assert isinstance(cube_inputs, iris.cube.CubeList)
        assert len(cube_inputs) == 1
        assert cube_inputs[0].name() == "distance_to_water"
        np.testing.assert_almost_equal(cube_inputs[0].data, expected_cube.data)


@pytest.mark.parametrize("make_files", [False, True])
def test_load_for_qrf_no_paths(tmp_path, make_files):
    """Test the LoadForTrainQRF plugin when the no valid file paths are provided.
    Either the paths do not exist, or the paths exist but the directories are empty."""
    feature_config = {"air_temperature": ["mean", "std", "altitude"]}

    file_paths = [
        tmp_path / "partition" / "forecast_table/",
        tmp_path / "partition" / "truth_table/",
    ]
    if make_files:
        for file_path in file_paths:
            (tmp_path / file_path).mkdir(parents=True, exist_ok=True)

    plugin = LoadForTrainQRF(
        experiment="latestblend",
        feature_config=feature_config,
        parquet_diagnostic_names=["temperature_at_screen_level"],
        target_cf_name="air_temperature",
        forecast_periods="6:12:6",
        cycletime="20170102T0000Z",
        training_length=2,
    )
    result = plugin(file_paths)
    # Expecting None since no valid paths are provided
    assert result == (None, None, None)


@pytest.mark.parametrize(
    "forecast_creation,truth_creation,cycletime,forecast_periods",
    [
        (
            _create_multi_site_forecast_parquet_file,
            _create_multi_site_truth_parquet_file,
            "20200102T0000Z",
            "6:12:6",
        ),
        (
            _create_multi_site_forecast_parquet_file,
            _create_multi_site_truth_parquet_file,
            "20170102T0000Z",
            "30:36:6",
        ),
    ],
)
def test_load_for_qrf_mismatches(
    tmp_path,
    forecast_creation,
    truth_creation,
    cycletime,
    forecast_periods,
):
    """Test the LoadForTrainQRF plugin when the cycletime or forecast_periods
    requested are not present in the provided files."""
    feature_config = {"air_temperature": ["mean", "std", "altitude"]}
    representation = "percentile"

    forecast_path, expected_forecast_df, _ = forecast_creation(tmp_path, representation)
    expected_forecast_df = amend_expected_forecast_df(
        expected_forecast_df,
        forecast_periods,
        ["temperature_at_screen_level"],
        "percentile",
        "wmo_id",
    )

    truth_path, expected_truth_df = truth_creation(tmp_path)
    expected_truth_df = amend_expected_truth_df(
        expected_truth_df, "temperature_at_screen_level"
    )

    file_paths = [forecast_path, truth_path]

    plugin = LoadForTrainQRF(
        experiment="latestblend",
        feature_config=feature_config,
        parquet_diagnostic_names=["temperature_at_screen_level"],
        target_cf_name="air_temperature",
        forecast_periods=forecast_periods,
        cycletime=cycletime,
        training_length=2,
    )
    forecast_df, _, _ = plugin(file_paths)
    # Expecting an empty DataFrame since the cycletime or forecast_periods
    # requested are not present in the provided file.
    assert forecast_df.empty


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
    """Test LoadForTrainQRF plugin behaviour in atypical situations."""
    feature_config = {"air_temperature": ["mean", "std", "altitude"]}

    forecast_path, _, _ = forecast_creation(tmp_path, representation)
    truth_path, _ = truth_creation(tmp_path)
    file_paths = [forecast_path, truth_path]

    # Create an instance of LoadForTrainQRF with the required parameters
    plugin = LoadForTrainQRF(
        experiment="latestblend",
        feature_config=feature_config,
        parquet_diagnostic_names=["temperature_at_screen_level"],
        target_cf_name="air_temperature",
        forecast_periods=forecast_periods,
        cycletime="20170103T0000Z",
        training_length=2,
    )

    if exception == "non_matching_truth":
        with pytest.raises(IOError, match="The requested filepath"):
            plugin(file_paths)
    elif exception == "missing_static_feature":
        feature_config = {
            "wind_speed_at_10m": ["mean", "std"],
            "distance_to_water": ["static"],
        }
        plugin.feature_config = feature_config
        with pytest.raises(ValueError, match="The features requested in the"):
            plugin.process(file_paths=file_paths)
    elif exception == "missing_dynamic_feature":
        feature_config = {
            "wind_speed_at_10m": ["mean", "std"],
            "air_temperature": ["mean", "std"],
        }
        plugin.feature_config = feature_config
        with pytest.raises(ValueError, match="The features requested in the"):
            plugin.process(file_paths=file_paths)
    elif exception == "no_percentile_realization":
        with pytest.raises(ValueError, match="The forecast parquet file"):
            plugin(file_paths)
    elif exception == "alternative_forecast_period":
        with pytest.raises(ValueError, match="The forecast_periods argument"):
            plugin(file_paths)
    elif exception == "no_quantile_forest_package":
        plugin.quantile_forest_installed = False
        result = plugin(file_paths)
        assert result is None
    else:
        raise ValueError(f"Unknown exception type: {exception}")


@pytest.mark.parametrize("representation", ["percentile", "realization"])
@pytest.mark.parametrize("include_dynamic", [True, False])
@pytest.mark.parametrize("include_static", [True, False])
@pytest.mark.parametrize("include_noncube_static", [True, False])
@pytest.mark.parametrize("remove_target", [True, False])
@pytest.mark.parametrize("include_nans", [True, False])
@pytest.mark.parametrize("include_latlon_nans", [True, False])
@pytest.mark.parametrize(
    "site_id", ["wmo_id", "station_id", ["wmo_id"], ["latitude", "longitude"]]
)
@pytest.mark.parametrize(
    "forecast_creation,truth_creation,forecast_periods",
    [
        (
            _create_multi_site_forecast_parquet_file,
            _create_multi_site_truth_parquet_file,
            "6:18:6",
        ),
        (
            _create_multi_percentile_forecast_parquet_file,
            _create_multi_percentile_truth_parquet_file,
            "6:18:6",
        ),
        (
            _create_multi_forecast_period_forecast_parquet_file,
            _create_multi_forecast_period_truth_parquet_file,
            "6:18:6",
        ),
        (
            _create_multi_forecast_period_forecast_parquet_file,
            _create_multi_forecast_period_truth_parquet_file,
            "12",
        ),
    ],
)
def test_prepare_and_train_qrf(
    tmp_path,
    representation,
    include_dynamic,
    include_static,
    include_noncube_static,
    remove_target,
    include_nans,
    include_latlon_nans,
    site_id,
    forecast_creation,
    truth_creation,
    forecast_periods,
):
    """Test the PrepareAndTrainQRF plugin."""
    feature_config = {"air_temperature": ["mean", "std", "altitude"]}
    n_estimators = 2
    max_depth = 5
    random_state = 46
    target_cf_name = "air_temperature"

    if isinstance(site_id, str):
        site_id = [site_id]

    _, forecast_df, site_ids = forecast_creation(tmp_path, representation)
    forecast_df = amend_expected_forecast_df(
        forecast_df,
        forecast_periods,
        ["temperature_at_screen_level"],
        representation,
        site_id,
    )
    _, truth_df = truth_creation(tmp_path)

    truth_df = amend_expected_truth_df(truth_df, "temperature_at_screen_level")

    if include_dynamic:
        forecast_df["wind_speed_at_10m"] = [10.0, 20.0, 15.0, 12.0, 11.0][
            : len(forecast_df)
        ]
        feature_config["wind_speed_at_10m"] = ["mean", "std"]

    if include_static:
        _, ancil_cube = _create_ancil_file(tmp_path, sorted(list(set(site_ids))))
        feature_config["distance_to_water"] = ["static"]

    if include_noncube_static:
        forecast_df["wind_from_direction"] = [90.0, 100.0, 110.0, 120.0, 130.0][
            : len(forecast_df)
        ]
        feature_config["wind_from_direction"] = ["static"]

    if remove_target:
        feature_config.pop("air_temperature")

    if include_nans:
        # Insert a NaN will result in this row being dropped.
        truth_df.loc[0, "ob_value"] = pd.NA

    if include_latlon_nans:
        # As latitude is not a feature, this NaN should be ignored.
        truth_df.loc[1, "latitude"] = pd.NA

    if feature_config == {}:
        pytest.skip("No features to train on")

    # Create an instance of PrepareAndTrainQRF with the required parameters
    plugin = PrepareAndTrainQRF(
        feature_config=feature_config,
        target_cf_name=target_cf_name,
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        transformation="log",
        pre_transform_addition=1,
        unique_site_id_keys=site_id,
    )
    if truth_df["ob_value"].isna().all():
        with pytest.raises(ValueError, match="Empty truth DataFrame"):
            plugin(forecast_df, truth_df)
        return
    elif include_static:
        qrf_model, transformation, pre_transform_addition = plugin(
            forecast_df, truth_df, iris.cube.CubeList([ancil_cube])
        )
    else:
        qrf_model, transformation, pre_transform_addition = plugin(
            forecast_df, truth_df
        )

    assert qrf_model.n_estimators == n_estimators
    assert qrf_model.max_depth == max_depth
    assert qrf_model.random_state == random_state
    assert transformation == "log"
    assert pre_transform_addition == 1

    current_forecast = [5.64, 3, 55]

    if remove_target:
        current_forecast = []

    if include_dynamic:
        current_forecast.extend([7, 1])

    if include_noncube_static:
        current_forecast.append(100)

    if include_static:
        current_forecast.append(2.5)

    result = qrf_model.predict(
        np.expand_dims(np.array(current_forecast), 0), quantiles=[0.5]
    )
    expected = 5.6
    np.testing.assert_almost_equal(result, expected, decimal=1)
