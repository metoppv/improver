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
"""
Unit tests for the utilities within the `calibration.dataframe_utilities`
module.

"""
import unittest

import iris
import numpy as np
import pandas as pd
import pytest

from improver.calibration.dataframe_utilities import (
    forecast_and_truth_dataframes_to_cubes,
    forecast_dataframe_to_cube,
    truth_dataframe_to_cube,
)
from improver.metadata.constants.time_types import TIME_COORDS
from improver.spotdata.build_spotdata_cube import build_spotdata_cube
from improver_tests import ImproverTest


def _chunker(seq, size):
    """Helper function to iterate through a sequence in chunks.

    Args:
        seq:
            The sequence to be chunked.
        size:
            The size of the chunks.

    Return:
        A sequence split into chunks.
    """
    return (seq[pos : pos + size] for pos in range(0, len(seq), size))


class SetupSharedDataFrames(ImproverTest):

    """A shared dataframe creation class."""

    def setUp(self):
        """Set-up forecast and truth dataframes."""
        pytest.importorskip("pandas")

        data = np.array(
            [5.2, 0.3, 20.4, 6.5, 3.1, 21.5, 7.2, 4.2, 24.3], dtype=np.float32
        )
        self.forecast_data = np.tile(data, 3)

        self.frt1 = pd.Timestamp("2017-07-20T12:00:00", tz="UTC")
        self.frt2 = pd.Timestamp("2017-07-21T12:00:00", tz="UTC")
        self.frt3 = pd.Timestamp("2017-07-22T12:00:00", tz="UTC")

        self.fp = pd.Timedelta(6 * 3600, unit="s")

        self.time1 = pd.Timestamp("2017-07-20T18:00:00", tz="UTC")
        self.time2 = pd.Timestamp("2017-07-21T18:00:00", tz="UTC")
        self.time3 = pd.Timestamp("2017-07-22T18:00:00", tz="UTC")

        self.wmo_ids = ["03002", "03003", "03004"]
        self.percentiles = np.array([25.0, 50.0, 75.0], dtype=np.float32)
        self.realizations = np.array([0, 1, 2], dtype=np.int32)
        diag = "air_temperature"
        self.cf_name = "air_temperature"
        self.latitudes = np.array([50.0, 60.0, 70.0], dtype=np.float32)
        self.longitudes = np.array([-10.0, 0.0, 10.0], dtype=np.float32)
        self.altitudes = np.array([10.0, 20.0, 30.0], dtype=np.float32)
        self.period = pd.Timedelta(1, unit="h")
        self.height = np.array([1.5], dtype=np.float32)
        self.units = "Celsius"
        self.experiment = "standardise"

        df_dict = {
            "forecast": self.forecast_data,
            "blend_time": np.repeat([self.frt1, self.frt2, self.frt3], 9),
            "forecast_period": np.repeat(self.fp, 27),
            "forecast_reference_time": np.repeat([self.frt1, self.frt2, self.frt3], 9),
            "time": np.repeat([self.time1, self.time2, self.time3], 9),
            "wmo_id": np.tile(self.wmo_ids, 9),
            "percentile": np.tile(np.repeat(self.percentiles, 3), 3),
            "diagnostic": [diag] * 27,
            "latitude": np.tile(self.latitudes, 9),
            "longitude": np.tile(self.longitudes, 9),
            "altitude": np.tile(self.altitudes, 9),
            "period": [self.period] * 27,
            "height": np.tile(self.height, 27),
            "cf_name": [self.cf_name] * 27,
            "units": [self.units] * 27,
            "experiment": [self.experiment] * 27,
        }

        self.forecast_df = pd.DataFrame(df_dict)

        realization_df = self.forecast_df.drop(columns=["percentile"])
        realization_df["realization"] = np.tile(np.repeat(self.realizations, 3), 3)
        self.forecast_df_realization = realization_df

        data = np.array([6.8, 2.7, 21.2], dtype=np.float32)
        self.truth_data = np.tile(data, 3)

        df_dict = {
            "ob_value": self.truth_data,
            "time": np.repeat([self.time1, self.time2, self.time3], 3),
            "wmo_id": self.wmo_ids * 3,
            "diagnostic": [diag] * 9,
            "latitude": np.tile(self.latitudes, 3),
            "longitude": np.tile(self.longitudes, 3),
            "altitude": np.tile(self.altitudes, 3),
            "period": [self.period] * 9,
            "height": np.tile(self.height, 9),
            "cf_name": [self.cf_name] * 9,
            "units": [self.units] * 9,
        }

        self.truth_df = pd.DataFrame(df_dict)
        self.truth_subset_df = self.truth_df.drop(
            columns=["period", "height", "cf_name", "units"]
        )

        self.validity_time = self.time3
        self.forecast_period = 6 * 3600
        self.training_length = 3
        self.date_range = pd.date_range(
            end=self.validity_time,
            periods=int(self.training_length),
            freq="D",
            tz="UTC",
        )
        self.date_range_two_days = pd.date_range(
            end=self.validity_time, periods=2, freq="D", tz="UTC"
        )

        self.height_coord = iris.coords.AuxCoord(
            np.array(self.height, dtype=np.float32), "height", units="m",
        )

        # Modify the forecast and truth DataFrames so that one of the sites is
        # only available at the most recent time point. These DataFrames also have
        # a station_id column.
        df = self.forecast_df
        condition = (df["time"].isin([self.time1, self.time2])) & (
            df["wmo_id"] == self.wmo_ids[2]
        )
        self.forecast_df_multi_station_id = df.drop(df[condition].index)
        self.forecast_df_multi_station_id["station_id"] = (
            self.forecast_df_multi_station_id["wmo_id"] + "0"
        )
        df = self.truth_subset_df
        condition = (df["time"].isin([self.time1, self.time2])) & (
            df["wmo_id"] == self.wmo_ids[2]
        )
        self.truth_df_multi_station_id = df.drop(df[condition].index)
        self.truth_df_multi_station_id["station_id"] = (
            self.truth_df_multi_station_id["wmo_id"] + "0"
        )

        # The forecast DataFrame has two sites. The truth DataFrame has two sites
        # with one of these sites being in common with the forecast DataFrame.
        # These DataFrames also have a station_id column.
        self.forecast_df_one_station_id = self.forecast_df.copy()
        self.forecast_df_one_station_id = self.forecast_df_one_station_id.loc[
            self.forecast_df_one_station_id["wmo_id"].isin(self.wmo_ids[:-1])
        ]
        self.forecast_df_one_station_id["station_id"] = (
            self.forecast_df_one_station_id["wmo_id"] + "0"
        )
        self.truth_df_one_station_id = self.truth_subset_df.copy()
        self.truth_df_one_station_id = self.truth_df_one_station_id.loc[
            self.truth_df_one_station_id["wmo_id"].isin(self.wmo_ids[1:])
        ]
        self.truth_df_one_station_id["station_id"] = (
            self.truth_df_one_station_id["wmo_id"] + "0"
        )


class SetupConstructedForecastCubes(SetupSharedDataFrames):

    """A constructed forecast cube class."""

    def setUp(self):
        """Set-up forecast cubes."""
        super().setUp()
        # Create a cube of the format expected based on the input dataframe.
        cubes = iris.cube.CubeList([])

        for frt, time in zip(
            [self.frt1, self.frt2, self.frt3], [self.time1, self.time2, self.time3]
        ):
            time_coord = iris.coords.DimCoord(
                np.array(time.timestamp(), dtype=TIME_COORDS["time"].dtype),
                "time",
                bounds=[
                    np.array(t.timestamp(), dtype=TIME_COORDS["time"].dtype)
                    for t in [time - self.period, time]
                ],
                units=TIME_COORDS["time"].units,
            )

            fp_coord = iris.coords.AuxCoord(
                np.array(
                    self.fp.total_seconds(), dtype=TIME_COORDS["forecast_period"].dtype
                ),
                "forecast_period",
                bounds=[
                    np.array(
                        f.total_seconds(), dtype=TIME_COORDS["forecast_period"].dtype
                    )
                    for f in [self.fp - self.period, self.fp]
                ],
                units=TIME_COORDS["forecast_period"].units,
            )

            frt_coord = iris.coords.AuxCoord(
                np.array(
                    frt.timestamp(), dtype=TIME_COORDS["forecast_reference_time"].dtype
                ),
                "forecast_reference_time",
                units=TIME_COORDS["forecast_reference_time"].units,
            )
            for index, data_slice in zip(
                range(len(self.percentiles)), _chunker(self.forecast_data, 3)
            ):
                realization_coord = iris.coords.DimCoord(
                    np.array(index, dtype=np.int32),
                    standard_name="realization",
                    units=1,
                )
                cube = build_spotdata_cube(
                    data_slice.astype(np.float32),
                    self.cf_name,
                    self.units,
                    np.array(self.altitudes, dtype=np.float32),
                    np.array(self.latitudes, dtype=np.float32),
                    np.array(self.longitudes, dtype=np.float32),
                    wmo_id=self.wmo_ids,
                    scalar_coords=[
                        time_coord,
                        frt_coord,
                        fp_coord,
                        realization_coord,
                        self.height_coord,
                    ],
                )
                cubes.append(cube)

        self.expected_period_forecast = cubes.merge_cube()
        self.expected_instantaneous_forecast = self.expected_period_forecast.copy()
        for coord in ["forecast_period", "time"]:
            self.expected_instantaneous_forecast.coord(coord).bounds = None

        # Modify the forecast cube, so that site and time combinations that are absent
        # from the input DataFrame have NaN values. A station_id coordinate is also
        # added.
        self.expected_forecast_multi_station_id = self.expected_period_forecast.copy()
        self.expected_forecast_multi_station_id.data[:, :2, -1] = np.nan
        unique_id_coord = iris.coords.AuxCoord(
            [w + "0" for w in self.wmo_ids],
            long_name="station_id",
            units="no_unit",
            attributes={"unique_site_identifier": "true"},
        )
        site_id_dim = self.expected_forecast_multi_station_id.coord_dims("spot_index")[
            0
        ]
        self.expected_forecast_multi_station_id.add_aux_coord(
            unique_id_coord, site_id_dim
        )

        # Modify the forecast cube by extracting a single site and adding a
        # station_id coordinate.
        self.expected_forecast_one_station_id = self.expected_period_forecast[
            :, :, [1]
        ].copy()
        unique_id_coord = iris.coords.AuxCoord(
            [self.wmo_ids[1] + "0"],
            long_name="station_id",
            units="no_unit",
            attributes={"unique_site_identifier": "true"},
        )
        site_id_dim = self.expected_forecast_one_station_id.coord_dims("spot_index")[0]
        self.expected_forecast_one_station_id.add_aux_coord(
            unique_id_coord, site_id_dim
        )
        self.expected_forecast_one_station_id.coord("spot_index").points = np.array(
            [0], dtype=np.int32
        )


class SetupConstructedTruthCubes(SetupSharedDataFrames):

    """A constructed truth cube class."""

    def setUp(self):
        """Set-up truth cubes."""
        super().setUp()
        # Create a cube of the format expected based on the input dataframe.
        cubes = iris.cube.CubeList([])

        for time, data_slice in zip(
            [self.time1, self.time2, self.time3], _chunker(self.truth_data, 3)
        ):
            time_coord = iris.coords.DimCoord(
                np.array(time.timestamp(), dtype=TIME_COORDS["time"].dtype),
                "time",
                bounds=[
                    np.array(t.timestamp(), dtype=TIME_COORDS["time"].dtype)
                    for t in [time - self.period, time]
                ],
                units=TIME_COORDS["time"].units,
            )
            cubes.append(
                build_spotdata_cube(
                    data_slice.astype(np.float32),
                    self.cf_name,
                    self.units,
                    np.array(self.altitudes, dtype=np.float32),
                    np.array(self.latitudes, dtype=np.float32),
                    np.array(self.longitudes, dtype=np.float32),
                    wmo_id=self.wmo_ids,
                    scalar_coords=[time_coord, self.height_coord],
                )
            )

        self.expected_period_truth = cubes.merge_cube()
        self.expected_instantaneous_truth = self.expected_period_truth.copy()
        self.expected_instantaneous_truth.coord("time").bounds = None

        # Modify the truth cube, so that site and time combinations that are absent
        # from the input DataFrame have NaN values. A station_id coordinate is also
        # added.
        unique_id_coord = iris.coords.AuxCoord(
            [w + "0" for w in self.wmo_ids],
            long_name="station_id",
            units="no_unit",
            attributes={"unique_site_identifier": "true"},
        )
        self.expected_truth_multi_station_id = self.expected_period_truth.copy()
        self.expected_truth_multi_station_id.data[:2, -1] = np.nan
        site_id_dim = self.expected_truth_multi_station_id.coord_dims("spot_index")[0]
        self.expected_truth_multi_station_id.add_aux_coord(unique_id_coord, site_id_dim)

        # Modify the truth cube by extracting a single site and adding a station_id
        # coordinate.
        unique_id_coord = iris.coords.AuxCoord(
            [self.wmo_ids[1] + "0"],
            long_name="station_id",
            units="no_unit",
            attributes={"unique_site_identifier": "true"},
        )
        self.expected_truth_one_station_id = self.expected_period_truth[:, [1]].copy()
        site_id_dim = self.expected_truth_one_station_id.coord_dims("spot_index")[0]
        self.expected_truth_one_station_id.add_aux_coord(unique_id_coord, site_id_dim)
        self.expected_truth_one_station_id.coord("spot_index").points = np.array(
            [0], dtype=np.int32
        )


class Test_forecast_dataframe_to_cube(SetupConstructedForecastCubes):

    """Test the forecast_dataframe_to_cube function."""

    def setUp(self):
        """Set up for testing."""
        super().setUp()

    def test_three_day_training_period_diag(self):
        """Test an input DataFrame is converted correctly into an Iris Cube
        for a three day training length for a period diagnostic."""
        result = forecast_dataframe_to_cube(
            self.forecast_df, self.date_range, self.forecast_period
        )
        self.assertCubeEqual(result, self.expected_period_forecast)

    def test_three_day_training_instantaneous_diag(self):
        """Test an input DataFrame is converted correctly into an Iris Cube
        for a three day training length for an instantaneous diagnostic."""
        self.forecast_df["period"] = pd.Timedelta("NaT")
        result = forecast_dataframe_to_cube(
            self.forecast_df, self.date_range, self.forecast_period
        )
        self.assertCubeEqual(result, self.expected_instantaneous_forecast)

    def test_two_day_training(self):
        """Test an input DataFrame is converted correctly into an Iris Cube
        for a two day training length."""
        result = forecast_dataframe_to_cube(
            self.forecast_df, self.date_range_two_days, self.forecast_period
        )
        self.assertCubeEqual(result, self.expected_period_forecast[:, 1:])

    def test_empty_dataframe(self):
        """Test if none of the required data is available in the dataframe."""
        forecast_period = 7 * 3600
        result = forecast_dataframe_to_cube(
            self.forecast_df, self.date_range, forecast_period
        )
        self.assertIsNone(result)

    def test_nonunique_values_in_column(self):
        """Test if there are multiple non-unique values in a column of the
        dataframe."""
        self.forecast_df.at[0, "period"] = pd.Timedelta(7200, units="seconds")
        msg = "Multiple values provided for the period"
        with self.assertRaisesRegex(ValueError, msg):
            forecast_dataframe_to_cube(
                self.forecast_df, self.date_range, self.forecast_period
            )


class Test_truth_dataframe_to_cube(SetupConstructedTruthCubes):

    """Test the truth_dataframe_to_cube function."""

    def setUp(self):
        """Set up for testing."""
        super().setUp()

    def test_three_day_training_period_diag(self):
        """Test an input DataFrame is converted correctly into an Iris Cube
        for a three day training length for a period diagnostic."""
        result = truth_dataframe_to_cube(self.truth_df, self.date_range,)
        self.assertCubeEqual(result, self.expected_period_truth)

    def test_three_day_training_instantaneous_diag(self):
        """Test an input DataFrame is converted correctly into an Iris Cube
        for a three day training length for an instantaneous diagnostic."""
        self.truth_df["period"] = pd.Timedelta("NaT")
        result = truth_dataframe_to_cube(self.truth_df, self.date_range,)
        self.assertCubeEqual(result, self.expected_instantaneous_truth)

    def test_two_day_training(self):
        """Test an input DataFrame is converted correctly into an Iris Cube
        for a two day training length."""
        result = truth_dataframe_to_cube(self.truth_df, self.date_range_two_days,)
        self.assertCubeEqual(result, self.expected_period_truth[1:, :])

    def test_empty_dataframe(self):
        """Test if none of the required data is available in the dataframe."""
        validity_time = np.datetime64("2017-07-22T19:00:00")
        date_range = pd.date_range(
            end=validity_time, periods=int(self.training_length), freq="D", tz="UTC"
        )
        result = truth_dataframe_to_cube(self.truth_df, date_range,)
        self.assertIsNone(result)

    def test_nonunique_values_in_column(self):
        """Test if there are multiple non-unique values in a column of the
        dataframe."""
        self.truth_df.at[0, "diagnostic"] = "wind_speed_at_10m"
        msg = "Multiple values provided for the diagnostic"
        with self.assertRaisesRegex(ValueError, msg):
            truth_dataframe_to_cube(self.truth_df, self.date_range)


class Test_forecast_and_truth_dataframes_to_cubes(
    SetupConstructedForecastCubes, SetupConstructedTruthCubes
):

    """Test the forecast_and_truth_dataframes_to_cubes function."""

    def setUp(self):
        """Set up dataframes for testing."""
        super().setUp()
        self.cycletime = "20170723T1200Z"

    def test_basic(self):
        """Test the expected cubes are generated from the input dataframes."""
        result = forecast_and_truth_dataframes_to_cubes(
            self.forecast_df,
            self.truth_subset_df,
            self.cycletime,
            self.forecast_period,
            self.training_length,
        )
        self.assertEqual(len(result), 2)
        self.assertCubeEqual(result[0], self.expected_period_forecast)
        self.assertCubeEqual(result[1], self.expected_period_truth)

    def test_realization(self):
        """Test the expected realization cubes are generated from the input dataframes."""
        result = forecast_and_truth_dataframes_to_cubes(
            self.forecast_df_realization,
            self.truth_subset_df,
            self.cycletime,
            self.forecast_period,
            self.training_length,
        )
        self.assertEqual(len(result), 2)
        self.assertCubeEqual(result[0], self.expected_period_forecast)
        self.assertCubeEqual(result[1], self.expected_period_truth)

    def test_error_column_missing(self):
        """Test that an error is raised if dataframe does not contain one of
        REPRESENTATION_COLUMNS."""
        msg = "None of the columns"
        with self.assertRaisesRegex(ValueError, msg):
            forecast_and_truth_dataframes_to_cubes(
                self.forecast_df.drop(columns=["percentile"]),
                self.truth_subset_df,
                self.cycletime,
                self.forecast_period,
                self.training_length,
            )

    def test_error_multiple_columns(self):
        """Test that an error is raised if dataframe contains more than one of
        REPRESENTATION_COLUMNS."""
        msg = "More than one column"
        self.forecast_df["realization"] = 0
        with self.assertRaisesRegex(ValueError, msg):
            forecast_and_truth_dataframes_to_cubes(
                self.forecast_df,
                self.truth_subset_df,
                self.cycletime,
                self.forecast_period,
                self.training_length,
            )

    def test_station_id_and_wmo_id(self):
        """Test that when station_id is present in both forecast and truth dataframes,
        output cubes contain station_id coordinate."""
        result = forecast_and_truth_dataframes_to_cubes(
            self.forecast_df_one_station_id,
            self.truth_df_one_station_id,
            self.cycletime,
            self.forecast_period,
            self.training_length,
        )
        self.assertEqual(len(result), 2)
        self.assertCubeEqual(result[0], self.expected_forecast_one_station_id)
        self.assertCubeEqual(result[1], self.expected_truth_one_station_id)

    def test_station_id_forecast_df_only(self):
        """Test that when station_id is only present in the forecast dataframe,
        a warning is raised."""
        msg = "station_id is only within the forecast DataFrame"
        with self.assertWarnsRegex(UserWarning, msg):
            forecast_and_truth_dataframes_to_cubes(
                self.forecast_df_one_station_id,
                self.truth_subset_df,
                self.cycletime,
                self.forecast_period,
                self.training_length,
            )

    def test_station_id_truth_df_only(self):
        """Test that when station_id is only present in the truth dataframe,
        a warning is raised."""
        msg = "station_id is only within the truth DataFrame"
        with self.assertWarnsRegex(UserWarning, msg):
            forecast_and_truth_dataframes_to_cubes(
                self.forecast_df,
                self.truth_df_one_station_id,
                self.cycletime,
                self.forecast_period,
                self.training_length,
            )

    def test_station_id_dummy_wmo_id(self):
        """Test that when station_id is present and wmo_id contains dummy data,
        station_id is used to match forecast and truth cubes."""
        self.forecast_df_one_station_id["wmo_id"] = "00000"
        self.truth_df_one_station_id["wmo_id"] = "00000"
        self.expected_truth_one_station_id.coord("wmo_id").points = ["00000"]
        self.expected_forecast_one_station_id.coord("wmo_id").points = ["00000"]
        result = forecast_and_truth_dataframes_to_cubes(
            self.forecast_df_one_station_id,
            self.truth_df_one_station_id,
            self.cycletime,
            self.forecast_period,
            self.training_length,
        )
        self.assertEqual(len(result), 2)
        self.assertCubeEqual(result[0], self.expected_forecast_one_station_id)
        self.assertCubeEqual(result[1], self.expected_truth_one_station_id)

    def test_units_in_truth(self):
        """Test that if truth_df contains a units column, it is used
        for units of truth output cube."""
        self.truth_subset_df["units"] = "Fahrenheit"
        self.truth_subset_df["ob_value"] = self.truth_subset_df["ob_value"] + 30
        expected_truth = self.expected_period_truth
        expected_truth.units = "Fahrenheit"
        expected_truth.data = expected_truth.data + 30
        result = forecast_and_truth_dataframes_to_cubes(
            self.forecast_df,
            self.truth_subset_df,
            self.cycletime,
            self.forecast_period,
            self.training_length,
        )
        self.assertEqual(len(result), 2)
        self.assertCubeEqual(result[0], self.expected_period_forecast)
        self.assertCubeEqual(result[1], expected_truth)

    def test_multiday_forecast_period(self):
        """Test for a multi-day forecast period to ensure that the
        validity times within the training dataset are always in
        the past, relative to the cycletime."""
        forecast_period = 30 * 3600
        self.forecast_df["forecast_period"] = np.timedelta64(
            forecast_period, "s"
        ).astype("timedelta64[ns]")
        for coord in ["forecast_reference_time", "blend_time"]:
            self.forecast_df[coord] = self.forecast_df[coord] - pd.Timedelta(
                1, unit="days"
            )
        fp_int = pd.Timedelta(forecast_period, "s").total_seconds()
        self.expected_period_forecast.coord("forecast_period").points = np.array(
            fp_int, dtype=TIME_COORDS["forecast_period"].dtype
        )
        self.expected_period_forecast.coord("forecast_period").bounds = np.array(
            [fp_int - self.period.total_seconds(), fp_int],
            dtype=TIME_COORDS["forecast_period"].dtype,
        )
        # Subtract number of seconds in a day.
        self.expected_period_forecast.coord("forecast_reference_time").points = (
            self.expected_period_forecast.coord("forecast_reference_time").points
            - 86400
        )
        result = forecast_and_truth_dataframes_to_cubes(
            self.forecast_df,
            self.truth_subset_df,
            self.cycletime,
            forecast_period,
            self.training_length,
        )
        self.assertEqual(len(result), 2)
        self.assertCubeEqual(result[0], self.expected_period_forecast)
        self.assertCubeEqual(result[1], self.expected_period_truth)

    def test_empty_forecast_dataframe(self):
        """Test that a None is returned for the forecast cube and the truth cube
        if an empty forecast dataframe is provided."""
        forecast_df = self.forecast_df.drop(self.forecast_df.index)
        result = forecast_and_truth_dataframes_to_cubes(
            forecast_df,
            self.truth_subset_df,
            self.cycletime,
            self.forecast_period,
            self.training_length,
        )
        self.assertEqual(len(result), 2)
        self.assertIsNone(result[0])
        self.assertIsNone(result[1])

    def test_site_absent_from_forecast(self):
        """Test for when a site is absent from the forecast dataframe."""
        df = self.forecast_df
        df = df.loc[df["wmo_id"].isin(self.wmo_ids[:-1])]
        expected_forecast = self.expected_period_forecast[:, :, :-1]
        expected_truth = self.expected_period_truth[:, :-1]
        result = forecast_and_truth_dataframes_to_cubes(
            df,
            self.truth_subset_df,
            self.cycletime,
            self.forecast_period,
            self.training_length,
        )
        self.assertEqual(len(result), 2)
        self.assertCubeEqual(result[0], expected_forecast)
        self.assertCubeEqual(result[1], expected_truth)

    def test_site_absent_from_truth(self):
        """Test for when a site is absent from the truth dataframe."""
        df = self.truth_subset_df
        df = df.loc[df["wmo_id"].isin(self.wmo_ids[:-1])]
        expected_forecast = self.expected_period_forecast[:, :, :-1]
        expected_truth = self.expected_period_truth[:, :-1]
        result = forecast_and_truth_dataframes_to_cubes(
            self.forecast_df,
            df,
            self.cycletime,
            self.forecast_period,
            self.training_length,
        )
        self.assertEqual(len(result), 2)
        self.assertCubeEqual(result[0], expected_forecast)
        self.assertCubeEqual(result[1], expected_truth)

    def test_site_coord_mismatch(self):
        """Test for a mismatch in the location of a site between the truths
        and forecasts. In this case, the position (lat/lon/alt) from the
        forecast will be used."""
        self.truth_subset_df.loc[::3, "altitude"] = 45
        self.truth_subset_df.loc[::3, "latitude"] = 52
        self.truth_subset_df.loc[::3, "longitude"] = -12
        result = forecast_and_truth_dataframes_to_cubes(
            self.forecast_df,
            self.truth_subset_df,
            self.cycletime,
            self.forecast_period,
            self.training_length,
        )
        self.assertEqual(len(result), 2)
        self.assertCubeEqual(result[0], self.expected_period_forecast)
        self.assertCubeEqual(result[1], self.expected_period_truth)

    def test_no_observations_for_a_time(self):
        """Test for a time point having no observations."""
        truth_subset_df = self.truth_subset_df[
            self.truth_subset_df["time"].isin([self.time2, self.time3])
        ]
        result = forecast_and_truth_dataframes_to_cubes(
            self.forecast_df,
            truth_subset_df,
            self.cycletime,
            self.forecast_period,
            self.training_length,
        )
        self.assertEqual(len(result), 2)
        self.assertCubeEqual(result[0], self.expected_period_forecast[:, 1:])
        self.assertCubeEqual(result[1], self.expected_period_truth[1:])

    def test_no_forecasts_for_a_time(self):
        """Test for a time point having no forecasts."""
        forecast_df = self.forecast_df[
            self.forecast_df["time"].isin([self.time2, self.time3])
        ]
        result = forecast_and_truth_dataframes_to_cubes(
            forecast_df,
            self.truth_subset_df,
            self.cycletime,
            self.forecast_period,
            self.training_length,
        )
        self.assertEqual(len(result), 2)
        self.assertCubeEqual(result[0], self.expected_period_forecast[:, 1:])
        self.assertCubeEqual(result[1], self.expected_period_truth[1:])

    def test_new_site_with_only_one_forecast_and_truth(self):
        """Test for a site that has a forecast and truth data point for the most
        recent time only. Other sites are present at all forecast and truth times.
        This mimics the situation when a new site is added to the forecast and truth
        DataFrames. The forecast and truth cubes generated have three sites
        with the 'new' site having NaNs for all time points except for the most
        recent time."""
        self.expected_period_forecast.data[:, :2, -1] = np.nan
        self.expected_period_truth.data[:2, -1] = np.nan

        df = self.forecast_df
        condition = (df["time"].isin([self.time1, self.time2])) & (
            df["wmo_id"] == self.wmo_ids[2]
        )
        forecast_df = df.drop(df[condition].index)
        df = self.truth_subset_df
        condition = (df["time"].isin([self.time1, self.time2])) & (
            df["wmo_id"] == self.wmo_ids[2]
        )
        truth_df = df.drop(df[condition].index)

        result = forecast_and_truth_dataframes_to_cubes(
            forecast_df,
            truth_df,
            self.cycletime,
            self.forecast_period,
            self.training_length,
        )
        self.assertEqual(len(result), 2)
        self.assertCubeEqual(result[0], self.expected_period_forecast)
        self.assertCubeEqual(result[1], self.expected_period_truth)

    def test_old_site_with_only_one_forecast_and_truth(self):
        """Test for a site that has a forecast and truth data point for the oldest
        time only. Other sites are present at all forecast and truth times. This
        mimics the situation when forecasts and truths are no longer available
        for a particular site. The forecast and truth cubes generated have three sites
        with the 'old' site having NaNs for all time points except for the oldest
        time."""
        self.expected_period_forecast.data[:, 1:, -1] = np.nan
        self.expected_period_truth.data[1:, -1] = np.nan

        df = self.forecast_df
        condition = (df["time"].isin([self.time2, self.time3])) & (
            df["wmo_id"] == self.wmo_ids[2]
        )
        forecast_df = df.drop(df[condition].index)
        df = self.truth_subset_df
        condition = (df["time"].isin([self.time2, self.time3])) & (
            df["wmo_id"] == self.wmo_ids[2]
        )
        truth_df = df.drop(df[condition].index)

        result = forecast_and_truth_dataframes_to_cubes(
            forecast_df,
            truth_df,
            self.cycletime,
            self.forecast_period,
            self.training_length,
        )
        self.assertEqual(len(result), 2)
        self.assertCubeEqual(result[0], self.expected_period_forecast)
        self.assertCubeEqual(result[1], self.expected_period_truth)

    def test_station_id_new_site_with_only_one_forecast(self):
        """Test for a site that has a forecast and truth data point for the most
        recent time only. Other sites are present at all forecast and truth times.
        Only the forecast DataFrame has a station_id column. This mimics the
        situation when a new site is added to the forecast and truth
        DataFrames. The forecast and truth cubes generated have three sites
        with the 'new' site having NaNs for all time points except for the most
        recent time. The resulting forecast cube has a station_id coordinate."""
        self.truth_df_multi_station_id.drop(columns="station_id", inplace=True)
        self.expected_truth_multi_station_id.remove_coord("station_id")

        result = forecast_and_truth_dataframes_to_cubes(
            self.forecast_df_multi_station_id,
            self.truth_df_multi_station_id,
            self.cycletime,
            self.forecast_period,
            self.training_length,
        )

        self.assertEqual(len(result), 2)
        self.assertCubeEqual(result[0], self.expected_forecast_multi_station_id)
        self.assertCubeEqual(result[1], self.expected_truth_multi_station_id)

    def test_station_id_new_site_with_only_one_truth(self):
        """Test for a site that has a forecast and truth data point for the most
        recent time only. Other sites are present at all forecast and truth times.
        Only the truth DataFrame has a station_id column. This mimics the situation
        when a new site is added to the forecast and truth DataFrames. The forecast
        and truth cubes generated have three sites with the 'new' site having NaNs
        for all time points except for the most recent time. The resulting truth cube
        has a station_id coordinate."""
        self.forecast_df_multi_station_id.drop(columns="station_id", inplace=True)
        self.expected_forecast_multi_station_id.remove_coord("station_id")

        result = forecast_and_truth_dataframes_to_cubes(
            self.forecast_df_multi_station_id,
            self.truth_df_multi_station_id,
            self.cycletime,
            self.forecast_period,
            self.training_length,
        )

        self.assertEqual(len(result), 2)
        self.assertCubeEqual(result[0], self.expected_forecast_multi_station_id)
        self.assertCubeEqual(result[1], self.expected_truth_multi_station_id)

    def test_station_id_new_site_with_only_one_forecast_and_truth(self):
        """Test for a site that has a forecast and truth data point for the most
        recent time only. Other sites are present at all forecast and truth times.
        Both the forecast and truth DataFrames have a station_id column. This mimics
        the situation when a new site is added to the forecast and truth DataFrames.
        The forecast and truth cubes generated have three sites with the 'new' site
        having NaNs for all time points except for the most recent time. Both the
        resulting forecast and truth cubes have a station_id coordinate."""
        result = forecast_and_truth_dataframes_to_cubes(
            self.forecast_df_multi_station_id,
            self.truth_df_multi_station_id,
            self.cycletime,
            self.forecast_period,
            self.training_length,
        )

        self.assertEqual(len(result), 2)
        self.assertCubeEqual(result[0], self.expected_forecast_multi_station_id)
        self.assertCubeEqual(result[1], self.expected_truth_multi_station_id)

    def test_sites_no_overlapping_dates(self):
        """Test for when there are sites with no overlapping dates within the
        forecasts and the truths. This tests for when sites are potentially
        being both added and removed from the forecast and truth DataFrames.
        In the resulting forecast and truth cubes, NaN values are present
        for site and time pairs that were not in the input forecasts and
        truth DataFrames.
        """
        self.expected_period_forecast.data[:, 0, 0] = np.nan
        self.expected_period_forecast.data[:, 2, 2] = np.nan
        self.expected_period_truth.data[0, 0] = np.nan
        self.expected_period_truth.data[2, 2] = np.nan

        df = self.forecast_df
        condition1 = (df["time"].isin([self.time1])) & (df["wmo_id"] == self.wmo_ids[0])
        condition2 = (df["time"].isin([self.time3])) & (df["wmo_id"] == self.wmo_ids[2])
        forecast_df = df.drop(df[condition1].index.union(df[condition2].index))

        df = self.truth_subset_df
        condition1 = (df["time"].isin([self.time1])) & (df["wmo_id"] == self.wmo_ids[0])
        condition2 = (df["time"].isin([self.time3])) & (df["wmo_id"] == self.wmo_ids[2])
        truth_df = df.drop(df[condition1].index.union(df[condition2].index))

        result = forecast_and_truth_dataframes_to_cubes(
            forecast_df,
            truth_df,
            self.cycletime,
            self.forecast_period,
            self.training_length,
        )

        self.assertEqual(len(result), 2)
        self.assertCubeEqual(result[0], self.expected_period_forecast)
        self.assertCubeEqual(result[1], self.expected_period_truth)

    def test_percentile_extract(self):
        """Test the desired percentiles are extracted."""
        expected_period_forecast = self.expected_period_forecast[::2]
        expected_period_forecast.coord("realization").points = np.array(
            [0, 1], dtype=np.int32
        )
        forecast_df = self.forecast_df.replace(
            {"percentile": self.percentiles[0]}, 100 / 3
        )
        forecast_df = forecast_df.replace(
            {"percentile": self.percentiles[2]}, (2 / 3) * 100
        )
        result = forecast_and_truth_dataframes_to_cubes(
            forecast_df,
            self.truth_subset_df,
            self.cycletime,
            self.forecast_period,
            self.training_length,
            percentiles=["33.333333", "66.666666"],
        )
        self.assertEqual(len(result), 2)
        self.assertCubeEqual(result[0], expected_period_forecast)
        self.assertCubeEqual(result[1], self.expected_period_truth)

    def test_not_quantiles(self):
        """Test if the percentiles can not be considered to be quantiles."""
        self.forecast_df = self.forecast_df.replace(
            {"percentile": self.percentiles[0]}, 10.0
        )
        msg = "The forecast percentiles can not be considered as quantiles"
        with self.assertRaisesRegex(ValueError, msg):
            forecast_and_truth_dataframes_to_cubes(
                self.forecast_df,
                self.truth_subset_df,
                self.cycletime,
                self.forecast_period,
                self.training_length,
            )

    def test_missing_observation_at_start(self):
        """Test a truth DataFrame with one missing observation
        within the first row of the dataframe is converted correctly
        into an iris Cube."""
        df = self.truth_subset_df.drop(0)
        self.expected_period_truth.data[0, 0] = np.nan
        result = forecast_and_truth_dataframes_to_cubes(
            self.forecast_df,
            df,
            self.cycletime,
            self.forecast_period,
            self.training_length,
        )
        self.assertEqual(len(result), 2)
        self.assertCubeEqual(result[0], self.expected_period_forecast)
        self.assertCubeEqual(result[1], self.expected_period_truth)

    def test_missing_observation_in_middle(self):
        """Test a truth DataFrame with one missing observation
        within a central row within the dataframe is converted correctly
        into an iris Cube."""
        df = self.truth_subset_df.drop(4)
        self.expected_period_truth.data[1, 1] = np.nan
        result = forecast_and_truth_dataframes_to_cubes(
            self.forecast_df,
            df,
            self.cycletime,
            self.forecast_period,
            self.training_length,
        )
        self.assertEqual(len(result), 2)
        self.assertCubeEqual(result[0], self.expected_period_forecast)
        self.assertCubeEqual(result[1], self.expected_period_truth)

    def test_missing_observation_at_end(self):
        """Test a truth DataFrame with one missing observation
        within the last row of the dataframe is converted correctly
        into an iris Cube."""
        df = self.truth_subset_df.drop(8)
        self.expected_period_truth.data[-1, -1] = np.nan
        result = forecast_and_truth_dataframes_to_cubes(
            self.forecast_df,
            df,
            self.cycletime,
            self.forecast_period,
            self.training_length,
        )
        self.assertEqual(len(result), 2)
        self.assertCubeEqual(result[0], self.expected_period_forecast)
        self.assertCubeEqual(result[1], self.expected_period_truth)

    def test_forecast_missing_compulsory_columns(self):
        """Test if there are missing compulsory columns in the forecast
        dataframe."""
        df = self.forecast_df.rename(columns={"diagnostic": "diag"})
        msg = "The following compulsory column\\(s\\) are missing"
        with self.assertRaisesRegex(ValueError, msg):
            forecast_and_truth_dataframes_to_cubes(
                df,
                self.truth_subset_df,
                self.cycletime,
                self.forecast_period,
                self.training_length,
            )

    def test_truth_missing_compulsory_columns(self):
        """Test if there are missing compulsory columns in the truth
        dataframe."""
        df = self.truth_subset_df.rename(columns={"diagnostic": "diag"})
        msg = "The following compulsory column\\(s\\) are missing"
        with self.assertRaisesRegex(ValueError, msg):
            forecast_and_truth_dataframes_to_cubes(
                self.forecast_df,
                df,
                self.cycletime,
                self.forecast_period,
                self.training_length,
            )

    def test_duplicate_cycle_forecasts(self):
        """Test that a forecast cube is still produced if a duplicated
        cycle of forecasts is provided."""
        forecast_df_with_duplicates = pd.concat(
            [self.forecast_df, self.forecast_df.iloc[:9]], ignore_index=True
        )
        result = forecast_and_truth_dataframes_to_cubes(
            forecast_df_with_duplicates,
            self.truth_subset_df,
            self.cycletime,
            self.forecast_period,
            self.training_length,
        )
        self.assertEqual(len(result), 2)
        self.assertCubeEqual(result[0], self.expected_period_forecast)
        self.assertCubeEqual(result[1], self.expected_period_truth)

    def test_duplicate_cycle_truths(self):
        """Test that a truth cube is still produced if duplicate
        truths for a given validity time are provided."""
        truth_df_with_duplicates = pd.concat(
            [self.truth_subset_df, self.truth_subset_df.iloc[:3]], ignore_index=True
        )
        result = forecast_and_truth_dataframes_to_cubes(
            self.forecast_df,
            truth_df_with_duplicates,
            self.cycletime,
            self.forecast_period,
            self.training_length,
        )
        self.assertEqual(len(result), 2)
        self.assertCubeEqual(result[0], self.expected_period_forecast)
        self.assertCubeEqual(result[1], self.expected_period_truth)

    def test_duplicate_row_forecasts(self):
        """Test that a forecast cube is still produced if duplicated
        forecasts are provided."""
        # Use results from the first realization,
        # equivalent to the 50th percentile.
        expected_period_forecast = self.expected_period_forecast[1, :, :]
        expected_period_forecast.coord("realization").points = np.array([0], np.int32)

        # Use 50th percentile only
        forecast_subset_df = self.forecast_df[self.forecast_df["percentile"] == 50.0]

        # Duplicate first row twice.
        forecast_df_with_duplicates = pd.concat(
            [
                forecast_subset_df,
                forecast_subset_df.iloc[[0]],
                forecast_subset_df.iloc[[0]],
            ],
            ignore_index=True,
        )
        forecast_df_with_duplicates.at[0, "forecast"] = 6.0
        forecast_df_with_duplicates.at[9, "forecast"] = 8.0

        result = forecast_and_truth_dataframes_to_cubes(
            forecast_df_with_duplicates,
            self.truth_subset_df,
            self.cycletime,
            self.forecast_period,
            self.training_length,
        )

        self.assertEqual(len(result), 2)
        self.assertCubeEqual(result[0], expected_period_forecast)
        self.assertCubeEqual(result[1], self.expected_period_truth)

    def test_duplicate_row_truths(self):
        """Test that a truth cube is still produced if duplicated
        truths for a given validity time are provided."""
        # Duplicate first row twice.
        truth_df_with_duplicates = pd.concat(
            [
                self.truth_subset_df,
                self.truth_subset_df.iloc[[0]],
                self.truth_subset_df.iloc[[0]],
            ],
            ignore_index=True,
        )
        truth_df_with_duplicates.at[0, "ob_value"] = 6.0
        truth_df_with_duplicates.at[9, "ob_value"] = 8.0
        result = forecast_and_truth_dataframes_to_cubes(
            self.forecast_df,
            truth_df_with_duplicates,
            self.cycletime,
            self.forecast_period,
            self.training_length,
        )

        self.assertEqual(len(result), 2)
        self.assertCubeEqual(result[0], self.expected_period_forecast)
        self.assertCubeEqual(result[1], self.expected_period_truth)

    def test_forecast_additional_columns_present(self):
        """Test that if there are additional columns present
        in the forecast dataframe, these have no impact."""
        self.forecast_df["surface_type"] = "grass"
        result = forecast_and_truth_dataframes_to_cubes(
            self.forecast_df,
            self.truth_subset_df,
            self.cycletime,
            self.forecast_period,
            self.training_length,
        )
        self.assertEqual(len(result), 2)

    def test_truth_additional_columns_present(self):
        """Test that if there are additional columns present
        in the truth dataframe, these have no impact."""
        self.truth_subset_df["surface_type"] = "grass"
        result = forecast_and_truth_dataframes_to_cubes(
            self.forecast_df,
            self.truth_subset_df,
            self.cycletime,
            self.forecast_period,
            self.training_length,
        )
        self.assertEqual(len(result), 2)

    def test_error_multiple_experiment_values(self):
        """Test an error is raised if multiple experiment values are in
        the dataframe."""
        experiment2 = self.forecast_df.copy()
        experiment2["experiment"] = "threshold"
        forecast_df = pd.concat([self.forecast_df, experiment2])
        msg = "More than one value for the experiment column found"
        with self.assertRaisesRegex(ValueError, msg):
            forecast_and_truth_dataframes_to_cubes(
                forecast_df,
                self.truth_subset_df,
                self.cycletime,
                self.forecast_period,
                self.training_length,
            )

    def test_select_single_experiment_value(self):
        """Test selecting a single experiment value from the dataframe"""
        experiment2 = self.forecast_df.copy()
        experiment2["experiment"] = "threshold"
        # Set original data to different values to make sure the correct experiment
        # is picked up
        self.forecast_df["forecast"] = 0.0
        forecast_df = pd.concat([self.forecast_df, experiment2])
        result = forecast_and_truth_dataframes_to_cubes(
            forecast_df,
            self.truth_subset_df,
            self.cycletime,
            self.forecast_period,
            self.training_length,
            experiment="threshold",
        )
        self.assertEqual(len(result), 2)
        self.assertCubeEqual(result[0], self.expected_period_forecast)
        self.assertCubeEqual(result[1], self.expected_period_truth)

    def test_forecast_missing_columns_and_additional_columns(self):
        """Test if there are missing compulsory columns in the forecast
        dataframe and there are additional non-compulsory columns."""
        self.forecast_df["station_id"] = "11111"
        df = self.forecast_df.rename(columns={"diagnostic": "diag"})
        msg = "The following compulsory column\\(s\\) are missing"
        with self.assertRaisesRegex(ValueError, msg):
            forecast_and_truth_dataframes_to_cubes(
                df,
                self.truth_subset_df,
                self.cycletime,
                self.forecast_period,
                self.training_length,
            )

    def test_truth_missing_columns_and_additional_columns(self):
        """Test if there are missing compulsory columns in the truth
        dataframe and there are additional non-compulsory columns."""
        self.truth_subset_df["station_id"] = "11111"
        df = self.truth_subset_df.rename(columns={"diagnostic": "diag"})
        msg = "The following compulsory column\\(s\\) are missing"
        with self.assertRaisesRegex(ValueError, msg):
            forecast_and_truth_dataframes_to_cubes(
                self.forecast_df,
                df,
                self.cycletime,
                self.forecast_period,
                self.training_length,
            )


if __name__ == "__main__":
    unittest.main()
