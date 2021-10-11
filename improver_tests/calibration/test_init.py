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
"""Unit tests for calibration.__init__"""

import unittest
from datetime import datetime

import iris
import numpy as np
import pandas as pd
import pytest

from improver.calibration import (
    forecast_and_truth_tables_to_cubes,
    forecast_table_to_cube,
    split_forecasts_and_truth,
    truth_table_to_cube,
)
from improver.metadata.constants.time_types import TIME_COORDS
from improver.spotdata.build_spotdata_cube import build_spotdata_cube
from improver.synthetic_data.set_up_test_cubes import (
    set_up_probability_cube,
    set_up_variable_cube,
)


class Test_split_forecasts_and_truth(unittest.TestCase):

    """Test the split_forecasts_and_truth method."""

    def setUp(self):
        """Create cubes for testing the split_forecasts_and_truth method.
        Forecast data is all set to 1, and truth data to 0, allowing for a
        simple check that the cubes have been separated as expected."""

        thresholds = [283, 288]
        probability_data = np.ones((2, 4, 4), dtype=np.float32)
        realization_data = np.ones((4, 4), dtype=np.float32)

        self.truth_attribute = "mosg__model_configuration=uk_det"
        truth_attributes = {"mosg__model_configuration": "uk_det"}

        probability_forecast_1 = set_up_probability_cube(probability_data, thresholds)
        probability_forecast_2 = set_up_probability_cube(
            probability_data,
            thresholds,
            time=datetime(2017, 11, 11, 4, 0),
            frt=datetime(2017, 11, 11, 0, 0),
        )
        self.probability_forecasts = [probability_forecast_1, probability_forecast_2]

        probability_truth_1 = probability_forecast_1.copy(
            data=np.zeros((2, 4, 4), dtype=np.float32)
        )
        probability_truth_2 = probability_forecast_2.copy(
            data=np.zeros((2, 4, 4), dtype=np.float32)
        )
        probability_truth_1.attributes.update(truth_attributes)
        probability_truth_2.attributes.update(truth_attributes)
        self.probability_truths = [probability_truth_1, probability_truth_2]

        realization_forecast_1 = set_up_variable_cube(realization_data)
        realization_forecast_2 = set_up_variable_cube(
            realization_data,
            time=datetime(2017, 11, 11, 4, 0),
            frt=datetime(2017, 11, 11, 0, 0),
        )
        self.realization_forecasts = [realization_forecast_1, realization_forecast_2]

        realization_truth_1 = realization_forecast_1.copy(
            data=np.zeros((4, 4), dtype=np.float32)
        )
        realization_truth_2 = realization_forecast_2.copy(
            data=np.zeros((4, 4), dtype=np.float32)
        )
        realization_truth_1.attributes.update(truth_attributes)
        realization_truth_2.attributes.update(truth_attributes)
        self.realization_truths = [realization_truth_1, realization_truth_2]

        self.landsea_mask = realization_truth_1.copy()
        self.landsea_mask.rename("land_binary_mask")

    def test_probability_data(self):
        """Test that when multiple probability forecast cubes and truth cubes
        are provided, the groups are created as expected."""

        forecast, truth, land_sea_mask = split_forecasts_and_truth(
            self.probability_forecasts + self.probability_truths, self.truth_attribute
        )

        self.assertIsInstance(forecast, iris.cube.Cube)
        self.assertIsInstance(truth, iris.cube.Cube)
        self.assertIsNone(land_sea_mask, None)
        self.assertSequenceEqual((2, 2, 4, 4), forecast.shape)
        self.assertSequenceEqual((2, 2, 4, 4), truth.shape)
        self.assertTrue(np.all(forecast.data))
        self.assertFalse(np.any(truth.data))

    def test_probability_data_with_land_sea_mask(self):
        """Test that when multiple probability forecast cubes, truth cubes, and
        a single land-sea mask are provided, the groups are created as
        expected."""

        forecast, truth, land_sea_mask = split_forecasts_and_truth(
            self.probability_forecasts + self.probability_truths + [self.landsea_mask],
            self.truth_attribute,
        )

        self.assertIsInstance(forecast, iris.cube.Cube)
        self.assertIsInstance(truth, iris.cube.Cube)
        self.assertIsInstance(land_sea_mask, iris.cube.Cube)
        self.assertEqual("land_binary_mask", land_sea_mask.name())
        self.assertSequenceEqual((2, 2, 4, 4), forecast.shape)
        self.assertSequenceEqual((2, 2, 4, 4), truth.shape)
        self.assertTrue(np.all(forecast.data))
        self.assertFalse(np.any(truth.data))
        self.assertSequenceEqual((4, 4), land_sea_mask.shape)

    def test_realization_data(self):
        """Test that when multiple forecast cubes and truth cubes are provided,
        the groups are created as expected."""

        forecast, truth, land_sea_mask = split_forecasts_and_truth(
            self.realization_forecasts + self.realization_truths, self.truth_attribute
        )

        self.assertIsInstance(forecast, iris.cube.Cube)
        self.assertIsInstance(truth, iris.cube.Cube)
        self.assertIsNone(land_sea_mask, None)
        self.assertSequenceEqual((2, 4, 4), forecast.shape)
        self.assertSequenceEqual((2, 4, 4), truth.shape)
        self.assertTrue(np.all(forecast.data))
        self.assertFalse(np.any(truth.data))

    def test_realization_data_with_land_sea_mask(self):
        """Test that when multiple forecast cubes, truth cubes, and
        a single land-sea mask are provided, the groups are created as
        expected."""

        forecast, truth, land_sea_mask = split_forecasts_and_truth(
            self.realization_forecasts + self.realization_truths + [self.landsea_mask],
            self.truth_attribute,
        )

        self.assertIsInstance(forecast, iris.cube.Cube)
        self.assertIsInstance(truth, iris.cube.Cube)
        self.assertIsInstance(land_sea_mask, iris.cube.Cube)
        self.assertEqual("land_binary_mask", land_sea_mask.name())
        self.assertSequenceEqual((2, 4, 4), forecast.shape)
        self.assertSequenceEqual((2, 4, 4), truth.shape)
        self.assertTrue(np.all(forecast.data))
        self.assertFalse(np.any(truth.data))
        self.assertSequenceEqual((4, 4), land_sea_mask.shape)

    def test_exception_for_multiple_land_sea_masks(self):
        """Test that when multiple land-sea masks are provided an exception is
        raised."""

        msg = "Expected one cube for land-sea mask."
        with self.assertRaisesRegex(IOError, msg):
            split_forecasts_and_truth(
                self.realization_forecasts
                + self.realization_truths
                + [self.landsea_mask, self.landsea_mask],
                self.truth_attribute,
            )

    def test_exception_for_unintended_cube_combination(self):
        """Test that when the forecast and truth cubes have different names,
        indicating different diagnostics, an exception is raised."""

        self.realization_truths[0].rename("kitten_density")

        msg = "Must have cubes with 1 or 2 distinct names."
        with self.assertRaisesRegex(ValueError, msg):
            split_forecasts_and_truth(
                self.realization_forecasts
                + self.realization_truths
                + [self.landsea_mask, self.landsea_mask],
                self.truth_attribute,
            )

    def test_exception_for_missing_truth_inputs(self):
        """Test that when all truths are missing an exception is raised."""

        self.realization_truths = []

        msg = "Missing truth input."
        with self.assertRaisesRegex(IOError, msg):
            split_forecasts_and_truth(
                self.realization_forecasts
                + self.realization_truths
                + [self.landsea_mask],
                self.truth_attribute,
            )

    def test_exception_for_missing_forecast_inputs(self):
        """Test that when all forecasts are missing an exception is raised."""

        self.realization_forecasts = []

        msg = "Missing historical forecast input."
        with self.assertRaisesRegex(IOError, msg):
            split_forecasts_and_truth(
                self.realization_forecasts
                + self.realization_truths
                + [self.landsea_mask],
                self.truth_attribute,
            )


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


class Test_shared_dataframes(unittest.TestCase):

    """A shared dataframe creation class."""

    def setUp(self):
        """Set-up forecast and truth dataframes."""
        pytest.importorskip("pandas")

        data = np.array(
            [5.2, 0.3, 20.4, 6.5, 3.1, 21.5, 7.2, 4.2, 24.3], dtype=np.float32
        )
        self.forecast_data = np.tile(data, 3)

        self.frt1 = np.datetime64("2017-07-20T12:00:00")
        self.frt2 = np.datetime64("2017-07-21T12:00:00")
        self.frt3 = np.datetime64("2017-07-22T12:00:00")

        self.fp = np.timedelta64("6", "h")

        self.time1 = np.datetime64("2017-07-20T18:00:00")
        self.time2 = np.datetime64("2017-07-21T18:00:00")
        self.time3 = np.datetime64("2017-07-22T18:00:00")

        self.wmo_ids = ["03002", "03003", "03004"]
        self.percentiles = [25, 50.0, 75.0]
        diag = "air_temperature"
        self.cf_name = "air_temperature"
        self.latitudes = [50, 60, 70]
        self.longitudes = [-10, 0, 10]
        self.altitudes = [10, 20, 30]
        self.period = 3600
        self.height = 1.5
        self.units = "Celsius"

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
            "height": [self.height] * 27,
            "cf_name": [self.cf_name] * 27,
            "units": [self.units] * 27,
        }

        self.forecast_df = pd.DataFrame(df_dict)

        data = np.array([6.8, 2.7, 21.2], dtype=np.float32)
        self.truth_data = np.tile(data, 3)

        df_dict = {
            "ob_value": self.truth_data,
            "time": np.repeat([self.time1, self.time2, self.time3], 3),
            "wmo_id": self.wmo_ids * 3,
            "diagnostic": [diag] * 9,
            "latitude": self.latitudes * 3,
            "longitude": self.longitudes * 3,
            "altitude": self.altitudes * 3,
            "period": [self.period] * 9,
            "height": [self.height] * 9,
            "cf_name": [self.cf_name] * 9,
            "units": [self.units] * 9,
        }

        self.truth_df = pd.DataFrame(df_dict)

        self.validity_time = self.time3
        self.forecast_period = 6
        self.training_length = 3
        self.date_range = pd.date_range(
            end=self.validity_time, periods=int(self.training_length), freq="D"
        )

        self.height_coord = iris.coords.AuxCoord(
            np.array(self.height, dtype=np.float32), "height", units="m",
        )


class Test_constructed_forecast_cubes(Test_shared_dataframes):

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
                time.astype(TIME_COORDS["time"].dtype),
                "time",
                bounds=[
                    t.astype(TIME_COORDS["time"].dtype)
                    for t in [time - self.period, time]
                ],
                units=TIME_COORDS["time"].units,
            )

            fp_point = self.fp.astype("timedelta64[s]")
            fp_coord = iris.coords.AuxCoord(
                fp_point.astype(TIME_COORDS["forecast_period"].dtype),
                "forecast_period",
                bounds=[
                    f.astype(TIME_COORDS["forecast_period"].dtype)
                    for f in [fp_point - self.period, fp_point]
                ],
                units=TIME_COORDS["forecast_period"].units,
            )

            frt_coord = iris.coords.AuxCoord(
                frt.astype(TIME_COORDS["forecast_reference_time"].dtype),
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
                    data_slice,
                    self.cf_name,
                    self.units,
                    self.altitudes,
                    self.latitudes,
                    self.longitudes,
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


class Test_constructed_truth_cubes(Test_shared_dataframes):

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
                time.astype(TIME_COORDS["time"].dtype),
                "time",
                bounds=[
                    t.astype(TIME_COORDS["time"].dtype)
                    for t in [time - self.period, time]
                ],
                units=TIME_COORDS["time"].units,
            )
            cubes.append(
                build_spotdata_cube(
                    data_slice,
                    self.cf_name,
                    self.units,
                    self.altitudes,
                    self.latitudes,
                    self.longitudes,
                    wmo_id=self.wmo_ids,
                    scalar_coords=[time_coord, self.height_coord],
                )
            )
        self.expected_period_truth = cubes.merge_cube()
        self.expected_instantaneous_truth = self.expected_period_truth.copy()
        self.expected_instantaneous_truth.coord("time").bounds = None


class Test_forecast_table_to_cube(Test_constructed_forecast_cubes):

    """Test the forecast_table_to_cube function."""

    def setUp(self):
        """Set-up forecast table for testing."""
        super().setUp()

    def test_three_day_training_period_diag(self):
        """Test an input DataFrame is converted correctly into an Iris Cube
        for a three day training length for a period diagnostic."""
        result = forecast_table_to_cube(
            self.forecast_df, self.date_range, self.forecast_period
        )
        self.assertEqual(result, self.expected_period_forecast)

    def test_three_day_training_instantaneous_diag(self):
        """Test an input DataFrame is converted correctly into an Iris Cube
        for a three day training length for an instantaneous diagnostic."""
        self.forecast_df["period"] = np.nan
        result = forecast_table_to_cube(
            self.forecast_df, self.date_range, self.forecast_period
        )
        self.assertEqual(result, self.expected_instantaneous_forecast)

    def test_two_day_training(self):
        """Test an input DataFrame is converted correctly into an Iris Cube
        for a two day training length."""
        training_length = 2
        date_range = pd.date_range(
            end=self.validity_time, periods=int(training_length), freq="D"
        )
        result = forecast_table_to_cube(
            self.forecast_df, date_range, self.forecast_period
        )
        self.assertEqual(result, self.expected_period_forecast[:, 1:])

    def test_empty_table(self):
        """Test if none of the required data is available in the dataframe."""
        forecast_period = 7
        msg = "No entries matching these dates"
        with self.assertRaisesRegex(ValueError, msg):
            forecast_table_to_cube(self.forecast_df, self.date_range, forecast_period)

    def test_nonunique_values_in_column(self):
        """Test if there are multiple non-unique values in a column of the
        dataframe."""
        df = self.forecast_df.copy()
        df.at[0, "period"] = 7200
        msg = "Multiple values provided for the period"
        with self.assertRaisesRegex(ValueError, msg):
            forecast_table_to_cube(df, self.date_range, self.forecast_period)


class Test_truth_table_to_cube(Test_constructed_truth_cubes):

    """Test the truth_table_to_cube function."""

    def setUp(self):
        super().setUp()

    def test_three_day_training_period_diag(self):
        """Test an input DataFrame is converted correctly into an Iris Cube
        for a three day training length for a period diagnostic."""
        result = truth_table_to_cube(self.truth_df, self.date_range)
        self.assertEqual(result, self.expected_period_truth)

    def test_three_day_training_instantaneous_diag(self):
        """Test an input DataFrame is converted correctly into an Iris Cube
        for a three day training length for an instantaneous diagnostic."""
        self.truth_df["period"] = np.nan
        result = truth_table_to_cube(self.truth_df, self.date_range)
        self.assertEqual(result, self.expected_instantaneous_truth)

    def test_two_day_training(self):
        """Test an input DataFrame is converted correctly into an Iris Cube
        for a two day training length."""
        training_length = 2
        date_range = pd.date_range(
            end=self.validity_time, periods=int(training_length), freq="D"
        )
        result = truth_table_to_cube(self.truth_df, date_range)
        self.assertEqual(result, self.expected_period_truth[1:, :])

    def test_missing_observation(self):
        """Test an input DataFrame is converted correctly into an Iris Cube
        if an observation is missing at a particular time."""
        df = self.truth_df.head(-1)
        self.expected_period_truth.data[-1, -1] = np.nan
        result = truth_table_to_cube(df, self.date_range)
        np.testing.assert_array_equal(result.data, self.expected_period_truth.data)
        for coord in ["altitude", "latitude", "longitude"]:
            self.assertEqual(
                result.coord(coord), self.expected_period_truth.coord(coord)
            )

    def test_moving_sites(self):
        """Test an input DataFrame is converted correctly into an Iris Cube
        if the position of a particular site varies during the training period."""
        df = self.truth_df.copy()
        df.at[0, "altitude"] = 45
        df.at[0, "latitude"] = 52
        df.at[0, "longitude"] = -12
        result = truth_table_to_cube(df, self.date_range)
        self.assertEqual(result, self.expected_period_truth)

    def test_empty_table(self):
        """Test if none of the required data is available in the dataframe."""
        validity_time = np.datetime64("2017-07-22T19:00:00")
        date_range = pd.date_range(
            end=validity_time, periods=int(self.training_length), freq="D"
        )
        msg = "No entries matching these dates"
        with self.assertRaisesRegex(ValueError, msg):
            truth_table_to_cube(self.truth_df, date_range)

    def test_nonunique_values_in_column(self):
        """Test if there are multiple non-unique values in a column of the
        dataframe."""
        df = self.truth_df.copy()
        df.at[0, "period"] = 7200
        msg = "Multiple values provided for the period"
        with self.assertRaisesRegex(ValueError, msg):
            truth_table_to_cube(df, self.date_range)


class Test_forecast_and_truth_tables_to_cubes(
    Test_constructed_forecast_cubes, Test_constructed_truth_cubes
):

    """Test the forecast_and_truth_tables_to_cubes function."""

    def setUp(self):
        """Set up dataframes for testing."""
        super().setUp()
        self.cycletime = "20170723T1200Z"

    def test_basic(self):
        """Test the expected cubes are generated from the input dataframes."""
        result = forecast_and_truth_tables_to_cubes(
            self.forecast_df,
            self.truth_df,
            self.cycletime,
            self.forecast_period,
            self.training_length,
        )
        self.assertEqual(len(result), 2)
        self.assertEqual(
            result, (self.expected_period_forecast, self.expected_period_truth)
        )

    def test_site_mismatch(self):
        """Test for a mismatch in the sites available as truths and forecasts."""
        df = self.truth_df.copy()
        df = df.loc[df["wmo_id"].isin(self.wmo_ids[:-1])]
        expected_forecast = self.expected_period_forecast[:, :, :-1]
        expected_truth = self.expected_period_truth[:, :-1]
        result = forecast_and_truth_tables_to_cubes(
            self.forecast_df,
            df,
            self.cycletime,
            self.forecast_period,
            self.training_length,
        )
        self.assertEqual(len(result), 2)
        self.assertEqual(result, (expected_forecast, expected_truth))

    def test_site_coord_mismatch(self):
        """Test for a mismatch in the location of a site between the truths
        and forecasts. In this case, the position (lat/lon/alt) from the
        forecast will be used."""
        df = self.truth_df.copy()
        df.at[::3, "altitude"] = 45
        df.at[::3, "latitude"] = 52
        df.at[::3, "longitude"] = -12
        result = forecast_and_truth_tables_to_cubes(
            self.forecast_df,
            df,
            self.cycletime,
            self.forecast_period,
            self.training_length,
        )
        self.assertEqual(len(result), 2)
        self.assertEqual(
            result, (self.expected_period_forecast, self.expected_period_truth)
        )


if __name__ == "__main__":
    unittest.main()
