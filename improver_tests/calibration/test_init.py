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
from iris.cube import CubeList
import pandas as pd
import pytest

from improver.calibration import (
    forecast_and_truth_dataframes_to_cubes,
    forecast_dataframe_to_cube,
    split_forecasts_and_coeffs,
    split_forecasts_and_truth,
    truth_dataframe_to_cube,
)
from improver.metadata.constants.time_types import TIME_COORDS
from improver.spotdata.build_spotdata_cube import build_spotdata_cube
from improver.synthetic_data.set_up_test_cubes import (
    set_up_percentile_cube,
    set_up_probability_cube,
    set_up_variable_cube,
)
from improver_tests import ImproverTest


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


class Test_split_forecasts_and_coeffs(ImproverTest):

    """Test the split_forecasts_and_coeffs function."""

    def setUp(self):
        """Set-up cubes for testing."""
        thresholds = [283, 288]
        percentiles = [25, 75]
        probability_data = np.ones((2, 4, 4), dtype=np.float32)
        realization_data = np.ones((4, 4), dtype=np.float32)

        self.truth_attribute = "mosg__model_configuration=uk_det"

        # Set-up probability and realization forecast cubes
        self.probability_forecast = CubeList(
            [set_up_probability_cube(probability_data, thresholds)]
        )
        self.realization_forecast = CubeList([set_up_variable_cube(realization_data)])
        self.percentile_forecast = CubeList(
            [set_up_percentile_cube(probability_data, percentiles)]
        )

        # Set-up coefficient cubes
        fp_names = [self.realization_forecast[0].name()]
        predictor_index = iris.coords.DimCoord(
            np.array(range(len(fp_names)), dtype=np.int32),
            long_name="predictor_index",
            units="1",
        )
        dim_coords_and_dims = ((predictor_index, 0),)
        predictor_name = iris.coords.AuxCoord(
            fp_names, long_name="predictor_name", units="no_unit"
        )
        aux_coords_and_dims = ((predictor_name, 0),)

        attributes = {
            "diagnostic_standard_name": self.realization_forecast[0].name(),
            "distribution": "norm",
        }
        alpha = iris.cube.Cube(
            np.array(0, dtype=np.float32),
            long_name="emos_coefficients_alpha",
            units="K",
            attributes=attributes,
        )
        beta = iris.cube.Cube(
            np.array([0.5], dtype=np.float32),
            long_name="emos_coefficients_beta",
            units="1",
            attributes=attributes,
            dim_coords_and_dims=dim_coords_and_dims,
            aux_coords_and_dims=aux_coords_and_dims,
        )
        gamma = iris.cube.Cube(
            np.array(0, dtype=np.float32),
            long_name="emos_coefficients_gamma",
            units="K",
            attributes=attributes,
        )
        delta = iris.cube.Cube(
            np.array(1, dtype=np.float32),
            long_name="emos_coefficients_delta",
            units="1",
            attributes=attributes,
        )

        self.coefficient_cubelist = CubeList([alpha, beta, gamma, delta])

        # Set-up land-sea mask.
        self.land_sea_mask_name = "land_binary_mask"
        self.land_sea_mask = set_up_variable_cube(
            np.zeros((4, 4), dtype=np.float32), name=self.land_sea_mask_name
        )
        for coord in ["time", "forecast_reference_time", "forecast_period"]:
            self.land_sea_mask.remove_coord(coord)
        self.land_sea_mask = CubeList([self.land_sea_mask])

    def test_realization_forecast_and_coefficients(self):
        """Test a realization forecast input."""
        forecast, coeffs, land_sea_mask, template = split_forecasts_and_coeffs(
            CubeList([self.realization_forecast, self.coefficient_cubelist]),
            self.land_sea_mask_name,
        )

        self.assertCubeEqual(forecast, self.realization_forecast[0])
        self.assertCubeListEqual(coeffs, self.coefficient_cubelist)
        self.assertEqual(land_sea_mask, None)
        self.assertEqual(template, None)

    def test_percentile_forecast_and_coefficients(self):
        """Test a percentile forecast input."""
        forecast, coeffs, land_sea_mask, template = split_forecasts_and_coeffs(
            CubeList([self.percentile_forecast, self.coefficient_cubelist]),
            self.land_sea_mask_name,
        )
        self.assertCubeEqual(forecast, self.percentile_forecast[0])
        self.assertCubeListEqual(coeffs, self.coefficient_cubelist)
        self.assertEqual(land_sea_mask, None)
        self.assertEqual(template, None)

    def test_probability_forecast_and_coefficients(self):
        """Test a probability forecast input."""
        forecast, coeffs, land_sea_mask, template = split_forecasts_and_coeffs(
            CubeList([self.probability_forecast, self.coefficient_cubelist]),
            self.land_sea_mask_name,
        )
        self.assertCubeEqual(forecast, self.probability_forecast[0])
        self.assertCubeListEqual(coeffs, self.coefficient_cubelist)
        self.assertEqual(land_sea_mask, None)
        self.assertEqual(template, None)

    def test_forecast_coefficients_and_land_sea_mask(self):
        """Test the addition of a land-sea mask."""
        forecast, coeffs, land_sea_mask, template = split_forecasts_and_coeffs(
            CubeList(
                [
                    self.realization_forecast,
                    self.coefficient_cubelist,
                    self.land_sea_mask,
                ]
            ),
            self.land_sea_mask_name,
        )

        self.assertCubeEqual(forecast, self.realization_forecast[0])
        self.assertCubeListEqual(coeffs, self.coefficient_cubelist)
        self.assertCubeEqual(land_sea_mask, self.land_sea_mask[0])
        self.assertEqual(template, None)

    def test_no_land_sea_mask_name(self):
        """Test when not providing the land_sea_mask_name option."""
        forecast, coeffs, land_sea_mask, template = split_forecasts_and_coeffs(
            CubeList([self.realization_forecast, self.coefficient_cubelist]),
        )

        self.assertCubeEqual(forecast, self.realization_forecast[0])
        self.assertCubeListEqual(coeffs, self.coefficient_cubelist)
        self.assertEqual(land_sea_mask, None)
        self.assertEqual(template, None)

    def test_forecast_coefficients_prob_template(self):
        """Test the addition of a probability template cube."""
        forecast, coeffs, land_sea_mask, template = split_forecasts_and_coeffs(
            CubeList(
                [
                    self.realization_forecast,
                    self.coefficient_cubelist,
                    self.probability_forecast,
                ]
            ),
            self.land_sea_mask_name,
        )
        self.assertCubeEqual(forecast, self.realization_forecast[0])
        self.assertCubeListEqual(coeffs, self.coefficient_cubelist)
        self.assertEqual(land_sea_mask, None)
        self.assertCubeEqual(template, self.probability_forecast[0])

    def test_all_options(self):
        """Test providing a forecast, coefficients, land-sea mask and a
        probability template."""
        forecast, coeffs, land_sea_mask, template = split_forecasts_and_coeffs(
            CubeList(
                [
                    self.realization_forecast,
                    self.coefficient_cubelist,
                    self.land_sea_mask,
                    self.probability_forecast,
                ]
            ),
            self.land_sea_mask_name,
        )
        self.assertCubeEqual(forecast, self.realization_forecast[0])
        self.assertCubeListEqual(coeffs, self.coefficient_cubelist)
        self.assertCubeEqual(land_sea_mask, self.land_sea_mask[0])
        self.assertCubeEqual(template, self.probability_forecast[0])

    def test_probability_forecast_coefficients_prob_template(self):
        """Test providing a probability template with a probability forecast."""
        msg = "Providing multiple probability cubes"
        with self.assertRaisesRegex(ValueError, msg):
            split_forecasts_and_coeffs(
                CubeList(
                    [
                        self.probability_forecast,
                        self.coefficient_cubelist,
                        self.probability_forecast,
                    ]
                ),
                self.land_sea_mask_name,
            )

    def test_no_coefficients(self):
        """Test if no EMOS coefficients are provided."""
        _, coeffs, _, _ = split_forecasts_and_coeffs(
            CubeList([self.percentile_forecast]), self.land_sea_mask_name
        )
        self.assertIsNone(coeffs)

    def test_no_forecast(self):
        """Test if no forecast is present."""
        msg = "No forecast is present"
        with self.assertRaisesRegex(ValueError, msg):
            split_forecasts_and_coeffs(
                CubeList([self.coefficient_cubelist]), self.land_sea_mask_name
            )

    def test_duplicate_forecasts(self):
        """Test if a duplicate forecast is provided."""
        msg = "Multiple items have been provided"
        with self.assertRaisesRegex(ValueError, msg):
            split_forecasts_and_coeffs(
                CubeList(
                    [
                        self.percentile_forecast,
                        self.coefficient_cubelist,
                        self.land_sea_mask,
                        self.probability_forecast,
                        self.percentile_forecast,
                    ]
                ),
                self.land_sea_mask_name,
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


class Test_shared_dataframes(ImproverTest):

    """A shared dataframe creation class."""

    def setUp(self):
        """Set-up forecast and truth dataframes."""
        pytest.importorskip("pandas")

        data = np.array(
            [5.2, 0.3, 20.4, 6.5, 3.1, 21.5, 7.2, 4.2, 24.3], dtype=np.float32
        )
        self.forecast_data = np.tile(data, 3)

        self.frt1 = np.datetime64("2017-07-20T12:00:00+00:00")
        self.frt2 = np.datetime64("2017-07-21T12:00:00+00:00")
        self.frt3 = np.datetime64("2017-07-22T12:00:00+00:00")

        self.fp = np.timedelta64("6", "h")

        self.time1 = np.datetime64("2017-07-20T18:00:00+00:00")
        self.time2 = np.datetime64("2017-07-21T18:00:00+00:00")
        self.time3 = np.datetime64("2017-07-22T18:00:00+00:00")

        self.wmo_ids = ["03002", "03003", "03004"]
        self.percentiles = np.array([25, 50.0, 75.0], dtype=np.float32)
        diag = "air_temperature"
        self.cf_name = "air_temperature"
        self.latitudes = np.array([50.0, 60.0, 70.0], dtype=np.float32)
        self.longitudes = np.array([-10.0, 0.0, 10.0], dtype=np.float32)
        self.altitudes = np.array([10.0, 20.0, 30.0], dtype=np.float32)
        self.period = np.timedelta64(1, "h").astype("timedelta64[ns]")
        self.height = np.array([1.5], dtype=np.float32)
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
            "height": np.tile(self.height, 27),
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
        self.forecast_period = 6
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
                    for t in [time - self.period.astype("timedelta64[s]"), time]
                ],
                units=TIME_COORDS["time"].units,
            )

            fp_point = self.fp.astype("timedelta64[s]")
            fp_coord = iris.coords.AuxCoord(
                fp_point.astype(TIME_COORDS["forecast_period"].dtype),
                "forecast_period",
                bounds=[
                    f.astype(TIME_COORDS["forecast_period"].dtype)
                    for f in [fp_point - self.period.astype("timedelta64[s]"), fp_point]
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
                    for t in [time - self.period.astype("timedelta64[s]"), time]
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


class Test_forecast_dataframe_to_cube(Test_constructed_forecast_cubes):

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
        forecast_period = 7
        result = forecast_dataframe_to_cube(
            self.forecast_df, self.date_range, forecast_period
        )
        self.assertIsNone(result)

    def test_nonunique_values_in_column(self):
        """Test if there are multiple non-unique values in a column of the
        dataframe."""
        df = self.forecast_df.copy()
        df.at[0, "period"] = pd.Timedelta(7200, units="seconds")
        msg = "Multiple values provided for the period"
        with self.assertRaisesRegex(ValueError, msg):
            forecast_dataframe_to_cube(df, self.date_range, self.forecast_period)


class Test_truth_dataframe_to_cube(Test_constructed_truth_cubes):

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
        df = self.truth_df.copy()
        df.at[0, "diagnostic"] = "wind_speed_at_10m"
        msg = "Multiple values provided for the diagnostic"
        with self.assertRaisesRegex(ValueError, msg):
            truth_dataframe_to_cube(df, self.date_range)


class Test_forecast_and_truth_dataframes_to_cubes(
    Test_constructed_forecast_cubes, Test_constructed_truth_cubes
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

    def test_multiday_forecast_period(self):
        """Test for a multi-day forecast period to ensure that the
        validity times within the training dataset are always in
        the past, relative to the cycletime."""
        forecast_period = 30
        forecast_df = self.forecast_df.copy()
        forecast_df["forecast_period"] = np.timedelta64(forecast_period, "h").astype(
            "timedelta64[ns]"
        )
        for coord in ["forecast_reference_time", "blend_time"]:
            forecast_df[coord] = forecast_df[coord].replace(
                to_replace={
                    self.frt1: self.frt1 - pd.Timedelta(1, days=1),
                    self.frt2: self.frt2 - pd.Timedelta(1, days=1),
                    self.frt3: self.frt3 - pd.Timedelta(1, days=1),
                }
            )

        fp_int = np.int32(np.timedelta64(forecast_period, "h").astype("timedelta64[s]"))
        self.expected_period_forecast.coord("forecast_period").points = fp_int
        self.expected_period_forecast.coord("forecast_period").bounds = [
            fp_int - np.int32(self.period.astype("timedelta64[s]")),
            fp_int,
        ]

        result = forecast_and_truth_dataframes_to_cubes(
            forecast_df,
            self.truth_subset_df,
            self.cycletime,
            forecast_period,
            self.training_length,
        )
        self.assertEqual(len(result), 2)
        self.assertCubeEqual(result[0], self.expected_period_forecast)
        self.assertCubeEqual(result[1], self.expected_period_truth)

    def test_site_absent_from_forecast(self):
        """Test for when a site is absent from the forecast dataframe."""
        df = self.forecast_df.copy()
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
        df = self.truth_subset_df.copy()
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
        df = self.truth_subset_df.copy()
        df.at[::3, "altitude"] = 45
        df.at[::3, "latitude"] = 52
        df.at[::3, "longitude"] = -12
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

    def test_missing_observation(self):
        """Test a truth DataFrame with a missing observation at
        a particular time is converted correctly into an iris Cube."""
        df = self.truth_subset_df.head(-1)
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
        df = self.forecast_df.copy()
        df = df.rename(columns={"diagnostic": "diag"})
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
        df = self.truth_subset_df.copy()
        df = df.rename(columns={"diagnostic": "diag"})
        msg = "The following compulsory column\\(s\\) are missing"
        with self.assertRaisesRegex(ValueError, msg):
            forecast_and_truth_dataframes_to_cubes(
                self.forecast_df,
                df,
                self.cycletime,
                self.forecast_period,
                self.training_length,
            )

    def test_forecast_additional_columns_present(self):
        """Test that if there are additional columns present
        in the forecast dataframe, these have no impact."""
        df = self.forecast_df.copy()
        df["station_id"] = "11111"
        result = forecast_and_truth_dataframes_to_cubes(
            df,
            self.truth_subset_df,
            self.cycletime,
            self.forecast_period,
            self.training_length,
        )
        self.assertEqual(len(result), 2)

    def test_truth_additional_columns_present(self):
        """Test that if there are additional columns present
        in the truth dataframe, these have no impact."""
        df = self.truth_subset_df.copy()
        df["station_id"] = "11111"
        result = forecast_and_truth_dataframes_to_cubes(
            self.forecast_df,
            df,
            self.cycletime,
            self.forecast_period,
            self.training_length,
        )
        self.assertEqual(len(result), 2)

    def test_forecast_missing_columns_and_additional_columns(self):
        """Test if there are missing compulsory columns in the forecast
        dataframe and there are additional non-compulsory columns."""
        df = self.forecast_df.copy()
        df["station_id"] = "11111"
        df = df.rename(columns={"diagnostic": "diag"})
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
        df = self.truth_subset_df.copy()
        df["station_id"] = "11111"
        df = df.rename(columns={"diagnostic": "diag"})
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
