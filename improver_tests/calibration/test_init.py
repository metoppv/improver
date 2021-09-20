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
from iris.cube import Cube, CubeList
import numpy as np

from improver.calibration import filter_obs, split_forecasts_and_coeffs, split_forecasts_and_truth
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

        # Set-up probability forecast test cubes
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

        # Set-up realization forecast test cubes
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

        self.landsea_mask_name = "land_binary_mask"
        self.landsea_mask = set_up_variable_cube(
            np.zeros((4, 4), dtype=np.float32), name=self.landsea_mask_name)
        for coord in ["time", "forecast_reference_time", "forecast_period"]:
            self.landsea_mask.remove_coord(coord)

        self.altitude = set_up_variable_cube(
            np.ones((4, 4), dtype=np.float32), name="surface_altitude", units="m"
        )
        for coord in ["time", "forecast_reference_time", "forecast_period"]:
            self.altitude.remove_coord(coord)

    def test_probability_data(self):
        """Test that when multiple probability forecast cubes and truth cubes
        are provided, the groups are created as expected."""

        forecast, truth, others, land_sea_mask = split_forecasts_and_truth(
            self.probability_forecasts + self.probability_truths, self.truth_attribute
        )

        self.assertIsInstance(forecast, iris.cube.Cube)
        self.assertIsInstance(truth, iris.cube.Cube)
        self.assertIsInstance(others, CubeList)
        self.assertIsNone(land_sea_mask, None)
        self.assertSequenceEqual((2, 2, 4, 4), forecast.shape)
        self.assertSequenceEqual((2, 2, 4, 4), truth.shape)
        self.assertTrue(np.all(forecast.data))
        self.assertFalse(np.any(truth.data))

    def test_probability_data_with_land_sea_mask(self):
        """Test that when multiple probability forecast cubes, truth cubes, and
        a single land-sea mask are provided, the groups are created as
        expected."""

        forecast, truth, others, land_sea_mask = split_forecasts_and_truth(
            self.probability_forecasts + self.probability_truths + [self.landsea_mask],
            self.truth_attribute, land_sea_mask_name=self.landsea_mask_name
        )

        self.assertIsInstance(forecast, iris.cube.Cube)
        self.assertIsInstance(truth, iris.cube.Cube)
        self.assertIsInstance(others, CubeList)
        self.assertIsInstance(land_sea_mask, iris.cube.Cube)
        self.assertEqual("land_binary_mask", land_sea_mask.name())
        self.assertSequenceEqual((2, 2, 4, 4), forecast.shape)
        self.assertSequenceEqual((2, 2, 4, 4), truth.shape)
        self.assertTrue(np.all(forecast.data))
        self.assertFalse(np.any(truth.data))
        self.assertSequenceEqual((4, 4), land_sea_mask.shape)

    def test_probability_data_with_additional_fields(self):
        """Test that when multiple probability forecast cubes and truth cubes
        are provided, the groups are created as expected."""

        forecast, truth, others, land_sea_mask = split_forecasts_and_truth(
            self.probability_forecasts + self.probability_truths + [self.altitude],
            self.truth_attribute,
        )

        self.assertIsInstance(forecast, iris.cube.Cube)
        self.assertIsInstance(truth, iris.cube.Cube)
        self.assertIsInstance(others, CubeList)
        self.assertIsNone(land_sea_mask, None)
        self.assertSequenceEqual((2, 2, 4, 4), forecast.shape)
        self.assertSequenceEqual((2, 2, 4, 4), truth.shape)
        self.assertEqual(len(others), 1)
        self.assertSequenceEqual((4, 4), others[0].shape)
        self.assertTrue(np.all(forecast.data))
        self.assertFalse(np.any(truth.data))

    def test_realization_data(self):
        """Test that when multiple forecast cubes and truth cubes are provided,
        the groups are created as expected."""

        forecast, truth, others, land_sea_mask = split_forecasts_and_truth(
            self.realization_forecasts + self.realization_truths, self.truth_attribute
        )

        self.assertIsInstance(forecast, iris.cube.Cube)
        self.assertIsInstance(truth, iris.cube.Cube)
        self.assertIsInstance(others, CubeList)
        self.assertIsNone(land_sea_mask, None)
        self.assertSequenceEqual((2, 4, 4), forecast.shape)
        self.assertSequenceEqual((2, 4, 4), truth.shape)
        self.assertTrue(np.all(forecast.data))
        self.assertFalse(np.any(truth.data))

    def test_realization_data_with_land_sea_mask(self):
        """Test that when multiple forecast cubes, truth cubes, and
        a single land-sea mask are provided, the groups are created as
        expected."""

        forecast, truth, others, land_sea_mask = split_forecasts_and_truth(
            self.realization_forecasts + self.realization_truths + [self.landsea_mask],
            self.truth_attribute, self.landsea_mask_name
        )

        self.assertIsInstance(forecast, iris.cube.Cube)
        self.assertIsInstance(truth, iris.cube.Cube)
        self.assertIsInstance(others, CubeList)
        self.assertIsInstance(land_sea_mask, iris.cube.Cube)
        self.assertEqual("land_binary_mask", land_sea_mask.name())
        self.assertSequenceEqual((2, 4, 4), forecast.shape)
        self.assertSequenceEqual((2, 4, 4), truth.shape)
        self.assertTrue(np.all(forecast.data))
        self.assertFalse(np.any(truth.data))
        self.assertSequenceEqual((4, 4), land_sea_mask.shape)

    def test_realization_data_with_additional_fields(self):
        """Test that when multiple forecast cubes, truth cubes, and
        a single land-sea mask are provided, the groups are created as
        expected."""

        forecast, truth, others, land_sea_mask = split_forecasts_and_truth(
            self.realization_forecasts + self.realization_truths + [self.altitude],
            self.truth_attribute
        )

        self.assertIsInstance(forecast, iris.cube.Cube)
        self.assertIsInstance(truth, iris.cube.Cube)
        self.assertIsInstance(others, CubeList)
        self.assertIsNone(land_sea_mask, None)
        self.assertSequenceEqual((2, 4, 4), forecast.shape)
        self.assertSequenceEqual((2, 4, 4), truth.shape)
        self.assertEqual(len(others), 1)
        self.assertSequenceEqual((4, 4), others[0].shape)
        self.assertTrue(np.all(forecast.data))
        self.assertFalse(np.any(truth.data))

    def test_exception_for_multiple_land_sea_masks(self):
        """Test that when multiple land-sea masks are provided an exception is
        raised."""

        msg = "Expected one cube for land-sea mask"
        with self.assertRaisesRegex(IOError, msg):
            split_forecasts_and_truth(
                self.realization_forecasts
                + self.realization_truths
                + [self.landsea_mask, self.landsea_mask],
                self.truth_attribute, self.landsea_mask_name
            )

    def test_exception_for_misnamed_land_sea_mask(self):
        """Test that when multiple land-sea masks are provided an exception is
        raised."""
        land_sea_mask = self.landsea_mask
        land_sea_mask.rename("misnamed_land_sea_mask")

        msg = "Expected one cube for land-sea mask"
        with self.assertRaisesRegex(IOError, msg):
            split_forecasts_and_truth(
                self.realization_forecasts
                + self.realization_truths
                + [self.landsea_mask],
                self.truth_attribute, self.landsea_mask_name
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

        msg = "Missing historic forecast input."
        with self.assertRaisesRegex(IOError, msg):
            split_forecasts_and_truth(
                self.realization_forecasts
                + self.realization_truths
                + [self.landsea_mask],
                self.truth_attribute,
            )


class Test_filter_obs(unittest.TestCase):

    """Test the filter_obs method."""

    def setUp(self):
        """Set-up cubes for testing."""
        data = np.array([280, 270, 280, 275], dtype=np.float32)
        altitudes = np.array([0, 1, 3, 2], dtype=np.float32)
        latitudes = np.array([10, 10, 20, 20], dtype=np.float32)
        longitudes = np.array([10, 10, 20, 20], dtype=np.float32)
        wmo_ids = np.arange(4)

        times = [
            datetime(2017, 11, 10, 4, 0),
            datetime(2017, 11, 11, 4, 0),
            datetime(2017, 11, 12, 4, 0),
        ]
        self.spot_obs = CubeList()
        for time in times:
            time_coord = [iris.coords.DimCoord(
                    (time-datetime(1970, 1, 1)).total_seconds(), "time", units=TIME_COORDS["time"].units
                )]
            self.spot_obs.append(build_spotdata_cube(
                data,
                "air_temperature",
                "K",
                altitudes.copy(),
                latitudes.copy(),
                longitudes.copy(),
                wmo_ids,
                scalar_coords=time_coord,
            ))

    def test_matching_coords(self):
        """Test the site output is consistent, if the input is consistent."""
        result = filter_obs(self.spot_obs)
        self.assertEqual(result, self.spot_obs)

    def test_nans_present(self):
        """Test handling of NaN within the site data."""
        modified_spot_obs = self.spot_obs.copy()
        modified_spot_obs[0].coord("altitude").points[0] = np.nan
        result = filter_obs(modified_spot_obs)
        self.assertEqual(result, self.spot_obs)

    def test_mismatching_coords(self):
        """Test mismatch in the coordinate points within the site data."""
        modified_spot_obs = self.spot_obs.copy()
        modified_spot_obs[1].coord("altitude").points[1] = 2
        modified_spot_obs[1].coord("latitude").points[1] = 20.1
        modified_spot_obs[1].coord("longitude").points[1] = 19.9
        result = filter_obs(modified_spot_obs)
        self.assertEqual(result, self.spot_obs)

    def test_gridded_input(self):
        """Test the function has no impact on gridded data."""
        gridded_cubes = CubeList([set_up_variable_cube(
                np.ones((3, 3), dtype=np.float32),
        )])
        result = filter_obs(gridded_cubes)
        self.assertEqual(result, gridded_cubes)


class Test_split_forecasts_and_coeffs(unittest.TestCase):

    """Test the split_forecasts_and_coeffs method."""

    def setUp(self):
        """Set-up cubes for testing."""
        thresholds = [283, 288]
        probability_data = np.ones((2, 4, 4), dtype=np.float32)
        realization_data = np.ones((4, 4), dtype=np.float32)

        self.truth_attribute = "mosg__model_configuration=uk_det"
        truth_attributes = {"mosg__model_configuration": "uk_det"}

        # Set-up probability and realization forecast cubes
        self.probability_forecast = CubeList([set_up_probability_cube(probability_data, thresholds)])
        self.realization_forecast = CubeList([set_up_variable_cube(realization_data)])

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
            "distribution": "norm"
        }
        alpha = iris.cube.Cube(
            np.array(0, dtype=np.float32), long_name="emos_coefficients_alpha", units="K",
            attributes=attributes
        )
        beta = iris.cube.Cube(
            np.array([0.5], dtype=np.float32), long_name="emos_coefficients_beta", units="1",
            attributes=attributes, dim_coords_and_dims=dim_coords_and_dims,
            aux_coords_and_dims=aux_coords_and_dims
        )
        gamma = iris.cube.Cube(
            np.array(0, dtype=np.float32), long_name="emos_coefficients_gamma", units="K",
            attributes=attributes
        )
        delta = iris.cube.Cube(
            np.array(1, dtype=np.float32), long_name="emos_coefficients_delta", units="1",
            attributes=attributes
        )

        self.coefficient_cubelist = CubeList([alpha, beta, gamma, delta])

        self.landsea_mask_name = "land_binary_mask"
        self.landsea_mask = set_up_variable_cube(
            np.zeros((4, 4), dtype=np.float32), name=self.landsea_mask_name)
        for coord in ["time", "forecast_reference_time", "forecast_period"]:
            self.landsea_mask.remove_coord(coord)
        self.landsea_mask = CubeList([self.landsea_mask])

        self.altitude = set_up_variable_cube(
            np.ones((4, 4), dtype=np.float32), name="surface_altitude", units="m"
        )
        for coord in ["time", "forecast_reference_time", "forecast_period"]:
            self.altitude.remove_coord(coord)
        self.altitude = CubeList([self.altitude])


    def test_realization(self):
        """Test a realization forecast input."""
        forecast, coeffs, others, landsea_mask = split_forecasts_and_coeffs(CubeList([self.realization_forecast, self.coefficient_cubelist]))
        self.assertEqual(forecast, self.realization_forecast[0])
        self.assertEqual(coeffs, self.coefficient_cubelist)
        self.assertEqual(others, CubeList())
        self.assertEqual(landsea_mask, None)

    def test_probability(self):
        """Test a probability forecast input."""
        forecast, coeffs, others, landsea_mask = split_forecasts_and_coeffs(CubeList([self.probability_forecast, self.coefficient_cubelist]))
        self.assertEqual(forecast, self.probability_forecast[0])
        self.assertEqual(coeffs, self.coefficient_cubelist)
        self.assertEqual(others, CubeList())
        self.assertEqual(landsea_mask, None)

    def test_land_sea_mask(self):
        """Test the addition of a land-sea mask input."""
        forecast, coeffs, others, landsea_mask = split_forecasts_and_coeffs(CubeList([self.realization_forecast, self.coefficient_cubelist, self.landsea_mask]), self.landsea_mask_name)
        self.assertEqual(forecast, self.realization_forecast[0])
        self.assertEqual(coeffs, self.coefficient_cubelist)
        self.assertEqual(others, CubeList())
        self.assertEqual(landsea_mask, self.landsea_mask[0])

    def test_additional_field(self):
        """Test the addition of an additional field (altitude)."""
        fp_names = [self.realization_forecast[0].name(), self.altitude[0].name()]
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
            "distribution": "norm"
        }

        beta = iris.cube.Cube(
            np.array([0.5, 0.5], dtype=np.float32), long_name="emos_coefficients_beta", units="1",
            attributes=attributes, dim_coords_and_dims=dim_coords_and_dims,
            aux_coords_and_dims=aux_coords_and_dims
        )

        self.coefficient_cubelist = CubeList([self.coefficient_cubelist[0], beta, *self.coefficient_cubelist[2:]])

        forecast, coeffs, others, landsea_mask = split_forecasts_and_coeffs(CubeList([self.realization_forecast, self.coefficient_cubelist, self.altitude]))
        self.assertEqual(forecast, self.realization_forecast[0])
        self.assertEqual(coeffs, self.coefficient_cubelist)
        self.assertEqual(others, self.altitude)
        self.assertEqual(landsea_mask, None)

    def test_missing_land_sea_mask(self):
        """Test a missing land-sea mask input."""
        msg = "Expected one cube for land-sea mask"
        with self.assertRaisesRegex(IOError, msg):
            split_forecasts_and_coeffs(CubeList([self.realization_forecast, self.coefficient_cubelist]), self.landsea_mask_name)

    def test_multiple_land_sea_masks(self):
        """Test multiple land-sea mask inputs."""
        msg = "Expected one cube for land-sea mask"
        with self.assertRaisesRegex(IOError, msg):
            split_forecasts_and_coeffs(CubeList([self.realization_forecast, self.coefficient_cubelist, self.landsea_mask, self.landsea_mask]), self.landsea_mask_name)

    def test_coefficient_mismatch(self):
        """Test a mismatch between the forecast and coefficient cube attributes."""
        self.coefficient_cubelist[0].attributes["diagnostic_standard_name"] = "wind_speed"
        msg = "The coefficients cubes are expected to"
        with self.assertRaisesRegex(AttributeError, msg):
            split_forecasts_and_coeffs(CubeList([self.realization_forecast, self.coefficient_cubelist]))

    def test_forecast_and_coefficient_mismatch(self):
        """Test a mismatch between the forecast and coefficient cube attributes."""
        for cube in self.coefficient_cubelist:
            cube.attributes["diagnostic_standard_name"] = "wind_speed"
        msg = "A forecast corresponding to"
        with self.assertRaisesRegex(KeyError, msg):
            split_forecasts_and_coeffs(CubeList([self.realization_forecast, self.coefficient_cubelist]))


if __name__ == "__main__":
    unittest.main()
