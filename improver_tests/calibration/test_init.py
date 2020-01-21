# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown Copyright 2017-2019 Met Office.
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
from improver.calibration import split_forecasts_and_truth
from improver_tests.set_up_test_cubes import (set_up_probability_cube,
                                              set_up_variable_cube)


class Test_split_forecasts_and_truth(unittest.TestCase):

    """Test the split_forecasts_and_truth method."""

    def setUp(self):
        """Create forecast and truth cubes for use in testing the reliability
        calibration plugin. Two forecast and two truth cubes are created, each
        pair containing the same data but given different forecast reference
        times and validity times. These times maintain the same forecast period
        for each forecast cube.
        Each forecast cube in conjunction with the contemporaneous truth cube
        will be used to produce a reliability calibration table. When testing
        the process method here we expect the final reliability calibration
        table for a given threshold (we are only using 283K in the value
        comparisons) to be the sum of two of these identical tables."""

        thresholds = [283, 288]
        forecast_data = np.ones((2, 4, 4), dtype=np.float32)
        truth_data = np.zeros((4, 4), dtype=np.float32)

        forecast_1 = set_up_probability_cube(forecast_data, thresholds)
        forecast_2 = set_up_probability_cube(
            forecast_data, thresholds, time=datetime(2017, 11, 11, 4, 0),
            frt=datetime(2017, 11, 11, 0, 0))
        self.probability_forecasts = [forecast_1, forecast_2]
        self.truth_attribute = "mosg__model_configuration=uk_det"
        truth_attributes = {'mosg__model_configuration': 'uk_det'}
        truth_1 = set_up_variable_cube(truth_data,
                                       attributes=truth_attributes)
        truth_2 = set_up_variable_cube(
            truth_data, time=datetime(2017, 11, 11, 4, 0),
            frt=datetime(2017, 11, 11, 0, 0),
            attributes=truth_attributes)
        self.truths = [truth_1, truth_2]

        realization_forecast_1 = truth_1.copy()
        realization_forecast_2 = truth_2.copy()
        realization_forecast_1.attributes.pop('mosg__model_configuration')
        realization_forecast_2.attributes.pop('mosg__model_configuration')
        self.realization_forecasts = [realization_forecast_1,
                                      realization_forecast_2]

        self.landsea_mask = truth_1.copy()
        self.landsea_mask.rename('land_binary_mask')

    def test_probability_data(self):
        """Test that when multiple probability forecast cubes and truth cubes
        are provided, the groups are created as expected."""

        forecast, truth, land_sea_mask = split_forecasts_and_truth(
            self.probability_forecasts + self.truths,
            self.truth_attribute)

        self.assertIsInstance(forecast, iris.cube.Cube)
        self.assertIsInstance(truth, iris.cube.Cube)
        self.assertIsNone(land_sea_mask, None)
        self.assertEqual('probability_of_air_temperature_above_threshold',
                         forecast.name())
        self.assertEqual('air_temperature', truth.name())
        self.assertSequenceEqual((2, 2, 4, 4), forecast.shape)
        self.assertSequenceEqual((2, 4, 4), truth.shape)

    def test_probability_data_with_land_sea_mask(self):
        """Test that when multiple probability forecast cubes, truth cubes, and
        a single land-sea mask are provided, the groups are created as
        expected."""

        forecast, truth, land_sea_mask = split_forecasts_and_truth(
            self.probability_forecasts + self.truths + [self.landsea_mask],
            self.truth_attribute)

        self.assertIsInstance(forecast, iris.cube.Cube)
        self.assertIsInstance(truth, iris.cube.Cube)
        self.assertIsInstance(land_sea_mask, iris.cube.Cube)
        self.assertEqual('probability_of_air_temperature_above_threshold',
                         forecast.name())
        self.assertEqual('air_temperature', truth.name())
        self.assertEqual('land_binary_mask', land_sea_mask.name())
        self.assertSequenceEqual((2, 2, 4, 4), forecast.shape)
        self.assertSequenceEqual((2, 4, 4), truth.shape)
        self.assertSequenceEqual((4, 4), land_sea_mask.shape)

    def test_realization_data(self):
        """Test that when multiple forecast cubes and truth cubes are provided,
        he groups are created as expected."""

        forecast, truth, land_sea_mask = split_forecasts_and_truth(
            self.realization_forecasts + self.truths,
            self.truth_attribute)

        self.assertIsInstance(forecast, iris.cube.Cube)
        self.assertIsInstance(truth, iris.cube.Cube)
        self.assertIsNone(land_sea_mask, None)
        self.assertEqual('air_temperature', forecast.name())
        self.assertEqual('air_temperature', truth.name())
        self.assertSequenceEqual((2, 4, 4), forecast.shape)
        self.assertSequenceEqual((2, 4, 4), truth.shape)

    def test_realization_data_with_land_sea_mask(self):
        """Test that when multiple forecast cubes, truth cubes, and
        a single land-sea mask are provided, the groups are created as
        expected."""

        forecast, truth, land_sea_mask = split_forecasts_and_truth(
            self.realization_forecasts + self.truths + [self.landsea_mask],
            self.truth_attribute)

        self.assertIsInstance(forecast, iris.cube.Cube)
        self.assertIsInstance(truth, iris.cube.Cube)
        self.assertIsInstance(land_sea_mask, iris.cube.Cube)
        self.assertEqual('air_temperature', forecast.name())
        self.assertEqual('air_temperature', truth.name())
        self.assertEqual('land_binary_mask', land_sea_mask.name())
        self.assertSequenceEqual((2, 4, 4), forecast.shape)
        self.assertSequenceEqual((2, 4, 4), truth.shape)
        self.assertSequenceEqual((4, 4), land_sea_mask.shape)

    def test_exception_for_multiple_land_sea_masks(self):
        """Test that when multiple land-sea masks are provided an exception is
        raised."""

        msg = 'Expected one cube for land-sea mask.'
        with self.assertRaisesRegex(IOError, msg):
            split_forecasts_and_truth(
                self.realization_forecasts + self.truths +
                [self.landsea_mask, self.landsea_mask],
                self.truth_attribute)

    def test_exception_for_unintended_cube_combination(self):
        """Test that when the forecast and truth cubes have different names,
        indicating different diagnostics, an exception is raised."""

        self.truths[0].rename('kitten_density')

        msg = 'Must have cubes with 1 or 2 distinct names.'
        with self.assertRaisesRegex(ValueError, msg):
            split_forecasts_and_truth(
                self.realization_forecasts + self.truths +
                [self.landsea_mask, self.landsea_mask],
                self.truth_attribute)

    def test_exception_for_missing_truth_inputs(self):
        """Test that when all truths are missing an exception is raised."""

        self.truths = []

        msg = 'Missing truth input.'
        with self.assertRaisesRegex(IOError, msg):
            split_forecasts_and_truth(
                self.realization_forecasts + self.truths +
                [self.landsea_mask], self.truth_attribute)

    def test_exception_for_missing_forecast_inputs(self):
        """Test that when all forecasts are missing an exception is raised."""

        self.realization_forecasts = []

        msg = 'Missing historical forecast input.'
        with self.assertRaisesRegex(IOError, msg):
            split_forecasts_and_truth(
                self.realization_forecasts + self.truths +
                [self.landsea_mask], self.truth_attribute)


if __name__ == '__main__':
    unittest.main()
