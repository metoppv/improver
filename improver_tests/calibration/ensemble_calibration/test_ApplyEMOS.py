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
"""Unit tests for the `ensemble_calibration.ApplyEMOS` class."""

import unittest

import iris
import numpy as np
from iris.tests import IrisTest

from improver.calibration.ensemble_calibration import ApplyEMOS
from improver.utilities.cube_manipulation import get_dim_coord_names

from ...set_up_test_cubes import (
    set_up_percentile_cube,
    set_up_probability_cube,
    set_up_variable_cube,
)


def build_coefficients_cube(data, template):
    """Make a cube of coefficients with expected metadata"""
    index = iris.coords.Coord(np.arange(4), long_name="coefficient_index")
    name = iris.coords.Coord(["gamma", "delta", "alpha", "beta"],
                             long_name="coefficient_name")
    coefficients = iris.cube.Cube(data, long_name="emos_coefficients",
                                  dim_coords_and_dims=[(index, 0)],
                                  aux_coords_and_dims=[(name, 0)])

    # add spatial and temporal coords from forecast to be calibrated
    for coord in ["time", "forecast_period", "forecast_reference_time"]:
        coefficients.add_aux_coord(template.coord(coord).copy())

    for coord in [template.coord(axis='x'), template.coord(axis='y')]:
        bounds = [min(coord.points), max(coord.points)]
        point = np.median(bounds)
        new_coord = coord.copy(points=[point], bounds=[bounds])
        coefficients.add_aux_coord(new_coord)

    coefficients.attributes['diagnostic_standard_name'] = 'air_temperature'

    return coefficients


class Test_process(IrisTest):
    """Tests for the ApplyEMOS callable plugin"""

    def setUp(self):
        """Set up some "uncalibrated forecast" inputs"""
        attributes = {"title": "MOGREPS-UK Forecast",
                      "source": "Met Office Unified Model",
                      "institution": "Met Office"}

        forecast = np.array([np.full((3, 3), 10.4),
                             np.full((3, 3), 10.8),
                             np.full((3, 3), 10.1)], dtype=np.float32)
        self.realizations = set_up_variable_cube(
            forecast, units='degC', attributes=attributes)

        percentiles = np.array([np.full((3, 3), 10.2),
                                np.full((3, 3), 10.4),
                                np.full((3, 3), 10.6)], dtype=np.float32)
        self.percentiles = set_up_percentile_cube(
            percentiles, np.array([25, 50, 75], dtype=np.float32),
            units='degC', attributes=attributes)

        probabilities = np.array([np.full((3, 3), 1),
                                  np.full((3, 3), 0.9),
                                  np.full((3, 3), 0)], dtype=np.float32)
        self.probabilities = set_up_probability_cube(
            probabilities, np.array([9, 10, 11], dtype=np.float32),
            threshold_units='degC', attributes=attributes)

        self.coefficients = build_coefficients_cube(
            [0, 1, 0, 1], self.realizations)

    def test_null_percentiles(self):
        """Test effect of "neutral" emos coefficients in percentile space
        (this is small but non-zero due to limited sampling of the
        distribution)"""
        expected_mean = np.mean(self.percentiles.data)
        expected_data = np.array([np.full((3, 3), 10.265101),
                                  np.full((3, 3), 10.4),
                                  np.full((3, 3), 10.534898)])
        result = ApplyEMOS()(
            self.percentiles, self.coefficients, realizations_count=3)
        self.assertIn("percentile", get_dim_coord_names(result))
        self.assertArrayAlmostEqual(result.data, expected_data)
        self.assertAlmostEqual(np.mean(result.data), expected_mean)

    def test_null_realizations(self):
        """Test effect of "neutral" emos coefficients in realization space"""
        expected_mean = np.mean(self.realizations.data)
        expected_data = np.array([np.full((3, 3), 10.433333),
                                  np.full((3, 3), 10.670206),
                                  np.full((3, 3), 10.196461)])
        result = ApplyEMOS()(self.realizations, self.coefficients)
        self.assertIn("realization", get_dim_coord_names(result))
        self.assertArrayAlmostEqual(result.data, expected_data)
        self.assertAlmostEqual(np.mean(result.data), expected_mean)

    def test_null_probabilities(self):
        """Test effect of "neutral" emos coefficients in probability space.
        Mean, 0 and 1 probabilities are not preserved."""
        expected_data = np.array([np.full((3, 3), 0.9999999),
                                  np.full((3, 3), 0.9452005),
                                  np.full((3, 3), 0.02274995)])
        result = ApplyEMOS()(
            self.probabilities, self.coefficients, realizations_count=3)
        self.assertIn("probability_of", result.name())
        self.assertArrayAlmostEqual(result.data, expected_data)

    def test_bias(self):
        """Test emos coefficients that correct a bias"""
        self.coefficients.data = [0, 1, 1, 1]
        expected_mean = np.mean(self.percentiles.data + 1.)
        expected_data = np.array([np.full((3, 3), 11.265101),
                                  np.full((3, 3), 11.4),
                                  np.full((3, 3), 11.534898)])
        result = ApplyEMOS()(
            self.percentiles, self.coefficients, realizations_count=3)
        self.assertArrayAlmostEqual(result.data, expected_data)
        self.assertAlmostEqual(np.mean(result.data), expected_mean)

    def test_spread(self):
        """Test emos coefficients that correct underspread"""
        self.coefficients.data = [1, 1, 0, 1]
        expected_mean = np.mean(self.percentiles.data)
        expected_data = np.array([np.full((3, 3), 9.7121525),
                                  np.full((3, 3), 10.4),
                                  np.full((3, 3), 11.087847)])
        result = ApplyEMOS()(
            self.percentiles, self.coefficients, realizations_count=3)
        self.assertArrayAlmostEqual(result.data, expected_data)
        self.assertAlmostEqual(np.mean(result.data), expected_mean)

    def test_error_realizations_count(self):
        """Test an error is raised if the realizations_count is not set"""
        msg = "The 'realizations_count' argument must be defined"
        with self.assertRaisesRegex(ValueError, msg):
            ApplyEMOS()(self.percentiles, self.coefficients)

    def test_land_sea_mask(self):
        """Test that coefficients can be effectively applied to "land" points
        only"""
        land_sea_data = np.array([[1, 1, 0],
                                  [1, 1, 0],
                                  [1, 0, 0]], dtype=np.int32)
        land_sea_mask = set_up_variable_cube(
            land_sea_data, name="land_binary_mask", units="1")
        self.coefficients.data = [1, 1, 0, 1]
        expected_data_slice = np.array([[9.7121525, 9.7121525, 10.2],
                                        [9.7121525, 9.7121525, 10.2],
                                        [9.7121525, 10.2, 10.2]])
        result = ApplyEMOS()(
            self.percentiles, self.coefficients, land_sea_mask=land_sea_mask,
            realizations_count=3)
        self.assertArrayAlmostEqual(result.data[0], expected_data_slice)


if __name__ == '__main__':
    unittest.main()
