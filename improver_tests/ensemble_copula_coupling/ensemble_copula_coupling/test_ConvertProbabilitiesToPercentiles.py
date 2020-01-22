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
"""
Unit tests for the
`ensemble_copula_coupling.ConvertProbabilitiesToPercentiles` class.
"""
import unittest

import cf_units as unit
import numpy as np
from iris.coords import DimCoord
from iris.cube import Cube
from iris.tests import IrisTest

from improver.ensemble_copula_coupling.ensemble_copula_coupling import (
    ConvertProbabilitiesToPercentiles as Plugin)
from improver.metadata.probabilistic import find_threshold_coordinate
from improver.utilities.warnings_handler import ManageWarnings

from ...calibration.ensemble_calibration.helper_functions import (
    add_forecast_reference_time_and_forecast_period,
    set_up_probability_above_threshold_spot_temperature_cube,
    set_up_probability_above_threshold_temperature_cube,
    set_up_probability_threshold_cube)


class Test__add_bounds_to_thresholds_and_probabilities(IrisTest):

    """
    Test the _add_bounds_to_thresholds_and_probabilities method of the
    ConvertProbabilitiesToPercentiles.
    """

    def setUp(self):
        """Set up current_temperature_forecast_cube for testing."""
        self.current_temperature_forecast_cube = (
            add_forecast_reference_time_and_forecast_period(
                set_up_probability_above_threshold_temperature_cube()))
        self.threshold_points = find_threshold_coordinate(
            self.current_temperature_forecast_cube).points

    def test_basic(self):
        """Test that the plugin returns two numpy arrays."""
        cube = self.current_temperature_forecast_cube
        probabilities_for_cdf = cube.data.reshape(3, 9)
        bounds_pairing = (-40, 50)
        plugin = Plugin()
        result = plugin._add_bounds_to_thresholds_and_probabilities(
            self.threshold_points, probabilities_for_cdf, bounds_pairing)
        self.assertIsInstance(result[0], np.ndarray)
        self.assertIsInstance(result[1], np.ndarray)

    def test_bounds_of_threshold_points(self):
        """
        Test that the plugin returns the expected results for the
        threshold_points, where they've been padded with the values from
        the bounds_pairing.
        """
        cube = self.current_temperature_forecast_cube
        probabilities_for_cdf = cube.data.reshape(3, 9)
        bounds_pairing = (-40, 50)
        plugin = Plugin()
        result = plugin._add_bounds_to_thresholds_and_probabilities(
            self.threshold_points, probabilities_for_cdf, bounds_pairing)
        self.assertArrayAlmostEqual(result[0][0], bounds_pairing[0])
        self.assertArrayAlmostEqual(result[0][-1], bounds_pairing[1])

    def test_probability_data(self):
        """
        Test that the plugin returns the expected results for the
        probabilities, where they've been padded with zeros and ones to
        represent the extreme ends of the Cumulative Distribution Function.
        """
        cube = self.current_temperature_forecast_cube
        probabilities_for_cdf = cube.data.reshape(3, 9)
        zero_array = np.zeros(probabilities_for_cdf[:, 0].shape)
        one_array = np.ones(probabilities_for_cdf[:, 0].shape)
        bounds_pairing = (-40, 50)
        plugin = Plugin()
        result = plugin._add_bounds_to_thresholds_and_probabilities(
            self.threshold_points, probabilities_for_cdf, bounds_pairing)
        self.assertArrayAlmostEqual(result[1][:, 0], zero_array)
        self.assertArrayAlmostEqual(result[1][:, -1], one_array)

    def test_endpoints_of_distribution_exceeded(self):
        """
        Test that the plugin raises a ValueError when the constant
        end points of the distribution are exceeded by a threshold value
        used in the forecast.
        """
        probabilities_for_cdf = np.array([[0.05, 0.7, 0.95]])
        threshold_points = np.array([8, 10, 60])
        bounds_pairing = (-40, 50)
        plugin = Plugin()
        msg = "The calculated threshold values"
        with self.assertRaisesRegex(ValueError, msg):
            plugin._add_bounds_to_thresholds_and_probabilities(
                threshold_points, probabilities_for_cdf, bounds_pairing)

    @ManageWarnings(record=True)
    def test_endpoints_of_distribution_exceeded_warning(
            self, warning_list=None):
        """
        Test that the plugin raises a warning message when the constant
        end points of the distribution are exceeded by a threshold value
        used in the forecast and the ecc_bounds_warning keyword argument
        has been specified.
        """
        probabilities_for_cdf = np.array([[0.05, 0.7, 0.95]])
        threshold_points = np.array([8, 10, 60])
        bounds_pairing = (-40, 50)
        plugin = Plugin(ecc_bounds_warning=True)
        warning_msg = "The calculated threshold values"
        plugin._add_bounds_to_thresholds_and_probabilities(
            threshold_points, probabilities_for_cdf, bounds_pairing)
        self.assertTrue(any(warning_msg in str(item)
                            for item in warning_list))

    @ManageWarnings(ignored_messages=['The calculated threshold values'])
    def test_new_endpoints_generation(self):
        """Test that the plugin re-applies the threshold bounds using the
        maximum and minimum threshold points values when the original bounds
        have been exceeded and ecc_bounds_warning has been set."""
        probabilities_for_cdf = np.array([[0.05, 0.7, 0.95]])
        threshold_points = np.array([-50, 10, 60])
        bounds_pairing = (-40, 50)
        plugin = Plugin(ecc_bounds_warning=True)
        result = plugin._add_bounds_to_thresholds_and_probabilities(
            threshold_points, probabilities_for_cdf, bounds_pairing)
        self.assertEqual(max(result[0]), max(threshold_points))
        self.assertEqual(min(result[0]), min(threshold_points))


class Test__probabilities_to_percentiles(IrisTest):

    """Test the _probabilities_to_percentiles method of the
    ConvertProbabilitiesToPercentiles plugin."""

    def setUp(self):
        """Set up temperature cube."""
        self.current_temperature_forecast_cube = (
            add_forecast_reference_time_and_forecast_period(
                set_up_probability_above_threshold_temperature_cube()))
        self.current_temperature_spot_forecast_cube = (
            add_forecast_reference_time_and_forecast_period(
                set_up_probability_above_threshold_spot_temperature_cube()))

    def test_basic(self):
        """Test that the plugin returns an Iris.cube.Cube."""
        cube = self.current_temperature_forecast_cube
        percentiles = [10, 50, 90]
        bounds_pairing = (-40, 50)
        plugin = Plugin()
        result = plugin._probabilities_to_percentiles(
            cube, percentiles, bounds_pairing)
        self.assertIsInstance(result, Cube)

    def test_unknown_thresholding(self):
        """Test that the plugin returns an Iris.cube.Cube."""
        cube = self.current_temperature_forecast_cube
        cube.coord(var_name="threshold"
                   ).attributes["spp__relative_to_threshold"] = "between"
        percentiles = [10, 50, 90]
        bounds_pairing = (-40, 50)
        plugin = Plugin()
        msg = "Probabilities to percentiles only implemented for"
        with self.assertRaisesRegex(NotImplementedError, msg):
            plugin._probabilities_to_percentiles(
                cube, percentiles, bounds_pairing)

    def test_return_name(self):
        """Test that the plugin returns an Iris.cube.Cube with an appropriate
        name.
        """
        cube = self.current_temperature_forecast_cube
        percentiles = [10, 50, 90]
        bounds_pairing = (-40, 50)
        plugin = Plugin()
        result = plugin._probabilities_to_percentiles(
            cube, percentiles, bounds_pairing)
        self.assertEqual(result.name(), 'air_temperature')

    def test_return_coord_units(self):
        """Test that the plugin returns an Iris.cube.Cube with an appropriate
        percentile coordinate with suitable units.
        """
        cube = self.current_temperature_forecast_cube
        percentiles = [10, 50, 90]
        bounds_pairing = (-40, 50)
        plugin = Plugin()
        result = plugin._probabilities_to_percentiles(
            cube, percentiles, bounds_pairing)
        self.assertIsInstance(result.coord('percentile'), DimCoord)
        self.assertArrayEqual(result.coord('percentile').points, percentiles)
        self.assertEqual(result.coord('percentile').units, unit.Unit("%"))

    def test_transpose_cube_dimensions(self):
        """
        Test that the plugin returns an the expected data, when comparing
        input cubes which have dimensions in a different order.
        """
        # Calculate result for nontransposed cube.
        cube = self.current_temperature_forecast_cube
        percentiles = [10, 50, 90]
        bounds_pairing = (-40, 50)
        plugin = Plugin()
        nontransposed_result = plugin._probabilities_to_percentiles(
            cube, percentiles, bounds_pairing)

        # Calculate result for transposed cube.
        # Original cube dimensions are [P, T, Y, X].
        # Transposed cube dimensions are [X, Y, T, P].
        cube.transpose([3, 2, 1, 0])
        transposed_result = plugin._probabilities_to_percentiles(
            cube, percentiles, bounds_pairing)

        # Result cube will be [P, X, Y, T]
        # Transpose cube to be [P, T, Y, X]
        transposed_result.transpose([0, 3, 2, 1])
        self.assertArrayAlmostEqual(
            nontransposed_result.data, transposed_result.data)

    def test_simple_check_data_above(self):
        """
        Test that the plugin returns an Iris.cube.Cube with the expected
        data values for the percentiles when input probabilities are given
        for being above a threshold.
        The input cube contains probabilities that values are above a given
        threshold.
        """
        expected = np.array([8.15384615, 9.38461538, 11.6])
        expected = expected[:, np.newaxis, np.newaxis, np.newaxis]

        data = np.array([0.95, 0.3, 0.05])
        data = data[:, np.newaxis, np.newaxis, np.newaxis]

        self.current_temperature_forecast_cube = (
            add_forecast_reference_time_and_forecast_period(
                set_up_probability_threshold_cube(
                    data, "air_temperature", "degreesC",
                    forecast_thresholds=[8, 10, 12], y_dimension_length=1,
                    x_dimension_length=1, spp__relative_to_threshold='above')))
        cube = self.current_temperature_forecast_cube
        percentiles = [10, 50, 90]
        bounds_pairing = (-40, 50)
        plugin = Plugin()
        result = plugin._probabilities_to_percentiles(
            cube, percentiles, bounds_pairing)
        self.assertArrayAlmostEqual(result.data, expected)

    def test_simple_check_data_below(self):
        """
        Test that the plugin returns an Iris.cube.Cube with the expected
        data values for the percentiles when input probabilities are given
        for being below a threshold.
        The input cube contains probabilities that values are below a given
        threshold.
        """
        expected = np.array([8.4, 10.61538462, 11.84615385])
        expected = expected[:, np.newaxis, np.newaxis, np.newaxis]

        data = np.array([0.95, 0.3, 0.05])[::-1]
        data = data[:, np.newaxis, np.newaxis, np.newaxis]

        self.current_temperature_forecast_cube = (
            add_forecast_reference_time_and_forecast_period(
                set_up_probability_threshold_cube(
                    data, "air_temperature", "degreesC",
                    forecast_thresholds=[8, 10, 12], y_dimension_length=1,
                    x_dimension_length=1, spp__relative_to_threshold='above')))
        cube = self.current_temperature_forecast_cube
        cube.coord(var_name="threshold"
                   ).attributes["spp__relative_to_threshold"] = "below"
        percentiles = [10, 50, 90]
        bounds_pairing = (-40, 50)
        plugin = Plugin()
        result = plugin._probabilities_to_percentiles(
            cube, percentiles, bounds_pairing)
        self.assertArrayAlmostEqual(result.data, expected)

    def test_check_data_multiple_timesteps(self):
        """
        Test that the plugin returns an Iris.cube.Cube with the expected
        data values for the percentiles.
        """
        expected = np.array([[[[8., 8.],
                               [-8., 8.66666667]],
                              [[8., -16.],
                               [8., -16.]]],
                             [[[12., 12.],
                               [12., 12.]],
                              [[10.5, 10.],
                               [10.5, 10.]]],
                             [[[31., 31.],
                               [31., 31.]],
                              [[11.5, 11.33333333],
                               [11.5, 12.]]]], dtype=np.float32)

        data = np.array([[[[0.8, 0.8],
                           [0.7, 0.9]],
                          [[0.8, 0.6],
                           [0.8, 0.6]]],
                         [[[0.6, 0.6],
                           [0.6, 0.6]],
                          [[0.5, 0.4],
                           [0.5, 0.4]]],
                         [[[0.4, 0.4],
                           [0.4, 0.4]],
                          [[0.1, 0.1],
                           [0.1, 0.2]]]], dtype=np.float32)

        cube = set_up_probability_threshold_cube(
            data, "air_temperature", "degreesC", timesteps=2,
            x_dimension_length=2, y_dimension_length=2,
            spp__relative_to_threshold='above')
        self.probability_cube = (
            add_forecast_reference_time_and_forecast_period(
                cube, time_point=np.array([402295.0, 402296.0]),
                fp_point=[2.0, 3.0]))
        cube = self.probability_cube
        percentiles = [20, 60, 80]
        bounds_pairing = (-40, 50)
        plugin = Plugin()
        result = plugin._probabilities_to_percentiles(
            cube, percentiles, bounds_pairing)
        self.assertArrayAlmostEqual(result.data, expected, decimal=5)

    @ManageWarnings(record=True)
    def test_probabilities_not_monotonically_increasing(self,
                                                        warning_list=None):
        """
        Test that the plugin raises a Warning when the probabilities
        of the Cumulative Distribution Function are not monotonically
        increasing.
        """
        data = np.array([0.05, 0.7, 0.95])
        data = data[:, np.newaxis, np.newaxis, np.newaxis]

        self.current_temperature_forecast_cube = (
            add_forecast_reference_time_and_forecast_period(
                set_up_probability_threshold_cube(
                    data, "air_temperature", "degreesC",
                    forecast_thresholds=[8, 10, 12], y_dimension_length=1,
                    x_dimension_length=1, spp__relative_to_threshold='above')))
        cube = self.current_temperature_forecast_cube
        percentiles = [10, 50, 90]
        bounds_pairing = (-40, 50)
        plugin = Plugin()
        warning_msg = "The probability values used to construct the"
        plugin._probabilities_to_percentiles(
            cube, percentiles, bounds_pairing)
        self.assertTrue(any(warning_msg in str(item)
                            for item in warning_list))

    def test_result_cube_has_no_air_temperature_threshold_coordinate(self):
        """
        Test that the plugin returns a cube with coordinates that
        do not include the air_temperature_threshold coordinate.
        """
        cube = self.current_temperature_forecast_cube
        percentiles = [10, 50, 90]
        bounds_pairing = (-40, 50)
        plugin = Plugin()
        result = plugin._probabilities_to_percentiles(
            cube, percentiles, bounds_pairing)
        for coord in result.coords():
            self.assertNotEqual(coord.name(), "threshold")

    def test_check_data(self):
        """
        Test that the plugin returns an Iris.cube.Cube with the expected
        data values for the percentiles.
        """
        data = np.array([[[[15.8, 8., 10.4],
                           [-16., 8., -30.4],
                           [-30.4, -34., -35.2]]],
                         [[[31., 10., 12.],
                           [10., 10., 8.],
                           [8., -10., -16.]]],
                         [[[46.2, 31., 42.4],
                           [31., 11.6, 12.],
                           [11., 9., 3.2]]]], dtype=np.float32)

        cube = self.current_temperature_forecast_cube
        percentiles = [10, 50, 90]
        bounds_pairing = (-40, 50)
        plugin = Plugin()
        result = plugin._probabilities_to_percentiles(
            cube, percentiles, bounds_pairing)
        self.assertArrayAlmostEqual(result.data, data, decimal=5)

    def test_check_single_threshold(self):
        """
        Test that the plugin returns an Iris.cube.Cube with the expected
        data values for the percentiles, if a single threshold is used for
        constructing the percentiles.
        """
        data = np.array([[[[12.2, 8., 12.2],
                           [-16., 8., -30.4],
                           [-30.4, -34., -35.2]]],
                         [[[29., 26.66666667, 29.],
                           [23.75, 26.66666667, 8.],
                           [8., -10., -16.]]],
                         [[[45.8, 45.33333333, 45.8],
                           [44.75, 45.33333333, 41.6],
                           [41.6, 29., 3.2]]]], dtype=np.float32)

        threshold_coord = find_threshold_coordinate(
            self.current_temperature_forecast_cube)
        for acube in self.current_temperature_forecast_cube.slices_over(
                threshold_coord):
            cube = acube
            break
        percentiles = [10, 50, 90]
        bounds_pairing = (-40, 50)
        plugin = Plugin()
        result = plugin._probabilities_to_percentiles(
            cube, percentiles, bounds_pairing)
        self.assertArrayAlmostEqual(result.data, data, decimal=5)

    def test_lots_of_probability_thresholds(self):
        """
        Test that the plugin returns an Iris.cube.Cube with the expected
        data values for the percentiles, if there are lots of thresholds.
        """
        input_probs_1d = np.linspace(1, 0, 30)
        input_probs = np.tile(input_probs_1d, (3, 3, 1, 1)).T

        data = np.array([[[[2.9, 2.9, 2.9],
                           [2.9, 2.9, 2.9],
                           [2.9, 2.9, 2.9]]],
                         [[[14.5, 14.5, 14.5],
                           [14.5, 14.5, 14.5],
                           [14.5, 14.5, 14.5]]],
                         [[[26.099998, 26.099998, 26.099998],
                           [26.099998, 26.099998, 26.099998],
                           [26.099998, 26.099998, 26.099998]]]],
                        dtype=np.float32)

        temperature_values = np.arange(0, 30)
        cube = (
            add_forecast_reference_time_and_forecast_period(
                set_up_probability_threshold_cube(
                    input_probs, "air_temperature", "degreesC",
                    forecast_thresholds=temperature_values,
                    spp__relative_to_threshold='above')))
        percentiles = [10, 50, 90]
        bounds_pairing = (-40, 50)
        plugin = Plugin()
        result = plugin._probabilities_to_percentiles(
            cube, percentiles, bounds_pairing)

        self.assertArrayAlmostEqual(result.data, data)

    def test_lots_of_percentiles(self):
        """
        Test that the plugin returns an Iris.cube.Cube with the expected
        data values for the percentiles, if lots of percentile values are
        requested.
        """
        data = np.array([[[[13.9, -16., 10.2],
                           [-28., -16., -35.2],
                           [-35.2, -37., -37.6]]],
                         [[[17.7, 8.25, 10.6],
                           [-4., 8.25, -25.6],
                           [-25.6, -31., -32.8]]],
                         [[[21.5, 8.75, 11.],
                           [8.33333333, 8.75, -16.],
                           [-16., -25., -28.]]],
                         [[[25.3, 9.25, 11.4],
                           [9., 9.25, -6.4],
                           [-6.4, -19., -23.2]]],
                         [[[29.1, 9.75, 11.8],
                           [9.66666667, 9.75, 3.2],
                           [3.2, -13., -18.4]]],
                         [[[32.9, 10.33333333, 15.8],
                           [10.33333333, 10.2, 8.5],
                           [8.33333333, -7., -13.6]]],
                         [[[36.7, 11., 23.4],
                           [11., 10.6, 9.5],
                           [9., -1., -8.8]]],
                         [[[40.5, 11.66666667, 31.],
                           [11.66666667, 11., 10.5],
                           [9.66666667, 5., -4.]]],
                         [[[44.3, 21.5, 38.6],
                           [21.5, 11.4, 11.5],
                           [10.5, 8.5, 0.8]]],
                         [[[48.1, 40.5, 46.2],
                           [40.5, 11.8, 31.],
                           [11.5, 9.5, 5.6]]]], dtype=np.float32)

        cube = self.current_temperature_forecast_cube
        percentiles = np.arange(5, 100, 10)
        bounds_pairing = (-40, 50)
        plugin = Plugin()
        result = plugin._probabilities_to_percentiles(
            cube, percentiles, bounds_pairing)
        self.assertArrayAlmostEqual(result.data, data, decimal=5)

    def test_check_data_spot_forecasts(self):
        """
        Test that the plugin returns an Iris.cube.Cube with the expected
        data values for the percentiles for spot forecasts.
        """
        data = np.array([[[15.8, 8., 10.4,
                           -16., 8., -30.4,
                           -30.4, -34., -35.2]],
                         [[31., 10., 12.,
                           10., 10., 8.,
                           8., -10., -16.]],
                         [[46.2, 31., 42.4,
                           31., 11.6, 12.,
                           11., 9., 3.2]]], dtype=np.float32)

        cube = self.current_temperature_spot_forecast_cube
        percentiles = [10, 50, 90]
        bounds_pairing = (-40, 50)
        plugin = Plugin()
        result = plugin._probabilities_to_percentiles(
            cube, percentiles, bounds_pairing)
        self.assertArrayAlmostEqual(result.data, data, decimal=5)


class Test_process(IrisTest):

    """
    Test the process method of the ConvertProbabilitiesToPercentiles plugin.
    """

    def setUp(self):
        """Set up temperature cube."""
        self.current_temperature_forecast_cube = (
            add_forecast_reference_time_and_forecast_period(
                set_up_probability_above_threshold_temperature_cube()))

        self.percentile_25 = np.array(
            [[[24., 8.75, 11.],
              [8.33333333, 8.75, -46.],
              [-46., -66.25, -73.]]], dtype=np.float32)
        self.percentile_50 = np.array(
            [[[36., 10., 12.],
              [10., 10., 8.],
              [8., -32.5, -46.]]], dtype=np.float32)
        self.percentile_75 = np.array(
            [[[48., 11.66666667, 36.],
              [11.66666667, 11., 10.5],
              [9.66666667, 1.25, -19.]]], dtype=np.float32)

    @ManageWarnings(
        ignored_messages=["Only a single cube so no differences"])
    def test_check_data_specifying_no_of_percentiles(self):
        """
        Test that the plugin returns an Iris.cube.Cube with the expected
        data values for a specific number of percentiles.
        """
        expected_data = np.array(
            [self.percentile_25, self.percentile_50, self.percentile_75])
        cube = self.current_temperature_forecast_cube
        plugin = Plugin()
        result = plugin.process(cube, no_of_percentiles=3)
        self.assertArrayAlmostEqual(result.data, expected_data, decimal=5)

    @ManageWarnings(
        ignored_messages=["Only a single cube so no differences"])
    def test_check_data_specifying_single_percentile(self):
        """
        Test that the plugin returns an Iris.cube.Cube with the expected
        data values for a specific percentile passes in as a single realization
        list.
        """
        expected_data = np.array([self.percentile_25])
        cube = self.current_temperature_forecast_cube
        plugin = Plugin()
        result = plugin.process(cube, percentiles=[25])
        self.assertArrayAlmostEqual(result.data, expected_data, decimal=5)

    @ManageWarnings(
        ignored_messages=["Only a single cube so no differences"])
    def test_check_data_specifying_single_percentile_not_as_list(self):
        """
        Test that the plugin returns an Iris.cube.Cube with the expected
        data values for a specific percentile passed in as a value.
        """
        expected_data = np.array([self.percentile_25])
        cube = self.current_temperature_forecast_cube
        plugin = Plugin()
        result = plugin.process(cube, percentiles=25)
        self.assertArrayAlmostEqual(result.data, expected_data, decimal=5)

    @ManageWarnings(
        ignored_messages=["Only a single cube so no differences"])
    def test_check_data_specifying_percentiles(self):
        """
        Test that the plugin returns an Iris.cube.Cube with the expected
        data values for a specific set of percentiles.
        """
        expected_data = np.array(
            [self.percentile_25, self.percentile_50, self.percentile_75])
        cube = self.current_temperature_forecast_cube
        percentiles = [25, 50, 75]
        plugin = Plugin()
        result = plugin.process(
            cube, percentiles=percentiles)
        self.assertArrayAlmostEqual(result.data, expected_data, decimal=5)

    @ManageWarnings(
        ignored_messages=["Only a single cube so no differences"])
    def test_check_data_not_specifying_percentiles(self):
        """
        Test that the plugin returns an Iris.cube.Cube with the expected
        data values without specifying the number of percentiles.
        """
        expected_data = np.array(
            [self.percentile_25, self.percentile_50, self.percentile_75])
        cube = self.current_temperature_forecast_cube
        plugin = Plugin()
        result = plugin.process(cube)
        self.assertArrayAlmostEqual(result.data, expected_data, decimal=5)

    def test_check_data_over_specifying_percentiles(self):
        """
        Test that the plugin raises a suitable error when both a number and set
        or percentiles are specified.
        """
        no_of_percentiles = 3
        percentiles = [25, 50, 75]
        cube = self.current_temperature_forecast_cube
        plugin = Plugin()
        msg = "Cannot specify both no_of_percentiles and percentiles"
        with self.assertRaisesRegex(ValueError, msg):
            plugin.process(cube, no_of_percentiles=no_of_percentiles,
                           percentiles=percentiles)


if __name__ == '__main__':
    unittest.main()
