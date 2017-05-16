# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown Copyright 2017 Met Office.
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
`plugins_ensemble_copula_coupling.GeneratePercentilesFromProbabilities`
class.

"""
import numpy as np
import unittest

from cf_units import Unit
from iris.coords import AuxCoord, DimCoord
from iris.cube import Cube
from iris.tests import IrisTest

from improver.ensemble_copula_coupling.ensemble_copula_coupling import (
    GeneratePercentilesFromProbabilities as Plugin)
from improver.ensemble_copula_coupling.ensemble_copula_coupling_constants \
    import bounds_for_ecdf, units_of_bounds_for_ecdf
from improver.tests.helper_functions_ensemble_calibration import(
    _add_forecast_reference_time_and_forecast_period)


def set_up_cube(data, phenomenon_standard_name, phenomenon_units,
                forecast_thresholds=[8, 10, 12],
                y_dimension_length=3, x_dimension_length=3):
    """Create a cube containing multiple realizations."""
    cube = Cube(data, standard_name=phenomenon_standard_name,
                units=phenomenon_units)
    cube.add_dim_coord(
        DimCoord(forecast_thresholds,
                 long_name='probability_above_threshold', units='degreesC'), 0)
    time_origin = "hours since 1970-01-01 00:00:00"
    calendar = "gregorian"
    tunit = Unit(time_origin, calendar)
    cube.add_dim_coord(DimCoord([402192.5],
                                "time", units=tunit), 1)
    cube.add_dim_coord(DimCoord(np.linspace(-45.0, 45.0, y_dimension_length),
                                'latitude', units='degrees'), 2)
    cube.add_dim_coord(DimCoord(np.linspace(120, 180, x_dimension_length),
                                'longitude', units='degrees'), 3)
    return cube


def set_up_temperature_cube():
    """Create a cube with metadata and values suitable for air temperature."""
    data = np.array([[[[1.0, 0.9, 1.0],
                       [0.8, 0.9, 0.5],
                       [0.5, 0.2, 0.0]]],
                     [[[1.0, 0.5, 1.0],
                       [0.5, 0.5, 0.3],
                       [0.2, 0.0, 0.0]]],
                     [[[1.0, 0.2, 0.5],
                       [0.2, 0.0, 0.1],
                       [0.0, 0.0, 0.0]]]])
    return set_up_cube(data, "air_temperature", "1")


def set_up_spot_cube(data, phenomenon_standard_name, phenomenon_units,
                     forecast_thresholds=[8, 10, 12],
                     y_dimension_length=9, x_dimension_length=9):
    """
    Create a cube containing multiple realizations, where one of the
    dimensions is an index used for spot forecasts.
    """
    cube = Cube(data, standard_name=phenomenon_standard_name,
                units=phenomenon_units)
    cube.add_dim_coord(
        DimCoord(forecast_thresholds,
                 long_name='probability_above_threshold', units='degreesC'), 0)
    time_origin = "hours since 1970-01-01 00:00:00"
    calendar = "gregorian"
    tunit = Unit(time_origin, calendar)
    cube.add_dim_coord(DimCoord([402192.5],
                                "time", units=tunit), 1)
    cube.add_dim_coord(DimCoord(np.arange(9), long_name='locnum',
                                units="1"), 2)
    cube.add_aux_coord(AuxCoord(np.linspace(-45.0, 45.0, y_dimension_length),
                                'latitude', units='degrees'), data_dims=2)
    cube.add_aux_coord(AuxCoord(np.linspace(120, 180, x_dimension_length),
                                'longitude', units='degrees'), data_dims=2)
    return cube


def set_up_spot_temperature_cube():
    """
    Create a cube with metadata and values suitable for air temperature
    for spot forecasts.
    """
    data = np.array([[[1.0, 0.9, 1.0,
                       0.8, 0.9, 0.5,
                       0.5, 0.2, 0.0]],
                     [[1.0, 0.5, 1.0,
                       0.5, 0.5, 0.3,
                       0.2, 0.0, 0.0]],
                     [[1.0, 0.2, 0.5,
                       0.2, 0.0, 0.1,
                       0.0, 0.0, 0.0]]])
    return set_up_spot_cube(data, "air_temperature", "1")


class Test__add_bounds_to_thresholds_and_probabilities(IrisTest):

    """Test the _add_bounds_to_thresholds_and_probabilities plugin."""

    def setUp(self):
        self.current_temperature_forecast_cube = (
            _add_forecast_reference_time_and_forecast_period(
                set_up_temperature_cube()))

    def test_basic(self):
        """Test that the plugin returns two numpy arrays."""
        cube = self.current_temperature_forecast_cube
        threshold_points = cube.coord("probability_above_threshold").points
        probabilities_for_cdf = cube.data.reshape(3, 9)
        bounds_pairing = (-40, 50)
        plugin = Plugin()
        result = plugin._add_bounds_to_thresholds_and_probabilities(
            threshold_points, probabilities_for_cdf, bounds_pairing)
        self.assertIsInstance(result[0], np.ndarray)
        self.assertIsInstance(result[1], np.ndarray)

    def test_bounds_of_threshold_points(self):
        """
        Test that the plugin returns the expected results for the
        threshold_points, where they've been padded with the values from
        the bounds_pairing.
        """
        cube = self.current_temperature_forecast_cube
        threshold_points = cube.coord("probability_above_threshold").points
        probabilities_for_cdf = cube.data.reshape(3, 9)
        bounds_pairing = (-40, 50)
        plugin = Plugin()
        result = plugin._add_bounds_to_thresholds_and_probabilities(
            threshold_points, probabilities_for_cdf, bounds_pairing)
        self.assertArrayAlmostEqual(result[0][0], bounds_pairing[0])
        self.assertArrayAlmostEqual(result[0][-1], bounds_pairing[1])

    def test_probability_data(self):
        """
        Test that the plugin returns the expected results for the
        probabilities, where they've been padded with zeros and ones to
        represent the extreme ends of the Cumulative Distribution Function.
        """
        cube = self.current_temperature_forecast_cube
        threshold_points = cube.coord("probability_above_threshold").points
        probabilities_for_cdf = cube.data.reshape(3, 9)
        zero_array = np.zeros(probabilities_for_cdf[:, 0].shape)
        one_array = np.ones(probabilities_for_cdf[:, 0].shape)
        bounds_pairing = (-40, 50)
        plugin = Plugin()
        result = plugin._add_bounds_to_thresholds_and_probabilities(
            threshold_points, probabilities_for_cdf, bounds_pairing)
        self.assertArrayAlmostEqual(result[1][:, 0], zero_array)
        self.assertArrayAlmostEqual(result[1][:, -1], one_array)


class Test__probabilities_to_percentiles(IrisTest):

    """Test the _create_cube_with_percentiles plugin."""

    def setUp(self):
        """Set up temperature cube."""
        self.current_temperature_forecast_cube = (
            _add_forecast_reference_time_and_forecast_period(
                set_up_temperature_cube()))
        self.current_temperature_spot_forecast_cube = (
            _add_forecast_reference_time_and_forecast_period(
                set_up_spot_temperature_cube()))

    def test_basic(self):
        """Test that the plugin returns an Iris.cube.Cube."""
        cube = self.current_temperature_forecast_cube
        percentiles = [0.1, 0.5, 0.9]
        bounds_pairing = (-40, 50)
        plugin = Plugin()
        result = plugin._probabilities_to_percentiles(
            cube, percentiles, bounds_pairing)
        self.assertIsInstance(result, Cube)

    def test_simple_check_data(self):
        """
        Test that the plugin returns an Iris.cube.Cube with the expected
        data values for the percentiles.

        The input cube contains probabilities greater than a given threshold.
        """
        expected = np.array([8.15384615, 9.38461538, 11.6])
        expected = expected[:, np.newaxis, np.newaxis, np.newaxis]

        data = np.array([0.95, 0.3, 0.05])
        data = data[:, np.newaxis, np.newaxis, np.newaxis]

        self.current_temperature_forecast_cube = (
            _add_forecast_reference_time_and_forecast_period(
                set_up_cube(
                    data, "air_temperature", "1",
                    forecast_thresholds=[8, 10, 12], y_dimension_length=1,
                    x_dimension_length=1)))
        cube = self.current_temperature_forecast_cube
        percentiles = [0.1, 0.5, 0.9]
        bounds_pairing = (-40, 50)
        plugin = Plugin()
        result = plugin._probabilities_to_percentiles(
            cube, percentiles, bounds_pairing)
        self.assertArrayAlmostEqual(result.data, expected)

    def test_probabilities_not_monotonically_increasing(self):
        """
        Test that the plugin raises a ValueError when the probabilities
        of the Cumulative Distribution Function are not monotonically
        increasing.
        """
        data = np.array([0.05, 0.7, 0.95])
        data = data[:, np.newaxis, np.newaxis, np.newaxis]

        self.current_temperature_forecast_cube = (
            _add_forecast_reference_time_and_forecast_period(
                set_up_cube(
                    data, "air_temperature", "1",
                    forecast_thresholds=[8, 10, 12], y_dimension_length=1,
                    x_dimension_length=1)))
        cube = self.current_temperature_forecast_cube
        percentiles = [0.1, 0.5, 0.9]
        bounds_pairing = (-40, 50)
        plugin = Plugin()
        msg = "The probability values used to construct the"
        with self.assertRaisesRegexp(ValueError, msg):
            result = plugin._probabilities_to_percentiles(
                cube, percentiles, bounds_pairing)

    def test_thresholds_not_monotonically_increasing(self):
        """
        Test that the plugin raises a ValueError, if threshold points
        are added to the cube, which are non monotonically increasing.
        """
        data = 1 - np.array([0.05, 0.7, 0.95])
        data = data[:, np.newaxis, np.newaxis, np.newaxis]
        msg = "The points array must be strictly monotonic"
        with self.assertRaisesRegexp(ValueError, msg):
            self.current_temperature_forecast_cube = (
                _add_forecast_reference_time_and_forecast_period(
                    set_up_cube(
                        data, "air_temperature", "1",
                        forecast_thresholds=[8, 12, 10], y_dimension_length=1,
                        x_dimension_length=1)))

    def test_endpoints_of_distribution_exceeded(self):
        """
        Test that the plugin raises a ValueError when the constant
        end points of the distribution are exceeded by a threshold value
        used in the forecast.
        """
        data = 1 - np.array([0.05, 0.7, 0.95])
        data = data[:, np.newaxis, np.newaxis, np.newaxis]

        self.current_temperature_forecast_cube = (
            _add_forecast_reference_time_and_forecast_period(
                set_up_cube(
                    data, "air_temperature", "1",
                    forecast_thresholds=[8, 10, 60], y_dimension_length=1,
                    x_dimension_length=1)))
        cube = self.current_temperature_forecast_cube
        percentiles = [0.1, 0.5, 0.9]
        bounds_pairing = (-40, 50)
        plugin = Plugin()
        msg = "The end points added to the threshold values for"
        with self.assertRaisesRegexp(ValueError, msg):
            result = plugin._probabilities_to_percentiles(
                cube, percentiles, bounds_pairing)

    def test_result_cube_has_no_probability_above_threshold_coordinate(self):
        """
        Test that the plugin returns a cube with coordinates that
        do not include the probability_above_threshold coordinate.
        """
        cube = self.current_temperature_forecast_cube
        percentiles = [0.1, 0.5, 0.9]
        bounds_pairing = (-40, 50)
        plugin = Plugin()
        result = plugin._probabilities_to_percentiles(
            cube, percentiles, bounds_pairing)
        for coord in result.coords():
            self.assertNotEqual(coord.name(), "probability_above_threshold")

    def test_check_data(self):
        """
        Test that the plugin returns an Iris.cube.Cube with the expected
        data values for the percentiles.
        """
        data = np.array([[[[15.8, 31., 46.2],
                           [8., 10., 31.],
                           [10.4, 12., 42.4]]],
                         [[[-16., 10, 31.],
                           [8., 10., 11.6],
                           [-30.4, 8., 12.]]],
                         [[[-30.4, 8., 11.],
                           [-34., -10., 9],
                           [-35.2, -16., 3.2]]]])

        cube = self.current_temperature_forecast_cube
        percentiles = [0.1, 0.5, 0.9]
        bounds_pairing = (-40, 50)
        plugin = Plugin()
        result = plugin._probabilities_to_percentiles(
            cube, percentiles, bounds_pairing)
        self.assertArrayAlmostEqual(result.data, data)

    def test_check_single_threshold(self):
        """
        Test that the plugin returns an Iris.cube.Cube with the expected
        data values for the percentiles, if a single threshold is used for
        constructing the percentiles.
        """
        data = np.array([[[[12.2, 29., 45.8],
                           [8., 26.66666667, 45.33333333],
                           [12.2, 29., 45.8]]],
                         [[[-16., 23.75, 44.75],
                           [8., 26.66666667, 45.33333333],
                           [-30.4, 8., 41.6]]],
                         [[[-30.4, 8., 41.6],
                           [-34., -10., 29.],
                           [-35.2, -16., 3.2]]]])

        for acube in self.current_temperature_forecast_cube.slices_over(
                "probability_above_threshold"):
            cube = acube
            break
        percentiles = [0.1, 0.5, 0.9]
        bounds_pairing = (-40, 50)
        plugin = Plugin()
        result = plugin._probabilities_to_percentiles(
            cube, percentiles, bounds_pairing)
        self.assertArrayAlmostEqual(result.data, data)

    def test_lots_of_probability_thresholds(self):
        """
        Test that the plugin returns an Iris.cube.Cube with the expected
        data values for the percentiles, if there are lots of thresholds.
        """
        input_probs_1d = np.linspace(1, 0, 30)
        input_probs = np.tile(input_probs_1d, (3, 3, 1, 1)).T

        data = np.array([[[[2.9, 14.5, 26.1],
                           [2.9, 14.5, 26.1],
                           [2.9, 14.5, 26.1]]],
                         [[[2.9, 14.5, 26.1],
                           [2.9, 14.5, 26.1],
                           [2.9, 14.5, 26.1]]],
                         [[[2.9, 14.5, 26.1],
                           [2.9, 14.5, 26.1],
                           [2.9, 14.5, 26.1]]]])

        temperature_values = np.arange(0, 30)
        cube = (
            _add_forecast_reference_time_and_forecast_period(
                set_up_cube(input_probs, "air_temperature", "1",
                            forecast_thresholds=temperature_values)))
        percentiles = [0.1, 0.5, 0.9]
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
        data = np.array([[[[13.9, 15.8, 17.7],
                           [19.6, 21.5, 23.4],
                           [25.3, 27.2, 29.1]]],
                         [[[31., 32.9, 34.8],
                           [36.7, 38.6, 40.5],
                           [42.4, 44.3, 46.2]]],
                         [[[48.1, -16., 8.],
                           [8.25, 8.5, 8.75],
                           [9., 9.25, 9.5]]],
                         [[[9.75, 10., 10.33333333],
                           [10.66666667, 11., 11.33333333],
                           [11.66666667, 12., 21.5]]],
                         [[[31., 40.5, 10.2],
                           [10.4, 10.6, 10.8],
                           [11., 11.2, 11.4]]],
                         [[[11.6, 11.8, 12.],
                           [15.8, 19.6, 23.4],
                           [27.2, 31., 34.8]]],
                         [[[38.6, 42.4, 46.2],
                           [-28., -16., -4.],
                           [8., 8.33333333, 8.66666667]]],
                         [[[9., 9.33333333, 9.66666667],
                           [10., 10.33333333, 10.66666667],
                           [11., 11.33333333, 11.66666667]]],
                         [[[12., 21.5, 31.],
                           [40.5, -16., 8.],
                           [8.25, 8.5, 8.75]]],
                         [[[9., 9.25, 9.5],
                           [9.75, 10., 10.2],
                           [10.4, 10.6, 10.8]]],
                         [[[11., 11.2, 11.4],
                           [11.6, 11.8, -35.2],
                           [-30.4, -25.6, -20.8]]],
                         [[[-16., -11.2, -6.4],
                           [-1.6, 3.2, 8.],
                           [8.5, 9., 9.5]]],
                         [[[10., 10.5, 11.],
                           [11.5, 12., 31.],
                           [-35.2, -30.4, -25.6]]],
                         [[[-20.8, -16., -11.2],
                           [-6.4, -1.6, 3.2],
                           [8., 8.33333333, 8.66666667]]],
                         [[[9., 9.33333333, 9.66666667],
                           [10., 10.5, 11.],
                           [11.5, -37., -34.]]],
                         [[[-31., -28., -25.],
                           [-22., -19., -16.],
                           [-13., -10., -7.]]],
                         [[[-4., -1., 2.],
                           [5., 8., 8.5],
                           [9., 9.5, -37.6]]],
                         [[[-35.2, -32.8, -30.4],
                           [-28., -25.6, -23.2],
                           [-20.8, -18.4, -16.]]],
                         [[[-13.6, -11.2, -8.8],
                           [-6.4, -4., -1.6],
                           [0.8, 3.2, 5.6]]]])
        cube = self.current_temperature_forecast_cube
        percentiles = np.arange(0.05, 1.0, 0.05)
        bounds_pairing = (-40, 50)
        plugin = Plugin()
        result = plugin._probabilities_to_percentiles(
            cube, percentiles, bounds_pairing)
        self.assertArrayAlmostEqual(result.data, data)

    def test_check_data_spot_forecasts(self):
        """
        Test that the plugin returns an Iris.cube.Cube with the expected
        data values for the percentiles for spot forecasts.
        """
        data = np.array([[[15.8, 31., 46.2,
                           8., 10., 31.,
                           10.4, 12., 42.4]],
                         [[-16., 10, 31.,
                           8., 10., 11.6,
                           -30.4, 8., 12.]],
                         [[-30.4, 8., 11.,
                           -34., -10., 9,
                           -35.2, -16., 3.2]]])
        cube = self.current_temperature_spot_forecast_cube
        percentiles = [0.1, 0.5, 0.9]
        bounds_pairing = (-40, 50)
        plugin = Plugin()
        result = plugin._probabilities_to_percentiles(
            cube, percentiles, bounds_pairing)
        self.assertArrayAlmostEqual(result.data, data)


class Test__get_bounds_of_distribution(IrisTest):

    """Test the _get_bounds_of_distribution plugin."""

    def setUp(self):
        self.current_temperature_forecast_cube = (
            _add_forecast_reference_time_and_forecast_period(
                set_up_temperature_cube()))

    def test_basic(self):
        """Test that the result is a numpy array."""
        cube = self.current_temperature_forecast_cube
        plugin = Plugin()
        result = plugin._get_bounds_of_distribution(cube)
        self.assertIsInstance(result, np.ndarray)

    def test_check_data(self):
        """
        Test that the expected results are returned for the bounds_pairing.
        """
        cube = self.current_temperature_forecast_cube
        bounds_pairing = (-40, 50)
        plugin = Plugin()
        result = plugin._get_bounds_of_distribution(cube)
        self.assertArrayAlmostEqual(result, bounds_pairing)

    def test_check_unit_conversion(self):
        """
        Test that the expected results are returned for the bounds_pairing,
        if the units of the bounds_pairings need to be converted to match
        the units of the forecast.
        """
        cube = self.current_temperature_forecast_cube
        cube.coord("probability_above_threshold").convert_units("fahrenheit")
        bounds_pairing = (-40, 122)  # In fahrenheit
        plugin = Plugin()
        result = plugin._get_bounds_of_distribution(cube)
        self.assertArrayAlmostEqual(result, bounds_pairing)

    def test_check_exception_is_raised(self):
        """
        Test that the expected results are returned for the bounds_pairing.
        """
        cube = self.current_temperature_forecast_cube
        cube.standard_name = None
        cube.long_name = "Nonsense"
        bounds_pairing = bounds_for_ecdf["air_temperature"]
        plugin = Plugin()
        msg = "The forecast_probabilities name"
        with self.assertRaisesRegexp(KeyError, msg):
            result = plugin._get_bounds_of_distribution(cube)


class Test_process(IrisTest):

    """Test the _create_cube_with_percentiles plugin."""

    def setUp(self):
        """Set up temperature cube."""
        self.current_temperature_forecast_cube = (
            _add_forecast_reference_time_and_forecast_period(
                set_up_temperature_cube()))

    def test_check_data_specifying_percentiles(self):
        """
        Test that the plugin returns an Iris.cube.Cube with the expected
        data values for a specific number of percentiles.
        """
        data = np.array([[[[21.5, 31., 40.5],
                           [8.75, 10., 11.66666667],
                           [11., 12., 31.]]],
                         [[[8.33333333, 10., 11.66666667],
                           [8.75, 10., 11.],
                           [-16., 8., 10.5]]],
                         [[[-16., 8., 9.66666667],
                           [-25., -10., 5.],
                           [-28., -16., -4.]]]])

        cube = self.current_temperature_forecast_cube
        percentiles = [0.1, 0.5, 0.9]
        plugin = Plugin()
        result = plugin.process(
            cube, no_of_percentiles=len(percentiles))
        self.assertArrayAlmostEqual(result.data, data)

    def test_check_data_not_specifying_percentiles(self):
        """
        Test that the plugin returns an Iris.cube.Cube with the expected
        data values without specifying the number of percentiles.
        """
        data = np.array([[[[21.5, 31., 40.5],
                           [8.75, 10., 11.66666667],
                           [11., 12., 31.]]],
                         [[[8.33333333, 10., 11.66666667],
                           [8.75, 10., 11.],
                           [-16., 8., 10.5]]],
                         [[[-16., 8., 9.66666667],
                           [-25., -10., 5.],
                           [-28., -16., -4.]]]])

        cube = self.current_temperature_forecast_cube
        plugin = Plugin()
        result = plugin.process(cube)
        self.assertArrayAlmostEqual(result.data, data)


if __name__ == '__main__':
    unittest.main()
