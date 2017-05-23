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
`ensemble_copula_coupling.EnsemeblCopulaCouplingUtilities` class.
"""
import unittest

from iris.coords import DimCoord
from iris.cube import Cube
from iris.exceptions import CoordinateNotFoundError
from iris.tests import IrisTest
import numpy as np

from improver.ensemble_copula_coupling.ensemble_copula_coupling_utilities \
    import (create_percentiles, create_cube_with_percentiles,
            insert_lower_and_upper_endpoint_to_1d_array,
            concatenate_2d_array_with_2d_array_endpoints,
            get_bounds_of_distribution,
            reshape_array_to_have_probabilistic_dimension_at_the_front)
from improver.ensemble_copula_coupling.ensemble_copula_coupling_constants \
    import bounds_for_ecdf
from improver.tests.helper_functions_ensemble_calibration import (
    set_up_cube,
    set_up_temperature_cube, set_up_spot_temperature_cube,
    _add_forecast_reference_time_and_forecast_period,
    set_up_probability_above_threshold_temperature_cube)


class Test_concatenate_2d_array_with_2d_array_endpoints(IrisTest):

    """Test the concatenate_2d_array_with_2d_array_endpoints."""

    def test_basic(self):
        """
        Basic test that the result is a numpy array with the expected contents.
        """
        expected = np.array([[0, 0.2, 0.5, 0.8, 1]])
        percentiles = np.array([[0.2, 0.5, 0.8]])
        result = concatenate_2d_array_with_2d_array_endpoints(
            percentiles, 0, 1)
        self.assertIsInstance(result, np.ndarray)
        self.assertArrayAlmostEqual(result, expected)

    def test_another_example(self):
        """
        Another basic test that the result is a numpy array with the
        expected contents.
        """
        expected = np.array(
           [[-100, -40, 200, 1000, 10000], [-100, -40, 200, 1000, 10000]])
        percentiles = np.array([[-40, 200, 1000], [-40, 200, 1000]])
        result = concatenate_2d_array_with_2d_array_endpoints(
            percentiles, -100, 10000)
        self.assertIsInstance(result, np.ndarray)
        self.assertArrayAlmostEqual(result, expected)

    def test_1d_input(self):
        """
        Test that a 1d input array results in the expected error.
        """
        expected = np.array([-100, -40, 200, 1000, 10000])
        percentiles = np.array([-40, 200, 1000])
        msg = "all the input arrays must have same number of dimensions"
        with self.assertRaisesRegexp(ValueError, msg):
            concatenate_2d_array_with_2d_array_endpoints(
                percentiles, -100, 10000)

    def test_3d_input(self):
        """
        Test that a 3d input array results in the expected error.
        """
        expected = np.array([[[-100, -40, 200, 1000, 10000]]])
        percentiles = np.array([[[-40, 200, 1000]]])
        msg = "all the input arrays must have same number of dimensions"
        with self.assertRaisesRegexp(ValueError, msg):
            concatenate_2d_array_with_2d_array_endpoints(
                percentiles, -100, 10000)


class Test_create_cube_with_percentiles(IrisTest):

    """Test the _create_cube_with_percentiles plugin."""

    def setUp(self):
        """Set up temperature cube."""
        current_temperature_forecast_cube = (
            _add_forecast_reference_time_and_forecast_period(
                set_up_temperature_cube()))

        self.cube_data = current_temperature_forecast_cube.data

        current_temperature_spot_forecast_cube = (
            _add_forecast_reference_time_and_forecast_period(
                set_up_spot_temperature_cube()))
        self.cube_spot_data = (
            current_temperature_spot_forecast_cube.data)

        for cube in current_temperature_forecast_cube.slices_over(
                "realization"):
            cube.remove_coord("realization")
            break
        self.current_temperature_forecast_cube = cube

        for cube in current_temperature_spot_forecast_cube.slices_over(
                "realization"):
            cube.remove_coord("realization")
            break
        self.current_temperature_spot_forecast_cube = cube

    def test_basic(self):
        """Test that the plugin returns an Iris.cube.Cube."""
        cube = self.current_temperature_forecast_cube
        cube_data = self.cube_data + 2
        percentiles = [0.1, 0.5, 0.9]
        result = create_cube_with_percentiles(
            percentiles, cube, cube_data)
        self.assertIsInstance(result, Cube)

    def test_many_percentiles(self):
        """
        Test that the plugin returns an Iris.cube.Cube with many
        percentiles.
        """
        cube = self.current_temperature_forecast_cube
        percentiles = np.linspace(0, 1, 100)
        cube_data = np.zeros(
            [len(percentiles), len(cube.coord("time").points),
             len(cube.coord("latitude").points),
             len(cube.coord("longitude").points)])
        result = create_cube_with_percentiles(
            percentiles, cube, cube_data)
        self.assertEqual(cube_data.shape, result.data.shape)

    def test_incompatible_percentiles(self):
        """
        Test that the plugin fails if the percentile values requested
        are not numbers.
        """
        cube = self.current_temperature_forecast_cube
        percentiles = ["cat", "dog", "elephant"]
        cube_data = np.zeros(
            [len(percentiles), len(cube.coord("time").points),
             len(cube.coord("latitude").points),
             len(cube.coord("longitude").points)])
        msg = "could not convert string to float"
        with self.assertRaisesRegexp(ValueError, msg):
            create_cube_with_percentiles(percentiles, cube, cube_data)

    def test_percentile_points(self):
        """
        Test that the plugin returns an Iris.cube.Cube
        with a percentile coordinate with the desired points.
        """
        cube = self.current_temperature_forecast_cube
        cube_data = self.cube_data + 2
        percentiles = [0.1, 0.5, 0.9]
        result = create_cube_with_percentiles(percentiles, cube, cube_data)
        self.assertIsInstance(result.coord("percentile"), DimCoord)
        self.assertArrayAlmostEqual(
            result.coord("percentile").points, percentiles)

    def test_spot_forecasts_percentile_points(self):
        """
        Test that the plugin returns a Cube with a percentile dimension
        coordinate and that the percentile dimension has the expected points
        for an input spot forecast.
        """
        cube = self.current_temperature_spot_forecast_cube
        cube_data = self.cube_spot_data + 2
        percentiles = [0.1, 0.5, 0.9]
        result = create_cube_with_percentiles(
            percentiles, cube, cube_data)
        self.assertIsInstance(result, Cube)
        self.assertIsInstance(result.coord("percentile"), DimCoord)
        self.assertArrayAlmostEqual(
            result.coord("percentile").points, percentiles)

    def test_percentile_length_too_short(self):
        """
        Test that the plugin raises the default ValueError, if the number
        of percentiles is fewer than the length of the zeroth dimension within
        the cube.
        """
        cube = self.current_temperature_forecast_cube
        cube_data = self.cube_data + 2
        percentiles = [0.1, 0.5]
        msg = "Unequal lengths"
        with self.assertRaisesRegexp(ValueError, msg):
            create_cube_with_percentiles(
                percentiles, cube, cube_data)

    def test_percentile_length_too_long(self):
        """
        Test that the plugin raises the default ValueError, if the number
        of percentiles exceeds the length of the zeroth dimension within
        the cube.
        """
        cube = self.current_temperature_forecast_cube
        cube = cube[0, :, :, :]
        cube_data = self.cube_data + 2
        percentiles = [0.1, 0.5, 0.9]
        msg = "Unequal lengths"
        with self.assertRaisesRegexp(ValueError, msg):
            create_cube_with_percentiles(
                percentiles, cube, cube_data)

    def test_metadata_copy(self):
        """
        Test that the metadata dictionaries within the input cube, are
        also present on the output cube.
        """
        cube = self.current_temperature_forecast_cube
        cube.attributes = {"source": "ukv"}
        cube_data = self.cube_data + 2
        percentiles = [0.1, 0.5, 0.9]
        result = create_cube_with_percentiles(
            percentiles, cube, cube_data)
        self.assertDictEqual(
            cube.metadata._asdict(), result.metadata._asdict())

    def test_coordinate_copy(self):
        """
        Test that the coordinates within the input cube, are
        also present on the output cube.
        """
        cube = self.current_temperature_forecast_cube
        cube.attributes = {"source": "ukv"}
        cube_data = self.cube_data + 2
        percentiles = [0.1, 0.5, 0.9]
        result = create_cube_with_percentiles(
            percentiles, cube, cube_data)
        for coord in cube.coords():
            if coord not in result.coords():
                msg = (
                    "Coordinate: {} not found in cube {}".format(
                        coord, result))
                raise CoordinateNotFoundError(msg)


class Test_create_percentiles(IrisTest):

    """Test the create_percentiles plugin."""

    def test_basic(self):
        """
        Test that the plugin returns a list with the expected number of
        percentiles.
        """
        no_of_percentiles = 3
        result = create_percentiles(no_of_percentiles)
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), no_of_percentiles)

    def test_data(self):
        """
        Test that the plugin returns a list with the expected data values
        for the percentiles.
        """
        data = np.array([0.25, 0.5, 0.75])
        no_of_percentiles = 3
        result = create_percentiles(no_of_percentiles)
        self.assertArrayAlmostEqual(result, data)

    def test_random(self):
        """
        Test that the plugin returns a list with the expected number of
        percentiles, if the random sampling option is selected.
        """
        no_of_percentiles = 3
        result = create_percentiles(no_of_percentiles, sampling="random")
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), no_of_percentiles)

    def test_unknown_sampling_option(self):
        """
        Test that the plugin returns the expected error message,
        if an unknown sampling option is selected.
        """
        no_of_percentiles = 3
        msg = "The unknown sampling option is not yet implemented"
        with self.assertRaisesRegexp(ValueError, msg):
            create_percentiles(no_of_percentiles, sampling="unknown")


class Test_get_bounds_of_distribution(IrisTest):

    """Test the get_bounds_of_distribution plugin."""

    def setUp(self):
        self.current_temperature_forecast_cube = (
            _add_forecast_reference_time_and_forecast_period(
                set_up_probability_above_threshold_temperature_cube()))

    def test_basic(self):
        """Test that the result is a numpy array."""
        cube = self.current_temperature_forecast_cube
        cube_units = cube.coord("probability_above_threshold").units
        result = get_bounds_of_distribution(cube.name(), cube_units)
        self.assertIsInstance(result, np.ndarray)

    def test_check_data(self):
        """
        Test that the expected results are returned for the bounds_pairing.
        """
        cube = self.current_temperature_forecast_cube
        cube_units = cube.coord("probability_above_threshold").units
        bounds_pairing = (-40, 50)
        result = (
            get_bounds_of_distribution(cube.name(), cube_units))
        self.assertArrayAlmostEqual(result, bounds_pairing)

    def test_check_unit_conversion(self):
        """
        Test that the expected results are returned for the bounds_pairing,
        if the units of the bounds_pairings need to be converted to match
        the units of the forecast.
        """
        cube = self.current_temperature_forecast_cube
        cube.coord("probability_above_threshold").convert_units("fahrenheit")
        cube_units = cube.coord("probability_above_threshold").units
        bounds_pairing = (-40, 122)  # In fahrenheit
        result = (
            get_bounds_of_distribution(cube.name(), cube_units))
        self.assertArrayAlmostEqual(result, bounds_pairing)

    def test_check_exception_is_raised(self):
        """
        Test that the expected results are returned for the bounds_pairing.
        """
        cube = self.current_temperature_forecast_cube
        cube.standard_name = None
        cube.long_name = "Nonsense"
        cube_units = cube.coord("probability_above_threshold").units
        msg = "The forecast_cube name"
        with self.assertRaisesRegexp(KeyError, msg):
            get_bounds_of_distribution(cube.name(), cube_units)


class Test_insert_lower_and_upper_endpoint_to_1d_array(IrisTest):

    """Test the insert_lower_and_upper_endpoint_to_1d_array."""

    def test_basic(self):
        """
        Basic test that the result is a numpy array with the expected contents.
        """
        expected = [0, 0.2, 0.5, 0.8, 1]
        percentiles = [0.2, 0.5, 0.8]
        result = insert_lower_and_upper_endpoint_to_1d_array(
            percentiles, 0, 1)
        self.assertIsInstance(result, np.ndarray)
        self.assertArrayAlmostEqual(result, expected)

    def test_another_example(self):
        """
        Another basic test that the result is a numpy array with the
        expected contents.
        """
        expected = [-100, -40, 200, 1000, 10000]
        percentiles = [-40, 200, 1000]
        result = insert_lower_and_upper_endpoint_to_1d_array(
            percentiles, -100, 10000)
        self.assertIsInstance(result, np.ndarray)
        self.assertArrayAlmostEqual(result, expected)


class Test_reshape_array_to_have_probabilistic_dimension_at_the_front(
        IrisTest):

    """Test the insert_lower_and_upper_endpoint_to_1d_array."""

    def setUp(self):
        """Set up temperature cube."""
        cube = (
            _add_forecast_reference_time_and_forecast_period(
                set_up_temperature_cube()))
        percentile_points = np.arange(len(cube.coord("realization").points))
        cube.coord("realization").points = percentile_points
        cube.coord("realization").rename("percentile")
        self.current_temperature_forecast_cube = cube

    def test_basic(self):
        """
        Basic test that the result is a numpy array with the expected contents.
        """
        cube = self.current_temperature_forecast_cube
        input_array = cube.data
        plen = len(cube.coord("percentile").points)
        reshaped_array = (
            reshape_array_to_have_probabilistic_dimension_at_the_front(
                cube.data, cube, "percentile", plen))
        self.assertIsInstance(reshaped_array, np.ndarray)

    def test_size_of_array(self):
        """
        Test that the result have the expected size for the
       probabilistic dimension and is generally of the expected size.
        """
        cube = self.current_temperature_forecast_cube
        input_array = cube.data
        plen = len(cube.coord("percentile").points)
        reshaped_array = (
            reshape_array_to_have_probabilistic_dimension_at_the_front(
                cube.data, cube, "percentile", plen))
        self.assertEqual(reshaped_array.shape[0], plen)
        self.assertEqual(reshaped_array.shape, (3, 1, 3, 3))

    def test_data_check(self):
        """
        Test that the data has been reshaped correctly.
        """
        expected = np.array([[[[ 4., 6.],
                               [8., 6.85714286]],
                              [[8.85714286, 10.85714286],
                               [5.42857143, 7.42857143]]],
                             [[[9.42857143, 8.28571429],
                               [10.28571429, 12.28571429]],
                              [[4.71428571, 6.71428571],
                               [8.71428571, 7.57142857]]],
                             [[[9.57142857, 11.57142857],
                               [6.14285714, 8.14285714]],
                              [[10.14285714, 9.],
                               [11., 13.]]]])

        data = np.tile(np.linspace(5, 10, 8), 3).reshape(3, 2, 2, 2)
        data[0] -= 1
        data[1] += 1
        data[2] += 3
        cube = set_up_cube(data, "air_temperature", "degreesC",
                           timesteps=2, x_dimension_length=2,
                           y_dimension_length=2)
        cube.coord("realization").rename("percentile")
        cube.coord("percentile").points = np.array([0.1, 0.5, 0.9])
        plen = len(cube.coord("percentile").points)
        percentile_cube = (
            _add_forecast_reference_time_and_forecast_period(
                cube, time_point=np.array([402295.0, 402296.0])))
        reshaped_array = (
            reshape_array_to_have_probabilistic_dimension_at_the_front(
                percentile_cube.data, percentile_cube, "percentile", plen))
        self.assertArrayAlmostEqual(reshaped_array, expected)

    def test_percentile_is_not_a_dimension_coordinate(self):
        """
        Test the array size, if the percentile coordinate is not a dimension
        coordinate on the cube.
        """
        cube = self.current_temperature_forecast_cube
        for cube_slice in cube.slices_over("percentile"):
            break
        input_array = cube_slice.data
        plen = len(cube_slice.coord("percentile").points)
        reshaped_array = (
            reshape_array_to_have_probabilistic_dimension_at_the_front(
                cube_slice.data, cube_slice, "percentile", plen))
        self.assertEqual(reshaped_array.shape[0], plen)
        self.assertEqual(reshaped_array.shape, (1, 1, 3, 3))

    def test_missing_coordinate(self):
        """
        Basic test that the result is a numpy array with the expected contents.
        """
        cube = self.current_temperature_forecast_cube
        input_array = cube.data
        plen = len(cube.coord("percentile").points)
        msg = "coordinate is not available"
        with self.assertRaisesRegexp(CoordinateNotFoundError, msg):
            reshape_array_to_have_probabilistic_dimension_at_the_front(
                cube.data, cube, "nonsense", plen)


if __name__ == '__main__':
    unittest.main()
