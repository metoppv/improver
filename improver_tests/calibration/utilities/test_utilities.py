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
Unit tests for the utilities within the `ensemble_calibration_utilities`
module.

"""
import unittest

import iris
import numpy as np
from iris.tests import IrisTest

from improver.calibration.utilities import (
    check_predictor, convert_cube_data_to_2d,
    flatten_ignoring_masked_data, filter_non_matching_cubes)

from ..ensemble_calibration.helper_functions import (set_up_temperature_cube,
                                                     SetupCubes)


class Test_convert_cube_data_to_2d(IrisTest):

    """Test the convert_cube_data_to_2d utility."""

    def setUp(self):
        """Use temperature cube to test with."""
        self.cube = set_up_temperature_cube()
        self.data = np.array([[226.15, 230.15, 232.15],
                              [237.4, 241.4, 243.4],
                              [248.65, 252.65, 254.65],
                              [259.9, 263.9, 265.9],
                              [271.15, 275.15, 277.15],
                              [282.4, 286.4, 288.4],
                              [293.65, 297.65, 299.65],
                              [304.9, 308.9, 310.9],
                              [316.15, 320.15, 322.15]],
                             dtype=np.float32)

    def test_basic(self):
        """Test that the utility returns an iris.cube.Cube."""
        result = convert_cube_data_to_2d(self.cube)
        self.assertIsInstance(result, np.ndarray)

    def test_check_values(self):
        """Test that the utility returns the expected data values."""
        result = convert_cube_data_to_2d(self.cube)
        self.assertArrayAlmostEqual(result, self.data)

    def test_change_coordinate(self):
        """
        Test that the utility returns the expected data values
        when the cube is sliced along the longitude dimension.
        """
        data = self.data.flatten().reshape(9, 3).T.reshape(9, 3)

        result = convert_cube_data_to_2d(
            self.cube, coord="longitude")
        self.assertArrayAlmostEqual(result, data)

    def test_no_transpose(self):
        """
        Test that the utility returns the expected data values
        when the cube is not transposed after slicing.
        """
        data = self.data.T

        result = convert_cube_data_to_2d(self.cube, transpose=False)
        self.assertArrayAlmostEqual(result, data)

    def test_3d_cube(self):
        """
        Test that the utility returns the expected data values
        when a 3d cube is input.
        """
        cube = set_up_temperature_cube()
        cube = cube[0]
        data = np.array([[226.15, 237.4, 248.65, 259.9, 271.15,
                          282.4, 293.65, 304.9, 316.15]]).T

        result = convert_cube_data_to_2d(cube)
        self.assertArrayAlmostEqual(result, data, decimal=5)

    def test_2d_cube(self):
        """
        Test that the utility returns the expected data values
        when a 2d cube is input.
        """
        cube = set_up_temperature_cube()
        cube = cube[0, 0, :, :]
        data = np.array([[226.15, 237.4, 248.65, 259.9, 271.15,
                          282.4, 293.65, 304.9, 316.15]]).T

        result = convert_cube_data_to_2d(cube)
        self.assertArrayAlmostEqual(result, data, decimal=5)

    def test_1d_cube(self):
        """
        Test that the utility returns the expected data values
        when a 1d cube is input.
        """
        cube = set_up_temperature_cube()
        cube = cube[0, 0, 0, :]
        data = np.array([[226.15, 237.4, 248.65]]).T

        result = convert_cube_data_to_2d(cube)
        self.assertArrayAlmostEqual(result, data, decimal=5)

    def test_5d_cube(self):
        """
        Test that the utility returns the expected data values
        when a 5d cube is input.
        """
        cube1 = set_up_temperature_cube()
        height_coord = iris.coords.AuxCoord([5], standard_name="height")
        cube1.add_aux_coord(height_coord)

        cube2 = set_up_temperature_cube()
        height_coord = iris.coords.AuxCoord([10], standard_name="height")
        cube2.add_aux_coord(height_coord)

        cubes = iris.cube.CubeList([cube1, cube2])
        cube = cubes.merge_cube()

        data = np.array([[226.15, 230.15, 232.15],
                         [237.4, 241.4, 243.4],
                         [248.65, 252.65, 254.65],
                         [259.9, 263.9, 265.9],
                         [271.15, 275.15, 277.15],
                         [282.4, 286.4, 288.4],
                         [293.65, 297.65, 299.65],
                         [304.9, 308.9, 310.9],
                         [316.15, 320.15, 322.15],
                         [226.15, 230.15, 232.15],
                         [237.4, 241.4, 243.4],
                         [248.65, 252.65, 254.65],
                         [259.9, 263.9, 265.9],
                         [271.15, 275.15, 277.15],
                         [282.4, 286.4, 288.4],
                         [293.65, 297.65, 299.65],
                         [304.9, 308.9, 310.9],
                         [316.15, 320.15, 322.15]])

        result = convert_cube_data_to_2d(cube)
        self.assertArrayAlmostEqual(result, data, decimal=5)


class Test_flatten_ignoring_masked_data(IrisTest):

    """Test the flatten_ignoring_masked_data utility."""
    def setUp(self):
        """Set up a basic 3D data array to use in the tests."""
        self.data_array = np.array([[[0., 1., 2., 3.],
                                     [4., 5., 6., 7.]],
                                    [[8., 9., 10., 11.],
                                     [12., 13., 14., 15.]],
                                    [[16., 17., 18., 19.],
                                     [20., 21., 22., 23.]]], dtype=np.float32)
        self.mask = np.array([[[True, False, True, True],
                               [True, False, True, True]],
                              [[True, False, True, True],
                               [True, False, True, True]],
                              [[True, False, True, True],
                               [True, False, True, True]]])
        self.expected_result_preserve_leading_dim = np.array(
            [[0., 1., 2., 3., 4., 5., 6., 7.],
             [8., 9., 10., 11., 12., 13., 14., 15.],
             [16., 17., 18., 19., 20., 21., 22., 23.]],
            dtype=np.float32)

    def test_basic_not_masked(self):
        """Test a basic unmasked array"""
        expected_result = np.arange(0, 24, 1, dtype=np.float32)
        result = flatten_ignoring_masked_data(self.data_array)
        self.assertArrayAlmostEqual(result, expected_result)
        self.assertEqual(result.dtype, np.float32)

    def test_basic_masked(self):
        """Test a basic masked array"""
        masked_data_array = np.ma.MaskedArray(self.data_array, self.mask)
        expected_result = np.array([1., 5., 9., 13., 17., 21.],
                                   dtype=np.float32)
        result = flatten_ignoring_masked_data(masked_data_array)
        self.assertArrayAlmostEqual(result, expected_result)
        self.assertEqual(result.dtype, np.float32)

    def test_basic_not_masked_preserver_leading_dim(self):
        """Test a basic unmasked array, with preserve_leading_dimension"""
        result = flatten_ignoring_masked_data(
            self.data_array, preserve_leading_dimension=True)
        self.assertArrayAlmostEqual(
            result, self.expected_result_preserve_leading_dim)
        self.assertEqual(result.dtype, np.float32)

    def test_basic_masked_preserver_leading_dim(self):
        """Test a basic masked array, with preserve_leading_dimension"""

        masked_data_array = np.ma.MaskedArray(self.data_array, self.mask)
        expected_result = np.array([[1., 5.],
                                    [9., 13.],
                                    [17., 21.]],
                                   dtype=np.float32)
        result = flatten_ignoring_masked_data(
            masked_data_array, preserve_leading_dimension=True)
        self.assertArrayAlmostEqual(result, expected_result)
        self.assertEqual(result.dtype, np.float32)

    def test_all_masked(self):
        """Test empty array is returned when all points are masked."""
        mask = np.ones((3, 2, 4)) * True
        masked_data_array = np.ma.MaskedArray(self.data_array, mask)
        expected_result = np.array([], dtype=np.float32)
        result = flatten_ignoring_masked_data(masked_data_array)
        self.assertArrayAlmostEqual(result, expected_result)
        self.assertEqual(result.dtype, np.float32)

    def test_1D_input(self):
        """Test input array is unchanged when input in 1D"""
        data_array = self.data_array.flatten()
        expected_result = data_array.copy()
        result = flatten_ignoring_masked_data(data_array)
        self.assertArrayAlmostEqual(result, expected_result)
        self.assertEqual(result.dtype, np.float32)

    def test_4D_input_not_masked_preserve_leading_dim(self):
        """Test input array is unchanged when input in 4D.
           This should give the same answer as the corresponding 3D array."""
        data_array = self.data_array.reshape((3, 2, 2, 2))
        result = flatten_ignoring_masked_data(
            data_array, preserve_leading_dimension=True)
        self.assertArrayAlmostEqual(
            result, self.expected_result_preserve_leading_dim)
        self.assertEqual(result.dtype, np.float32)

    def test_inconsistent_mask_along_leading_dim(self):
        """Test an inconsistently masked array raises an error."""
        mask = np.array([[[True, False, False, True],
                          [True, False, True, True]],
                         [[True, False, True, True],
                          [True, False, True, True]],
                         [[True, False, True, True],
                          [True, False, True, False]]])
        masked_data_array = np.ma.MaskedArray(self.data_array, mask)
        expected_message = "The mask on the input array is not the same"
        with self.assertRaisesRegex(ValueError, expected_message):
            flatten_ignoring_masked_data(
                masked_data_array, preserve_leading_dimension=True)


class Test_check_predictor(IrisTest):

    """
    Test to check the predictor.
    """

    @staticmethod
    def test_mean():
        """
        Test that the utility does not raise an exception when
        predictor = "mean".
        """
        check_predictor("mean")

    @staticmethod
    def test_realizations():
        """
        Test that the utility does not raise an exception when
        predictor = "realizations".
        """
        check_predictor("realizations")

    def test_invalid_predictor(self):
        """
        Test that the utility raises an exception when
        predictor = "foo", a name not present in the list of
        accepted values for the predictor.
        """
        msg = "The requested value for the predictor"
        with self.assertRaisesRegex(ValueError, msg):
            check_predictor("foo")


class Test__filter_non_matching_cubes(SetupCubes):
    """Test the _filter_non_matching_cubes method."""

    def setUp(self):
        super().setUp()
        # Create historical forecasts and truth cubes where some items
        # are missing.
        self.partial_historic_forecasts = (
            self.historic_forecasts[:2] +
            self.historic_forecasts[3:]).merge_cube()
        self.partial_truth = (self.truth[:2] + self.truth[3:]).merge_cube()

    def test_all_matching(self):
        """Test for when the historic forecast and truth cubes all match."""
        hf_result, truth_result = filter_non_matching_cubes(
            self.historic_temperature_forecast_cube,
            self.temperature_truth_cube)
        self.assertEqual(hf_result, self.historic_temperature_forecast_cube)
        self.assertEqual(truth_result, self.temperature_truth_cube)

    def test_bounded_variables(self):
        """Test for when the historic forecast and truth cubes all match
        inclusive of both the points and bounds on the time coordinate."""
        # Define bounds so that the lower bound is one hour preceding the point
        # whilst the upper bound is equal to the point.
        points = self.historic_temperature_forecast_cube.coord("time").points
        bounds = []
        for point in points:
            bounds.append([point - 1*60*60, point])

        self.historic_temperature_forecast_cube.coord("time").bounds = bounds
        self.temperature_truth_cube.coord("time").bounds = bounds

        hf_result, truth_result = filter_non_matching_cubes(
            self.historic_temperature_forecast_cube,
            self.temperature_truth_cube)
        self.assertEqual(hf_result, self.historic_temperature_forecast_cube)
        self.assertEqual(truth_result, self.temperature_truth_cube)

    def test_fewer_historic_forecasts(self):
        """Test for when there are fewer historic forecasts than truths,
        for example, if there is a missing forecast cycle."""
        hf_result, truth_result = filter_non_matching_cubes(
            self.partial_historic_forecasts, self.temperature_truth_cube)
        self.assertEqual(hf_result, self.partial_historic_forecasts)
        self.assertEqual(truth_result, self.partial_truth)

    def test_fewer_truths(self):
        """Test for when there are fewer truths than historic forecasts,
        for example, if there is a missing analysis."""
        hf_result, truth_result = filter_non_matching_cubes(
            self.historic_temperature_forecast_cube, self.partial_truth)
        self.assertEqual(hf_result, self.partial_historic_forecasts)
        self.assertEqual(truth_result, self.partial_truth)

    def test_mismatching(self):
        """Test for when there is both a missing historic forecasts and a
        missing truth at different validity times. This results in the
        expected historic forecasts and the expected truths containing cubes
        at three matching validity times."""
        partial_truth = self.truth[1:].merge_cube()
        expected_historical_forecasts = iris.cube.CubeList(
            [self.historic_forecasts[index]
             for index in (1, 3, 4)]).merge_cube()
        expected_truth = iris.cube.CubeList(
            [self.truth[index] for index in (1, 3, 4)]).merge_cube()
        hf_result, truth_result = filter_non_matching_cubes(
            self.partial_historic_forecasts, partial_truth)
        self.assertEqual(hf_result, expected_historical_forecasts)
        self.assertEqual(truth_result, expected_truth)

    def test_no_matches_exception(self):
        """Test for when no matches in validity time are found between the
        historic forecasts and the truths. In this case, an exception is
        raised."""
        partial_truth = self.truth[2]
        msg = "The filtering has found no matches in validity time "
        with self.assertRaisesRegex(ValueError, msg):
            filter_non_matching_cubes(
                self.partial_historic_forecasts, partial_truth)


if __name__ == '__main__':
    unittest.main()
