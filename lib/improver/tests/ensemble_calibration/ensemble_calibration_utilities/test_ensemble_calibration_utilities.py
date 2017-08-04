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
Unit tests for the utilities within the `ensemble_calibration_utilities`
module.

"""
import unittest

import iris
from iris.cube import Cube
from iris.tests import IrisTest
import numpy as np

from improver.ensemble_calibration.ensemble_calibration_utilities import (
    convert_cube_data_to_2d, ensure_dimension_is_the_zeroth_dimension,
    rename_coordinate, _renamer, check_predictor_of_mean_flag)
from improver.tests.ensemble_calibration.ensemble_calibration.\
    helper_functions import set_up_temperature_cube


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
                              [316.15, 320.15, 322.15]])

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
        self.assertArrayAlmostEqual(result, data)

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
        self.assertArrayAlmostEqual(result, data)

    def test_1d_cube(self):
        """
        Test that the utility returns the expected data values
        when a 1d cube is input.
        """
        cube = set_up_temperature_cube()
        cube = cube[0, 0, 0, :]
        data = np.array([[226.15, 237.4, 248.65]]).T

        result = convert_cube_data_to_2d(cube)
        self.assertArrayAlmostEqual(result, data)

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
        self.assertArrayAlmostEqual(result, data)


class Test_ensure_dimension_is_the_zeroth_dimension(IrisTest):

    """
    Test the ensure_dimension_is_the_zeroth_dimension
    utility.
    """

    def setUp(self):
        """Use temperature cube to test with."""
        self.cube = set_up_temperature_cube()

    def test_basic(self):
        """Test that the function returns an iris.cube.Cube."""
        result = (
            ensure_dimension_is_the_zeroth_dimension(self.cube, "realization"))
        self.assertIsInstance(result, Cube)

    def test_if_probabilistic_dimension_is_first(self):
        """
        Test that a cube with the expected data contents is returned when
        the probabilistic dimension is the first dimension coordinate.
        """
        result = (
            ensure_dimension_is_the_zeroth_dimension(self.cube, "realization"))
        self.assertArrayAlmostEqual(result.data, self.cube.data)

    def test_if_probabilistic_dimension_is_not_first(self):
        """
        Test that a cube with the expected data contents is returned when
        the probabilistic dimension is a dimension coordinate but it is
        not the first dimension coordinate,.
        """
        expected = self.cube.copy()
        expected.transpose([0, 3, 2, 1])

        cube = self.cube.copy()
        cube.transpose([3, 2, 1, 0])
        result = (
            ensure_dimension_is_the_zeroth_dimension(cube, "realization"))
        self.assertArrayAlmostEqual(result.data, expected.data)

    def test_if_probabilistic_dimension_is_scalar(self):
        """
        Test that a cube with the expected data contents is returned when
        the probabilistic dimension is a scalar coordinate.
        """
        cube = self.cube[0, :, :, :]
        result = (
            ensure_dimension_is_the_zeroth_dimension(cube, "realization"))
        self.assertArrayAlmostEqual(result.data, [cube.data])

    def test_if_probabilistic_dimension_not_available(self):
        """
        Test that the expected error message is raised when the required
        probabilistic dimension is not available in the cube.
        """
        cube = self.cube[0, :, :, :]
        cube.remove_coord("realization")
        msg = "not a dimension coordinate"
        with self.assertRaisesRegexp(ValueError, msg):
            ensure_dimension_is_the_zeroth_dimension(cube, "realization")


class Test_rename_coordinate(IrisTest):

    """Test the rename_coordinate utility."""

    def setUp(self):
        """Use temperature cube to test with."""
        self.cube = set_up_temperature_cube()

    def test_basic_cube(self):
        """Test that the utility returns an iris.cube.Cube."""
        rename_coordinate(
            self.cube, "realization", "ensemble_member_id")
        self.assertIsInstance(self.cube, iris.cube.Cube)

    def test_basic_cubelist(self):
        """
        Test that the utility returns an iris.cube.CubeList and that
        the cubes in the cubelist have an ensemble_member_id coordinate.
        """
        cube1 = self.cube.copy()
        cube2 = self.cube.copy()
        cubes = iris.cube.CubeList([cube1, cube2])
        rename_coordinate(
            cubes, "realization", "ensemble_member_id")
        self.assertIsInstance(cubes, iris.cube.CubeList)
        for cube in cubes:
            self.assertTrue(cube.coord("ensemble_member_id"))

    def test_check_coordinate_name(self):
        """
        Test that the utility returns an iris.cube.Cube with an
        ensemble_member_id coordinate.
        """
        rename_coordinate(
            self.cube, "realization", "ensemble_member_id")
        self.assertTrue(self.cube.coord("ensemble_member_id"))

    def test_check_type_error(self):
        """
        Test that a TyoeError is raised, if the input variable is not an
        iris.cube.Cube.
        """
        fake_cube = "fake"
        msg = "A Cube or CubeList is not provided for renaming"
        with self.assertRaisesRegexp(TypeError, msg):
            rename_coordinate(
                fake_cube, "realization", "ensemble_member_id")


class Test__renamer(IrisTest):

    """Test the _renamer utility."""

    def setUp(self):
        """Use temperature cube to test with."""
        self.cube = set_up_temperature_cube()

    def test_check_coordinate_name(self):
        """
        Test that the utility returns an iris.cube.Cube with an
        ensemble_member_id coordinate following renaming.
        """
        rename_coordinate(
            self.cube, "realization", "ensemble_member_id")
        self.assertTrue(self.cube.coord("ensemble_member_id"))

    def test_absent_original_coord(self):
        """
        Test that the utility returns an iris.cube.Cube after the renaming
        was not successful, as the original coordinate was not found in
        the cube.
        """
        rename_coordinate(
            self.cube, "fake", "ensemble_member_id")
        self.assertFalse(self.cube.coords("ensemble_member_id"))


class Test_check_predictor_of_mean_flag(IrisTest):

    """
    Test to check the predictor_of_mean_flag.
    """

    def test_mean(self):
        """
        Test that the utility does not fail when the predictor_of_mean_flag
        is "mean".
        """
        predictor_of_mean_flag = "mean"

        try:
            check_predictor_of_mean_flag(predictor_of_mean_flag)
        except ValueError as err:
            msg = ("_check_predictor_of_mean_flag raised "
                   "ValueError unexpectedly."
                   "Message is {}").format(err)
            self.fail(msg)

    def test_members(self):
        """
        Test that the utility does not fail when the predictor_of_mean_flag
        is "members".
        """
        predictor_of_mean_flag = "members"

        try:
            check_predictor_of_mean_flag(predictor_of_mean_flag)
        except ValueError as err:
            msg = ("_check_predictor_of_mean_flag raised "
                   "ValueError unexpectedly."
                   "Message is {}").format(err)
            self.fail(msg)

    def test_foo(self):
        """
        Test that the utility fails when the predictor_of_mean_flag
        is "foo" i.e. a name not present in the list of accepted values
        for the predictor_of_mean_flag.
        """
        predictor_of_mean_flag = "foo"

        msg = "The requested value for the predictor_of_mean_flag"
        with self.assertRaisesRegexp(ValueError, msg):
            check_predictor_of_mean_flag(predictor_of_mean_flag)


if __name__ == '__main__':
    unittest.main()
