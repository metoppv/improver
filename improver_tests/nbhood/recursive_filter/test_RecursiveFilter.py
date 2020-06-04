# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown Copyright 2017-2020 Met Office.
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
"""Unit tests for the nbhood.RecursiveFilter plugin."""

import unittest

import iris
import numpy as np
from iris.cube import Cube
from iris.tests import IrisTest

from improver.nbhood.recursive_filter import RecursiveFilter
from improver.utilities.cube_manipulation import enforce_coordinate_ordering
from improver.utilities.pad_spatial import pad_cube_with_halo
from improver.utilities.warnings_handler import ManageWarnings

from ...set_up_test_cubes import set_up_variable_cube
from ..nbhood.test_BaseNeighbourhoodProcessing import set_up_cube


class Test__repr__(IrisTest):

    """Test the repr method."""

    def test_basic(self):
        """Test that the __repr__ returns the expected string."""
        iterations = None
        edge_width = 1
        result = str(RecursiveFilter(iterations, edge_width))
        msg = "<RecursiveFilter: iterations: {}, edge_width: {}".format(
            iterations, edge_width
        )
        self.assertEqual(result, msg)


class Test_RecursiveFilter(IrisTest):

    """Test class for the RecursiveFilter tests, setting up cubes."""

    def setUp(self):
        """Create test cubes."""

        self.iterations = 1

        # Generate data cube with dimensions 1 x 5 x 5
        data = np.array(
            [
                [
                    [0.00, 0.00, 0.10, 0.00, 0.00],
                    [0.00, 0.00, 0.25, 0.00, 0.00],
                    [0.10, 0.25, 0.50, 0.25, 0.10],
                    [0.00, 0.00, 0.25, 0.00, 0.00],
                    [0.00, 0.00, 0.10, 0.00, 0.00],
                ]
            ],
            dtype=np.float32,
        )
        self.cube = set_up_variable_cube(
            data, name="precipitation_amount", units="kg m^-2 s^-1"
        )

        mean_x_points = np.array([-15.0, -5.0, 5.0, 15.0], dtype=np.float32)
        mean_y_points = np.array([45.0, 55.0, 65.0, 75.0], dtype=np.float32)

        # Generate x smoothing_coefficients_cube with correct dimensions 5 x 4
        self.smoothing_coefficients_cube_x = set_up_variable_cube(
            np.full((5, 4), 0.5, dtype=np.float32), name="smoothing_coefficient_x"
        )
        self.smoothing_coefficients_cube_x.coord(axis="x").points = mean_x_points

        # Generate y smoothing_coefficients_cube with correct dimensions 5 x 4
        self.smoothing_coefficients_cube_y = set_up_variable_cube(
            np.full((4, 5), 0.5, dtype=np.float32), name="smoothing_coefficient_y"
        )
        self.smoothing_coefficients_cube_y.coord(axis="y").points = mean_y_points

        # Generate an alternative y smoothing_coefficients_cube with correct dimensions 5 x 4
        self.smoothing_coefficients_cube_y_half = (
            self.smoothing_coefficients_cube_y * 0.5
        )
        self.smoothing_coefficients_cube_y_half.rename("smoothing_coefficient_y")

        # Generate smoothing_coefficients_cube with incorrect dimensions 6 x 6
        self.smoothing_coefficients_cube_wrong_name = set_up_variable_cube(
            np.full((5, 4), 0.5, dtype=np.float32), name="air_temperature"
        )
        self.smoothing_coefficients_cube_wrong_name.coord(
            axis="x"
        ).points = mean_x_points

        # Generate x smoothing_coefficients_cube with incorrect dimensions 6 x 6
        self.smoothing_coefficients_cube_wrong_x = set_up_variable_cube(
            np.full((6, 6), 0.5, dtype=np.float32), name="smoothing_coefficient_x"
        )

        # Generate y smoothing_coefficients_cube with incorrect dimensions 6 x 6
        self.smoothing_coefficients_cube_wrong_y = set_up_variable_cube(
            np.full((6, 6), 0.5, dtype=np.float32), name="smoothing_coefficient_y"
        )

        # Generate smoothing_coefficients_cube with correct dimensions 5 x 4
        self.smoothing_coefficients_cube_wrong_x_points = (
            self.smoothing_coefficients_cube_x.copy()
        )
        self.smoothing_coefficients_cube_wrong_x_points.coord(axis="x").points = (
            self.smoothing_coefficients_cube_wrong_x_points.coord(axis="x").points + 10
        )

        # Generate smoothing_coefficients_cube with correct dimensions 4 x 5
        self.smoothing_coefficients_cube_wrong_y_points = (
            self.smoothing_coefficients_cube_y.copy()
        )
        self.smoothing_coefficients_cube_wrong_y_points.coord(axis="y").points = (
            self.smoothing_coefficients_cube_wrong_y_points.coord(axis="y").points + 10
        )


class Test__init__(Test_RecursiveFilter):

    """Test plugin initialisation."""

    def test_basic(self):
        """Test using the default arguments."""
        result = RecursiveFilter()
        self.assertIsNone(result.iterations)
        self.assertEqual(result.edge_width, 15)
        self.assertFalse(result.re_mask)

    def test_iterations(self):
        """Test when iterations value less than unity is given (invalid)."""
        iterations = 0
        msg = "Invalid number of iterations: must be >= 1: 0"
        with self.assertRaisesRegex(ValueError, msg):
            RecursiveFilter(
                iterations=iterations, edge_width=1,
            )

    @ManageWarnings(record=True)
    def test_iterations_warn(self, warning_list=None):
        """Test when the iteration value is more than 3 it warns."""
        iterations = 5
        warning_msg = (
            "More than two iterations degrades the conservation"
            "of probability assumption."
        )

        RecursiveFilter(iterations=iterations)
        self.assertTrue(any(item.category == UserWarning for item in warning_list))
        self.assertTrue(any(warning_msg in str(item) for item in warning_list))


class Test_set_up_cubes(IrisTest):

    """Test the set up of cubes prior to neighbourhooding."""

    def setUp(self):
        """Set up a cube."""
        self.cube = set_up_cube(zero_point_indices=((0, 0, 2, 2),), num_grid_points=5)
        self.cube = iris.util.squeeze(self.cube)

    def test_without_masked_data(self):
        """Test setting up cubes to be neighbourhooded when the input cube
        does not contain masked arrays."""
        expected_mask = np.ones((5, 5), dtype=np.bool)
        expected_nans = np.zeros((5, 5), dtype=np.bool)
        cube, mask, nan_array = RecursiveFilter.set_up_cubes(self.cube)
        self.assertIsInstance(cube, Cube)
        self.assertIsInstance(mask, Cube)
        self.assertEqual(cube, self.cube)
        self.assertArrayEqual(nan_array, expected_nans)
        self.assertArrayEqual(mask.data, expected_mask)

    def test_with_masked_data(self):
        """Test setting up cubes to be neighbourhooded when the input cube
        contains masked arrays."""
        cube = self.cube
        data = cube.data
        cube.data[1, 3] = 0.5
        cube.data[3, 3] = 0.5
        cube.data = np.ma.masked_equal(data, 0.5)
        mask = ~cube.data.mask
        expected_nans = np.zeros((5, 5), dtype=np.bool)
        data = cube.data.data * mask
        result_cube, result_mask, result_nan_array = RecursiveFilter.set_up_cubes(
            cube.copy()
        )
        self.assertArrayAlmostEqual(result_cube.data, data)
        self.assertArrayAlmostEqual(result_mask.data, mask)
        self.assertArrayEqual(result_nan_array, expected_nans)

    def test_with_separate_mask_cube(self):
        """Test for an input cube and an additional mask cube."""
        self.cube.data[1, 3] = 0.5
        self.cube.data[3, 3] = 0.5
        mask_cube = self.cube.copy()
        mask_cube.data = np.ones((5, 5))
        mask_cube.data[self.cube.data == 0.5] = 0
        expected_data = self.cube.data * mask_cube.data
        expected_mask = np.ones((5, 5))
        expected_mask[1, 3] = 0.0
        expected_mask[3, 3] = 0.0
        expected_nans = np.zeros((5, 5), dtype=np.bool)
        result_cube, result_mask, result_nan_array = RecursiveFilter.set_up_cubes(
            self.cube.copy(), mask_cube=mask_cube
        )
        self.assertIsInstance(result_cube, Cube)
        self.assertIsInstance(result_mask, Cube)
        self.assertArrayAlmostEqual(result_cube.data, expected_data)
        self.assertArrayAlmostEqual(result_mask.data, expected_mask)
        self.assertArrayEqual(result_nan_array, expected_nans)

    def test_with_separate_mask_cube_and_nan(self):
        """Test for an input cube and an additional mask cube."""
        mask_cube = self.cube.copy()
        self.cube.data[1, 3] = 0.5
        self.cube.data[3, 3] = 0.5
        self.cube.data[1, 2] = np.nan
        self.cube.data[3, 1] = np.nan
        mask_cube.data = np.ones((5, 5))
        mask_cube.data[self.cube.data == 0.5] = 0
        mask_cube.data = mask_cube.data.astype(int)

        expected_mask = np.ones((5, 5))
        expected_mask[1, 3] = 0.0
        expected_mask[3, 3] = 0.0
        expected_mask[1, 2] = 0.0
        expected_mask[3, 1] = 0.0
        expected_data = self.cube.data * expected_mask
        expected_data[1, 2] = 0.0
        expected_data[3, 1] = 0.0
        expected_nans = np.zeros((5, 5), dtype=np.bool)
        expected_nans[1, 2] = True
        expected_nans[3, 1] = True

        result_cube, result_mask, result_nan_array = RecursiveFilter.set_up_cubes(
            self.cube.copy(), mask_cube=mask_cube
        )

        self.assertIsInstance(result_cube, Cube)
        self.assertIsInstance(result_mask, Cube)
        self.assertArrayAlmostEqual(result_cube.data, expected_data)
        self.assertArrayAlmostEqual(result_mask.data, expected_mask)
        self.assertArrayEqual(result_nan_array, expected_nans)


class Test__set_smoothing_coefficients(Test_RecursiveFilter):

    """Test the _set_smoothing_coefficients function"""

    def test_smoothing_coefficients_cube(self):
        """Test that the returned smoothing_coefficients array has the expected
        result when smoothing_coefficients_cube is not None."""
        result = RecursiveFilter(edge_width=1)._set_smoothing_coefficients(
            self.smoothing_coefficients_cube_x
        )
        expected_result = 0.5
        self.assertIsInstance(result.data, np.ndarray)
        self.assertEqual(result.data[0][2], expected_result)
        # Check shape: Array should be padded with 4 extra rows/columns
        expected_shape = (9, 8)
        self.assertEqual(result.shape, expected_shape)


class Test__validate_smoothing_coefficients(Test_RecursiveFilter):
    def test_smoothing_coefficients_cube(self):
        """Test that correctly shaped smoothing_coefficients validate."""
        RecursiveFilter(edge_width=1)._validate_smoothing_coefficients(
            self.cube[0, :], self.smoothing_coefficients_cube_x
        )

    def test_smoothing_coefficients_wrong_name(self):
        """Test that an error is raised if the smoothing_coefficients_cube has
        an incorrect name"""
        msg = "The smoothing coefficients cube must be named either "
        with self.assertRaisesRegex(ValueError, msg):
            RecursiveFilter(edge_width=1)._validate_smoothing_coefficients(
                self.cube, self.smoothing_coefficients_cube_wrong_name
            )

    def test_smoothing_coefficients_mismatched_x_dimension(self):
        """Test that an error is raised if the x smoothing_coefficients_cube is
        of an incorrect shape compared to the data cube."""
        msg = "The x spatial dimension of the smoothing coefficients "
        with self.assertRaisesRegex(ValueError, msg):
            RecursiveFilter(edge_width=1)._validate_smoothing_coefficients(
                self.cube, self.smoothing_coefficients_cube_wrong_x
            )

    def test_smoothing_coefficients_mismatched_y_dimension(self):
        """Test that an error is raised if the y smoothing_coefficients_cube is
        of an incorrect shape compared to the data cube."""
        msg = "The y spatial dimension of the smoothing coefficients "
        with self.assertRaisesRegex(ValueError, msg):
            RecursiveFilter(edge_width=1)._validate_smoothing_coefficients(
                self.cube, self.smoothing_coefficients_cube_wrong_y
            )

    def test_smoothing_coefficients_mismatched_x_points(self):
        """Test that an error is raised if the x smoothing_coefficients_cube
        has mismatched coordinate points compared to the data cube."""
        msg = "The points of the x spatial dimension of the " "smoothing coefficients"
        with self.assertRaisesRegex(ValueError, msg):
            RecursiveFilter(edge_width=1)._validate_smoothing_coefficients(
                self.cube, self.smoothing_coefficients_cube_wrong_x_points
            )

    def test_smoothing_coefficients_mismatched_y_points(self):
        """Test that an error is raised if the y smoothing_coefficients_cube
        has mismatched coordinate points compared to the data cube."""
        msg = "The points of the y spatial dimension of the " "smoothing coefficients"
        with self.assertRaisesRegex(ValueError, msg):
            RecursiveFilter(edge_width=1)._validate_smoothing_coefficients(
                self.cube, self.smoothing_coefficients_cube_wrong_y_points
            )


class Test__recurse_forward(Test_RecursiveFilter):

    """Test the _recurse_forward method"""

    def test_first_axis(self):
        """Test that the returned _recurse_forward array has the expected
           type and result."""
        expected_result = np.array(
            [
                [0.0000, 0.00000, 0.100000, 0.00000, 0.0000],
                [0.0000, 0.00000, 0.175000, 0.00000, 0.0000],
                [0.0500, 0.12500, 0.337500, 0.12500, 0.0500],
                [0.0250, 0.06250, 0.293750, 0.06250, 0.0250],
                [0.0125, 0.03125, 0.196875, 0.03125, 0.0125],
            ]
        )
        result = RecursiveFilter(edge_width=1)._recurse_forward(
            self.cube.data[0, :], self.smoothing_coefficients_cube_y.data, 0
        )
        self.assertIsInstance(result, np.ndarray)
        self.assertArrayAlmostEqual(result, expected_result)

    def test_second_axis(self):
        """Test that the returned _recurse_forward array has the expected
           type and result."""
        expected_result = np.array(
            [
                [0.0, 0.000, 0.0500, 0.02500, 0.012500],
                [0.0, 0.000, 0.1250, 0.06250, 0.031250],
                [0.1, 0.175, 0.3375, 0.29375, 0.196875],
                [0.0, 0.000, 0.1250, 0.06250, 0.031250],
                [0.0, 0.000, 0.0500, 0.02500, 0.012500],
            ]
        )
        result = RecursiveFilter(edge_width=1)._recurse_forward(
            self.cube.data[0, :], self.smoothing_coefficients_cube_x.data, 1
        )
        self.assertIsInstance(result, np.ndarray)
        self.assertArrayAlmostEqual(result, expected_result)


class Test__recurse_backward(Test_RecursiveFilter):

    """Test the _recurse_backward method"""

    def test_first_axis(self):
        """Test that the returned _recurse_backward array has the expected
           type and result."""
        expected_result = np.array(
            [
                [0.0125, 0.03125, 0.196875, 0.03125, 0.0125],
                [0.0250, 0.06250, 0.293750, 0.06250, 0.0250],
                [0.0500, 0.12500, 0.337500, 0.12500, 0.0500],
                [0.0000, 0.00000, 0.175000, 0.00000, 0.0000],
                [0.0000, 0.00000, 0.100000, 0.00000, 0.0000],
            ]
        )
        result = RecursiveFilter(edge_width=1)._recurse_backward(
            self.cube.data[0, :], self.smoothing_coefficients_cube_y.data, 0
        )
        self.assertIsInstance(result, np.ndarray)
        self.assertArrayAlmostEqual(result, expected_result)

    def test_second_axis(self):
        """Test that the returned _recurse_backward array has the expected
           type and result."""
        expected_result = np.array(
            [
                [0.012500, 0.02500, 0.0500, 0.000, 0.0],
                [0.031250, 0.06250, 0.1250, 0.000, 0.0],
                [0.196875, 0.29375, 0.3375, 0.175, 0.1],
                [0.031250, 0.06250, 0.1250, 0.000, 0.0],
                [0.012500, 0.02500, 0.0500, 0.000, 0.0],
            ]
        )
        result = RecursiveFilter(edge_width=1)._recurse_backward(
            self.cube.data[0, :], self.smoothing_coefficients_cube_x.data, 1
        )
        self.assertIsInstance(result, np.ndarray)
        self.assertArrayAlmostEqual(result, expected_result)


class Test__run_recursion(Test_RecursiveFilter):

    """Test the _run_recursion method"""

    def test_return_type(self):
        """Test that the _run_recursion method returns an iris.cube.Cube."""
        edge_width = 1
        cube = iris.util.squeeze(self.cube)
        smoothing_coefficients_x = RecursiveFilter(
            edge_width=1
        )._set_smoothing_coefficients(self.smoothing_coefficients_cube_x)
        smoothing_coefficients_y = RecursiveFilter(
            edge_width=1
        )._set_smoothing_coefficients(self.smoothing_coefficients_cube_y)
        padded_cube = pad_cube_with_halo(cube, 2 * edge_width, 2 * edge_width)
        result = RecursiveFilter(edge_width=1)._run_recursion(
            padded_cube,
            smoothing_coefficients_x,
            smoothing_coefficients_y,
            self.iterations,
        )
        self.assertIsInstance(result, Cube)

    def test_result_basic(self):
        """Test that the _run_recursion method returns the expected value."""
        edge_width = 1
        cube = iris.util.squeeze(self.cube)
        smoothing_coefficients_x = RecursiveFilter(
            edge_width=edge_width
        )._set_smoothing_coefficients(self.smoothing_coefficients_cube_x)
        smoothing_coefficients_y = RecursiveFilter(
            edge_width=edge_width
        )._set_smoothing_coefficients(self.smoothing_coefficients_cube_y)
        padded_cube = pad_cube_with_halo(cube, 2 * edge_width, 2 * edge_width)
        result = RecursiveFilter(edge_width=edge_width)._run_recursion(
            padded_cube,
            smoothing_coefficients_x,
            smoothing_coefficients_y,
            self.iterations,
        )
        expected_result = 0.12302627
        self.assertAlmostEqual(result.data[4][4], expected_result)

    def test_different_smoothing_coefficients(self):
        """Test that the _run_recursion method returns expected values when
        smoothing_coefficient values are different in the x and y directions"""
        edge_width = 1
        cube = iris.util.squeeze(self.cube)
        smoothing_coefficients_x = RecursiveFilter(
            edge_width=edge_width
        )._set_smoothing_coefficients(self.smoothing_coefficients_cube_x)
        smoothing_coefficients_y = RecursiveFilter(
            edge_width=edge_width
        )._set_smoothing_coefficients(self.smoothing_coefficients_cube_y_half)
        padded_cube = pad_cube_with_halo(cube, 2 * edge_width, 2 * edge_width)
        result = RecursiveFilter(edge_width=edge_width)._run_recursion(
            padded_cube, smoothing_coefficients_x, smoothing_coefficients_y, 1
        )
        # slice back down to the source grid - easier to visualise!
        unpadded_result = result.data[2:-2, 2:-2]

        expected_result = np.array(
            [
                [0.01320939, 0.02454378, 0.04346254, 0.02469828, 0.01359563],
                [0.03405095, 0.06060188, 0.09870366, 0.06100013, 0.03504659],
                [0.0845406, 0.13908109, 0.18816182, 0.14006987, 0.08701254],
                [0.03405397, 0.06060749, 0.09871361, 0.06100579, 0.03504971],
                [0.01322224, 0.02456765, 0.04350482, 0.0247223, 0.01360886],
            ],
            dtype=np.float32,
        )
        self.assertArrayAlmostEqual(unpadded_result, expected_result)


class Test_process(Test_RecursiveFilter):

    """Test the process method. """

    # Test output from plugin returns expected values
    def test_return_type_and_shape(self):
        """Test that the RecursiveFilter plugin returns an iris.cube.Cube of
        the expected shape."""
        # Output data array should have same dimensions as input data array
        expected_shape = (1, 5, 5)
        plugin = RecursiveFilter(iterations=self.iterations,)
        result = plugin(
            self.cube,
            smoothing_coefficients_x=self.smoothing_coefficients_cube_x,
            smoothing_coefficients_y=self.smoothing_coefficients_cube_y,
        )
        self.assertIsInstance(result, Cube)
        self.assertEqual(result.shape, expected_shape)
        self.assertEqual(result.shape, expected_shape)

    def test_smoothing_coefficient_cubes(self):
        """Test that the RecursiveFilter plugin returns the correct data."""
        plugin = RecursiveFilter(iterations=self.iterations,)
        result = plugin(
            self.cube,
            smoothing_coefficients_x=self.smoothing_coefficients_cube_x,
            smoothing_coefficients_y=self.smoothing_coefficients_cube_y,
        )
        expected = 0.14994797
        self.assertAlmostEqual(result.data[0][2][2], expected)

    def test_smoothing_coefficient_nan_in_data(self):
        """Test that the RecursiveFilter plugin returns the correct data
        when the data contains nans."""
        plugin = RecursiveFilter(iterations=self.iterations,)
        self.cube.data[0][3][2] = np.nan
        result = plugin(
            self.cube,
            smoothing_coefficients_x=self.smoothing_coefficients_cube_x,
            smoothing_coefficients_y=self.smoothing_coefficients_cube_y,
        )
        expected = 0.13277836
        self.assertAlmostEqual(result.data[0][2][2], expected)

    def test_smoothing_coefficient_cubes_masked_data(self):
        """Test that the RecursiveFilter plugin returns the correct data
        when a masked data cube."""
        plugin = RecursiveFilter(iterations=self.iterations,)
        mask = np.zeros((self.cube.data.shape))
        mask[0][3][2] = 1
        self.cube.data = np.ma.MaskedArray(self.cube.data, mask=mask)
        result = plugin(
            self.cube,
            smoothing_coefficients_x=self.smoothing_coefficients_cube_x,
            smoothing_coefficients_y=self.smoothing_coefficients_cube_y,
        )
        expected = 0.13277836
        self.assertAlmostEqual(result.data[0][2][2], expected)

    def test_coordinate_reordering_with_different_smoothing_coefficients(self):
        """Test that x and y smoothing_coefficients still apply to the right
        coordinate when the input cube spatial dimensions are (x, y) not
        (y, x)"""
        enforce_coordinate_ordering(self.cube, ["realization", "longitude", "latitude"])
        plugin = RecursiveFilter(iterations=self.iterations,)
        result = plugin(
            self.cube,
            smoothing_coefficients_x=self.smoothing_coefficients_cube_x,
            smoothing_coefficients_y=self.smoothing_coefficients_cube_y_half,
        )

        expected_result = np.array(
            [
                [0.02554158, 0.05397786, 0.1312837, 0.05397786, 0.02554158],
                [0.03596632, 0.07334216, 0.1668669, 0.07334216, 0.03596632],
                [0.05850913, 0.11031596, 0.21073693, 0.11031596, 0.05850913],
                [0.03596632, 0.07334216, 0.1668669, 0.07334216, 0.03596632],
                [0.02554158, 0.05397786, 0.1312837, 0.05397786, 0.02554158],
            ],
            dtype=np.float32,
        )

        self.assertSequenceEqual(
            [x.name() for x in result.coords(dim_coords=True)],
            ["realization", "longitude", "latitude"],
        )
        self.assertArrayAlmostEqual(result.data[0], expected_result)


if __name__ == "__main__":
    unittest.main()
