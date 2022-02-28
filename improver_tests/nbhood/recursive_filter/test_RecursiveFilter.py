# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown Copyright 2017-2022 Met Office.
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
from datetime import timedelta

import iris
import numpy as np
from iris.cube import Cube
from iris.tests import IrisTest

from improver.nbhood.recursive_filter import RecursiveFilter
from improver.synthetic_data.set_up_test_cubes import (
    add_coordinate,
    set_up_variable_cube,
)
from improver.utilities.cube_manipulation import enforce_coordinate_ordering
from improver.utilities.pad_spatial import pad_cube_with_halo
from improver.utilities.warnings_handler import ManageWarnings


def _mean_points(points):
    """Create an array of the mean of adjacent points in original array"""
    return np.array((points[:-1] + points[1:]) / 2, dtype=np.float32)


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

        self.x_name = "smoothing_coefficient_x"
        self.y_name = "smoothing_coefficient_y"

        # Generate smoothing_coefficients_cubes with correct dimensions 5 x 4
        smoothing_coefficients_cube_x = set_up_variable_cube(
            np.full((5, 4), 0.5, dtype=np.float32), name=self.x_name
        )
        mean_x_points = _mean_points(
            smoothing_coefficients_cube_x.coord(axis="y").points
        )
        smoothing_coefficients_cube_x.coord(axis="x").points = mean_x_points
        smoothing_coefficients_cube_y = set_up_variable_cube(
            np.full((4, 5), 0.5, dtype=np.float32), name=self.y_name
        )
        mean_y_points = _mean_points(
            smoothing_coefficients_cube_y.coord(axis="x").points
        )
        smoothing_coefficients_cube_y.coord(axis="y").points = mean_y_points

        self.smoothing_coefficients = [
            smoothing_coefficients_cube_x,
            smoothing_coefficients_cube_y,
        ]

        # Generate an alternative y smoothing_coefficients_cube with correct dimensions 5 x 4
        smoothing_coefficients_cube_y_half = smoothing_coefficients_cube_y * 0.5
        smoothing_coefficients_cube_y_half.rename(self.y_name)
        self.smoothing_coefficients_alternative = [
            smoothing_coefficients_cube_x,
            smoothing_coefficients_cube_y_half,
        ]

        # Generate smoothing_coefficients_cube with incorrect name
        smoothing_coefficients_wrong_name = smoothing_coefficients_cube_x.copy()
        smoothing_coefficients_wrong_name.rename("air_temperature")
        self.smoothing_coefficients_wrong_name = [
            smoothing_coefficients_wrong_name,
            smoothing_coefficients_cube_y,
        ]

        # Generate smoothing_coefficients_cubes with incorrect dimensions 6 x 6
        smoothing_coefficients_cube_wrong_x = set_up_variable_cube(
            np.full((6, 6), 0.5, dtype=np.float32), name=self.x_name
        )
        smoothing_coefficients_cube_wrong_y = set_up_variable_cube(
            np.full((6, 6), 0.5, dtype=np.float32), name=self.y_name
        )
        self.smoothing_coefficients_wrong_dimensions = [
            smoothing_coefficients_cube_wrong_x,
            smoothing_coefficients_cube_wrong_y,
        ]

        # Generate smoothing_coefficients_cubes with incorrect coordinate values
        smoothing_coefficients_cube_wrong_x_points = (
            smoothing_coefficients_cube_x.copy()
        )
        smoothing_coefficients_cube_wrong_x_points.coord(axis="x").points = (
            smoothing_coefficients_cube_wrong_x_points.coord(axis="x").points + 10
        )
        smoothing_coefficients_cube_wrong_y_points = (
            smoothing_coefficients_cube_y.copy()
        )
        smoothing_coefficients_cube_wrong_y_points.coord(axis="y").points = (
            smoothing_coefficients_cube_wrong_y_points.coord(axis="y").points + 10
        )
        self.smoothing_coefficients_wrong_points = [
            smoothing_coefficients_cube_wrong_x_points,
            smoothing_coefficients_cube_wrong_y_points,
        ]


class Test__init__(Test_RecursiveFilter):

    """Test plugin initialisation."""

    def test_basic(self):
        """Test using the default arguments."""
        result = RecursiveFilter()
        self.assertIsNone(result.iterations)
        self.assertEqual(result.edge_width, 15)

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


class Test__validate_coefficients(Test_RecursiveFilter):

    """Test the _validate_coefficients method"""

    def test_return_order(self):
        """Test that the coefficients cubes are returned in x, y order."""
        x, y = RecursiveFilter()._validate_coefficients(
            self.cube, self.smoothing_coefficients
        )
        self.assertEqual(x.name(), self.x_name)
        self.assertEqual(y.name(), self.y_name)

        x, y = RecursiveFilter()._validate_coefficients(
            self.cube, self.smoothing_coefficients[::-1]
        )
        self.assertEqual(x.name(), self.x_name)
        self.assertEqual(y.name(), self.y_name)

    def test_smoothing_coefficients_wrong_name(self):
        """Test that an error is raised if the smoothing_coefficients_cube has
        an incorrect name"""
        msg = (
            "The smoothing coefficient cube name air_temperature does not "
            "match the expected name smoothing_coefficient_x"
        )
        with self.assertRaisesRegex(ValueError, msg):
            RecursiveFilter(edge_width=1)._validate_coefficients(
                self.cube, self.smoothing_coefficients_wrong_name
            )

    def test_smoothing_coefficients_mismatched_x_dimension(self):
        """Test that an error is raised if the x smoothing_coefficients_cube is
        of an incorrect shape compared to the data cube."""
        msg = (
            "The smoothing coefficients x dimension does not have the "
            "expected length or values"
        )
        with self.assertRaisesRegex(ValueError, msg):
            RecursiveFilter(edge_width=1)._validate_coefficients(
                self.cube, self.smoothing_coefficients_wrong_dimensions
            )

    def test_smoothing_coefficients_mismatched_x_points(self):
        """Test that an error is raised if the x smoothing_coefficients_cube
        has mismatched coordinate points compared to the data cube."""
        msg = (
            "The smoothing coefficients x dimension does not have the "
            "expected length or values"
        )
        with self.assertRaisesRegex(ValueError, msg):
            RecursiveFilter(edge_width=1)._validate_coefficients(
                self.cube, self.smoothing_coefficients_wrong_points
            )

    def test_smoothing_coefficients_exceed_max(self):
        """Test that an error is raised if any smoothing coefficient value
        exceeds the allowed maximum of 0.5."""
        self.smoothing_coefficients[0].data += 1.0
        msg = (
            "All smoothing_coefficient values must be less than 0.5. "
            "A large smoothing_coefficient value leads to poor "
            "conservation of probabilities"
        )
        with self.assertRaisesRegex(ValueError, msg):
            RecursiveFilter(edge_width=1)._validate_coefficients(
                self.cube, self.smoothing_coefficients
            )


class Test__pad_coefficients(Test_RecursiveFilter):
    """Test the _pad_coefficients method"""

    def test_padding_default(self):
        """Test that the returned smoothing_coefficients array is padded as
        expected with the default edge_width.

        Using default edge_width of 15 cells, which is doubled and applied to both
        sides of the array, so array should be padded with 15 * 4 extra rows/columns.
        """
        expected_shape_x = (65, 64)
        expected_shape_y = (64, 65)
        expected_result_x = np.full(expected_shape_x, 0.5)
        expected_result_y = np.full(expected_shape_y, 0.5)
        result_x, result_y = RecursiveFilter()._pad_coefficients(
            *self.smoothing_coefficients
        )
        self.assertIsInstance(result_x.data, np.ndarray)
        self.assertIsInstance(result_y.data, np.ndarray)
        self.assertArrayEqual(result_x.data, expected_result_x)
        self.assertArrayEqual(result_y.data, expected_result_y)
        self.assertEqual(result_x.shape, expected_shape_x)
        self.assertEqual(result_y.shape, expected_shape_y)

    def test_padding_set_edge_width(self):
        """Test that the returned smoothing_coefficients arrays are padded as
        expected with a set edge_width.

        Using an edge_width of 1 cell, which is doubled and applied to both
        sides of the array, so array should be padded with 1 * 4 extra rows/columns.
        """
        expected_shape_x = (9, 8)
        expected_shape_y = (8, 9)
        expected_result_x = np.full(expected_shape_x, 0.5)
        expected_result_y = np.full(expected_shape_y, 0.5)
        result_x, result_y = RecursiveFilter(edge_width=1)._pad_coefficients(
            *self.smoothing_coefficients
        )
        self.assertArrayEqual(result_x.data, expected_result_x)
        self.assertArrayEqual(result_y.data, expected_result_y)
        self.assertEqual(result_x.shape, expected_shape_x)
        self.assertEqual(result_y.shape, expected_shape_y)

    def test_padding_non_constant_values(self):
        """Test that the returned smoothing_coefficients array contains the
        expected values when padded symmetrically with non-constant smoothing
        coefficients.

        Using an edge_width of 1 cell, which is doubled and applied to both
        sides of the array, so array should be padded with 1 * 4 extra rows/columns.
        """
        expected_shape = (9, 8)
        expected_result = np.full(expected_shape, 0.5)
        expected_result[1:3, 1:3] = 0.25
        self.smoothing_coefficients[0].data[0, 0] = 0.25
        result, _ = RecursiveFilter(edge_width=1)._pad_coefficients(
            *self.smoothing_coefficients
        )
        self.assertArrayEqual(result.data, expected_result)
        self.assertEqual(result.shape, expected_shape)


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
            self.cube.data[0, :], self.smoothing_coefficients[1].data, 0
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
            self.cube.data[0, :], self.smoothing_coefficients[0].data, 1
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
            self.cube.data[0, :], self.smoothing_coefficients[1].data, 0
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
            self.cube.data[0, :], self.smoothing_coefficients[0].data, 1
        )
        self.assertIsInstance(result, np.ndarray)
        self.assertArrayAlmostEqual(result, expected_result)


class Test__run_recursion(Test_RecursiveFilter):

    """Test the _run_recursion method"""

    def test_return_type(self):
        """Test that the _run_recursion method returns an iris.cube.Cube."""
        edge_width = 1
        cube = iris.util.squeeze(self.cube)
        smoothing_coefficients_x, smoothing_coefficients_y = RecursiveFilter(
            edge_width=edge_width
        )._pad_coefficients(*self.smoothing_coefficients)
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
        smoothing_coefficients_x, smoothing_coefficients_y = RecursiveFilter(
            edge_width=edge_width
        )._pad_coefficients(*self.smoothing_coefficients)
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
        smoothing_coefficients_x, smoothing_coefficients_y = RecursiveFilter(
            edge_width=edge_width
        )._pad_coefficients(*self.smoothing_coefficients_alternative)
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
        result = plugin(self.cube, smoothing_coefficients=self.smoothing_coefficients,)
        self.assertIsInstance(result, Cube)
        self.assertEqual(result.shape, expected_shape)
        self.assertEqual(result.shape, expected_shape)

    def test_smoothing_coefficient_cubes(self):
        """Test that the RecursiveFilter plugin returns the correct data."""
        plugin = RecursiveFilter(iterations=self.iterations,)
        result = plugin(self.cube, smoothing_coefficients=self.smoothing_coefficients,)
        expected = 0.14994797
        self.assertAlmostEqual(result.data[0][2][2], expected)

    def test_smoothing_coefficient_cubes_masked_data(self):
        """Test that the RecursiveFilter plugin returns the correct data
        when a masked data cube.
        """
        plugin = RecursiveFilter(iterations=self.iterations,)
        mask = np.zeros(self.cube.data.shape)
        mask[0][3][2] = 1
        self.cube.data = np.ma.MaskedArray(self.cube.data, mask=mask)
        result = plugin(self.cube, smoothing_coefficients=self.smoothing_coefficients)
        expected = 0.184375
        self.assertAlmostEqual(result.data[0][2][2], expected)
        self.assertArrayEqual(result.data.mask, mask)

    def test_coordinate_reordering_with_different_smoothing_coefficients(self):
        """Test that x and y smoothing_coefficients still apply to the right
        coordinate when the input cube spatial dimensions are (x, y) not
        (y, x)"""
        enforce_coordinate_ordering(self.cube, ["realization", "longitude", "latitude"])
        plugin = RecursiveFilter(iterations=self.iterations,)
        result = plugin(
            self.cube, smoothing_coefficients=self.smoothing_coefficients_alternative
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

    def test_error_multiple_times_masked(self):
        """Test that the plugin raises an error when given a masked cube with
        multiple time points"""
        point = self.cube.coord("time").cell(0).point
        time_points = [point - timedelta(seconds=3600), point]
        cube = add_coordinate(self.cube, time_points, "time", is_datetime=True)
        mask = np.zeros(cube.data.shape, dtype=int)
        mask[0, 0, 2, 2] = 1
        mask[1, 0, 2, 3] = 1
        cube.data = np.ma.MaskedArray(cube.data, mask=mask)
        plugin = RecursiveFilter(iterations=self.iterations,)
        msg = "multiple time points is unsupported"
        with self.assertRaisesRegex(ValueError, msg):
            plugin(cube, smoothing_coefficients=self.smoothing_coefficients)


if __name__ == "__main__":
    unittest.main()
