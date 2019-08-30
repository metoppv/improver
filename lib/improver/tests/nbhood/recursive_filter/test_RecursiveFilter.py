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
"""Unit tests for the nbhood.RecursiveFilter plugin."""

import unittest

import iris
import numpy as np
from iris.cube import Cube
from iris.tests import IrisTest

from improver.nbhood.recursive_filter import RecursiveFilter
from improver.tests.set_up_test_cubes import set_up_variable_cube
from improver.utilities.cube_manipulation import enforce_coordinate_ordering
from improver.utilities.pad_spatial import pad_cube_with_halo


class Test__repr__(IrisTest):

    """Test the repr method."""

    def test_basic(self):
        """Test that the __repr__ returns the expected string."""
        alpha_x = None
        alpha_y = None
        iterations = None
        edge_width = 1
        result = str(RecursiveFilter(alpha_x, alpha_y, iterations, edge_width))
        msg = ('<RecursiveFilter: alpha_x: {}, alpha_y: {}, iterations: {},'
               ' edge_width: {}'.format(alpha_x, alpha_y,
                                        iterations, edge_width))
        self.assertEqual(result, msg)


class Test_RecursiveFilter(IrisTest):

    """Test class for the RecursiveFilter tests, setting up cubes."""

    def setUp(self):
        """Create test cubes."""

        self.alpha_x = 0.5
        self.alpha_y = 0.5
        self.iterations = 1

        # Generate data cube with dimensions 1 x 5 x 5
        data = np.array([[[0.00, 0.00, 0.10, 0.00, 0.00],
                          [0.00, 0.00, 0.25, 0.00, 0.00],
                          [0.10, 0.25, 0.50, 0.25, 0.10],
                          [0.00, 0.00, 0.25, 0.00, 0.00],
                          [0.00, 0.00, 0.10, 0.00, 0.00]]], dtype=np.float32)
        self.cube = set_up_variable_cube(
            data, name="precipitation_amount", units="kg m^-2 s^-1")

        # Generate alphas_cube with correct dimensions 5 x 5
        self.alphas_cube = set_up_variable_cube(
            np.full((5, 5), 0.5, dtype=np.float32))

        # Generate alphas_cube with incorrect dimensions 6 x 6
        self.alphas_cube_wrong_dims = set_up_variable_cube(
            np.full((6, 6), 0.5, dtype=np.float32))


class Test__init__(Test_RecursiveFilter):

    """Test plugin initialisation."""

    def test_alpha_x_gt_unity(self):
        """Test when an alpha_x value > unity is given (invalid)."""
        alpha_x = 1.1
        msg = "Invalid alpha_x: must be > 0 and < 1: 1.1"
        with self.assertRaisesRegex(ValueError, msg):
            RecursiveFilter(alpha_x=alpha_x, alpha_y=None,
                            iterations=None, edge_width=1)

    def test_alpha_x_lt_zero(self):
        """Test when an alpha_x value <= zero is given (invalid)."""
        alpha_x = -0.5
        msg = "Invalid alpha_x: must be > 0 and < 1: -0.5"
        with self.assertRaisesRegex(ValueError, msg):
            RecursiveFilter(alpha_x=alpha_x, alpha_y=None,
                            iterations=None, edge_width=1)

    def test_alpha_y_gt_unity(self):
        """Test when an alpha_y value > unity is given (invalid)."""
        alpha_y = 1.1
        msg = "Invalid alpha_y: must be > 0 and < 1: 1.1"
        with self.assertRaisesRegex(ValueError, msg):
            RecursiveFilter(alpha_x=None, alpha_y=alpha_y,
                            iterations=None, edge_width=1)

    def test_alpha_y_lt_zero(self):
        """Test when an alpha_y value <= zero is given (invalid)."""
        alpha_y = -0.5
        msg = "Invalid alpha_y: must be > 0 and < 1: -0.5"
        with self.assertRaisesRegex(ValueError, msg):
            RecursiveFilter(alpha_x=None, alpha_y=alpha_y,
                            iterations=None, edge_width=1)

    def test_iterations(self):
        """Test when iterations value less than unity is given (invalid)."""
        iterations = 0
        msg = "Invalid number of iterations: must be >= 1: 0"
        with self.assertRaisesRegex(ValueError, msg):
            RecursiveFilter(alpha_x=None, alpha_y=None,
                            iterations=iterations, edge_width=1)


class Test__set_alphas(Test_RecursiveFilter):

    """Test the _set_alphas function"""

    def test_alpha_x_used_result(self):
        """Test that the returned alphas array has the expected result
           when alphas_cube=None."""
        cube = iris.util.squeeze(self.cube)
        result = RecursiveFilter()._set_alphas(cube, self.alpha_x, None)
        expected_result = 0.5
        self.assertIsInstance(result.data, np.ndarray)
        self.assertEqual(result.data[0][2], expected_result)
        # Check shape: Array should be padded with 4 extra rows/columns
        expected_shape = (9, 9)
        self.assertEqual(result.shape, expected_shape)

    def test_alphas_cube_used_result(self):
        """Test that the returned alphas array has the expected result
           when alphas_cube is not None."""
        result = RecursiveFilter()._set_alphas(self.cube[0, :], None,
                                               self.alphas_cube)
        expected_result = 0.5
        self.assertIsInstance(result.data, np.ndarray)
        self.assertEqual(result.data[0][2], expected_result)
        # Check shape: Array should be padded with 4 extra rows/columns
        expected_shape = (9, 9)
        self.assertEqual(result.shape, expected_shape)

    def test_mismatched_dimensions_alphas_cube_data_cube(self):
        """Test that an error is raised if the alphas_cube is of a different
        shape to the data cube."""
        msg = "Dimensions of alphas array do not match dimensions "
        with self.assertRaisesRegex(ValueError, msg):
            RecursiveFilter()._set_alphas(self.cube, None,
                                          self.alphas_cube_wrong_dims)

    def test_alphas_cube_and_alpha_not_set(self):
        """Test error is raised when both alphas_cube and alpha are set
           to None (invalid)."""
        alpha = None
        alphas_cube = None
        msg = "A value for alpha must be set if alphas_cube is "
        with self.assertRaisesRegex(ValueError, msg):
            RecursiveFilter()._set_alphas(self.cube, alpha, alphas_cube)

    def test_alphas_cube_and_alpha_both_set(self):
        """Test error is raised when both alphas_cube and alpha are set."""
        alpha = 0.5
        alphas_cube = self.alphas_cube
        msg = "A cube of alpha values and a single float value for"
        with self.assertRaisesRegex(ValueError, msg):
            RecursiveFilter()._set_alphas(self.cube, alpha, alphas_cube)


class Test__recurse_forward(Test_RecursiveFilter):

    """Test the _recurse_forward method"""

    def test_first_axis(self):
        """Test that the returned _recurse_forward array has the expected
           type and result."""
        expected_result = np.array(
            [[0.0000, 0.00000, 0.100000, 0.00000, 0.0000],
             [0.0000, 0.00000, 0.175000, 0.00000, 0.0000],
             [0.0500, 0.12500, 0.337500, 0.12500, 0.0500],
             [0.0250, 0.06250, 0.293750, 0.06250, 0.0250],
             [0.0125, 0.03125, 0.196875, 0.03125, 0.0125]])
        result = RecursiveFilter()._recurse_forward(
            self.cube.data[0, :], self.alphas_cube.data, 0)
        self.assertIsInstance(result, np.ndarray)
        self.assertArrayAlmostEqual(result, expected_result)

    def test_second_axis(self):
        """Test that the returned _recurse_forward array has the expected
           type and result."""
        expected_result = np.array(
            [[0.0, 0.000, 0.0500, 0.02500, 0.012500],
             [0.0, 0.000, 0.1250, 0.06250, 0.031250],
             [0.1, 0.175, 0.3375, 0.29375, 0.196875],
             [0.0, 0.000, 0.1250, 0.06250, 0.031250],
             [0.0, 0.000, 0.0500, 0.02500, 0.012500]])
        result = RecursiveFilter()._recurse_forward(
            self.cube.data[0, :], self.alphas_cube.data, 1)
        self.assertIsInstance(result, np.ndarray)
        self.assertArrayAlmostEqual(result, expected_result)


class Test__recurse_backward(Test_RecursiveFilter):

    """Test the _recurse_backward method"""

    def test_first_axis(self):
        """Test that the returned _recurse_backward array has the expected
           type and result."""
        expected_result = np.array(
            [[0.0125, 0.03125, 0.196875, 0.03125, 0.0125],
             [0.0250, 0.06250, 0.293750, 0.06250, 0.0250],
             [0.0500, 0.12500, 0.337500, 0.12500, 0.0500],
             [0.0000, 0.00000, 0.175000, 0.00000, 0.0000],
             [0.0000, 0.00000, 0.100000, 0.00000, 0.0000]])
        result = RecursiveFilter()._recurse_backward(
            self.cube.data[0, :], self.alphas_cube.data, 0)
        self.assertIsInstance(result, np.ndarray)
        self.assertArrayAlmostEqual(result, expected_result)

    def test_second_axis(self):
        """Test that the returned _recurse_backward array has the expected
           type and result."""
        expected_result = np.array(
            [[0.012500, 0.02500, 0.0500, 0.000, 0.0],
             [0.031250, 0.06250, 0.1250, 0.000, 0.0],
             [0.196875, 0.29375, 0.3375, 0.175, 0.1],
             [0.031250, 0.06250, 0.1250, 0.000, 0.0],
             [0.012500, 0.02500, 0.0500, 0.000, 0.0]])
        result = RecursiveFilter()._recurse_backward(
            self.cube.data[0, :], self.alphas_cube.data, 1)
        self.assertIsInstance(result, np.ndarray)
        self.assertArrayAlmostEqual(result, expected_result)


class Test__run_recursion(Test_RecursiveFilter):

    """Test the _run_recursion method"""

    def test_return_type(self):
        """Test that the _run_recursion method returns an iris.cube.Cube."""
        edge_width = 1
        cube = iris.util.squeeze(self.cube)
        alphas_x = RecursiveFilter()._set_alphas(cube, self.alpha_x, None)
        alphas_y = RecursiveFilter()._set_alphas(cube, self.alpha_y, None)
        padded_cube = pad_cube_with_halo(cube, 2*edge_width, 2*edge_width)
        result = RecursiveFilter()._run_recursion(
            padded_cube, alphas_x, alphas_y, self.iterations)
        self.assertIsInstance(result, Cube)

    def test_result_basic(self):
        """Test that the _run_recursion method returns the expected value."""
        edge_width = 1
        cube = iris.util.squeeze(self.cube)
        alphas_x = RecursiveFilter()._set_alphas(cube, self.alpha_x, None)
        alphas_y = RecursiveFilter()._set_alphas(cube, self.alpha_y, None)
        padded_cube = pad_cube_with_halo(cube, 2*edge_width, 2*edge_width)
        result = RecursiveFilter()._run_recursion(
            padded_cube, alphas_x, alphas_y, self.iterations)
        expected_result = 0.13382206
        self.assertAlmostEqual(result.data[4][4], expected_result)

    def test_different_alphas(self):
        """Test that the _run_recursion method returns expected values when
        alpha values are different in the x and y directions."""
        cube = iris.util.squeeze(self.cube)
        alpha_y = 0.5*self.alpha_x
        alphas_x = RecursiveFilter()._set_alphas(cube, self.alpha_x, None)
        alphas_y = RecursiveFilter()._set_alphas(cube, alpha_y, None)
        padded_cube = pad_cube_with_halo(cube, 2, 2)
        result = RecursiveFilter()._run_recursion(
            padded_cube, alphas_x, alphas_y, 1)
        # slice back down to the source grid - easier to visualise!
        unpadded_result = result.data[2:-2, 2:-2]

        expected_result = np.array(
            [[0.01620921, 0.02866841, 0.05077430, 0.02881413, 0.01657352],
             [0.03978802, 0.06457599, 0.10290188, 0.06486591, 0.04051282],
             [0.10592333, 0.15184643, 0.19869247, 0.15238355, 0.10726611],
             [0.03978982, 0.06457873, 0.10290585, 0.06486866, 0.04051464],
             [0.01621686, 0.02868005, 0.05079120, 0.02882582, 0.01658128]])

        self.assertArrayAlmostEqual(unpadded_result, expected_result)


class Test_process(Test_RecursiveFilter):

    """Test the process method. """

    # Test output from plugin returns expected values
    def test_return_type(self):
        """Test that the RecursiveFilter plugin returns an iris.cube.Cube."""
        plugin = RecursiveFilter(alpha_x=self.alpha_x, alpha_y=self.alpha_y,
                                 iterations=self.iterations)
        result = plugin.process(self.cube, alphas_x=None, alphas_y=None)
        self.assertIsInstance(result, Cube)

    def test_alpha_floats(self):
        """Test that the RecursiveFilter plugin returns the correct data
        when using float alpha values."""
        plugin = RecursiveFilter(alpha_x=self.alpha_x, alpha_y=self.alpha_y,
                                 iterations=self.iterations)
        result = plugin.process(self.cube, alphas_x=None, alphas_y=None)
        expected = 0.13382206
        self.assertAlmostEqual(result.data[0][2][2], expected)

    def test_alpha_cubes(self):
        """Test that the RecursiveFilter plugin returns the correct data
        when using alpha cubes."""
        plugin = RecursiveFilter(alpha_x=None, alpha_y=None,
                                 iterations=self.iterations)
        result = plugin.process(self.cube, alphas_x=self.alphas_cube,
                                alphas_y=self.alphas_cube)
        expected = 0.13382206
        self.assertAlmostEqual(result.data[0][2][2], expected)

    def test_alpha_floats_nan_in_data(self):
        """Test that the RecursiveFilter plugin returns the correct data
        when using float alpha values and the data contains nans."""
        plugin = RecursiveFilter(alpha_x=self.alpha_x, alpha_y=self.alpha_y,
                                 iterations=self.iterations)
        self.cube.data[0][3][2] = np.nan
        result = plugin.process(self.cube, alphas_x=None, alphas_y=None)
        expected = 0.11979733
        self.assertAlmostEqual(result.data[0][2][2], expected)

    def test_alpha_floats_nan_in_masked_data(self):
        """Test that the RecursiveFilter plugin returns the correct data
        when using float alpha values, the data contains nans and the data
        is masked (but not the nan value)."""
        plugin = RecursiveFilter(alpha_x=self.alpha_x, alpha_y=self.alpha_y,
                                 iterations=self.iterations)
        self.cube.data[0][3][2] = np.nan
        mask = np.zeros((self.cube.data.shape))
        mask[0][1][2] = 1
        self.cube.data = np.ma.MaskedArray(self.cube.data, mask=mask)
        result = plugin.process(self.cube, alphas_x=None, alphas_y=None)
        expected = 0.105854129
        self.assertAlmostEqual(result.data[0][2][2], expected)

    def test_alpha_cubes_masked_data(self):
        """Test that the RecursiveFilter plugin returns the correct data
        when using alpha cubes and a masked data cube."""
        plugin = RecursiveFilter(alpha_x=None, alpha_y=None,
                                 iterations=self.iterations)
        mask = np.zeros((self.cube.data.shape))
        mask[0][3][2] = 1
        self.cube.data = np.ma.MaskedArray(self.cube.data, mask=mask)
        result = plugin.process(self.cube, alphas_x=self.alphas_cube,
                                alphas_y=self.alphas_cube)
        expected = 0.11979733
        self.assertAlmostEqual(result.data[0][2][2], expected)

    def test_dimensions_of_output_array_is_as_expected(self):
        """Test that the RecursiveFilter plugin returns a data array with
           the correct dimensions"""
        plugin = RecursiveFilter(alpha_x=self.alpha_x, alpha_y=self.alpha_y,
                                 iterations=self.iterations)
        result = plugin.process(self.cube, alphas_x=None, alphas_y=None)
        # Output data array should have same dimensions as input data array
        expected_shape = (1, 5, 5)
        self.assertEqual(result.data.shape, expected_shape)
        self.assertEqual(result.data.shape, expected_shape)

    def test_coordinate_reordering_with_different_alphas(self):
        """Test that x and y alphas still apply to the right coordinate when
        the input cube spatial dimensions are (x, y) not (y, x)"""
        alpha_y = 0.5*self.alpha_x
        cube = enforce_coordinate_ordering(
            self.cube, ["realization", "longitude", "latitude"])
        plugin = RecursiveFilter(alpha_x=self.alpha_x, alpha_y=alpha_y,
                                 iterations=self.iterations)
        result = plugin.process(cube)

        expected_result = np.array(
            [[0.01620921, 0.03978802, 0.10592333, 0.03978982, 0.01621686],
             [0.02866841, 0.06457599, 0.15184643, 0.06457873, 0.02868005],
             [0.05077430, 0.10290188, 0.19869247, 0.10290585, 0.05079120],
             [0.02881413, 0.06486591, 0.15238355, 0.06486866, 0.02882582],
             [0.01657352, 0.04051282, 0.10726611, 0.04051464, 0.01658128]])

        self.assertSequenceEqual(
            [x.name() for x in result.coords(dim_coords=True)],
            ["realization", "longitude", "latitude"])
        self.assertArrayAlmostEqual(result.data[0], expected_result)


if __name__ == '__main__':
    unittest.main()
