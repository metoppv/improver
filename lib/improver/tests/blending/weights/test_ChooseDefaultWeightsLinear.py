# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown Copyright 2017-2018 Met Office.
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
"""Unit tests for the weights.ChooseDefaultWeightsLinear plugin."""


import unittest

from iris.coords import AuxCoord
from iris.tests import IrisTest
import numpy as np

from improver.blending.weights import ChooseDefaultWeightsLinear \
    as LinearWeights
from improver.tests.blending.weights.helper_functions import (
    set_up_zero_cube, set_up_cube_with_scalar_coord, add_realizations)


class Test__init__(IrisTest):
    """Test the __init__ method."""

    def test_default_y0val_and_ynval(self):
        """Test default values of y0val and ynval are set correctly."""
        plugin = LinearWeights()
        self.assertEqual(plugin.y0val, 20.0)
        self.assertEqual(plugin.ynval, 2.0)

    def test_fails_y0val_not_float(self):
        """Test it fails if y0val not set to float """
        msg = ('y0val must be a float >= 0.0')
        with self.assertRaisesRegex(ValueError, msg):
            LinearWeights(y0val=2)

    def test_fails_y0val_lessthan_zero(self):
        """Test it raises a Value Error if y0val less than zero. """
        msg = ('y0val must be a float >= 0.0')
        with self.assertRaisesRegex(ValueError, msg):
            LinearWeights(y0val=-10.0)


class Test_linear_weights(IrisTest):
    """Test the linear weights function. """

    def test_basic(self):
        """Test that the function returns an array of weights. """
        result = LinearWeights().linear_weights(3)
        self.assertIsInstance(result, np.ndarray)

    def test_fails_ynval_and_slope_set(self):
        """Test it fails if y0val not set properly """
        msg = ('Relative end point weight or slope must be set'
               ' but not both.')
        with self.assertRaisesRegex(ValueError, msg):
            LinearWeights(ynval=3.0, slope=-1.0).linear_weights(3)

    def test_returns_correct_values_num_of_weights_one(self):
        """Test it returns the correct values, method is proportional."""
        result = LinearWeights().linear_weights(1)
        expected_result = np.array([1.0])
        self.assertArrayAlmostEqual(result, expected_result)

    def test_returns_correct_values_y0val_ynval_set(self):
        """Test it returns the correct values when y0val and ynval set"""
        result = LinearWeights(y0val=100.0, ynval=10.0).linear_weights(6)
        expected_result = np.array([0.3030303, 0.24848485,
                                    0.19393939, 0.13939394,
                                    0.08484848, 0.0303030])
        self.assertArrayAlmostEqual(result, expected_result)

    def test_returns_correct_values_y0val_slope_set(self):
        """Test it returns the correct values when y0val and slope set"""
        result = LinearWeights(y0val=10.0, slope=-1.0).linear_weights(6)
        expected_result = np.array([0.22222222, 0.2,
                                    0.17777778, 0.15555556,
                                    0.13333333, 0.11111111])
        self.assertArrayAlmostEqual(result, expected_result)

    def test_returns_correct_values_y0val_is_0_ynval_set(self):
        """Test it returns the correct values when y0val=0 and ynval set"""
        result = LinearWeights(y0val=0.0, ynval=5.0).linear_weights(5)
        expected_result = np.array([0.0, 0.1, 0.2, 0.3, 0.4])
        self.assertArrayAlmostEqual(result, expected_result)

    def test_returns_correct_values_y0val_is_0_slope_set(self):
        """Test it returns the correct values when y0val=0 and slope set."""
        result = LinearWeights(y0val=0.0, slope=1.0).linear_weights(5)
        expected_result = np.array([0.0, 0.1, 0.2, 0.3, 0.4])
        self.assertArrayAlmostEqual(result, expected_result)

    def test_returns_correct_values_y0val_is_0_ynval_is_0(self):
        """Test it raises an error when y0val=0 and ynval=0."""
        msg = "Sum of weights must be > 0.0"
        with self.assertRaisesRegex(ValueError, msg):
            LinearWeights(y0val=0.0, slope=0.0).linear_weights(5)

    def test_returns_correct_values_y0val_is_0_slope_is_0(self):
        """Test it raises an error when y0val=0 and slope=0."""
        msg = "Sum of weights must be > 0.0"
        with self.assertRaisesRegex(ValueError, msg):
            LinearWeights(y0val=0.0, slope=0.0).linear_weights(5)


class Test_process(IrisTest):
    """Test the Default Linear Weights plugin. """

    def setUp(self):
        self.cube = set_up_zero_cube()
        self.coord_name = "time"
        self.coord_vals = ','.join(
            [str(x) for x in self.cube.coord("time").points])

    def test_basic(self):
        """Test that the plugin returns an array of weights. """
        plugin = LinearWeights()
        result = plugin.process(self.cube, self.coord_name, self.coord_vals)
        self.assertIsInstance(result, np.ndarray)

    def test_array_sum_equals_one(self):
        """Test that the resulting weights add up to one. """
        plugin = LinearWeights()
        result = plugin.process(self.cube, self.coord_name, self.coord_vals)
        self.assertAlmostEqual(result.sum(), 1.0)

    def test_fails_coord_not_in_cube(self):
        """Test it raises a Value Error if coord not in the cube. """
        coord = AuxCoord([], long_name="notset")
        plugin = LinearWeights()
        msg = ('The coord for this plugin must be '
               'an existing coordinate in the input cube')
        with self.assertRaisesRegex(ValueError, msg):
            plugin.process(self.cube, coord)

    def test_fails_input_not_a_cube(self):
        """Test it raises a Value Error if not supplied with a cube. """
        plugin = LinearWeights()
        notacube = 0.0
        msg = ('The first argument must be an instance of '
               'iris.cube.Cube')
        with self.assertRaisesRegex(TypeError, msg):
            plugin.process(notacube, self.coord_name)

    def test_fails_ynval_and_slope_set(self):
        """Test it raises a Value Error if slope and ynval set. """
        plugin = LinearWeights(y0val=10.0, slope=-5.0, ynval=5.0)
        msg = ('Relative end point weight or slope must be set'
               ' but not both.')
        with self.assertRaisesRegex(ValueError, msg):
            plugin.process(self.cube, self.coord_name, self.coord_vals)

    def test_fails_weights_negative(self):
        """Test it raises a Value Error if weights become negative. """
        plugin = LinearWeights(y0val=10.0, slope=-5.0)
        cubenew = add_realizations(self.cube, 6)
        coord = cubenew.coord('realization')
        msg = 'Weights must be positive'
        with self.assertRaisesRegex(ValueError, msg):
            plugin.process(cubenew, coord)

    def test_works_scalar_coord(self):
        """Test it works if scalar coordinate. """
        cube = set_up_cube_with_scalar_coord()
        coord = cube.coord("scalar_coord")
        plugin = LinearWeights()
        result = plugin.process(cube, coord)
        self.assertArrayAlmostEqual(result, np.array([1.0]))

    def test_works_defaults_used(self):
        """Test it works if defaults used. """
        plugin = LinearWeights()
        result = plugin.process(self.cube, self.coord_name, self.coord_vals)
        expected_result = np.array([0.90909091, 0.09090909])
        self.assertArrayAlmostEqual(result, expected_result)

    def test_works_y0val_and_slope_set(self):
        """Test it works if y0val and slope_set. """
        plugin = LinearWeights(y0val=10.0, slope=-5.0)
        result = plugin.process(self.cube, self.coord_name, self.coord_vals)
        expected_result = np.array([0.66666667, 0.33333333])
        self.assertArrayAlmostEqual(result, expected_result)

    def test_works_y0val_and_ynval_set(self):
        """Test it works if y0val and ynval set. """
        plugin = LinearWeights(y0val=10.0, ynval=5.0)
        result = plugin.process(self.cube, self.coord_name, self.coord_vals)
        expected_result = np.array([0.66666667, 0.33333333])
        self.assertArrayAlmostEqual(result, expected_result)

    def test_works_with_larger_num(self):
        """Test it works with larger num_of_vals. """
        plugin = LinearWeights(y0val=10.0, ynval=5.0)
        cubenew = add_realizations(self.cube, 6)
        coord = cubenew.coord('realization')
        result = plugin.process(cubenew, coord)
        expected_result = np.array([0.22222222, 0.2,
                                    0.17777778, 0.15555556,
                                    0.13333333, 0.11111111])
        self.assertArrayAlmostEqual(result, expected_result)

    def test_works_with_missing_coord(self):
        """Test it works with missing coord """
        plugin = LinearWeights(y0val=10.0, ynval=5.0)
        cubenew = add_realizations(self.cube, 6)
        coord_vals = '0, 1, 2, 3, 4, 5, 6'
        coord_name = 'realization'
        result = plugin.process(cubenew, coord_name, coord_vals)
        expected_result = np.array([0.206349, 0.190476,
                                    0.174603, 0.15873,
                                    0.142857, 0.126984])
        self.assertArrayAlmostEqual(result, expected_result)


if __name__ == '__main__':
    unittest.main()
