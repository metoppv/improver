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
"""Unit tests for the weights.ChooseDefaultWeightsLinear plugin."""


import unittest
from datetime import datetime as dt

import iris
import numpy as np
from iris.coords import AuxCoord
from iris.tests import IrisTest

from improver.blending.weights import \
    ChooseDefaultWeightsLinear as LinearWeights

from ...set_up_test_cubes import add_coordinate, set_up_variable_cube


class Test__init__(IrisTest):
    """Test the __init__ method."""

    def test_basic(self):
        """Test values of y0val and ynval are set correctly"""
        plugin = LinearWeights(y0val=20.0, ynval=2.0)
        self.assertEqual(plugin.y0val, 20.0)
        self.assertEqual(plugin.ynval, 2.0)

    def test_fails_y0val_less_than_zero(self):
        """Test it raises a Value Error if y0val less than zero. """
        msg = ('y0val must be a float >= 0.0')
        with self.assertRaisesRegex(ValueError, msg):
            LinearWeights(y0val=-10.0, ynval=2.0)


class Test_linear_weights(IrisTest):
    """Test the linear weights function. """

    def test_basic(self):
        """Test that the function returns an array of weights"""
        result = LinearWeights(y0val=20.0, ynval=2.0).linear_weights(3)
        self.assertIsInstance(result, np.ndarray)

    def test_returns_correct_values_num_of_weights_one(self):
        """Test it returns the correct values, method is proportional"""
        result = LinearWeights(y0val=20.0, ynval=2.0).linear_weights(1)
        expected_result = np.array([1.0])
        self.assertArrayAlmostEqual(result, expected_result)

    def test_returns_correct_values_y0val_ynval_set(self):
        """Test it returns the correct values when y0val and ynval set"""
        result = LinearWeights(y0val=10.0, ynval=5.0).linear_weights(6)
        expected_result = np.array([0.22222222, 0.2,
                                    0.17777778, 0.15555556,
                                    0.13333333, 0.11111111])
        self.assertArrayAlmostEqual(result, expected_result)

    def test_returns_correct_values_y0val_is_0_ynval_set(self):
        """Test it returns the correct values when y0val=0 and ynval set"""
        result = LinearWeights(y0val=0.0, ynval=5.0).linear_weights(5)
        expected_result = np.array([0.0, 0.1, 0.2, 0.3, 0.4])
        self.assertArrayAlmostEqual(result, expected_result)

    def test_fails_if_total_weights_zero(self):
        """Test it raises an error when y0val=0 and ynval=0"""
        msg = "Sum of weights must be > 0.0"
        with self.assertRaisesRegex(ValueError, msg):
            LinearWeights(y0val=0.0, ynval=0.0).linear_weights(5)


class Test_process(IrisTest):
    """Test the Default Linear Weights plugin. """

    def setUp(self):
        """Set up for testing process method"""
        cube = set_up_variable_cube(
            np.zeros((2, 2), dtype=np.float32),
            name="lwe_thickness_of_precipitation_amount", units="m",
            time=dt(2017, 1, 10, 5, 0), frt=dt(2017, 1, 10, 3, 0))
        self.cube = add_coordinate(
            cube, [dt(2017, 1, 10, 5, 0), dt(2017, 1, 10, 6, 0)],
            "time", is_datetime=True)
        self.coord_name = "time"

    def test_basic(self):
        """Test that the plugin returns a cube of weights. """
        plugin = LinearWeights(y0val=20.0, ynval=2.0)
        result = plugin.process(self.cube, self.coord_name)
        self.assertIsInstance(result, iris.cube.Cube)

    def test_array_sum_equals_one(self):
        """Test that the resulting weights add up to one. """
        plugin = LinearWeights(y0val=20.0, ynval=2.0)
        result = plugin.process(self.cube, self.coord_name)
        self.assertAlmostEqual(result.data.sum(), 1.0)

    def test_fails_input_not_a_cube(self):
        """Test it raises a Value Error if not supplied with a cube. """
        plugin = LinearWeights(y0val=20.0, ynval=2.0)
        notacube = 0.0
        msg = ('The first argument must be an instance of '
               'iris.cube.Cube')
        with self.assertRaisesRegex(TypeError, msg):
            plugin.process(notacube, self.coord_name)

    def test_works_scalar_coord(self):
        """Test it works if scalar coordinate. """
        self.cube.add_aux_coord(
            AuxCoord(1, long_name='scalar_coord', units='no_unit'))
        coord = self.cube.coord("scalar_coord")
        plugin = LinearWeights(y0val=20.0, ynval=2.0)
        result = plugin.process(self.cube, coord)
        self.assertArrayAlmostEqual(result.data, np.array([1.0]))

    def test_works_defaults_used(self):
        """Test it works if defaults used. """
        plugin = LinearWeights(y0val=20.0, ynval=2.0)
        result = plugin.process(self.cube, self.coord_name)
        expected_result = np.array([0.90909091, 0.09090909])
        self.assertArrayAlmostEqual(result.data, expected_result)

    def test_works_y0val_and_ynval_set(self):
        """Test it works if y0val and ynval set. """
        plugin = LinearWeights(y0val=10.0, ynval=5.0)
        result = plugin.process(self.cube, self.coord_name)
        expected_result = np.array([0.66666667, 0.33333333])
        self.assertArrayAlmostEqual(result.data, expected_result)

    def test_works_with_larger_num(self):
        """Test it works with larger num_of_vals. """
        plugin = LinearWeights(y0val=10.0, ynval=5.0)
        cubenew = add_coordinate(
            self.cube, np.arange(6), "realization", dtype=np.int32)
        coord = cubenew.coord('realization')
        result = plugin.process(cubenew, coord)
        expected_result = np.array([0.22222222, 0.2,
                                    0.17777778, 0.15555556,
                                    0.13333333, 0.11111111])
        self.assertArrayAlmostEqual(result.data, expected_result)


if __name__ == '__main__':
    unittest.main()
