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
"""Unit tests for the weights.ChooseDefaultWeightsLinear plugin."""


import unittest

from iris.coords import AuxCoord
from iris.tests import IrisTest
import iris
import numpy as np

from improver.weights import ChooseDefaultWeightsLinear as LinearWeights
from improver.tests.weights.test_WeightsUtilities import (set_up_cube,
                                                          add_realizations)


class TestChooseDefaultWeightsLinear(IrisTest):
    """Test the Default Linear Weights plugin. """

    def setUp(self):
        self.cube = set_up_cube()
        self.coord = self.cube.coord("time")

    def test_basic(self):
        """Test that the plugin returns an array of weights. """
        plugin = LinearWeights()
        result = plugin.process(self.cube, self.coord)
        self.assertIsInstance(result, np.ndarray)

    def test_array_sum_equals_one(self):
        """Test that the resulting weights add up to one. """
        plugin = LinearWeights()
        result = plugin.process(self.cube, self.coord)
        self.assertAlmostEquals(result.sum(), 1.0)

    def test_fails_coord_not_in_cube(self):
        """Test it raises a Value Error if coord not in the cube. """
        coord = AuxCoord([], long_name="notset")
        plugin = LinearWeights()
        msg = ('The coord for this plugin must be '
               'an existing coordinate in the input cube')
        with self.assertRaisesRegexp(ValueError, msg):
            plugin.process(self.cube, coord)

    def test_fails_input_not_a_cube(self):
        """Test it raises a Value Error if not supplied with a cube. """
        plugin = LinearWeights()
        notacube = 0.0
        msg = ('The first argument must be an instance of '
               'iris.cube.Cube')
        with self.assertRaisesRegexp(ValueError, msg):
            plugin.process(notacube, self.coord)

    def test_fails_y0val_lessthan_zero(self):
        """Test it raises a Value Error if y0val less than zero. """
        plugin = LinearWeights(y0val=-10.0)
        msg = ('y0val must be a float > 0.0')
        with self.assertRaisesRegexp(ValueError, msg):
            plugin.process(self.cube, self.coord)

    def test_fails_ynval_and_slope_set(self):
        """Test it raises a Value Error if slope and ynval set. """
        plugin = LinearWeights(y0val=10.0, slope=-5.0, ynval=5.0)
        msg = ('Relative end point weight or slope must be set'
               ' but not both.')
        with self.assertRaisesRegexp(ValueError, msg):
            plugin.process(self.cube, self.coord)

    def test_fails_weights_negative(self):
        """Test it raises a Value Error if weights become negative. """
        plugin = LinearWeights(y0val=10.0, slope=-5.0)
        cubenew = add_realizations(self.cube, 6)
        coord = cubenew.coord('realization')
        msg = 'Weights must be positive, at least one value < 0.0'
        with self.assertRaisesRegexp(ValueError, msg):
            plugin.process(cubenew, coord)

    def test_works_scalar_coord(self):
        """Test it works if scalar coordinate. """
        coord = self.cube.coord("scalar_coord")
        plugin = LinearWeights()
        result = plugin.process(self.cube, coord)
        self.assertArrayAlmostEqual(result, np.array([1.0]))

    def test_works_defaults_used(self):
        """Test it works if defaults used. """
        plugin = LinearWeights()
        result = plugin.process(self.cube, self.coord)
        expected_result = np.array([0.90909091, 0.09090909])
        self.assertArrayAlmostEqual(result, expected_result)

    def test_works_y0val_and_slope_set(self):
        """Test it works if y0val and slope_set. """
        plugin = LinearWeights(y0val=10.0, slope=-5.0)
        result = plugin.process(self.cube, self.coord)
        expected_result = np.array([0.66666667, 0.33333333])
        self.assertArrayAlmostEqual(result, expected_result)

    def test_works_y0val_and_ynval_set(self):
        """Test it works if y0val and ynval set. """
        plugin = LinearWeights(y0val=10.0, ynval=5.0)
        result = plugin.process(self.cube, self.coord)
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
        coord = iris.coords.AuxCoord([0, 1, 2, 3, 4, 5, 6],
                                     standard_name='realization')
        result = plugin.process(cubenew, coord)
        expected_result = np.array([0.206349, 0.190476,
                                    0.174603, 0.15873,
                                    0.142857, 0.126984])
        self.assertArrayAlmostEqual(result, expected_result)


if __name__ == '__main__':
    unittest.main()
