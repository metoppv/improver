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

from cf_units import Unit
from iris.coords import AuxCoord, DimCoord
from iris.cube import Cube
from iris.tests import IrisTest
import iris
import numpy as np

from improver.weights import ChooseDefaultWeightsNonLinear as NonLinearWeights
from improver.tests.weights.test_WeightsUtilities import (set_up_cube,
                                                          add_realizations)


class TestChooseDefaultWeightsNonLinear(IrisTest):
    """Test the Default non-Linear Weights plugin. """

    def setUp(self):
        self.cube = set_up_cube()
        self.coord = self.cube.coord("time")

    def test_basic(self):
        """Test that the plugin returns an array of weights. """
        plugin = NonLinearWeights()
        result = plugin.process(self.cube, self.coord)
        self.assertIsInstance(result, np.ndarray)

    def test_array_sum_equals_one(self):
        """Test that the resulting weights add up to one. """
        plugin = NonLinearWeights()
        result = plugin.process(self.cube, self.coord)
        self.assertAlmostEquals(result.sum(), 1.0)

    def test_fails_coord_not_in_cube(self):
        """Test it raises a Value Error if coord not in the cube. """
        coord = AuxCoord([], long_name="notset")
        plugin = NonLinearWeights()
        msg = ('The coord for this plugin must be '
               'an existing coordinate in the input cube')
        with self.assertRaisesRegexp(ValueError, msg):
            plugin.process(self.cube, coord)

    def test_fails_input_not_a_cube(self):
        """Test it raises a Value Error if not supplied with a cube. """
        plugin = NonLinearWeights()
        notacube = 0.0
        msg = ('The first argument must be an instance of '
               'iris.cube.Cube')
        with self.assertRaisesRegexp(ValueError, msg):
            plugin.process(notacube, self.coord)

    def test_fails_if_cval_not_valid(self):
        """Test it raises a Value Error if cval is not in range,
            cval must be greater than 0.0 and less
            than or equal to 1.0
        """
        plugin = NonLinearWeights(cval=-1.0)
        msg = ('cval must be greater than 0.0 and less '
               'than or equal to 1.0')
        with self.assertRaisesRegexp(ValueError, msg):
            plugin.process(self.cube, self.coord)
        plugin2 = NonLinearWeights(cval=1.1)
        with self.assertRaisesRegexp(ValueError, msg):
            plugin2.process(self.cube, self.coord)

    def test_works_if_scalar_coord(self):
        """Test it works if scalar coordinate. """
        coord = self.cube.coord("scalar_coord")
        plugin = NonLinearWeights()
        result = plugin.process(self.cube, coord)
        self.assertArrayAlmostEqual(result, np.array([1.0]))

    def test_works_with_default_cval(self):
        """Test it works with default cval. """
        plugin = NonLinearWeights()
        result = plugin.process(self.cube, self.coord)
        expected_result = np.array([0.54054054, 0.45945946])
        self.assertArrayAlmostEqual(result, expected_result)

    def test_works_with_cval_equal_one(self):
        """Test it works with cval = 1.0, i.e. equal weights. """
        plugin = NonLinearWeights(cval=1.0)
        result = plugin.process(self.cube, self.coord)
        expected_result = np.array([0.5, 0.5])
        self.assertArrayAlmostEqual(result, expected_result)

    def test_works_with_larger_num(self):
        """Test it works with larger num_of_vals. """
        plugin = NonLinearWeights(cval=0.5)
        cubenew = add_realizations(self.cube, 6)
        coord = cubenew.coord('realization')
        result = plugin.process(cubenew, coord)
        expected_result = np.array([0.50793651, 0.25396825,
                                    0.12698413, 0.06349206,
                                    0.03174603, 0.01587302])
        self.assertArrayAlmostEqual(result, expected_result)

    def test_works_with_missing_coord(self):
        """Test it works with missing coord """
        plugin = NonLinearWeights(cval=0.6)
        cubenew = add_realizations(self.cube, 6)
        coord = iris.coords.AuxCoord([0, 1, 2, 3, 4, 5, 6],
                                     standard_name='realization')
        result = plugin.process(cubenew, coord)
        expected_result = np.array([0.41472, 0.250112,
                                    0.151347, 0.092088,
                                    0.056533, 0.0352])
        self.assertArrayAlmostEqual(result, expected_result)


if __name__ == '__main__':
    unittest.main()
