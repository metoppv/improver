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
"""Unit tests for the weights.ChooseDefaultWeightsNonLinear plugin."""


import unittest
from datetime import datetime as dt

import iris
import numpy as np
from iris.coords import AuxCoord
from iris.tests import IrisTest

from improver.blending.weights import \
    ChooseDefaultWeightsNonLinear as NonLinearWeights

from ...set_up_test_cubes import add_coordinate, set_up_variable_cube


class Test__init__(IrisTest):
    """Test the __init__ method"""
    def test_basic(self):
        """Test that cval is initialised correctly"""
        result = NonLinearWeights(0.85)
        self.assertAlmostEqual(result.cval, 0.85)

    def test_fails_cval_not_set(self):
        """Test plugin raises an error if cval is None"""
        msg = 'cval is a required argument'
        with self.assertRaisesRegex(ValueError, msg):
            NonLinearWeights(None)

    def test_fails_cval_set_wrong(self):
        """Test it fails if cval is negative or greater than 1"""
        msg = 'cval must be greater than 0.0'
        with self.assertRaisesRegex(ValueError, msg):
            NonLinearWeights(-0.1)
        with self.assertRaisesRegex(ValueError, msg):
            NonLinearWeights(1.85)


class Test_nonlinear_weights(IrisTest):
    """Test the nonlinear weights function. """

    def test_basic(self):
        """Test that the function returns an array of weights. """
        result = NonLinearWeights(0.85).nonlinear_weights(3)
        self.assertIsInstance(result, np.ndarray)

    def test_values(self):
        """Test it returns the correct values for num_of_weights 6, cval 0.6"""
        result = NonLinearWeights(0.6).nonlinear_weights(6)
        expected_result = np.array([0.41957573, 0.25174544,
                                    0.15104726, 0.09062836,
                                    0.05437701, 0.03262621])
        self.assertArrayAlmostEqual(result.data, expected_result)


class Test_process(IrisTest):
    """Test the Default non-Linear Weights plugin. """

    def setUp(self):
        """Set up test cube and coordinate"""
        cube = set_up_variable_cube(
            np.zeros((2, 2), dtype=np.float32),
            name="lwe_thickness_of_precipitation_amount", units="m",
            time=dt(2017, 1, 10, 5, 0), frt=dt(2017, 1, 10, 3, 0))
        self.cube = add_coordinate(
            cube, [dt(2017, 1, 10, 5, 0), dt(2017, 1, 10, 6, 0)],
            "time", is_datetime=True)
        self.coord_name = "time"

    def test_basic(self):
        """Test that the plugin returns an array of weights. """
        plugin = NonLinearWeights(0.85)
        result = plugin.process(self.cube, self.coord_name)
        self.assertIsInstance(result, iris.cube.Cube)

    def test_array_sum_equals_one(self):
        """Test that the resulting weights add up to one. """
        plugin = NonLinearWeights(0.85)
        result = plugin.process(self.cube, self.coord_name)
        self.assertAlmostEqual(result.data.sum(), 1.0)

    def test_fails_input_not_a_cube(self):
        """Test it raises a Value Error if not supplied with a cube. """
        plugin = NonLinearWeights(0.85)
        notacube = 0.0
        msg = ('The first argument must be an instance of '
               'iris.cube.Cube')
        with self.assertRaisesRegex(TypeError, msg):
            plugin.process(notacube, self.coord_name)

    def test_scalar_coord(self):
        """Test it works if blend coordinate is scalar. """
        self.cube.add_aux_coord(
            AuxCoord(1, long_name='scalar_coord', units='no_unit'))
        coord = self.cube.coord("scalar_coord")
        plugin = NonLinearWeights(0.85)
        result = plugin.process(self.cube, coord)
        self.assertArrayAlmostEqual(result.data, np.array([1.0]))

    def test_values(self):
        """Test weights values. """
        plugin = NonLinearWeights(cval=0.85)
        result = plugin.process(self.cube, self.coord_name)
        expected_result = np.array([0.54054054, 0.45945946])
        self.assertArrayAlmostEqual(result.data, expected_result)

    def test_values_inverse_ordering(self):
        """Test inverting the order of the input cube produces inverted weights
        order, with the cube and weights cube still matching in dimensions. """
        reference_cube = self.cube.copy()
        plugin = NonLinearWeights(cval=0.85)
        result = plugin.process(
            self.cube, self.coord_name, inverse_ordering=True)
        expected_result = np.array([0.45945946, 0.54054054])
        self.assertArrayAlmostEqual(result.data, expected_result)
        # check input cube blend coordinate order is unchanged
        self.assertArrayEqual(
            self.cube.coord(self.coord_name).points,
            reference_cube.coord(self.coord_name).points)
        # check weights cube and input cube blend coordinate orders match
        self.assertArrayEqual(
            result.coord(self.coord_name).points,
            reference_cube.coord(self.coord_name).points)

    def test_cval_equal_one(self):
        """Test it works with cval = 1.0, i.e. equal weights. """
        plugin = NonLinearWeights(cval=1.0)
        result = plugin.process(self.cube, self.coord_name)
        expected_result = np.array([0.5, 0.5])
        self.assertArrayAlmostEqual(result.data, expected_result)

    def test_larger_num(self):
        """Test it works with larger num_of_vals. """
        plugin = NonLinearWeights(cval=0.5)
        cubenew = add_coordinate(
            self.cube, np.arange(6), 'realization', dtype=np.int32)
        coord_name = 'realization'
        result = plugin.process(cubenew, coord_name)
        expected_result = np.array([0.50793651, 0.25396825,
                                    0.12698413, 0.06349206,
                                    0.03174603, 0.01587302])
        self.assertArrayAlmostEqual(result.data, expected_result)


if __name__ == '__main__':
    unittest.main()
