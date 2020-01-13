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

import iris
import numpy as np
from iris.tests import IrisTest

from improver.blending.weights import WeightsUtilities
from improver.metadata.probabilistic import find_threshold_coordinate

from ...metadata.test_amend import create_cube_with_threshold


class Test__repr__(IrisTest):

    """Test the repr method."""

    def test_basic(self):
        """Test that the __repr__ returns the expected string."""
        result = str(WeightsUtilities())
        msg = '<WeightsUtilities>'
        self.assertEqual(result, msg)


class Test_normalise_weights(IrisTest):
    """Test the normalise_weights function. """

    def test_basic(self):
        """Test that the function returns an array of weights. """
        weights_in = np.array([1.0, 2.0, 3.0])
        result = WeightsUtilities.normalise_weights(weights_in)
        self.assertIsInstance(result, np.ndarray)

    def test_array_sum_equals_one(self):
        """Test that the resulting weights add up to one. """
        weights_in = np.array([1.0, 2.0, 3.0])
        result = WeightsUtilities.normalise_weights(weights_in)
        self.assertAlmostEqual(result.sum(), 1.0)

    def test_fails_weight_less_than_zero(self):
        """Test it fails if weight less than zero. """
        weights_in = np.array([-1.0, 0.1])
        msg = ('Weights must be positive')
        with self.assertRaisesRegex(ValueError, msg):
            WeightsUtilities.normalise_weights(weights_in)

    def test_fails_sum_equals_zero(self):
        """Test it fails if sum of input weights is zero. """
        weights_in = np.array([0.0, 0.0, 0.0])
        msg = ('Sum of weights must be > 0.0')
        with self.assertRaisesRegex(ValueError, msg):
            WeightsUtilities.normalise_weights(weights_in)

    def test_returns_correct_values(self):
        """Test it returns the correct values. """
        weights_in = np.array([6.0, 3.0, 1.0])
        result = WeightsUtilities.normalise_weights(weights_in)
        expected_result = np.array([0.6, 0.3, 0.1])
        self.assertArrayAlmostEqual(result, expected_result)

    def test_returns_correct_values_2darray_axis0(self):
        """Test normalizing along the columns of the array."""
        weights_in = np.array([[6.0, 3.0, 1.0],
                               [4.0, 1.0, 3.0]])
        result = WeightsUtilities.normalise_weights(weights_in, axis=0)
        expected_result = np.array([[0.6, 0.75, 0.25],
                                    [0.4, 0.25, 0.75]])
        self.assertArrayAlmostEqual(result, expected_result)

    def test_returns_correct_values_2darray_axis1(self):
        """Test normalizing along the rows of the array."""
        weights_in = np.array([[6.0, 3.0, 1.0],
                               [4.0, 1.0, 3.0]])
        result = WeightsUtilities.normalise_weights(weights_in, axis=1)
        expected_result = np.array([[0.6, 0.3, 0.1],
                                    [0.5, 0.125, 0.375]])
        self.assertArrayAlmostEqual(result, expected_result)

    def test_returns_correct_values_2darray_zero_weights(self):
        """Test normalizing along the columns of the array when there are
           zeros in the input array."""
        weights_in = np.array([[6.0, 3.0, 0.0],
                               [0.0, 1.0, 3.0]])
        result = WeightsUtilities.normalise_weights(weights_in, axis=0)
        expected_result = np.array([[1.0, 0.75, 0.0],
                                    [0.0, 0.25, 1.0]])
        self.assertArrayAlmostEqual(result, expected_result)


class Test_build_weights_cube(IrisTest):
    """Test the build_weights_cube function. """

    def setUp(self):
        """Setup for testing cube creation."""
        self.cube = create_cube_with_threshold()

    def test_basic_weights(self):
        """Test building a cube with weights along the blending coordinate."""

        weights = np.array([0.4, 0.6])
        blending_coord = 'time'

        plugin = WeightsUtilities.build_weights_cube
        result = plugin(self.cube, weights, blending_coord)

        self.assertIsInstance(result, iris.cube.Cube)
        self.assertEqual(result.name(), 'weights')
        self.assertFalse(result.attributes)
        self.assertArrayEqual(result.data, weights)
        self.assertEqual(result.coords(dim_coords=True)[0].name(),
                         blending_coord)
        self.assertEqual(len(result.coords(dim_coords=True)), 1)

    def test_aux_coord_for_blending_coord(self):
        """Test building a cube with weights along the blending coordinate,
        when the blending coordinate is an auxillary coordinate. In this case
        we expect the associated dim_coord (time) to be on the output cube as
        well as the blending coordinate (forecast_period) as an auxillary
        coordinate."""

        weights = np.array([0.4, 0.6])
        blending_coord = 'forecast_period'
        plugin = WeightsUtilities.build_weights_cube
        result = plugin(self.cube, weights, blending_coord)
        self.assertIsInstance(result, iris.cube.Cube)
        self.assertEqual(result.name(), 'weights')
        self.assertFalse(result.attributes)
        self.assertArrayEqual(result.data, weights)
        self.assertEqual(result.coords(dim_coords=True)[0].name(),
                         "time")
        self.assertEqual(len(result.coords(dim_coords=True)), 1)
        coord_names = [coord.name() for coord in result.coords()]
        self.assertIn("forecast_period", coord_names)

    def test_weights_scalar_coord(self):
        """Test building a cube of weights where the blending coordinate is a
        scalar."""

        weights = [1.0]
        blending_coord = find_threshold_coordinate(self.cube).name()
        cube = iris.util.squeeze(self.cube)

        plugin = WeightsUtilities.build_weights_cube
        result = plugin(cube, weights, blending_coord)

        self.assertIsInstance(result, iris.cube.Cube)
        self.assertEqual(result.name(), 'weights')
        self.assertFalse(result.attributes)
        self.assertArrayEqual(result.data, weights)
        self.assertEqual(result.coords(dim_coords=True)[0].name(),
                         blending_coord)

    def test_incompatible_weights(self):
        """Test building a cube with weights that do not match the length of
        the blending coordinate."""

        weights = np.array([0.4, 0.4, 0.2])
        blending_coord = 'time'
        msg = ("Weights array provided is not the same size as the "
               "blending coordinate")
        plugin = WeightsUtilities.build_weights_cube
        with self.assertRaisesRegex(ValueError, msg):
            plugin(self.cube, weights, blending_coord)


if __name__ == '__main__':
    unittest.main()
