# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the weights.ChooseDefaultWeightsLinear plugin."""

import unittest
from datetime import datetime as dt

import iris
import numpy as np
from iris.tests import IrisTest

from improver.blending.weights import WeightsUtilities
from improver.metadata.probabilistic import find_threshold_coordinate
from improver.synthetic_data.set_up_test_cubes import (
    add_coordinate,
    set_up_probability_cube,
)


class Test__repr__(IrisTest):
    """Test the repr method."""

    def test_basic(self):
        """Test that the __repr__ returns the expected string."""
        result = str(WeightsUtilities())
        msg = "<WeightsUtilities>"
        self.assertEqual(result, msg)


class Test_normalise_weights(IrisTest):
    """Test the normalise_weights function."""

    def test_basic(self):
        """Test that the function returns an array of weights."""
        weights_in = np.array([1.0, 2.0, 3.0])
        result = WeightsUtilities.normalise_weights(weights_in)
        self.assertIsInstance(result, np.ndarray)

    def test_array_sum_equals_one(self):
        """Test that the resulting weights add up to one."""
        weights_in = np.array([1.0, 2.0, 3.0])
        result = WeightsUtilities.normalise_weights(weights_in)
        self.assertAlmostEqual(result.sum(), 1.0)

    def test_fails_weight_less_than_zero(self):
        """Test it fails if weight less than zero."""
        weights_in = np.array([-1.0, 0.1])
        msg = "Weights must be positive"
        with self.assertRaisesRegex(ValueError, msg):
            WeightsUtilities.normalise_weights(weights_in)

    def test_fails_sum_equals_zero(self):
        """Test it fails if sum of input weights is zero."""
        weights_in = np.array([0.0, 0.0, 0.0])
        msg = "Sum of weights must be > 0.0"
        with self.assertRaisesRegex(ValueError, msg):
            WeightsUtilities.normalise_weights(weights_in)

    def test_returns_correct_values(self):
        """Test it returns the correct values."""
        weights_in = np.array([6.0, 3.0, 1.0])
        result = WeightsUtilities.normalise_weights(weights_in)
        expected_result = np.array([0.6, 0.3, 0.1])
        self.assertArrayAlmostEqual(result, expected_result)

    def test_returns_correct_values_2darray_axis0(self):
        """Test normalizing along the columns of the array."""
        weights_in = np.array([[6.0, 3.0, 1.0], [4.0, 1.0, 3.0]])
        result = WeightsUtilities.normalise_weights(weights_in, axis=0)
        expected_result = np.array([[0.6, 0.75, 0.25], [0.4, 0.25, 0.75]])
        self.assertArrayAlmostEqual(result, expected_result)

    def test_returns_correct_values_2darray_axis1(self):
        """Test normalizing along the rows of the array."""
        weights_in = np.array([[6.0, 3.0, 1.0], [4.0, 1.0, 3.0]])
        result = WeightsUtilities.normalise_weights(weights_in, axis=1)
        expected_result = np.array([[0.6, 0.3, 0.1], [0.5, 0.125, 0.375]])
        self.assertArrayAlmostEqual(result, expected_result)

    def test_returns_correct_values_2darray_zero_weights(self):
        """Test normalizing along the columns of the array when there are
        zeros in the input array."""
        weights_in = np.array([[6.0, 3.0, 0.0], [0.0, 1.0, 3.0]])
        result = WeightsUtilities.normalise_weights(weights_in, axis=0)
        expected_result = np.array([[1.0, 0.75, 0.0], [0.0, 0.25, 1.0]])
        self.assertArrayAlmostEqual(result, expected_result)


class Test_build_weights_cube(IrisTest):
    """Test the build_weights_cube function."""

    def setUp(self):
        """Setup for testing cube creation."""
        cube = set_up_probability_cube(
            np.full((1, 2, 2), 0.5, dtype=np.float32),
            np.array([1.0], dtype=np.float32),
            variable_name="rainfall_rate",
            threshold_units="mm h-1",
            time=dt(2015, 11, 19, 1, 30),
            frt=dt(2015, 11, 18, 22, 0),
        )

        time_points = [dt(2015, 11, 19, 0, 30), dt(2015, 11, 19, 1, 30)]
        self.cube = add_coordinate(cube, time_points, "time", is_datetime=True)
        self.cube.data[1, :, :] = 0.6

    def test_basic_weights(self):
        """Test building a cube with weights along the blending coordinate."""

        weights = np.array([0.4, 0.6])
        blending_coord = "time"

        plugin = WeightsUtilities.build_weights_cube
        result = plugin(self.cube, weights, blending_coord)

        self.assertIsInstance(result, iris.cube.Cube)
        self.assertEqual(result.name(), "weights")
        self.assertFalse(result.attributes)
        self.assertArrayEqual(result.data, weights)
        self.assertEqual(result.coords(dim_coords=True)[0].name(), blending_coord)
        self.assertEqual(len(result.coords(dim_coords=True)), 1)

    def test_aux_coord_for_blending_coord(self):
        """Test building a cube with weights along the blending coordinate,
        when the blending coordinate is an auxillary coordinate. In this case
        we expect the associated dim_coord (time) to be on the output cube as
        well as the blending coordinate (forecast_period) as an auxillary
        coordinate."""

        weights = np.array([0.4, 0.6])
        blending_coord = "forecast_period"
        plugin = WeightsUtilities.build_weights_cube
        result = plugin(self.cube, weights, blending_coord)
        self.assertIsInstance(result, iris.cube.Cube)
        self.assertEqual(result.name(), "weights")
        self.assertFalse(result.attributes)
        self.assertArrayEqual(result.data, weights)
        self.assertEqual(result.coords(dim_coords=True)[0].name(), "time")
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
        self.assertEqual(result.name(), "weights")
        self.assertFalse(result.attributes)
        self.assertArrayEqual(result.data, weights)
        self.assertEqual(result.coords(dim_coords=True)[0].name(), blending_coord)

    def test_incompatible_weights(self):
        """Test building a cube with weights that do not match the length of
        the blending coordinate."""

        weights = np.array([0.4, 0.4, 0.2])
        blending_coord = "time"
        msg = "Weights array provided is not the same size as the blending coordinate"
        plugin = WeightsUtilities.build_weights_cube
        with self.assertRaisesRegex(ValueError, msg):
            plugin(self.cube, weights, blending_coord)


if __name__ == "__main__":
    unittest.main()
