# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""
Unit tests for ConvertLocationAndScaleParameters
"""

import unittest

import numpy as np
from iris.cube import CubeList
from scipy import stats

from improver.ensemble_copula_coupling.ensemble_copula_coupling import (
    ConvertLocationAndScaleParameters as Plugin,
)
from improver.synthetic_data.set_up_test_cubes import set_up_spot_variable_cube


class Test__init__(unittest.TestCase):
    """Test the __init__ method."""

    def test_valid_distribution(self):
        """Test for a valid distribution."""
        plugin = Plugin(distribution="norm")
        self.assertEqual(plugin.distribution, stats.norm)

    def test_invalid_distribution(self):
        """Test for an invalid distribution."""
        msg = "The distribution requested"
        with self.assertRaisesRegex(AttributeError, msg):
            Plugin(distribution="elephant")


class Test__repr__(unittest.TestCase):
    """Test string representation of plugin."""

    def test_basic(self):
        """Test string representation"""
        expected_string = "<ConvertLocationAndScaleParameters: distribution: norm>"
        result = str(Plugin())
        self.assertEqual(result, expected_string)


class Test__prepare_shape_parameter_truncnorm(unittest.TestCase):
    """Test the _rescale_shape_parameters"""

    def setUp(self):
        """Set up values for testing."""
        location_parameter_data = np.array([-1, 0, 1], dtype=np.float32)
        scale_parameter_data = np.array([1, 1.5, 2], dtype=np.float32)

        self.location_parameter = set_up_spot_variable_cube(location_parameter_data)
        self.scale_parameter = set_up_spot_variable_cube(scale_parameter_data)

    def test_truncated_at_zero(self):
        """Test scaling shape parameters implying a truncation at zero."""
        expected = [np.array([1.0, 0, -0.5]), np.array([np.inf, np.inf, np.inf])]
        shape_parameters = CubeList()
        for arr in [
            np.array([0, 0, 0], dtype=np.float32),
            np.array([np.inf, np.inf, np.inf], dtype=np.float32),
        ]:
            shape_parameters.append(set_up_spot_variable_cube(arr))

        plugin = Plugin(distribution="truncnorm")
        plugin._prepare_shape_parameter_truncnorm(
            shape_parameters, self.location_parameter, self.scale_parameter
        )

        for res, exp in zip(shape_parameters, expected):
            np.testing.assert_array_almost_equal(res.data, exp)

    def test_discrete_shape_parameters(self):
        """Test scaling discrete shape parameters."""
        expected = [np.array([-3, -2.666667, -2.5]), np.array([7, 4, 2.5])]
        shape_parameters = CubeList()
        for arr in [
            np.array([-4, -4, -4], dtype=np.float32),
            np.array([6, 6, 6], dtype=np.float32),
        ]:
            shape_parameters.append(set_up_spot_variable_cube(arr))

        plugin = Plugin(distribution="truncnorm")
        plugin._prepare_shape_parameter_truncnorm(
            shape_parameters, self.location_parameter, self.scale_parameter
        )

        for res, exp in zip(shape_parameters, expected):
            np.testing.assert_array_almost_equal(res.data, exp)

    def test_too_few_shape_parameters(self):
        """Test error raised if too few shape parameters are provided."""
        shape_parameters = CubeList()
        for arr in [
            np.array([0, 0, 0], dtype=np.float32),
        ]:
            shape_parameters.append(set_up_spot_variable_cube(arr))

        plugin = Plugin(distribution="truncnorm")
        msg = "For the truncated normal distribution, two shape parameters are"
        with self.assertRaisesRegex(ValueError, msg):
            plugin._prepare_shape_parameter_truncnorm(
                shape_parameters, self.location_parameter, self.scale_parameter
            )

    def test_too_many_shape_parameters(self):
        """Test error raised if too many shape parameters are provided."""
        shape_parameters = CubeList()
        for arr in [
            np.array([0, 0, 0], dtype=np.float32),
            np.array([1, 1, 1], dtype=np.float32),
            np.array([np.inf, np.inf, np.inf], dtype=np.float32),
        ]:
            shape_parameters.append(set_up_spot_variable_cube(arr))

        plugin = Plugin(distribution="truncnorm")
        msg = "For the truncated normal distribution, two shape parameters are"
        with self.assertRaisesRegex(ValueError, msg):
            plugin._prepare_shape_parameter_truncnorm(
                shape_parameters, self.location_parameter, self.scale_parameter
            )


if __name__ == "__main__":
    unittest.main()
