# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""
Unit tests for ConvertLocationAndScaleParameters
"""

import unittest

import numpy as np
from scipy import stats

from improver.ensemble_copula_coupling.ensemble_copula_coupling import (
    ConvertLocationAndScaleParameters as Plugin,
)


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
        expected_string = "<ConvertLocationAndScaleParameters: " "distribution: norm>"
        result = str(Plugin())
        self.assertEqual(result, expected_string)


class Test__prepare_shape_parameter_truncnorm(unittest.TestCase):
    """Test the _rescale_shape_parameters"""

    def setUp(self):
        """Set up values for testing."""
        self.location_parameter = np.array([-1, 0, 1])
        self.scale_parameter = np.array([1, 1.5, 2])

    def test_truncated_at_zero(self):
        """Test scaling shape parameters implying a truncation at zero."""
        expected = [np.array([1.0, 0, -0.5]), np.array([np.inf, np.inf, np.inf])]
        shape_parameters = [
            np.array([0, 0, 0], dtype=np.float32),
            np.array([np.inf, np.inf, np.inf], dtype=np.float32),
        ]
        result = []
        plugin = Plugin(distribution="truncnorm")
        for arr in shape_parameters:
            result.append(
                plugin._prepare_shape_parameter_truncnorm(
                    arr, self.location_parameter, self.scale_parameter
                )
            )

        np.testing.assert_array_almost_equal(result, expected)

    def test_discrete_shape_parameters(self):
        """Test scaling discrete shape parameters."""
        expected = [np.array([-3, -2.666667, -2.5]), np.array([7, 4, 2.5])]
        shape_parameters = np.array([-4, 6], dtype=np.float32)
        shape_parameters = [
            np.array([-4, -4, -4], dtype=np.float32),
            np.array([6, 6, 6], dtype=np.float32),
        ]
        result = []
        plugin = Plugin(distribution="truncnorm")
        for arr in shape_parameters:
            result.append(
                plugin._prepare_shape_parameter_truncnorm(
                    arr, self.location_parameter, self.scale_parameter
                )
            )

        np.testing.assert_array_almost_equal(result, expected)


if __name__ == "__main__":
    unittest.main()
