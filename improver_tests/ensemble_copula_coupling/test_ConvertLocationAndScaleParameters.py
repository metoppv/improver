# (C) Crown copyright, Met Office. All rights reserved.
#
# This file is part of IMPROVER and is released under a BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""
Unit tests for ConvertLocationAndScaleParameters
"""
import unittest

import numpy as np
from iris.tests import IrisTest
from scipy import stats

import improver.ensemble_copula_coupling._scipy_continuous_distns as scipy_cont_distns
from improver.ensemble_copula_coupling.ensemble_copula_coupling import (
    ConvertLocationAndScaleParameters as Plugin,
)


class Test__init__(IrisTest):

    """Test the __init__ method."""

    def test_valid_distribution(self):
        """Test for a valid distribution."""
        plugin = Plugin(distribution="norm")
        self.assertEqual(plugin.distribution, stats.norm)
        self.assertEqual(plugin.shape_parameters, [])

    def test_valid_distribution_with_shape_parameters(self):
        """Test for a valid distribution with shape parameters."""
        plugin = Plugin(distribution="truncnorm", shape_parameters=[0, np.inf])
        self.assertEqual(plugin.distribution, scipy_cont_distns.truncnorm)
        self.assertEqual(plugin.shape_parameters, [0, np.inf])

    def test_error_shape_parameters_required(self):
        """Test error is raised when shape parameters are needed"""
        msg = "shape parameters must be specified"
        with self.assertRaisesRegex(ValueError, msg):
            Plugin(distribution="truncnorm")

    def test_invalid_distribution(self):
        """Test for an invalid distribution."""
        msg = "The distribution requested"
        with self.assertRaisesRegex(AttributeError, msg):
            Plugin(distribution="elephant")


class Test__repr__(IrisTest):

    """Test string representation of plugin."""

    def test_basic(self):
        """Test string representation"""
        expected_string = (
            "<ConvertLocationAndScaleParameters: "
            "distribution: norm; shape_parameters: []>"
        )
        result = str(Plugin())
        self.assertEqual(result, expected_string)


class Test__rescale_shape_parameters(IrisTest):

    """Test the _rescale_shape_parameters"""

    def setUp(self):
        """Set up values for testing."""
        self.location_parameter = np.array([-1, 0, 1])
        self.scale_parameter = np.array([1, 1.5, 2])

    def test_truncated_at_zero(self):
        """Test scaling shape parameters implying a truncation at zero."""
        expected = [np.array([1.0, 0, -0.5]), np.array([np.inf, np.inf, np.inf])]
        shape_parameters = np.array([0, np.inf], dtype=np.float32)
        plugin = Plugin(distribution="truncnorm", shape_parameters=shape_parameters)
        plugin._rescale_shape_parameters(self.location_parameter, self.scale_parameter)
        self.assertArrayAlmostEqual(plugin.shape_parameters, expected)

    def test_discrete_shape_parameters(self):
        """Test scaling discrete shape parameters."""
        expected = [np.array([-3, -2.666667, -2.5]), np.array([7, 4, 2.5])]
        shape_parameters = np.array([-4, 6], dtype=np.float32)
        plugin = Plugin(distribution="truncnorm", shape_parameters=shape_parameters)
        plugin._rescale_shape_parameters(self.location_parameter, self.scale_parameter)
        self.assertArrayAlmostEqual(plugin.shape_parameters, expected)

    def test_alternative_distribution(self):
        """Test specifying a distribution other than truncated normal. In
        this instance, no rescaling is applied."""
        shape_parameters = np.array([0, np.inf], dtype=np.float32)
        plugin = Plugin(distribution="norm", shape_parameters=shape_parameters)
        plugin._rescale_shape_parameters(self.location_parameter, self.scale_parameter)
        self.assertArrayEqual(plugin.shape_parameters, shape_parameters)


if __name__ == "__main__":
    unittest.main()
