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
"""
Unit tests for ConvertLocationAndScaleParameters
"""
import unittest

import numpy as np
from scipy import stats
from iris.tests import IrisTest

from improver.ensemble_copula_coupling.ensemble_copula_coupling import (
    ConvertLocationAndScaleParameters as Plugin)


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
        self.assertEqual(plugin.distribution, stats.truncnorm)
        self.assertEqual(plugin.shape_parameters, [0, np.inf])

    def test_invalid_distribution(self):
        """Test for an invalid distribution."""
        msg = "The distribution requested"
        with self.assertRaisesRegex(AttributeError, msg):
            Plugin(distribution="elephant")


class Test__repr__(IrisTest):

    """Test string representation of plugin."""

    def test_basic(self):
        """Test string representation"""
        expected_string = ("<ConvertLocationAndScaleParameters: "
                           "distribution: norm; shape_parameters: []>")
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
        expected = [np.array([1., 0, -0.5]),
                    np.array([np.inf, np.inf, np.inf])]
        shape_parameters = [0, np.inf]
        plugin = Plugin(distribution="truncnorm",
                        shape_parameters=shape_parameters)
        plugin._rescale_shape_parameters(
            self.location_parameter, self.scale_parameter)
        self.assertArrayAlmostEqual(plugin.shape_parameters, expected)

    def test_discrete_shape_parameters(self):
        """Test scaling discrete shape parameters."""
        expected = [np.array([-3, -2.666667, -2.5]), np.array([7, 4, 2.5])]
        shape_parameters = [-4, 6]
        plugin = Plugin(distribution="truncnorm",
                        shape_parameters=shape_parameters)
        plugin._rescale_shape_parameters(
            self.location_parameter, self.scale_parameter)
        self.assertArrayAlmostEqual(plugin.shape_parameters, expected)

    def test_alternative_distribution(self):
        """Test specifying a distribution other than truncated normal. In
        this instance, no rescaling is applied."""
        shape_parameters = [0, np.inf]
        plugin = Plugin(distribution="norm",
                        shape_parameters=shape_parameters)
        plugin._rescale_shape_parameters(
            self.location_parameter, self.scale_parameter)
        self.assertArrayEqual(plugin.shape_parameters, shape_parameters)

    def test_no_shape_parameters_exception(self):
        """Test raising an exception when shape parameters are not specified
        for the truncated normal distribution."""
        plugin = Plugin(distribution="truncnorm")
        msg = "For the truncated normal distribution"
        with self.assertRaisesRegex(ValueError, msg):
            plugin._rescale_shape_parameters(
                self.location_parameter, self.scale_parameter)


if __name__ == '__main__':
    unittest.main()
