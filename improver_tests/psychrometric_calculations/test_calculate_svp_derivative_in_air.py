# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for psychrometric_calculations calculate_svp_derivative_in_air"""

import unittest

import numpy as np
from iris.tests import IrisTest

from improver.psychrometric_calculations.psychrometric_calculations import (
    _svp_derivative_from_lookup,
    calculate_svp_derivative_in_air,
)


class Test_calculate_svp_derivative_in_air(IrisTest):
    """Test the calculate_svp_derivative_in_air function"""

    def setUp(self):
        """Set up test data"""
        self.temperature = np.array([[185.0, 260.65, 338.15]], dtype=np.float32)
        self.pressure = np.array([[1.0e5, 9.9e4, 9.8e4]], dtype=np.float32)

    def test_calculate_svp_derivative_in_air(self):
        """Test pressure-corrected SVP derivative values"""
        expected = np.array([[0.01362905, 208.47170252, 25187.76423485]])
        result = calculate_svp_derivative_in_air(self.temperature, self.pressure)
        np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)

    def test_values(self):
        """Basic extraction of SVP derivative values from lookup table"""
        self.temperature[0, 1] = 260.56833
        expected = [[1.350531e-02, 2.06000274e02, 2.501530e04]]
        result = _svp_derivative_from_lookup(self.temperature)
        np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)

    def test_beyond_table_bounds(self):
        """Extracting SVP derivative values from the table with temperatures beyond
        its valid range. Should return the nearest end of the table."""
        self.temperature[0, 0] = 150.0
        self.temperature[0, 2] = 400.0
        expected = [[9.664590e-03, 2.075279e02, 2.501530e04]]
        result = _svp_derivative_from_lookup(self.temperature)
        np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
