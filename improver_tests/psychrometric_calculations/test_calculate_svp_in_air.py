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
"""Unit tests for psychrometric_calculations calculate_svp_in_air"""

import unittest

import numpy as np
from iris.tests import IrisTest

from improver.psychrometric_calculations.psychrometric_calculations import (
    _svp_from_lookup, calculate_svp_in_air)


class Test_calculate_svp_in_air(IrisTest):
    """Test the calculate_svp_in_air function"""

    def setUp(self):
        """Set up test data"""
        self.temperature = np.array(
            [[185.0, 260.65, 338.15]], dtype=np.float32)
        self.pressure = np.array([[1.E5, 9.9E4, 9.8E4]], dtype=np.float32)

    def test_calculate_svp_in_air(self):
        """Test pressure-corrected SVP values"""
        expected = np.array([[0.01362905, 208.47170252, 25187.76423485]])
        result = calculate_svp_in_air(self.temperature, self.pressure)
        self.assertArrayAlmostEqual(result, expected)

    def test_values(self):
        """Basic extraction of SVP values from lookup table"""
        self.temperature[0, 1] = 260.56833
        expected = [[1.350531e-02, 2.06000274e+02, 2.501530e+04]]
        result = _svp_from_lookup(self.temperature)
        self.assertArrayAlmostEqual(result, expected)

    def test_beyond_table_bounds(self):
        """Extracting SVP values from the table with temperatures beyond
        its valid range. Should return the nearest end of the table."""
        self.temperature[0, 0] = 150.
        self.temperature[0, 2] = 400.
        expected = [[9.664590e-03, 2.075279e+02, 2.501530e+04]]
        result = _svp_from_lookup(self.temperature)
        self.assertArrayAlmostEqual(result, expected)


if __name__ == '__main__':
    unittest.main()
