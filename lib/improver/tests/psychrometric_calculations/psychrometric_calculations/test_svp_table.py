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
"""Unit tests for saturated vapour pressure table."""

import unittest

import numpy as np
from iris.tests import IrisTest

from improver.psychrometric_calculations import svp_table
from improver.utilities.ancillary_creation import SaturatedVapourPressureTable


class Test_svp_table(IrisTest):

    """Test that the values in the static svp_table are as expected,
    agreeing with values produced by the creation plugin. Does this piecewise
    to avoid the testing being too slow."""

    @staticmethod
    def check_svp_table(t_min, t_max, t_increment, expected):
        """Recreate part of table and compare with expected values."""
        result = SaturatedVapourPressureTable(
            t_min=t_min, t_max=t_max, t_increment=t_increment).process()
        np.testing.assert_allclose(result.data, expected, rtol=1.e-5)

    def test_cube_values_bottom(self):
        """Test the lower end of the SVP table"""
        t_min, t_max, t_increment = (svp_table.T_MIN, 185.15,
                                     svp_table.T_INCREMENT)
        expected = svp_table.DATA[0:21]
        self.check_svp_table(t_min, t_max, t_increment, expected)

    def test_cube_values_middle(self):
        """Test the middle of the SVP table"""
        t_min, t_max, t_increment = 273.15, 275.15, svp_table.T_INCREMENT
        expected = svp_table.DATA[900:921]
        self.check_svp_table(t_min, t_max, t_increment, expected)

    def test_cube_values_top(self):
        """Test the upper end of the SVP table"""
        t_min, t_max, t_increment = (336.15, svp_table.T_MAX,
                                     svp_table.T_INCREMENT)
        expected = svp_table.DATA[1530:]
        self.check_svp_table(t_min, t_max, t_increment, expected)


if __name__ == '__main__':
    unittest.main()
