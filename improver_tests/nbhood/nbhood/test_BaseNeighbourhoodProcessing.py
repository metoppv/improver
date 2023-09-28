# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown copyright. The Met Office.
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
"""Unit tests for the nbhood.BaseNeighbourhoodProcessing plugin."""


import unittest

import numpy as np
from iris.tests import IrisTest

from improver.nbhood.nbhood import BaseNeighbourhoodProcessing as NBHood
from improver.synthetic_data.set_up_test_cubes import set_up_variable_cube


class Test__init__(IrisTest):

    """Test the __init__ method of NeighbourhoodProcessing"""

    def test_radii_varying_with_lead_time_mismatch(self):
        """
        Test that the desired error message is raised, if there is a mismatch
        between the number of radii and the number of lead times.
        """
        radii = [10000, 20000, 30000]
        lead_times = [2, 3]
        msg = "There is a mismatch in the number of radii"
        with self.assertRaisesRegex(ValueError, msg):
            NBHood(radii, lead_times=lead_times)


class Test__find_radii(IrisTest):

    """Test the internal _find_radii function is working correctly."""

    def test_basic_array_cube_lead_times_an_array(self):
        """Test _find_radii returns an array with the correct values."""
        fp_points = np.array([2, 3, 4])
        radii = [10000, 20000, 30000]
        lead_times = [1, 3, 5]
        plugin = NBHood(radii, lead_times=lead_times)
        result = plugin._find_radii(cube_lead_times=fp_points)
        expected_result = np.array([15000.0, 20000.0, 25000.0])
        self.assertIsInstance(result, np.ndarray)
        self.assertArrayAlmostEqual(result, expected_result)

    def test_interpolation(self):
        """Test that interpolation is working as expected in _find_radii."""
        fp_points = np.array([2, 3, 4])
        radii = [10000, 30000]
        lead_times = [2, 4]
        plugin = NBHood(radii=radii, lead_times=lead_times)
        result = plugin._find_radii(cube_lead_times=fp_points)
        expected_result = np.array([10000.0, 20000.0, 30000.0])
        self.assertArrayAlmostEqual(result, expected_result)


class Test_process(IrisTest):

    """Tests for the process method of NeighbourhoodProcessing."""

    RADIUS = 6300  # Gives 3 grid cells worth.

    def setUp(self):
        """Set up cube."""
        data = np.ones((16, 16), dtype=np.float32)
        data[7, 7] = 0
        self.cube = set_up_variable_cube(
            data,
            spatial_grid="equalarea",
        )

    def test_single_point_nan(self):
        """Test behaviour for a single NaN grid cell."""
        self.cube.data[6][7] = np.NAN
        msg = "NaN detected in input cube data"
        with self.assertRaisesRegex(ValueError, msg):
            NBHood(self.RADIUS)(self.cube)

    def test_correct_radii_set(self):
        """Test that the correct neighbourhood radius is set when interpolation
        is required"""

        radii = [5600, 9500]
        lead_times = [3, 5]
        expected_radius = 7550
        plugin = NBHood(radii, lead_times)
        plugin(self.cube)
        self.assertEqual(plugin.radius, expected_radius)


if __name__ == "__main__":
    unittest.main()
