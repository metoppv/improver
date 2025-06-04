# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
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
        self.cube = set_up_variable_cube(data, spatial_grid="equalarea")

    def test_single_point_nan(self):
        """Test behaviour for a single NaN grid cell."""
        self.cube.data[6][7] = np.nan
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
