# (C) Crown copyright, Met Office. All rights reserved.
#
# This file is part of IMPROVER and is released under a BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for nbhood.nbhood.check_radius_against_distance."""

import unittest

import numpy as np
from iris.tests import IrisTest

from improver.nbhood.nbhood import check_radius_against_distance
from improver.synthetic_data.set_up_test_cubes import set_up_variable_cube


class Test_check_radius_against_distance(IrisTest):

    """Test check_radius_against_distance function."""

    def setUp(self):
        """Set up the cube."""
        data = np.ones((4, 4), dtype=np.float32)
        self.cube = set_up_variable_cube(data, spatial_grid="equalarea")

    def test_error(self):
        """Test correct exception raised when the distance is larger than the
        corner-to-corner distance of the domain."""
        distance = 550000.0
        msg = "Distance of 550000.0m exceeds max domain distance of "
        with self.assertRaisesRegex(ValueError, msg):
            check_radius_against_distance(self.cube, distance)

    def test_passes(self):
        """Test no exception raised when the distance is smaller than the
        corner-to-corner distance of the domain."""
        distance = 4000
        check_radius_against_distance(self.cube, distance)


if __name__ == "__main__":
    unittest.main()
