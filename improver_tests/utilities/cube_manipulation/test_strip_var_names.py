# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""
Unit tests for the function "cube_manipulation.strip_var_names".
"""

import unittest

import iris
import numpy as np

from improver.synthetic_data.set_up_test_cubes import set_up_variable_cube
from improver.utilities.cube_manipulation import strip_var_names


class Test_strip_var_names(unittest.TestCase):

    """Test the _slice_var_names utility."""

    def setUp(self):
        """Use temperature cube to test with."""
        data = 281 * np.ones((3, 3, 3), dtype=np.float32)
        self.cube = set_up_variable_cube(data)
        self.cube.var_name = "air_temperature"

    def test_basic(self):
        """Test that the utility returns an iris.cube.CubeList."""
        result = strip_var_names(self.cube)
        self.assertIsInstance(result, iris.cube.CubeList)

    def test_cube_var_name_is_none(self):
        """
        Test that the utility returns an iris.cube.Cube with a
        var_name of None.
        """
        result = strip_var_names(self.cube)
        self.assertIsNone(result[0].var_name, None)

    def test_cube_coord_var_name_is_none(self):
        """
        Test that the coordinates have var_names of None.
        """
        self.cube.coord("time").var_name = "time"
        self.cube.coord("latitude").var_name = "latitude"
        self.cube.coord("longitude").var_name = "longitude"
        result = strip_var_names(self.cube)
        for cube in result:
            for coord in cube.coords():
                self.assertIsNone(coord.var_name, None)

    def test_cubelist(self):
        """Test that the utility returns an iris.cube.CubeList."""
        cubes = iris.cube.CubeList([self.cube, self.cube])
        result = strip_var_names(cubes)
        self.assertIsInstance(result, iris.cube.CubeList)
        for cube in result:
            for coord in cube.coords():
                self.assertIsNone(coord.var_name, None)


if __name__ == "__main__":
    unittest.main()
