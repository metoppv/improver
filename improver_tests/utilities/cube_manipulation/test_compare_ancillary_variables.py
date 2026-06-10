# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""
Unit tests for the function "cube_manipulation.compare_ancillary_variables".
"""

import unittest

import iris
import numpy as np
import pytest
from iris.coords import AncillaryVariable, DimCoord

from improver.synthetic_data.set_up_test_cubes import set_up_variable_cube
from improver.utilities.cube_manipulation import compare_ancillary_variables


class Test_compare_ancillary_variables(unittest.TestCase):
    """Test the compare_ancillary_variables utility."""

    def setUp(self):
        """Use temperature cube to test with."""
        data = 275 * np.ones((3, 3, 3), dtype=np.float32)
        ancillary_data = np.ones((3, 3, 3), dtype=np.int32)
        self.cube = set_up_variable_cube(data)
        self.extra_dim_coord = DimCoord(
            np.array([5.0], dtype=np.float32), standard_name="height", units="m"
        )
        self.extra_ancillary_variable = AncillaryVariable(
            ancillary_data, long_name="some_kind_of_mask", units="1"
        )

    def test_basic(self):
        """Test that the utility returns a list."""
        cube1 = self.cube.copy()
        cube2 = self.cube.copy()
        cubelist = iris.cube.CubeList([cube1, cube2])
        result = compare_ancillary_variables(cubelist)
        self.assertIsInstance(result, list)
        self.assertEqual(result, [{}, {}])

    def test_catch_warning(self):
        """Test warning is raised if the input is cubelist of length 1."""
        cube = self.cube.copy()
        warning_msg = "Only a single cube so no differences will be found "

        with pytest.warns(UserWarning, match=warning_msg):
            result = compare_ancillary_variables(iris.cube.CubeList([cube]))

        self.assertEqual(result, [])

    def test_first_cube_has_extra_ancillary_variable(self):
        """Test for comparing coordinate between cubes, where the first
        cube in the list has an extra ancillary variable."""
        cube1 = self.cube.copy()
        cube2 = self.cube.copy()
        cube1.add_ancillary_variable(self.extra_ancillary_variable, data_dims=[0, 1, 2])
        cubelist = iris.cube.CubeList([cube1, cube2])
        result = compare_ancillary_variables(cubelist)
        self.assertIsInstance(result, list)
        self.assertEqual(len(result[0]), 1)
        self.assertEqual(len(result[1]), 0)
        self.assertEqual(
            result[0]["some_kind_of_mask"]["ancillary_variable"],
            self.extra_ancillary_variable,
        )

    def test_second_cube_has_extra_ancillary_variable(self):
        """Test for comparing coordinate between cubes, where the second
        cube in the list has an extra ancillary variable."""
        cube1 = self.cube.copy()
        cube2 = self.cube.copy()
        cube2.add_ancillary_variable(self.extra_ancillary_variable, data_dims=[0, 1, 2])
        cubelist = iris.cube.CubeList([cube1, cube2])
        result = compare_ancillary_variables(cubelist)
        self.assertIsInstance(result, list)
        self.assertEqual(len(result[0]), 0)
        self.assertEqual(len(result[1]), 1)
        self.assertEqual(
            result[1]["some_kind_of_mask"]["ancillary_variable"],
            self.extra_ancillary_variable,
        )


if __name__ == "__main__":
    unittest.main()
