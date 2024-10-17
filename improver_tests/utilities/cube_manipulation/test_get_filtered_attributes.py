# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""
Unit tests for the function "cube_manipulation.get_filtered_attributes".
"""

import unittest

import numpy as np
from iris.tests import IrisTest

from improver.synthetic_data.set_up_test_cubes import set_up_variable_cube
from improver.utilities.cube_manipulation import get_filtered_attributes


class Test_get_filtered_attributes(IrisTest):
    """Test the get_filtered_attributes function."""

    def setUp(self):
        """Use temperature cube to test with."""

        self.attributes = {
            "mosg__grid_domain": "uk",
            "mosg__grid_type": "standard",
            "mosg__grid_version": "1.2.0",
            "mosg__model_configuration": "uk_det",
        }

        data = np.arange(25).reshape(5, 5).astype(np.float32)
        self.cube = set_up_variable_cube(
            data, attributes=self.attributes, spatial_grid="equalarea"
        )

    def test_no_filter(self):
        """Test a case in which all the attributes of the cube passed in are
        returned as no filter is specified."""
        result = get_filtered_attributes(self.cube)
        self.assertEqual(result, self.cube.attributes)

    def test_all_matches(self):
        """Test a case in which the cube passed in contains attributes that
        all partially match the attribute_filter string."""
        attribute_filter = "mosg"
        result = get_filtered_attributes(self.cube, attribute_filter=attribute_filter)
        self.assertEqual(result, self.attributes)

    def test_subset_of_matches(self):
        """Test a case in which the cube passed in contains some attributes
        that partially match the attribute_filter string, and some that do
        not."""
        attribute_filter = "mosg__grid"
        expected = self.attributes
        expected.pop("mosg__model_configuration")
        result = get_filtered_attributes(self.cube, attribute_filter=attribute_filter)
        self.assertEqual(result, expected)

    def test_without_matches(self):
        """Test a case in which the cube passed in does not contain any
        attributes that partially match the expected string."""
        attribute_filter = "test"
        result = get_filtered_attributes(self.cube, attribute_filter=attribute_filter)
        self.assertFalse(result)


if __name__ == "__main__":
    unittest.main()
