# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""
Unit tests for the function "cube_manipulation.get_filtered_ancillary_variable_names".
"""

import unittest

import numpy as np
from iris.coords import AncillaryVariable
from iris.tests import IrisTest

from improver.synthetic_data.set_up_test_cubes import set_up_variable_cube
from improver.utilities.cube_manipulation import get_ancillary_variable_names


class Test_get_ancillary_variable_names(IrisTest):
    """Test the get_ancillary_variable_names function."""

    def setUp(self):
        """Define some attributes to test with."""

        self.attributes = {
            "mosg__grid_domain": "uk",
            "mosg__grid_type": "standard",
            "mosg__grid_version": "1.2.0",
            "mosg__model_configuration": "uk_det",
        }

    def test_get_expected_ancillary_variable_names(self):
        """Test that the expected ancillary variable names are returned."""
        data1 = np.arange(25).reshape(5, 5).astype(np.float32)
        cube_with_ancillary = set_up_variable_cube(
            data1, attributes=self.attributes, spatial_grid="equalarea"
        )
        # data for an ancillary variable
        data2 = np.arange(25).reshape(5, 5).astype(np.int32)
        ancillary_variable = AncillaryVariable(
            data2, long_name="ancillary_var_to_test", units="1"
        )
        cube_with_ancillary.add_ancillary_variable(ancillary_variable, data_dims=[0, 1])
        expected_names = ["ancillary_var_to_test"]
        result = get_ancillary_variable_names(cube_with_ancillary)
        self.assertEqual(result, expected_names)

    def test_no_ancillary_variables(self):
        """Test that an empty list is returned when there are no ancillary
        variables."""
        data = np.arange(25).reshape(5, 5).astype(np.float32)
        cube_without_ancillaries = set_up_variable_cube(
            data, attributes=self.attributes, spatial_grid="equalarea"
        )
        result = get_ancillary_variable_names(cube_without_ancillaries)
        self.assertEqual(result, [])

    def test_multiple_ancillary_variables(self):
        """Test that multiple ancillary variable names are returned."""
        data1 = np.arange(25).reshape(5, 5).astype(np.float32)
        cube_with_ancillary = set_up_variable_cube(
            data1,
            attributes=self.attributes,
            spatial_grid="equalarea",
        )
        # data for an ancillary variable
        data = np.arange(25).reshape(5, 5).astype(np.int32)
        ancillary_variable_1 = AncillaryVariable(
            data, long_name="ancillary_var_to_test", units="1"
        )
        ancillary_variable_2 = AncillaryVariable(
            data, long_name="another_ancillary_var", units="1"
        )
        cube_with_ancillary.add_ancillary_variable(
            ancillary_variable_1, data_dims=[0, 1]
        )
        cube_with_ancillary.add_ancillary_variable(
            ancillary_variable_2, data_dims=[0, 1]
        )
        expected_names = ["ancillary_var_to_test", "another_ancillary_var"]
        result = get_ancillary_variable_names(cube_with_ancillary)
        self.assertEqual(result, expected_names)


if __name__ == "__main__":
    unittest.main()
