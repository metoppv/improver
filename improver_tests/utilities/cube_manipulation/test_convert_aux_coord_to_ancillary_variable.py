# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""
Unit tests for the function "cube_manipulation.convert_aux_coord_to_ancillary_variable".
"""

import unittest

import iris
from iris.coords import AuxCoord, AncillaryVariable
import numpy as np
import pytest

from improver.synthetic_data.set_up_test_cubes import set_up_variable_cube
from improver.utilities.cube_manipulation import convert_aux_coord_to_ancillary_variable

class Test_convert_aux_coord_to_ancillary_variable(unittest.TestCase):
    """Test the convert_aux_coord_to_ancillary_variable utility."""
    aux_coord_name = "relative_humidity status_flag"
    ancillary_var_name = "status_flag"

    def setUp(self):
        """Set up test cubes and auxiliary coordinates for use in the tests."""
        data = 275 * np.ones((3, 3, 3), dtype=np.float32)
        flag_data = 275 * np.ones((3, 3, 3), dtype=np.int32)
        
        aux_coord = AuxCoord(
            flag_data,
            standard_name=self.aux_coord_name,
            units=None,
            var_name="flag",
        )

        self.cube_with_aux_coord = set_up_variable_cube(data)
        self.cube_with_aux_coord.add_aux_coord(aux_coord, (0,1,2))
        self.cube_without_aux_coord = set_up_variable_cube(data)


    # Test with the cube that has an auxiliary coordinate should convert it to an ancillary variable
    def test_conversion_with_aux_coord(self):
        """Test the conversion of auxiliary coordinate to ancillary variable."""
        result = convert_aux_coord_to_ancillary_variable(self.cube_with_aux_coord, self.aux_coord_name, self.ancillary_var_name)
        self.assertIsNotNone(result)
        # The result should not have an Auxiliary Coordinate
        self.assertFalse(result.coords(self.aux_coord_name))
        # The result should have an Ancillary Variable
        self.assertTrue(result.ancillary_variables(self.ancillary_var_name))


    # Test that the Ancillary Variable has the same expected data as the original Auxiliary Coordinate
    def test_ancillary_variable_data(self):
        """Test that the Ancillary Variable has the same data as the original Auxiliary Coordinate."""
        result = convert_aux_coord_to_ancillary_variable(self.cube_with_aux_coord, self.aux_coord_name, self.ancillary_var_name)
        anc_var = result.ancillary_variables(self.ancillary_var_name)[0]
        aux_coord = self.cube_with_aux_coord.coords(self.aux_coord_name)[0]
        np.testing.assert_array_equal(anc_var.data, aux_coord.points)


    # Test with the cube that has no auxiliary coordinates should fail with ValueError
    def test_conversion_with_no_aux_coord(self):
        """Test the conversion of auxiliary coordinates to ancillary variables."""
        with self.assertRaises(ValueError):
            convert_aux_coord_to_ancillary_variable(self.cube_without_aux_coord, self.aux_coord_name, self.ancillary_var_name)


if __name__ == "__main__":
    unittest.main()
