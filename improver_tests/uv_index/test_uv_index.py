# (C) Crown copyright, Met Office. All rights reserved.
#
# This file is part of IMPROVER and is released under a BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the uv_index function."""

import unittest

import numpy as np
from cf_units import Unit
from iris.tests import IrisTest

from improver.synthetic_data.set_up_test_cubes import set_up_variable_cube
from improver.uv_index import calculate_uv_index


class Test_uv_index(IrisTest):
    """ Tests that the uv_index plugin calculates the UV index
    correctly. """

    def setUp(self):
        """Set up cubes for downward uv flux."""
        data_down = np.full((3, 2), dtype=np.float32, fill_value=0.1)
        uv_down_name = "surface_downwelling_ultraviolet_flux_in_air"

        self.cube_uv_down = set_up_variable_cube(
            data_down, name=uv_down_name, units="W m-2"
        )
        self.cube_down_badname = set_up_variable_cube(
            data_down, name="Wrong name", units="W m-2"
        )

    def test_basic(self):
        """ Test that the a basic uv calculation works, using the
        default scaling factor. Make sure the output is a cube
        with the expected data."""
        scale_factor = 1.0
        expected = self.cube_uv_down.data.copy()
        result = calculate_uv_index(self.cube_uv_down, scale_factor)
        self.assertArrayEqual(result.data, expected)

    def test_scale_factor(self):
        """ Test the uv calculation works when changing the scale factor. Make
        sure the output is a cube with the expected data."""
        expected = np.ones_like(self.cube_uv_down.data, dtype=np.float32)
        result = calculate_uv_index(self.cube_uv_down, scale_factor=10)
        self.assertArrayEqual(result.data, expected)

    def test_metadata(self):
        """ Tests that the uv index output has the correct metadata (no units,
        and name = ultraviolet index)."""
        result = calculate_uv_index(self.cube_uv_down)
        self.assertEqual(str(result.standard_name), "ultraviolet_index")
        self.assertIsNone(result.var_name)
        self.assertIsNone(result.long_name)
        self.assertEqual((result.units), Unit("1"))

    def test_badname_down(self):
        """Tests that a ValueError is raised if the input uv down
        file has the wrong name. """
        msg = "The radiation flux in UV downward has the wrong name"
        with self.assertRaisesRegex(ValueError, msg):
            calculate_uv_index(self.cube_down_badname)

    def test_negative_input(self):
        """Tests that a ValueError is raised if the input contains
        negative values. """
        negative_data_down = np.full_like(
            self.cube_uv_down.data, dtype=np.float32, fill_value=-0.1
        )
        negative_uv_down = self.cube_uv_down.copy(data=negative_data_down)
        msg = (
            "The radiation flux in UV downward contains data "
            "that is negative or NaN. Data should be >= 0."
        )
        with self.assertRaisesRegex(ValueError, msg):
            calculate_uv_index(negative_uv_down)

    def test_nan_input(self):
        """Tests that a ValueError is raised if the input contains
        values that are not a number. """
        self.cube_uv_down.data.fill(np.nan)
        msg = (
            "The radiation flux in UV downward contains data "
            "that is negative or NaN. Data should be >= 0."
        )
        with self.assertRaisesRegex(ValueError, msg):
            calculate_uv_index(self.cube_uv_down)

    def test_unit_conversion(self):
        """Test that the units are successfully converted to
        W m-2."""
        self.cube_uv_down.convert_units("kW m-2")
        scale_factor = 1.0
        expected = np.full_like(
            self.cube_uv_down.data, dtype=np.float32, fill_value=0.1
        )
        result = calculate_uv_index(self.cube_uv_down, scale_factor)
        self.assertArrayEqual(result.data, expected)


if __name__ == "__main__":
    unittest.main()
