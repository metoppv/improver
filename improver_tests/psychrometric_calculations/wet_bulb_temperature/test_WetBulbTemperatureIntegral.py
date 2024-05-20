# (C) Crown copyright, Met Office. All rights reserved.
#
# This file is part of improver and is released under a BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for psychrometric_calculations WetBulbTemperatureIntegral."""

import unittest

import iris
import numpy as np
from iris.tests import IrisTest

from improver.psychrometric_calculations.wet_bulb_temperature import (
    WetBulbTemperatureIntegral,
)
from improver.synthetic_data.set_up_test_cubes import (
    add_coordinate,
    set_up_variable_cube,
)


class Test_process(IrisTest):

    """Test the calculation of the wet bulb temperature integral from
    temperature, pressure, and relative humidity information using the
    process function. Integration is calculated in the vertical.

    The wet bulb temperature calculated at each level, and the difference in
    height between the levels are used to calculate an integrated wet bulb
    temperature. This is used to ascertain the degree of melting that initially
    frozen precipitation undergoes.
    """

    def setUp(self):
        """Set up temperature, pressure, and relative humidity cubes that
        contain multiple height levels; in this case the values of these
        diagnostics are identical on each level."""
        super().setUp()

        self.height_points = np.array([5.0, 10.0, 20.0])
        height_attribute = {"positive": "up"}

        data = np.array([[-88.15, -13.266943, 60.81063]], dtype=np.float32)
        self.wet_bulb_temperature = set_up_variable_cube(
            data, name="wet_bulb_temperature", units="Celsius"
        )
        self.wet_bulb_temperature = add_coordinate(
            self.wet_bulb_temperature,
            self.height_points,
            "height",
            coord_units="m",
            attributes=height_attribute,
        )

    def test_basic(self):
        """Test that the wet bulb temperature integral returns a cube
        with the expected name."""
        wb_temp_int = WetBulbTemperatureIntegral().process(self.wet_bulb_temperature)
        self.assertIsInstance(wb_temp_int, iris.cube.Cube)
        self.assertEqual(wb_temp_int.name(), "wet_bulb_temperature_integral")
        self.assertEqual(str(wb_temp_int.units), "K m")

    def test_model_id_attr(self):
        """Test that the wet bulb temperature integral returns a cube
        with the expected name and model_id_attr attribute."""
        self.wet_bulb_temperature.attributes["mosg__model_configuration"] = "uk_ens"
        wb_temp_int = WetBulbTemperatureIntegral(
            model_id_attr="mosg__model_configuration"
        ).process(self.wet_bulb_temperature)
        self.assertIsInstance(wb_temp_int, iris.cube.Cube)
        self.assertEqual(wb_temp_int.name(), "wet_bulb_temperature_integral")
        self.assertEqual(str(wb_temp_int.units), "K m")
        self.assertEqual(wb_temp_int.attributes["mosg__model_configuration"], "uk_ens")

    def test_data(self):
        """Test that the wet bulb temperature integral returns a cube
        containing the expected data."""
        expected_wb_int = np.array(
            [[[0.0, 0.0, 608.1063]], [[0.0, 0.0, 912.1595]]], dtype=np.float32
        )
        wb_temp_int = WetBulbTemperatureIntegral().process(self.wet_bulb_temperature)
        self.assertIsInstance(wb_temp_int, iris.cube.Cube)
        self.assertArrayAlmostEqual(wb_temp_int.data, expected_wb_int)


if __name__ == "__main__":
    unittest.main()
