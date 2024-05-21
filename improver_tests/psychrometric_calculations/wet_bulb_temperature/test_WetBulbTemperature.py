# (C) Crown copyright, Met Office. All rights reserved.
#
# This file is part of IMPROVER and is released under a BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for psychrometric_calculations WetBulbTemperature"""

import unittest

import iris
import numpy as np
from cf_units import Unit
from iris.cube import Cube, CubeList
from iris.tests import IrisTest

from improver.psychrometric_calculations.wet_bulb_temperature import WetBulbTemperature
from improver.synthetic_data.set_up_test_cubes import set_up_variable_cube


class Test_psychrometric_variables(IrisTest):
    """Test calculations of one-line variables: svp in air, latent heat,
    mixing ratios, etc"""

    def setUp(self):
        """Set up shared input data"""
        self.mixing_ratio = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        self.specific_heat = np.array([1089.5, 1174.0, 1258.5], dtype=np.float32)
        self.latent_heat = np.array([2531771.0, 2508371.0, 2484971.0], dtype=np.float32)
        self.temperature = np.array([185.0, 260.65, 338.15], dtype=np.float32)

    def test_calculate_specific_heat(self):
        """Test specific heat calculation"""
        expected = np.array([1089.5, 1174.0, 1258.5], dtype=np.float32)
        result = WetBulbTemperature()._calculate_specific_heat(self.mixing_ratio)
        self.assertArrayAlmostEqual(result, expected, decimal=2)

    def test_calculate_enthalpy(self):
        """Basic calculation of some enthalpies. Comparison adjusted for
        32-bit precision."""
        expected = [454734.6, 807677.3, 1171053.1]
        result = WetBulbTemperature()._calculate_enthalpy(
            self.mixing_ratio, self.specific_heat, self.latent_heat, self.temperature
        )
        self.assertArrayAlmostEqual(result, expected, decimal=1)

    def test_calculate_enthalpy_gradient(self):
        """Test calculation of enthalpy gradient with temperature. Comparison
        adjusted for 32-bit precision."""
        expected = [41662.730, 41300.594, 36356.254]
        result = WetBulbTemperature()._calculate_enthalpy_gradient(
            self.mixing_ratio, self.specific_heat, self.latent_heat, self.temperature
        )
        self.assertArrayAlmostEqual(result.data, expected, decimal=3)


class Test_WetBulbTemperature(IrisTest):
    """Class to set up inputs for WetBulbTemperature tests."""

    def setUp(self):
        """Set up test input cubes and output data values."""
        temperature = np.array([[185.0, 260.65, 273.15, 338.15]], dtype=np.float32)
        self.temperature = set_up_variable_cube(temperature)
        humidity = np.array([[60.0, 70.0, 75.0, 80.0]], dtype=np.float32)
        self.relative_humidity = set_up_variable_cube(
            humidity, name="relative_humidity", units="%"
        )
        pressure = np.array([[1.0e5, 9.9e4, 9.85e4, 9.8e4]], dtype=np.float32)
        self.pressure = set_up_variable_cube(pressure, name="air_pressure", units="Pa")
        self.expected_wbt_data = np.array(
            [[185.0, 259.88306, 271.78006, 333.96066]], dtype=np.float32
        )


class Test_create_wet_bulb_temperature_cube(Test_WetBulbTemperature):
    """Test the calculation of wet bulb temperatures from temperature,
    pressure, and relative humidity information."""

    def test_cube_metadata(self):
        """Check metadata of returned cube."""
        result = WetBulbTemperature().create_wet_bulb_temperature_cube(
            self.temperature, self.relative_humidity, self.pressure
        )
        self.assertIsInstance(result, Cube)
        self.assertEqual(result.units, Unit("K"))
        self.assertEqual(result.name(), "wet_bulb_temperature")

    def test_values(self):
        """Basic wet bulb temperature calculation."""
        result = WetBulbTemperature().create_wet_bulb_temperature_cube(
            self.temperature, self.relative_humidity, self.pressure
        )
        self.assertArrayAlmostEqual(result.data, self.expected_wbt_data, decimal=3)

    def test_different_units(self):
        """Wet bulb temperature calculation with unit conversion."""
        self.temperature.convert_units("celsius")
        self.relative_humidity.convert_units("1")
        self.pressure.convert_units("kPa")
        result = WetBulbTemperature().create_wet_bulb_temperature_cube(
            self.temperature, self.relative_humidity, self.pressure
        )
        self.assertArrayAlmostEqual(result.data, self.expected_wbt_data, decimal=3)
        self.assertEqual(result.units, Unit("K"))


class Test_process(Test_WetBulbTemperature):
    """Test the calculation of wet bulb temperatures from temperature,
    pressure, and relative humidity information using the process function."""

    @staticmethod
    def _make_multi_level(cube, time_promote=False):
        """
        Take the input cube data and duplicate it to make a two height level
        cube for testing multi-level data.

        Args:
            cube (iris.cube.Cube):
                Cube to be made multi-level.
            time_promote (bool):
                Option to promote a scalar time coordinate to a dimension.
        """
        height1 = iris.coords.DimCoord([10], "height", units="m")
        height1.attributes["positive"] = "up"
        height2 = iris.coords.DimCoord([20], "height", units="m")
        height2.attributes["positive"] = "up"
        cube1 = cube.copy()
        cube2 = cube.copy()
        cube1.add_aux_coord(height1)
        cube2.add_aux_coord(height2)
        new_cube = iris.cube.CubeList([cube1, cube2]).merge_cube()
        if time_promote:
            new_cube = iris.util.new_axis(new_cube, "time")
        return new_cube

    def test_values_single_level(self):
        """Basic wet bulb temperature calculation as if calling the
        create_wet_bulb_temperature_cube function directly with single
        level data."""
        result = WetBulbTemperature().process(
            CubeList([self.temperature, self.relative_humidity, self.pressure])
        )
        self.assertArrayAlmostEqual(result.data, self.expected_wbt_data, decimal=3)
        self.assertEqual(result.units, Unit("K"))

    def test_model_id_attr(self):
        """Basic wet bulb temperature calculation as if calling the
        create_wet_bulb_temperature_cube function directly with single
        level data with the model_id_attr attribute present."""
        for cube in [self.temperature, self.relative_humidity, self.pressure]:
            cube.attributes["mosg__model_configuration"] = "uk_ens"

        result = WetBulbTemperature(model_id_attr="mosg__model_configuration").process(
            CubeList([self.temperature, self.relative_humidity, self.pressure])
        )
        self.assertArrayAlmostEqual(result.data, self.expected_wbt_data, decimal=3)
        self.assertEqual(result.units, Unit("K"))
        self.assertEqual(result.attributes["mosg__model_configuration"], "uk_ens")

    def test_values_single_level_reorder_cubes(self):
        """Same test as test_values_single_level but the cubes are in a
        different order."""
        result = WetBulbTemperature().process(
            CubeList([self.relative_humidity, self.temperature, self.pressure])
        )
        self.assertArrayAlmostEqual(result.data, self.expected_wbt_data, decimal=3)
        self.assertEqual(result.units, Unit("K"))

    def test_values_multi_level(self):
        """Basic wet bulb temperature calculation using multi-level
        data."""
        temperature = self._make_multi_level(self.temperature)
        relative_humidity = self._make_multi_level(self.relative_humidity)
        pressure = self._make_multi_level(self.pressure)
        result = WetBulbTemperature().process(
            CubeList([temperature, relative_humidity, pressure])
        )
        self.assertArrayAlmostEqual(result.data[0], self.expected_wbt_data, decimal=3)
        self.assertArrayAlmostEqual(result.data[1], self.expected_wbt_data, decimal=3)
        self.assertEqual(result.units, Unit("K"))
        self.assertArrayEqual(result.coord("height").points, [10, 20])

    def test_different_level_types(self):
        """Check an exception is raised if trying to work with data on a mix of
        height and pressure levels."""
        temperature = self._make_multi_level(self.temperature)
        relative_humidity = self._make_multi_level(self.relative_humidity)
        pressure = self._make_multi_level(self.pressure)
        temperature.coord("height").rename("pressure")
        msg = "WetBulbTemperature: Cubes have differing"
        with self.assertRaisesRegex(ValueError, msg):
            WetBulbTemperature().process(
                CubeList([temperature, relative_humidity, pressure])
            )

    def test_cube_multi_level(self):
        """Check the cube is returned with expected coordinate order after the
        data has been sliced and reconstructed."""
        temperature = self._make_multi_level(self.temperature, time_promote=True)
        relative_humidity = self._make_multi_level(
            self.relative_humidity, time_promote=True
        )
        pressure = self._make_multi_level(self.pressure, time_promote=True)
        result = WetBulbTemperature().process(
            CubeList([temperature, relative_humidity, pressure])
        )
        self.assertEqual(result.coord_dims("time")[0], 0)
        self.assertEqual(result.coord_dims("height")[0], 1)

    def test_too_many_cubes(self):
        """Tests that an error is raised if there are too many cubes."""
        temp = self.temperature
        humid = self.relative_humidity
        pressure = self.pressure
        msg = "Expected 3"
        with self.assertRaisesRegex(ValueError, msg):
            WetBulbTemperature().process(CubeList([temp, humid, pressure, temp]))

    def test_empty_cube_list(self):
        """Tests that an error is raised if there is an empty list."""
        msg = "Expected 3"
        with self.assertRaisesRegex(ValueError, msg):
            WetBulbTemperature().process(CubeList([]))


if __name__ == "__main__":
    unittest.main()
