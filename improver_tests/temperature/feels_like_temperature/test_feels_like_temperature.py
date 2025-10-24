# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the feels_like_temperature module"""

import unittest

import numpy as np
from iris.tests import IrisTest

from improver.synthetic_data.set_up_test_cubes import set_up_variable_cube
from improver.temperature.feels_like_temperature import (
    _calculate_apparent_temperature,
    _calculate_wind_chill,
    calculate_feels_like_temperature,
    calculate_wind_chill_cube,
)


class Test__calculate_apparent_temperature(IrisTest):
    """Test the apparent temperature function."""

    def test_values(self):
        """Test output values from apparent temperature equation."""
        temperature = np.full((1, 3), 22.0)
        wind_speed = np.full((1, 3), 5)
        pressure = np.array([[99998.0, 101248.0, 102498.0]])
        relh = np.array([[0.0, 0.075, 0.15]])
        expected_result = np.array([[16.9300, 17.3283, 17.7267]])
        result = _calculate_apparent_temperature(
            temperature, wind_speed, relh, pressure
        )
        self.assertArrayAlmostEqual(result, expected_result, decimal=4)


class Test__calculate_wind_chill(IrisTest):
    """Test the wind chill function."""

    def test_values(self):
        """Test output values when from the wind chill equation."""
        temperature = np.full((1, 3), 1.7)
        wind_speed = np.full((1, 3), 3) * 60 * 60 / 1000.0
        expected_result = np.full((1, 3), -1.4754, dtype=np.float32)
        result = _calculate_wind_chill(temperature, wind_speed)
        self.assertArrayAlmostEqual(result, expected_result, decimal=4)


class Test_calculate_feels_like_temperature(IrisTest):
    """Test the feels like temperature function."""

    def setUp(self):
        """Create cubes to input."""
        mandatory_attributes = {
            "source": "Met Office Unified Model",
            "institution": "Met Office",
            "title": "UKV Model Forecast on UK 2 km Standard Grid",
        }

        temperature = np.array(
            [
                [
                    [226.15, 237.4, 248.65],
                    [259.9, 271.15, 282.4],
                    [293.65, 304.9, 316.15],
                ],
                [
                    [230.15, 241.4, 252.65],
                    [263.9, 275.15, 286.4],
                    [297.65, 308.9, 320.15],
                ],
                [
                    [232.15, 243.4, 254.65],
                    [265.9, 277.15, 288.4],
                    [299.65, 310.9, 322.15],
                ],
            ],
            dtype=np.float32,
        )
        self.temperature_cube = set_up_variable_cube(
            temperature,
            standard_grid_metadata="uk_det",
            attributes=mandatory_attributes,
        )

        wind_speed = np.array(
            [
                [[0.0, 7.5, 15.0], [22.5, 30.0, 37.5], [45.0, 52.5, 60.0]],
                [[2.0, 9.5, 17.0], [24.5, 32.0, 39.5], [47.0, 54.5, 62.0]],
                [[4.0, 11.5, 19.0], [26.5, 34.0, 41.5], [49.0, 56.5, 64.0]],
            ],
            dtype=np.float32,
        )
        self.wind_speed_cube = set_up_variable_cube(
            wind_speed,
            name="wind_speed",
            units="m s-1",
            standard_grid_metadata="uk_det",
            attributes=mandatory_attributes,
        )

        # create cube with metadata and values suitable for pressure.
        pressure_data = np.tile(np.linspace(100000, 110000, 9), 3).reshape(3, 3, 3)
        pressure_data[0] -= 2
        pressure_data[1] += 2
        pressure_data[2] += 4
        self.pressure_cube = set_up_variable_cube(
            pressure_data.astype(np.float32),
            name="air_pressure",
            units="Pa",
            standard_grid_metadata="uk_det",
            attributes=mandatory_attributes,
        )

        # create cube with metadata and values suitable for relative humidity.
        relative_humidity_data = np.tile(np.linspace(0, 0.6, 9), 3).reshape(3, 3, 3)
        relative_humidity_data[0] += 0
        relative_humidity_data[1] += 0.2
        relative_humidity_data[2] += 0.4
        self.relative_humidity_cube = set_up_variable_cube(
            relative_humidity_data.astype(np.float32),
            name="relative_humidity",
            units="1",
            standard_grid_metadata="uk_det",
            attributes=mandatory_attributes,
        )

    def test_temperature_less_than_10(self):
        """Test values of feels like temperature when temperature < 10
        degrees C."""
        self.temperature_cube.data = np.full((3, 3, 3), 282.15, dtype=np.float32)
        expected_result = np.array(
            [281.82837, 278.644562, 277.094147], dtype=np.float32
        )
        result = calculate_feels_like_temperature(
            self.temperature_cube,
            self.wind_speed_cube,
            self.relative_humidity_cube,
            self.pressure_cube,
        )
        self.assertArrayAlmostEqual(result[0, 0].data, expected_result)

    def test_temperature_between_10_and_20(self):
        """Test values of feels like temperature when temperature is between 10
        and 20 degress C."""

        self.temperature_cube.data = np.full((3, 3, 3), 287.15)
        expected_result = np.array(
            [286.49557, 283.217041, 280.669495], dtype=np.float32
        )
        result = calculate_feels_like_temperature(
            self.temperature_cube,
            self.wind_speed_cube,
            self.relative_humidity_cube,
            self.pressure_cube,
        )
        self.assertArrayAlmostEqual(result[0, 0].data, expected_result)

    def test_temperature_greater_than_20(self):
        """Test values of feels like temperature when temperature > 20
        degrees C."""

        self.temperature_cube.data = np.full((3, 3, 3), 294.15)
        expected_result = np.array(
            [292.290009, 287.789673, 283.289398], dtype=np.float32
        )
        result = calculate_feels_like_temperature(
            self.temperature_cube,
            self.wind_speed_cube,
            self.relative_humidity_cube,
            self.pressure_cube,
        )
        self.assertArrayAlmostEqual(result[0, 0].data, expected_result)

    def test_temperature_range_and_bounds(self):
        """Test temperature values across the full range including boundary
        temperatures 10 degrees Celsius and 20 degrees Celsius"""

        temperature_cube = self.temperature_cube[0]
        data = np.linspace(-10, 30, 9).reshape(3, 3)
        data = data + 273.15
        temperature_cube.data = data
        expected_result = np.array(
            [
                [260.32947, 260.53790283, 264.74499512],
                [270.41448975, 276.82208252, 273.33273315],
                [264.11410522, 265.66778564, 267.76950073],
            ],
            dtype=np.float32,
        )
        result = calculate_feels_like_temperature(
            temperature_cube,
            self.wind_speed_cube[0],
            self.relative_humidity_cube[0],
            self.pressure_cube[0],
        )
        self.assertArrayAlmostEqual(result.data, expected_result)

    def test_name_and_units(self):
        """Test correct outputs for name and units."""

        expected_name = "feels_like_temperature"
        expected_units = "K"
        result = calculate_feels_like_temperature(
            self.temperature_cube,
            self.wind_speed_cube,
            self.relative_humidity_cube,
            self.pressure_cube,
        )
        self.assertEqual(result.name(), expected_name)
        self.assertEqual(result.units, expected_units)

    def test_different_units(self):
        """Test that values are correct from input cubes with
        different units, and that inputs are unmodified"""

        self.temperature_cube.convert_units("fahrenheit")
        self.wind_speed_cube.convert_units("knots")
        self.relative_humidity_cube.convert_units("%")
        self.pressure_cube.convert_units("hPa")

        data = np.array(
            [
                [218.4631667, 220.76792908, 231.12779236],
                [244.45491028, 259.30001831, 275.13476562],
                [264.70050049, 274.29473877, 286.60421753],
            ]
        )
        # convert to fahrenheit
        expected_result = (data * (9.0 / 5.0) - 459.67).astype(np.float32)
        result = calculate_feels_like_temperature(
            self.temperature_cube[0],
            self.wind_speed_cube[0],
            self.relative_humidity_cube[0],
            self.pressure_cube[0],
        )
        self.assertArrayAlmostEqual(result.data, expected_result, decimal=4)
        # check inputs are unmodified
        self.assertEqual(self.temperature_cube.units, "fahrenheit")
        self.assertEqual(self.wind_speed_cube.units, "knots")
        self.assertEqual(self.relative_humidity_cube.units, "%")
        self.assertEqual(self.pressure_cube.units, "hPa")

    def test_model_id_attr(self):
        """Test model id attribute can be selectively inherited"""
        model_id_attr = "mosg__model_configuration"
        attrs = ["title", "source", "institution", model_id_attr]
        expected_attrs = {
            attr: self.temperature_cube.attributes[attr] for attr in attrs
        }
        result = calculate_feels_like_temperature(
            self.temperature_cube,
            self.wind_speed_cube,
            self.relative_humidity_cube,
            self.pressure_cube,
            model_id_attr=model_id_attr,
        )
        self.assertDictEqual(result.attributes, expected_attrs)



class Test_calculate_wind_chill_cube(IrisTest):
    """Test the cube-based wind chill wrapper function."""

    def setUp(self):
        """Create simple test cubes for temperature and wind speed."""
        mandatory_attributes = {
            "source": "Met Office Unified Model",
            "institution": "Met Office",
            "title": "UKV Model Forecast on UK 2 km Standard Grid",
        }

        temperature = np.full((3, 3), 273.15, dtype=np.float32)  # 0°C
        self.temperature_cube = set_up_variable_cube(
            temperature,
            name="air_temperature",
            units="K",
            standard_grid_metadata="uk_det",
            attributes=mandatory_attributes,
        )

        wind_speed = np.full((3, 3), 10.0, dtype=np.float32)  # 10 m/s
        self.wind_speed_cube = set_up_variable_cube(
            wind_speed,
            name="wind_speed",
            units="m s-1",
            standard_grid_metadata="uk_det",
            attributes=mandatory_attributes,
        )

    def test_basic_functionality(self):
        """Test that the function runs and returns a valid cube."""
        result = calculate_wind_chill_cube(
            self.temperature_cube, self.wind_speed_cube
        )

        # Check type and name
        self.assertIsInstance(result, type(self.temperature_cube))
        self.assertEqual(result.name(), "wind_chill_temperature")

        # Check shape and units
        self.assertEqual(result.data.shape, self.temperature_cube.data.shape)
        self.assertEqual(str(result.units), str(self.temperature_cube.units))

        # Data validity
        self.assertTrue(np.isfinite(result.data).all())

    

    

    def test_unit_conversion_correctness(self):
        """Check that unit conversions (K↔°C, m/s↔km/h) are handled correctly."""
        #  Simple one-point cubes
        temperature_data = np.array([[273.15]], dtype=np.float32)  # 0°C
        wind_speed_data = np.array([[10.0]], dtype=np.float32)     # 10 m/s = 36 km/h

        mandatory_attributes = {
            "source": "Met Office Unified Model",
            "institution": "Met Office",
            "title": "UKV Model Forecast on UK 2 km Standard Grid",
        }

        temperature = set_up_variable_cube(
            temperature_data,
            name="air_temperature",
            units="K",
            standard_grid_metadata="uk_det",
            attributes=mandatory_attributes,
        )
        wind_speed = set_up_variable_cube(
            wind_speed_data,
            name="wind_speed",
            units="m s-1",
            standard_grid_metadata="uk_det",
            attributes=mandatory_attributes,
        )

        # Expected value: run the internal NumPy function directly
        expected_celsius = _calculate_wind_chill(np.array([[0.0]]), np.array([[36.0]]))
        expected_kelvin = expected_celsius + 273.15

        #  Run the cube-based wrapper
        result = calculate_wind_chill_cube(temperature, wind_speed)

        #  Assertions
        self.assertEqual(str(result.units), "K")                # converted back to Kelvin
        self.assertArrayAlmostEqual(result.data, expected_kelvin, decimal=3)
        # sanity: wind chill cooler than actual air temp
        self.assertLess(result.data, temperature.data)

        
    def test_metadata_attributes(self):
        """Ensure the output cube preserves metadata attributes."""
        temperature_data = np.ones((3, 4), dtype=np.float32) * 280.0  # ~7°C
        wind_speed_data = np.ones((3, 4), dtype=np.float32) * 8.0     # 8 m/s

        mandatory_attributes = {
            "source": "Met Office Unified Model",
            "institution": "Met Office",
            "title": "UKV Model Forecast on UK 2 km Standard Grid",
        }

        temperature = set_up_variable_cube(
            temperature_data,
            name="air_temperature",
            units="K",
            standard_grid_metadata="uk_det",
            attributes=mandatory_attributes,
        )
        wind_speed = set_up_variable_cube(
            wind_speed_data,
            name="wind_speed",
            units="m s-1",
            standard_grid_metadata="uk_det",
            attributes=mandatory_attributes,
        )

        result = calculate_wind_chill_cube(temperature, wind_speed)


        # Attributes propagated correctly
        for key, value in mandatory_attributes.items():
            self.assertIn(key, result.attributes)
            self.assertEqual(result.attributes[key], value)

        



if __name__ == "__main__":
    unittest.main()
