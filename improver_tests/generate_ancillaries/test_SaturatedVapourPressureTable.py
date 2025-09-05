# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""
Unit tests for the SaturatedVapourPressureTable utility.

"""

import unittest
import warnings

import numpy as np
import pytest
from cf_units import Unit
from iris.tests import IrisTest

from improver.generate_ancillaries.generate_svp_table import (
    SaturatedVapourPressureTable,
)


class Test__init__(unittest.TestCase):
    """Test the init method"""

    def test_raise_error_if_both_flags_true(self):
        """If both the water and ice flags are true, the plugin should raise an exception."""
        with self.assertRaises(ValueError):
            SaturatedVapourPressureTable(water_only=True, ice_only=True)


class Test__repr__(IrisTest):
    """Test the repr method."""

    def test_basic(self):
        """Test that the __repr__ returns the expected string."""
        result = str(SaturatedVapourPressureTable())
        msg = (
            "<SaturatedVapourPressureTable: t_min: {}; t_max: {}; "
            "t_increment: {}>".format(183.15, 338.25, 0.1)
        )
        self.assertEqual(result, msg)


@pytest.mark.skip(
    reason="This test requires Numpy 2.3.2 to pass, which is in the IMPROVER latest environment"
)
class Test_saturation_vapour_pressure_goff_gratch(IrisTest):
    """Test calculations of the saturated vapour pressure using the Goff-Gratch
    method."""

    def test_basic(self):
        """Basic calculation of some saturated vapour pressures."""
        data = np.array([[260.0, 270.0, 280.0]], dtype=np.float32)
        plugin = SaturatedVapourPressureTable()
        result = plugin.saturation_vapour_pressure_goff_gratch(data)
        expected = np.array([[1.956417, 4.696705, 9.909414]])
        self.assertArrayAlmostEqual(result, expected)


class Test_temperature_data_limits(unittest.TestCase):
    """
    Test that a warning message is raised if the temperature input values are outside
    the range for which the method is considered valid.
    MAX_VALID_TEMPERATURE_WATER = 373.0
    MAX_VALID_TEMPERATURE_ICE = 273.15
    MIN_VALID_TEMPERATURE_WATER = 223.0
    MIN_VALID_TEMPERATURE_ICE = 173.0
    """

    def setUp(self):
        """Set up the plugins for testing."""
        self.plugins = (
            SaturatedVapourPressureTable(),
            SaturatedVapourPressureTable(water_only=True),
            SaturatedVapourPressureTable(ice_only=True),
        )
        self.messages = (
            "Temperatures out of SVP table range",
            "Temperatures out of SVP table range for water",
            "Temperatures out of SVP table range for ice",
        )

    def test_warning_on_temperature_below_min(self):
        """Test that a warning is raised if the temperature is below the minimum."""
        temps = np.array(
            [[150.0, 180.0, 200.0], [150.0, 270.0, 370.0], [150.0, 180.0, 270.0]]
        )  # One temperature for each plugin is below the minimum
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            for plugin, temperature, message in zip(self.plugins, temps, self.messages):
                plugin._check_temperature_limits(temperature)
                self.assertTrue(any(message in str(warn.message) for warn in w))

    def test_warning_on_temperature_above_max(self):
        """Test that a warning is raised if the temperature is above the maximum."""
        temps = np.array(
            [[370.0, 374.0, 380.0], [370.0, 374.0, 380.0], [270.0, 274.0, 380.0]]
        )  # Two temperatures for each plugin are above the maximum
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            for plugin, temperature, message in zip(self.plugins, temps, self.messages):
                plugin._check_temperature_limits(temperature)
                self.assertTrue(any(message in str(warn.message) for warn in w))

    def test_no_warning_on_temperature_within_bounds(self):
        """Test that no warning is raised if all temperatures are within bounds."""
        temps = np.array(
            [
                [180.0, 200.0, 300.0, 370.0],
                [230.0, 270.0, 300.0, 370.0],
                [180.0, 200.0, 230.0, 270.0],
            ]
        )  # All temperatures are within bounds
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            for plugin, temperature, message in zip(self.plugins, temps, self.messages):
                plugin._check_temperature_limits(temperature)
                self.assertFalse(any(message in str(warn.message) for warn in w))


class Test_process(IrisTest):
    """Test that the plugin functions as expected."""

    def test_cube_attributes(self):
        """Test that returned cube has appropriate attributes."""
        t_min, t_max, t_increment = 200.15, 220.15, 10.0
        result = SaturatedVapourPressureTable(
            t_min=t_min, t_max=t_max, t_increment=t_increment
        ).process()
        self.assertEqual(result.attributes["minimum_temperature"], t_min)
        self.assertEqual(result.attributes["maximum_temperature"], t_max)
        self.assertEqual(result.attributes["temperature_increment"], t_increment)
        self.assertEqual(result.units, Unit("Pa"))

    def test_cube_values(self):
        """Test that returned cube has expected values."""
        t_min, t_max, t_increment = 183.15, 338.15, 10.0
        expected = [
            0.0096646,
            0.0546844,
            0.2613554,
            1.0799927,
            3.9333663,
            12.8286096,
            37.9714586,
            103.1532749,
            259.6617372,
            610.6359361,
            1227.0888425,
            2337.0801979,
            4242.7259947,
            7377.3294046,
            12338.9996048,
            19925.4362844,
        ]
        result = SaturatedVapourPressureTable(
            t_min=t_min, t_max=t_max, t_increment=t_increment
        ).process()

        self.assertArrayAlmostEqual(result.data, expected)

    def test_cube_values_water_only(self):
        """
        Test that returned cube has expected values when the table
        is constructed with respect to water only.
        """
        t_min, t_max, t_increment = 183.15, 338.15, 10.0
        expected = [
            0.018690,
            0.107194,
            0.491913,
            1.897283,
            6.354202,
            18.909257,
            50.868046,
            125.375832,
            286.221982,
            610.695096,
            1227.088842,
            2337.080198,
            4242.725995,
            7377.329405,
            12338.999605,
            19925.436284,
        ]
        result = SaturatedVapourPressureTable(
            t_min=t_min, t_max=t_max, t_increment=t_increment, water_only=True
        ).process()

        self.assertArrayAlmostEqual(result.data, expected)

    def test_cube_values_ice_only(self):
        """
        Test that returned cube has expected values when the table
        is constructed with respect to ice only.
        """
        t_min, t_max, t_increment = 183.15, 338.15, 10.0
        expected = [
            0.009665,
            0.054684,
            0.261355,
            1.079993,
            3.933366,
            12.828610,
            37.971459,
            103.153275,
            259.661737,
            610.635936,
            1351.017491,
            2829.351643,
            5638.441933,
            10742.038523,
            19644.167698,
            34606.291988,
        ]
        result = SaturatedVapourPressureTable(
            t_min=t_min, t_max=t_max, t_increment=t_increment, ice_only=True
        ).process()

        self.assertArrayAlmostEqual(result.data, expected)

    def test_coordinate_values(self):
        """Test that returned cube temperature coordinate has expected
        values."""
        t_min, t_max, t_increment = 183.15, 338.15, 10.0
        expected = np.arange(t_min, t_max, t_increment)
        result = SaturatedVapourPressureTable(
            t_min=t_min, t_max=t_max, t_increment=t_increment
        ).process()

        self.assertArrayAlmostEqual(result.coord("air_temperature").points, expected)


if __name__ == "__main__":
    unittest.main()
