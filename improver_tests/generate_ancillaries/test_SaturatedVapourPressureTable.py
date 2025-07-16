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
from cf_units import Unit
from iris.tests import IrisTest

from improver.generate_ancillaries.generate_svp_table import (
    SaturatedVapourPressureTable,
)


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


class Test_saturation_vapour_pressure_goff_gratch(IrisTest):
    """Test calculations of the saturated vapour pressure using the Goff-Gratch
    method."""

    def test_basic(self):
        """Basic calculation of some saturated vapour pressures."""
        data = np.array([[260.0, 270.0, 280.0]], dtype=np.float32)
        plugin = SaturatedVapourPressureTable()
        result = plugin.saturation_vapour_pressure_goff_gratch(data)
        expected = np.array([[1.956417, 4.696705, 9.909414]], dtype=np.float32)
        self.assertArrayAlmostEqual(expected, result)


class Test_temperature_data_limits(unittest.TestCase):
    """
    Test that a warning message is raised if the temperature input values are outside
    the range for which the method is considered valid.
    MAX_VALID_TEMPERATURE = 373.0
    MIN_VALID_TEMPERATURE = 173.0
    """

    def setUp(self):
        """Set up the plugin for testing."""
        self.plugin = SaturatedVapourPressureTable()

    def test_warning_on_temperature_below_min(self):
        """Test that a warning is raised if the temperature is below the minimum."""
        temps = np.array([150.0, 180.0, 200.0])  # One temperature is below the minimum
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            self.plugin._check_temperature_limits(temps)
            self.assertTrue(
                any(
                    "Temperatures out of SVP table range" in str(warn.message)
                    for warn in w
                )
            )

    def test_warning_on_temperature_above_max(self):
        """Test that a warning is raised if the temperature is above the maximum."""
        temps = np.array(
            [370.0, 374.0, 380.0]
        )  # Two temperatures are above the maximum
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            self.plugin._check_temperature_limits(temps)
            self.assertTrue(
                any(
                    "Temperatures out of SVP table range" in str(warn.message)
                    for warn in w
                )
            )

    def test_no_warning_on_temperature_within_bounds(self):
        """Test that no warning is raised if all temperatures are within bounds."""
        temps = np.array(
            [180.0, 200.0, 300.0, 370.0]
        )  # All temperatures are within bounds
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            self.plugin._check_temperature_limits(temps)
            self.assertFalse(
                any(
                    "Temperatures out of SVP table range" in str(warn.message)
                    for warn in w
                )
            )


class Test_temperature_data_limits(unittest.TestCase):
    """
    Test that a warning message is raised if the temperature input values are outside
    the range for which the method is considered valid.
    MAX_VALID_TEMPERATURE = 373.0
    MIN_VALID_TEMPERATURE = 173.0
    """

    def setUp(self):
        """Set up the plugin for testing."""
        self.plugin = SaturatedVapourPressureTable()

    def test_warning_on_temperature_below_min(self):
        """Test that a warning is raised if the temperature is below the minimum."""
        temps = np.array([150.0, 180.0, 200.0])  # One temperature is below the minimum
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            self.plugin._check_temperature_limits(temps)
            self.assertTrue(
                any(
                    "Temperatures out of SVP table range" in str(warn.message)
                    for warn in w
                )
            )

    def test_warning_on_temperature_above_max(self):
        """Test that a warning is raised if the temperature is above the maximum."""
        temps = np.array(
            [370.0, 374.0, 380.0]
        )  # Two temperatures are above the maximum
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            self.plugin._check_temperature_limits(temps)
            self.assertTrue(
                any(
                    "Temperatures out of SVP table range" in str(warn.message)
                    for warn in w
                )
            )

    def test_no_warning_on_temperature_within_bounds(self):
        """Test that no warning is raised if all temperatures are within bounds."""
        temps = np.array(
            [180.0, 200.0, 300.0, 370.0]
        )  # All temperatures are within bounds
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            self.plugin._check_temperature_limits(temps)
            self.assertFalse(
                any(
                    "Temperatures out of SVP table range" in str(warn.message)
                    for warn in w
                )
            )


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
