# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""
Unit tests for the SaturatedVapourPressureDerivativeTable utility.
"""

import unittest

import numpy as np
from cf_units import Unit
from iris.tests import IrisTest

from improver.generate_ancillaries.generate_svp_derivative_table import (
    SaturatedVapourPressureDerivativeTable,
)


class Test__init__(unittest.TestCase):
    """Test the init method"""

    def test_raise_error_if_both_flags_true(self):
        """If both the water and ice flags are true, the plugin should raise an exception."""
        with self.assertRaises(ValueError):
            SaturatedVapourPressureDerivativeTable(water_only=True, ice_only=True)


class Test__repr__(IrisTest):
    """Test the repr method."""

    def test_basic(self):
        """Test that the __repr__ returns the expected string."""
        result = str(SaturatedVapourPressureDerivativeTable())
        msg = (
            "<SaturatedVapourPressureDerivativeTable: t_min: {}; t_max: {}; "
            "t_increment: {}>".format(183.15, 338.25, 0.1)
        )
        self.assertEqual(result, msg)


class Test_process(IrisTest):
    """Test that the plugin functions as expected."""

    def test_cube_attributes(self):
        """Test that returned cube has appropriate attributes."""
        t_min, t_max, t_increment = 200.15, 220.15, 10.0
        result = SaturatedVapourPressureDerivativeTable(
            t_min=t_min, t_max=t_max, t_increment=t_increment
        ).process()
        self.assertEqual(result.attributes["minimum_temperature"], t_min)
        self.assertEqual(result.attributes["maximum_temperature"], t_max)
        self.assertEqual(result.attributes["temperature_increment"], t_increment)
        self.assertEqual(result.units, Unit("Pa/K"))

    def test_cube_values(self):
        """Test that returned cube has expected saturated vapour
        pressure derivative values."""
        # Sample 17 temperature values [183.15, 193.15, 203.15, 213.15, 223.15, 233.15, 243.15, 253.15,
        # 263.15, 273.15, 283.15, 293.15, 303.15, 313.15, 323.15, 333.15, 343.15]
        t_min, t_max, t_increment = 183.15, 338.25, 10.0
        # These temperature values should yield the following
        # saturated vapour pressure derivative values:
        expected = [
            0.00176528667,
            0.00899240573,
            0.0388911564,
            0.146099665,
            0.485748136,
            1.45173134,
            3.95110728,
            9.90069022,
            23.0549049,
            50.2876506,
            82.2186162,
            144.762205,
            243.532384,
            393.298427,
            612.272861,
            922.155215,
            1348.01875,
        ]
        result = SaturatedVapourPressureDerivativeTable(
            t_min=t_min, t_max=t_max, t_increment=t_increment
        ).process()

        self.assertArrayAlmostEqual(result.data, expected, decimal=5)

    def test_cube_values_water_only(self):
        """
        Test that returned cube has expected saturated vapour
        pressure derivative values. Table constructed with respect
        to water only.
        """
        t_min, t_max, t_increment = 183.15, 338.25, 10.0
        expected = [
            0.003516,
            0.017420,
            0.070356,
            0.241868,
            0.728414,
            1.961939,
            4.801103,
            10.809001,
            22.619253,
            44.376241,
            82.218616,
            144.762205,
            243.532384,
            393.298427,
            612.272861,
            922.155215,
            1348.018754,
        ]
        result = SaturatedVapourPressureDerivativeTable(
            t_min=t_min, t_max=t_max, t_increment=t_increment, water_only=True
        ).process()

        self.assertArrayAlmostEqual(result.data, expected, decimal=5)

    def test_cube_values_ice_only(self):
        """
        Test that returned cube has expected saturated vapour
        pressure derivative values. Table constructed with respect
        to ice only.
        """
        t_min, t_max, t_increment = 183.15, 338.25, 10.0
        expected = [
            0.001765,
            0.008992,
            0.038891,
            0.146100,
            0.485748,
            1.451731,
            3.951107,
            9.900690,
            23.054905,
            50.287651,
            103.448396,
            201.888953,
            375.710661,
            669.714555,
            1147.955795,
            1898.717510,
            3039.638789,
        ]
        result = SaturatedVapourPressureDerivativeTable(
            t_min=t_min, t_max=t_max, t_increment=t_increment, ice_only=True
        ).process()

        self.assertArrayAlmostEqual(result.data, expected, decimal=5)

    def test_coordinate_values(self):
        """Test that returned cube temperature coordinate has expected
        values."""
        t_min, t_max, t_increment = 183.15, 338.15, 10.0
        expected = np.arange(t_min, t_max, t_increment)
        result = SaturatedVapourPressureDerivativeTable(
            t_min=t_min, t_max=t_max, t_increment=t_increment
        ).process()

        self.assertArrayAlmostEqual(result.coord("air_temperature").points, expected)


if __name__ == "__main__":
    unittest.main()
