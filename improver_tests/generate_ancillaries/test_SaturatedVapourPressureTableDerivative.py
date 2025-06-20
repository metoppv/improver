# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""
Unit tests for the SaturatedVapourPressureTableDerivative utility.
"""

import unittest

import numpy as np
from cf_units import Unit
from iris.tests import IrisTest

from improver.generate_ancillaries.generate_svp_derivative_table import (
    SaturatedVapourPressureTableDerivative,
)


class Test__repr__(IrisTest):
    """Test the repr method."""

    def test_basic(self):
        """Test that the __repr__ returns the expected string."""
        result = str(SaturatedVapourPressureTableDerivative())
        msg = (
            "<SaturatedVapourPressureTableDerivative: t_min: {}; t_max: {}; "
            "t_increment: {}>".format(183.15, 338.25, 0.1)
        )
        self.assertEqual(result, msg)

class Test_process(IrisTest):
    """Test that the plugin functions as expected."""

    def test_cube_attributes(self):
        """Test that returned cube has appropriate attributes."""
        t_min, t_max, t_increment = 200.15, 220.15, 10.0
        result = SaturatedVapourPressureTableDerivative(
            t_min=t_min, t_max=t_max, t_increment=t_increment
        ).process()
        self.assertEqual(result.attributes["minimum_temperature"], t_min)
        self.assertEqual(result.attributes["maximum_temperature"], t_max)
        self.assertEqual(result.attributes["temperature_increment"], t_increment)
        self.assertEqual(result.units, Unit("Pa/K"))

    def test_cube_values(self):
        """Test that returned cube has expected saturated vapour
        pressure derivative values."""
        t_min, t_max, t_increment = 183.15, 338.15, 10.0
        expected = [
            0.0017652866680592896,
            0.008992405731777998,
            0.038891156374921014,
            0.14609966478107075,
            0.48574813585576376,
            1.4517313404121803,
            3.9511072777595935,
            9.90069021599656,
            23.05490494630426,
            50.28765059645862,
            82.20383066377845,
            144.74821296486527,
            243.5197615309319,
            393.28752111543724,
            612.2637976764966,
            922.1479428240805,
        ]
        result = SaturatedVapourPressureTableDerivative(
            t_min=t_min, t_max=t_max, t_increment=t_increment
        ).process()

        self.assertArrayAlmostEqual(result.data, expected)

    def test_coordinate_values(self):
        """Test that returned cube temperature coordinate has expected
        values."""
        t_min, t_max, t_increment = 183.15, 338.15, 10.0
        expected = np.arange(t_min, t_max, t_increment)
        result = SaturatedVapourPressureTableDerivative(
            t_min=t_min, t_max=t_max, t_increment=t_increment
        ).process()

        self.assertArrayAlmostEqual(result.coord("air_temperature").points, expected)


if __name__ == "__main__":
    unittest.main()
