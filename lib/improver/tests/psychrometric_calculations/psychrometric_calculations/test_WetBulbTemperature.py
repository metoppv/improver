# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown Copyright 2017 Met Office.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
"""Unit tests for psychrometric_calculations WetBulbTemperature"""

import unittest
import warnings
from iris.cube import Cube
from iris.tests import IrisTest
from iris.coords import DimCoord
from cf_units import Unit

from improver.psychrometric_calculations.psychrometric_calculations import (
    WetBulbTemperature)


class Test_WetBulbTemperature(IrisTest):

    """Test class for the WetBulbTemperature tests, setting up cubes."""

    def setUp(self):
        """Set up the initial conditions for tests."""

        longitude = DimCoord([0, 10, 20], 'longitude', units='degrees')
        temperature = Cube([183.15, 260.65, 338.15], 'air_temperature',
                           units='K',
                           dim_coords_and_dims=[(longitude, 0)])
        pressure = Cube([1.E5, 9.9E4, 9.8E4], 'air_pressure', units='Pa',
                        dim_coords_and_dims=[(longitude, 0)])
        relative_humidity = Cube([60, 70, 80], 'relative_humidity', units='%',
                                 dim_coords_and_dims=[(longitude, 0)])
        mixing_ratio = Cube([0.1, 0.2, 0.3], long_name='mixing_ratio',
                            units='1',
                            dim_coords_and_dims=[(longitude, 0)])

        self.temperature = temperature
        self.pressure = pressure
        self.relative_humidity = relative_humidity
        self.mixing_ratio = mixing_ratio


class Test__repr__(IrisTest):

    """Test the repr method."""

    def test_basic(self):
        """Test that the __repr__ returns the expected string."""
        result = str(WetBulbTemperature())
        msg = '<WetBulbTemperature: precision: 0.005>'
        self.assertEqual(result, msg)


class Test__check_range(Test_WetBulbTemperature):

    """Test function that checks temperatures fall within a suitable range."""

    def test_basic(self):
        """Basic test that a warning is raised if temperatures fall outside the
        allowed range."""

        with warnings.catch_warnings(record=True) as w_messages:
            WetBulbTemperature._check_range(self.temperature,
                                            270., 360.)
            assert len(w_messages) == 1
            assert issubclass(w_messages[0].category, UserWarning)
            assert "Wet bulb temperatures are" in str(w_messages[0])


class Test__lookup_svp(Test_WetBulbTemperature):

    """Test the lookup of saturated vapour pressures."""

    def test_values(self):
        """Basic extraction of some SVP values from the lookup table."""
        self.temperature.data[1] = 260.5683203
        expected = [9.664590e-03, 206., 2.501530e+04]
        result = WetBulbTemperature()._lookup_svp(self.temperature)
        self.assertArrayAlmostEqual(result.data, expected)
        self.assertEqual(result.units, Unit('Pa'))


class Test__pressure_correct_svp(Test_WetBulbTemperature):

    """Test the conversion of saturated vapour pressures in a pure water
    vapour system into SVPs in air."""

    def test_values(self):
        """Basic pressure correction of water vapour SVPs to give SVPs in
        air."""
        svp = self.pressure.copy(data=[197.41815, 474.1368, 999.5001])
        expected = [199.265984, 476.293085, 1006.390954]
        result = WetBulbTemperature()._pressure_correct_svp(
            svp, self.temperature, self.pressure)

        self.assertArrayAlmostEqual(result.data, expected)
        self.assertEqual(result.units, Unit('Pa'))


class Test__mixing_ratio(Test_WetBulbTemperature):

    """Test the calculation of the specific mixing ratio from temperature,
    and pressure information using the SVP."""

    def test_values(self):
        """Basic mixing ratio calculation."""

        expected = [6.067447e-08, 1.310793e-03, 0.1770631]
        result = WetBulbTemperature()._mixing_ratio(
            self.temperature, self.pressure)

        self.assertArrayAlmostEqual(result.data, expected)
        self.assertEqual(result.units, Unit('1'))


class Test_calculate_wet_bulb_temperature(Test_WetBulbTemperature):

    """Test the calculation of wet bulb temperatures from temperature,
    pressure, and relative humidity information."""

    def test_cube_metadata(self):
        """Check metadata of returned cube."""

        result = WetBulbTemperature().calculate_wet_bulb_temperature(
            self.temperature, self.relative_humidity, self.pressure)

        self.assertIsInstance(result, Cube)
        self.assertEqual(result.units, Unit('K'))
        self.assertEqual(result.name(), 'wet_bulb_temperature')

    def test_values(self):
        """Basic wet bulb temperature calculation."""

        expected = [183.15, 259.883055, 333.960651]
        result = WetBulbTemperature().calculate_wet_bulb_temperature(
            self.temperature, self.relative_humidity, self.pressure)

        self.assertArrayAlmostEqual(result.data, expected)
        self.assertEqual(result.units, Unit('K'))

    def test_different_units(self):
        """Basic wet bulb temperature calculation with a unit conversion
        required."""

        self.temperature.convert_units('celsius')
        self.relative_humidity.convert_units('1')
        self.pressure.convert_units('kPa')

        expected = [183.15, 259.883055, 333.960651]
        result = WetBulbTemperature().calculate_wet_bulb_temperature(
            self.temperature, self.relative_humidity, self.pressure)

        self.assertArrayAlmostEqual(result.data, expected)
        self.assertEqual(result.units, Unit('K'))


if __name__ == '__main__':
    unittest.main()
