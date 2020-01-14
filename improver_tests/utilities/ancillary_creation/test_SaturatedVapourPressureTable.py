# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown Copyright 2017-2019 Met Office.
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
"""
Unit tests for the SaturatedVapourPressureTable utility.

"""
import unittest

import numpy as np
from cf_units import Unit
from iris.tests import IrisTest

from improver.utilities.ancillary_creation import SaturatedVapourPressureTable


class Test__repr__(IrisTest):

    """Test the repr method."""

    def test_basic(self):
        """Test that the __repr__ returns the expected string."""
        result = str(SaturatedVapourPressureTable())
        msg = ('<SaturatedVapourPressureTable: t_min: {}; t_max: {}; '
               't_increment: {}>'.format(183.15, 338.25, 0.1))
        self.assertEqual(result, msg)


class Test_saturation_vapour_pressure_goff_gratch(IrisTest):

    """Test calculations of the saturated vapour pressure using the Goff-Gratch
    method."""

    def test_basic(self):
        """Basic calculation of some saturated vapour pressures."""
        data = np.array([[260., 270., 280.]], dtype=np.float32)
        plugin = SaturatedVapourPressureTable()
        result = plugin.saturation_vapour_pressure_goff_gratch(data)
        expected = 0.01*np.array([[195.6419, 469.67078, 990.9421]])
        self.assertArrayAlmostEqual(result, expected)


class Test_process(IrisTest):

    """Test that the plugin functions as expected."""

    def test_cube_attributes(self):
        """Test that returned cube has appropriate attributes."""
        t_min, t_max, t_increment = 200.15, 220.15, 10.
        result = SaturatedVapourPressureTable(
            t_min=t_min, t_max=t_max, t_increment=t_increment).process()
        self.assertEqual(result.attributes['minimum_temperature'], t_min)
        self.assertEqual(result.attributes['maximum_temperature'], t_max)
        self.assertEqual(result.attributes['temperature_increment'],
                         t_increment)
        self.assertEqual(result.units, Unit('Pa'))

    def test_cube_values(self):
        """Test that returned cube has expected values."""
        t_min, t_max, t_increment = 183.15, 338.15, 10.
        expected = [0.0096646, 0.0546844, 0.2613554, 1.0799927, 3.9333663,
                    12.8286096, 37.9714586, 103.1532749, 259.6617372,
                    610.6359361, 1227.0888425, 2337.0801979, 4242.7259947,
                    7377.3294046, 12338.9996048, 19925.4362844]
        result = SaturatedVapourPressureTable(
            t_min=t_min, t_max=t_max, t_increment=t_increment).process()

        self.assertArrayAlmostEqual(result.data, expected)

    def test_coordinate_values(self):
        """Test that returned cube temperature coordinate has expected
        values."""
        t_min, t_max, t_increment = 183.15, 338.15, 10.
        expected = np.arange(t_min, t_max, t_increment)
        result = SaturatedVapourPressureTable(
            t_min=t_min, t_max=t_max, t_increment=t_increment).process()

        self.assertArrayAlmostEqual(result.coord('air_temperature').points,
                                    expected)


if __name__ == '__main__':
    unittest.main()
