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
"""Unit tests for the windgust_diagnostic.WindGustDiagnostic plugin."""


import iris
from iris.tests import IrisTest
from iris.cube import Cube
from cf_units import Unit
from iris.coords import DimCoord
from iris.exceptions import CoordinateNotFoundError
import numpy as np

from improver.wind_gust_diagnostic import WindGustDiagnostic


class Test__init__(IrisTest):

    """Test the __init__ method."""

    def test_basic(self):
        """Test that the __init__ sets things up correctly"""
        plugin = (WindGustDiagnostic(50.0, 95.0))
        self.assertEqual(plugin.percentile_gust, 50.0)
        self.assertEqual(plugin.percentile_windspeed, 95.0)


class Test__repr__(IrisTest):

    """Test the repr method."""

    def test_basic(self):
        """Test that the __repr__ returns the expected string."""
        result = str(WindGustDiagnostic(50.0, 95.0))
        msg = ('<WindGustDiagnostic: wind-gust perc=50.0, '
               'wind-speed perc=95.0>')
        self.assertEqual(result, msg)


class Test_add_metadata(IrisTest):

    """Test the add_metadata method."""

    def setUp(self):
        """Create a cube with a single non-zero point."""
        data = np.zeros((2, 2, 2))
        data[0, :, :] = 1.0
        data[1, :, :] = 2.0
        cube = Cube(data, standard_name="wind_speed_of_gust",
                    units="m s^-1")
        cube.add_dim_coord(DimCoord(np.linspace(-45.0, 45.0, 2), 'latitude',
                                    units='degrees'), 1)
        cube.add_dim_coord(DimCoord(np.linspace(120, 180, 2), 'longitude',
                                    units='degrees'), 2)
        time_origin = "hours since 1970-01-01 00:00:00"
        calendar = "gregorian"
        tunit = Unit(time_origin, calendar)
        cube.add_dim_coord(DimCoord([402192.5, 402193.5],
                                    "time", units=tunit), 0)
        self.cube_wg = cube

    def test_basic(self):
        """Test that the plugin returns a Cube. """
        plugin = WindGustDiagnostic(50.0, 95.0)
        result = plugin.add_metadata(self.cube_wg)
        self.assertIsInstance(result, Cube)


class Test_process(IrisTest):

    """Test the creation of wind-gust diagnostic by the plugin."""

    def setUp(self):
        """Create a wind-speed and wind-gust cube with percentile coord."""
        data = np.zeros((1, 2, 2, 2))
        data[0, 0, :, :] = 1.0
        data[0, 1, :, :] = 2.0
        cube = Cube(data, standard_name="wind_speed",
                    units="m s^-1")
        cube.add_dim_coord(DimCoord(np.linspace(-45.0, 45.0, 2), 'latitude',
                                    units='degrees'), 2)
        cube.add_dim_coord(DimCoord(np.linspace(120, 180, 2), 'longitude',
                                    units='degrees'), 3)
        time_origin = "hours since 1970-01-01 00:00:00"
        calendar = "gregorian"
        tunit = Unit(time_origin, calendar)
        cube.add_dim_coord(DimCoord([402192.5, 402193.5],
                                    "time", units=tunit), 1)
        cube.add_dim_coord(DimCoord([50.0],
                                    long_name="percentile_over_nbhood",
                                    units="%"), 0)
        self.cube_ws = cube
        self.cube_wg = cube.copy()
        self.cube_wg.standard_name = "wind_speed_of_gust"
        self.cube_wg.data[0, 0, :, :] = 3.0
        self.cube_wg.data[0, 1, :, :] = 1.5

    def test_basic(self):
        """Test that the plugin returns a Cube. """
        plugin = WindGustDiagnostic(50.0, 95.0)
        result = plugin.process(self.cube_wg, self.cube_ws)
        self.assertIsInstance(result, Cube)


if __name__ == '__main__':
    unittest.main()
