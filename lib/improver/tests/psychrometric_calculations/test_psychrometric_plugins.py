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
"""Unit tests for the the stand alone functions
in psychrometric_calculations.py"""

from iris.cube import Cube
from iris.tests import IrisTest
from iris.coords import DimCoord
import numpy as np
from cf_units import Unit


from improver.psychrometric_calculations.psychrometric_plugins import (
    WetBulb)


def set_up_cube(data, phenomenon_standard_name, phenomenon_units,
                realizations=np.array([0, 1, 2]),
                timesteps=np.array([402192.5, 402193.5, 402194.5]),
                y_dimension_values=np.array([0., 2000., 4000.]),
                x_dimension_values=np.array([0., 2000., 4000.]),
                z_dimension_values=np.array([0., 2000., 4000.])):
    """Create a cube containing the required realizations, timesteps, heights
    y-dimension values and x-dimension values."""
    cube = Cube(data, standard_name=phenomenon_standard_name,
                units=phenomenon_units)
    cube.add_dim_coord(DimCoord(realizations, 'realization',
                                units='1'), 0)
    time_origin = "hours since 1970-01-01 00:00:00"
    calendar = "gregorian"
    tunit = Unit(time_origin, calendar)
    cube.add_dim_coord(DimCoord(timesteps, "time", units=tunit), 1)
    cube.add_dim_coord(DimCoord(y_dimension_values,
                                'projection_y_coordinate', units='m'), 3)
    cube.add_dim_coord(DimCoord(x_dimension_values,
                                'projection_x_coordinate', units='m'), 4)
    cube.add_dim_coord(DimCoord(z_dimension_values,
                                'height', units='m'), 2)
    return cube


class Test_wet_bulb_plugin(IrisTest):
    """Unit tests for the wet bulb temperature plugins.
    Fairly small because most scientific testing is carried out
    on the methods in psychrometric_calculations.py
    """
    def test_basic(self):
        """Test functioning of plugin on cubes of up to
        5 dimensions.
        """
        tdata = np.full((3, 3, 3, 3, 3), 280.)
        pdata = np.full((3, 3, 3, 3, 3), 1000.)
        rhdata = np.full((3, 3, 3, 3, 3), 0.8)
        temperature = set_up_cube(tdata, "air_temperature", "K")
        pressure = set_up_cube(pdata, "air_pressure", "hPa")
        rel_h = set_up_cube(rhdata, "relative_humidity", "%")
        result = WetBulb.process(temperature, rel_h, pressure)
        expectation = np.full((3, 3, 3, 3, 3), 264.52)
        self.assertArrayAlmostEqual(result.data, expectation, decimal=2)
        self.assertEqual(result.units, Unit('K'))
        self.assertEqual(result.name(), "wet_bulb_temperature")

    def test_raises_type_error(self):
        """Test that an error is raised when non cubes are passed to the
        plugin.
        """
        tdata = np.full((3, 3, 3, 3, 3), 280.)
        pdata = np.full((3, 3, 3, 3, 3), 1000.)
        rhdata = np.full((3, 3, 3, 3, 3), 0.8)
        temperature = set_up_cube(tdata, "air_temperature", "K")
        pressure = set_up_cube(pdata, "air_pressure", "hPa")
        rel_h = set_up_cube(rhdata, "relative_humidity", "%")
        emsg = "is not a cube, but"
        with self.assertRaisesRegexp(TypeError, emsg):
            WetBulb.process(temperature, rel_h, 42)
        with self.assertRaisesRegexp(TypeError, emsg):
            WetBulb.process(temperature, "Dave", pressure)
        with self.assertRaisesRegexp(TypeError, emsg):
            WetBulb.process([6, 7], rel_h, pressure)

    def test_raises_cube_shapes_unequal_error(self):
        """Test that exception is raised when cubes of differing sizes are
        passed to the cube.
        """
        tdata = np.full((3, 3, 3, 3, 3), 280.)
        pdata = np.full((3, 3, 3, 3, 3), 1000.)
        rhdata = np.full((3, 3, 3, 3, 3), 0.8)
        temperature = set_up_cube(tdata, "air_temperature", "K")
        temperature = temperature[0]
        pressure = set_up_cube(pdata, "air_pressure", "hPa")
        rel_h = set_up_cube(rhdata, "relative_humidity", "%")
        emsg = "input cubes must have the same shapes"
        with self.assertRaisesRegexp(ValueError, emsg):
            WetBulb.process(temperature, rel_h, pressure)
