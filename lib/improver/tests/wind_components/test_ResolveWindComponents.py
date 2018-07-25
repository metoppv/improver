# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown Copyright 2017-2018 Met Office.
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
"""Unit tests for the wind_components.ResolveWindComponents plugin."""

import unittest
import numpy as np
from cf_units import Unit

import iris
from iris.tests import IrisTest
from iris.coords import DimCoord

from improver.wind_components import ResolveWindComponents


def set_up_cube(data_2d, name, unit):
    """Set up a 2D test cube of wind direction or speed"""
    x_coord = DimCoord(np.arange(data_2d.shape[1]), "projection_x_coordinate",
                       units="metres")
    y_coord = DimCoord(np.arange(data_2d.shape[0]), "projection_y_coordinate",
                       units="metres")
    cube = iris.cube.Cube(data_2d, standard_name=name, units=unit,
                          dim_coords_and_dims=[(y_coord, 0), (x_coord, 1)])
    time_unit = Unit("hours since 1970-01-01 00:00:00", "gregorian")
    t_coord = DimCoord(402292.5, "time", units=time_unit)
    cube.add_aux_coord(t_coord)
    return cube

def expand_realizations(cube, nreal):
    """Add realization dimension with nreal points by copying cube data"""
    cubelist = iris.cube.CubeList([])
    for i in range(nreal):
        newcube = cube.copy(cube.data)
        newcube.add_aux_coord(DimCoord(i, "realization", 1))
        cubelist.append(newcube)
    merged_cube = cubelist.merge_cube()
    return merged_cube


class Test__repr__(IrisTest):
    """Tests the __repr__ method"""

    def test_basic(self):
        """Tests the output string is as expected"""
        result = str(ResolveWindComponents())
        self.assertEqual(result, '<ResolveWindComponents>')


class Test_reproject_angles(IrisTest):
    """Tests the reproject_angles method"""
    pass


class Test_resolve_wind_components(IrisTest):
    """Tests the resolve_wind_components method"""
    pass


class Test_process(IrisTest):
    """Tests the process method"""

    def setUp(self):
        """Create dummy cubes for tests"""
        wind_speed_data = np.array([[6, 5, 4, 3],
                                    [8, 6, 4, 4],
                                    [12, 8, 6, 5]])
        self.wind_speed_cube = set_up_cube(
            wind_speed_data, "wind_speed", "knots")

        wind_direction_data = np.array([[138, 142, 141, 141],
                                        [141, 143, 140, 142],
                                        [142, 146, 141, 142]])
        self.wind_direction_cube = set_up_cube(
            wind_direction_data, "wind_to_direction", "degrees")

    def test_basic(self):
        """Test plugin creates two output cubes with the correct metadata"""
        ucube, vcube = ResolveWindComponents().process(
            self.wind_speed_cube, self.wind_direction_cube)
        for cube in ucube, vcube:
            self.assertIsInstance(cube, iris.cube.Cube)
            self.assertEqual(cube.units, self.wind_speed_cube.units)
        self.assertEqual(ucube.name(), "grid_eastward_wind")
        self.assertEqual(vcube.name(), "grid_northward_wind")

    def test_values(self):
        """Test plugin generates expected wind values"""
        # TODO waiting on angle reprojection function
        pass

    def test_coordinate_value_mismatch(self):
        """Test an error is raised if coordinate values are different for wind
        speed and direction cubes"""
        self.wind_direction_cube.coord(axis='y').convert_units("km")
        msg = 'Wind speed and direction cubes have unmatched coordinates'
        with self.assertRaisesRegexp(ValueError, msg):
            _, _ = ResolveWindComponents().process(
                self.wind_speed_cube, self.wind_direction_cube)

    def test_projection_mismatch(self):
        """Test an error is raised if coordinate names are different for wind
        speed and direction cubes"""
        self.wind_speed_cube.coord(axis='x').rename('longitude')
        self.wind_speed_cube.coord(axis='y').rename('latitude')
        msg = 'Wind speed and direction cubes have unmatched coordinates'
        with self.assertRaisesRegexp(ValueError, msg):
            _, _ = ResolveWindComponents().process(
                self.wind_speed_cube, self.wind_direction_cube)

    def test_multiple_realizations(self):
        """Test a cube with more than one realization is correctly processed"""
        wind_speed_3d = expand_realizations(self.wind_speed_cube, 3)
        wind_direction_3d = expand_realizations(self.wind_direction_cube, 3)
        # TODO test values: waiting on angle reprojection function


if __name__ == '__main__':
    unittest.main()
