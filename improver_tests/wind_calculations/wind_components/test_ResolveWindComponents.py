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
"""Unit tests for the wind_components.ResolveWindComponents plugin."""

import unittest

import iris
import numpy as np
from cf_units import Unit
from iris.coord_systems import OSGB
from iris.coords import DimCoord
from iris.tests import IrisTest

from improver.wind_calculations.wind_components import ResolveWindComponents

RAD_TO_DEG = 180./np.pi


def set_up_cube(data_2d, name, unit):
    """Set up a 2D test cube of wind direction or speed"""

    x_coord = DimCoord(np.linspace(150000, 250000, data_2d.shape[1]),
                       "projection_x_coordinate", units="metres",
                       coord_system=OSGB())
    y_coord = DimCoord(np.linspace(0, 600000, data_2d.shape[0]),
                       "projection_y_coordinate", units="metres",
                       coord_system=OSGB())
    cube = iris.cube.Cube(data_2d, standard_name=name, units=unit,
                          dim_coords_and_dims=[(y_coord, 0), (x_coord, 1)])
    time_unit = Unit("hours since 1970-01-01 00:00:00", "gregorian")
    t_coord = DimCoord(402292.5, "time", units=time_unit)
    cube.add_aux_coord(t_coord)
    return cube


def add_new_dimension(cube, npoints, name, unit):
    """Add a new dimension with npoints by copying cube data"""
    cubelist = iris.cube.CubeList([])
    for i in range(npoints):
        newcube = cube.copy(cube.data)
        newcube.add_aux_coord(DimCoord(i, name, unit))
        cubelist.append(newcube)
    merged_cube = cubelist.merge_cube()
    return merged_cube


class Test__repr__(IrisTest):
    """Tests the __repr__ method"""

    def test_basic(self):
        """Tests the output string is as expected"""
        result = str(ResolveWindComponents())
        self.assertEqual(result, '<ResolveWindComponents>')


class Test_calc_true_north_offset(IrisTest):
    """Tests the calc_true_north_offset function"""

    def setUp(self):
        """Set up a target cube with OSGB projection"""
        wind_angle = np.zeros((3, 5), dtype=np.float32)
        self.directions = set_up_cube(
            wind_angle, "wind_to_direction", "degrees")
        self.plugin = ResolveWindComponents()

    def test_basic(self):
        """Test function returns correct type"""
        result = self.plugin.calc_true_north_offset(self.directions)
        self.assertIsInstance(result, np.ndarray)

    def test_values(self):
        """Test that for UK National Grid coordinates the angle adjustments
        are sensible"""
        expected_result = np.array([
            [2.651483, 2.386892, 2.122119, 1.857182, 1.592121],
            [2.921058, 2.629620, 2.337963, 2.046132, 1.754138],
            [3.223816, 2.902300, 2.580523, 2.258494, 1.936247]],
            dtype=np.float32)
        result = self.plugin.calc_true_north_offset(self.directions)
        self.assertArrayAlmostEqual(RAD_TO_DEG*result, expected_result)


class Test_resolve_wind_components(IrisTest):
    """Tests the resolve_wind_components method"""

    def setUp(self):
        """Set up some arrays to convert"""
        self.plugin = ResolveWindComponents()
        wind_speed = 10.*np.ones((4, 4), dtype=np.float32)
        wind_angle = np.array([[0., 30., 45., 60.],
                               [90., 120., 135., 150.],
                               [180., 210., 225., 240.],
                               [270., 300., 315., 330.]])
        self.wind_cube = set_up_cube(wind_speed, "wind_speed", "knots")
        self.directions = set_up_cube(
            wind_angle, "wind_to_direction", "degrees")
        self.adjustments = np.zeros((4, 4), dtype=np.float32)

    def test_basic(self):
        """Test function returns correct type"""
        uspeed, vspeed = self.plugin.resolve_wind_components(
            self.wind_cube, self.directions, self.adjustments)
        self.assertIsInstance(uspeed, iris.cube.Cube)
        self.assertIsInstance(vspeed, iris.cube.Cube)

    def test_values(self):
        """Test correct values are returned for well-behaved angles"""
        expected_uspeed = 5.*np.array(
            [[0., 1., np.sqrt(2.), np.sqrt(3.)],
             [2., np.sqrt(3.), np.sqrt(2.), 1.],
             [0., -1., -np.sqrt(2.), -np.sqrt(3.)],
             [-2., -np.sqrt(3.), -np.sqrt(2.), -1.]])

        expected_vspeed = 5*np.array(
            [[2., np.sqrt(3.), np.sqrt(2.), 1.],
             [0., -1., -np.sqrt(2.), -np.sqrt(3.)],
             [-2., -np.sqrt(3.), -np.sqrt(2.), -1.],
             [0., 1., np.sqrt(2.), np.sqrt(3.)]])

        uspeed, vspeed = self.plugin.resolve_wind_components(
            self.wind_cube, self.directions, self.adjustments)
        self.assertArrayAlmostEqual(uspeed.data, expected_uspeed)
        self.assertArrayAlmostEqual(vspeed.data, expected_vspeed)


class Test_process(IrisTest):
    """Tests the process method"""

    def setUp(self):
        """Create dummy cubes for tests"""
        self.plugin = ResolveWindComponents()
        wind_speed_data = np.array(
            [[6, 5, 4, 3], [8, 6, 4, 4], [12, 8, 6, 5]], dtype=np.float32)
        self.wind_speed_cube = set_up_cube(
            wind_speed_data, "wind_speed", "knots")

        wind_direction_data = np.array(
            [[138, 142, 141, 141], [141, 143, 140, 142],
             [142, 146, 141, 142]], dtype=np.float32)
        self.wind_direction_cube = set_up_cube(
            wind_direction_data, "wind_to_direction", "degrees")

        self.expected_u = np.array([
            [3.804214, 2.917800, 2.410297, 1.822455],
            [4.711193, 3.395639, 2.454748, 2.365005],
            [6.844465, 4.144803, 3.580219, 2.943424]], dtype=np.float32)

        self.expected_v = np.array([
            [-4.639823, -4.060351, -3.1922507, -2.382994],
            [-6.465651, -4.946681, -3.1581972, -3.225949],
            [-9.856638, -6.842559, -4.8147717, -4.041813]], dtype=np.float32)

    def test_basic(self):
        """Test plugin creates two output cubes with the correct metadata"""
        ucube, vcube = self.plugin.process(
            self.wind_speed_cube, self.wind_direction_cube)
        for cube in ucube, vcube:
            self.assertIsInstance(cube, iris.cube.Cube)
            self.assertEqual(cube.units, self.wind_speed_cube.units)
        self.assertEqual(ucube.name(), "grid_eastward_wind")
        self.assertEqual(vcube.name(), "grid_northward_wind")

    def test_values(self):
        """Test plugin generates expected wind values"""
        ucube, vcube = self.plugin.process(
            self.wind_speed_cube, self.wind_direction_cube)
        self.assertArrayAlmostEqual(ucube.data, self.expected_u)
        self.assertArrayAlmostEqual(vcube.data, self.expected_v)

    def test_coordinate_value_mismatch(self):
        """Test an error is raised if coordinate values are different for wind
        speed and direction cubes"""
        self.wind_direction_cube.coord(axis='y').convert_units("km")
        msg = 'Wind speed and direction cubes have unmatched coordinates'
        with self.assertRaisesRegex(ValueError, msg):
            _, _ = self.plugin.process(
                self.wind_speed_cube, self.wind_direction_cube)

    def test_projection_mismatch(self):
        """Test an error is raised if coordinate names are different for wind
        speed and direction cubes"""
        self.wind_speed_cube.coord(axis='x').rename('longitude')
        self.wind_speed_cube.coord(axis='y').rename('latitude')
        msg = 'Wind speed and direction cubes have unmatched coordinates'
        with self.assertRaisesRegex(ValueError, msg):
            _, _ = self.plugin.process(
                self.wind_speed_cube, self.wind_direction_cube)

    def test_height_levels(self):
        """Test a cube on more than one height level is correctly processed"""
        wind_speed_3d = add_new_dimension(
            self.wind_speed_cube, 3, "height", "km")
        wind_direction_3d = add_new_dimension(
            self.wind_direction_cube, 3, "height", "km")
        ucube, vcube = self.plugin.process(wind_speed_3d, wind_direction_3d)
        self.assertSequenceEqual(ucube.shape, (3, 3, 4))
        self.assertArrayAlmostEqual(ucube[1].data, self.expected_u)
        self.assertArrayAlmostEqual(vcube[2].data, self.expected_v)

    def test_wind_from_direction(self):
        """Test correct behaviour when wind direction is 'from' not 'to'.
        We do not get perfect direction inversion to the 7th decimal place here
        because we ignore imprecision in the iris rotate_winds calcuation near
        the corners of the domain, and regrid the available data linearly to
        fill the gap.  The output wind speeds (in m s-1) compare equal to the
        5th decimal place.
        """
        expected_u = -1.*self.expected_u
        expected_v = -1.*self.expected_v
        self.wind_direction_cube.rename("wind_from_direction")
        ucube, vcube = self.plugin.process(
            self.wind_speed_cube, self.wind_direction_cube)
        self.assertArrayAllClose(ucube.data, expected_u, atol=1e-5)
        self.assertArrayAllClose(vcube.data, expected_v, atol=1e-5)


if __name__ == '__main__':
    unittest.main()
