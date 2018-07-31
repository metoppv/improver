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
from iris.coord_systems import OSGB

from improver.wind_components import ResolveWindComponents


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


class Test_calc_true_north_offset(IrisTest):
    """Tests the calc_true_north_offset function"""

    def setUp(self):
        """Set up a target cube with OSGB projection"""
        self.plugin = ResolveWindComponents()
        wind_angle = np.zeros((3, 5), dtype=np.float32)
        self.directions = set_up_cube(
            wind_angle, "wind_to_direction", "degrees")

    def test_basic(self):
        """Test function returns correct type"""
        result = self.plugin.calc_true_north_offset(self.directions)
        self.assertIsInstance(result, np.ndarray)

    def test_values(self):
        """Test that for UK National Grid coordinates the angle adjustments
        are sensible"""
        expected_result = np.array([
            [2.65185907, 2.38818715, 2.12407738, 1.85957757, 1.59473581],
            [2.91895735, 2.62891746, 2.33833627, 2.04727270, 1.75578612],
            [3.22072244, 2.90095846, 2.58051566, 2.25946772, 1.93788889]])
        result = self.plugin.calc_true_north_offset(self.directions)


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

        """
        # with no angle adjustments...
        self.expected_u = np.array([
            [4.01478386, 3.07830715, 2.51728201, 1.88796151],
            [5.03456402, 3.61089063, 2.57115054, 2.46264577],
            [7.38793755, 4.47354412, 3.77592301, 3.07830715]],
            dtype=np.float32)
        self.expected_v = np.array([
            [-4.45886898, -3.94005394, -3.10858345, -2.33143759],
            [-6.21716690, -4.79181290, -3.06417775, -3.15204310],
            [-9.45612907, -6.63229990, -4.66287518, -3.94005394]],
            dtype=np.float32)
        """
        self.expected_u = np.array([
            [4.216783, 3.233962, 2.621484, 1.952114],
            [5.344630, 3.819064, 2.684003, 2.558066],
            [7.907538, 4.791543, 3.965243, 3.209784]], dtype=np.float32)

        self.expected_v = np.array([
            [-4.268342, -3.813330, -3.021229, -2.277993],
            [-5.952724, -4.627607, -2.965827, -3.075109],
            [-9.026119, -6.406334, -4.502983, -3.833704]], dtype=np.float32)

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
        with self.assertRaisesRegexp(ValueError, msg):
            _, _ = self.plugin.process(
                self.wind_speed_cube, self.wind_direction_cube)

    def test_projection_mismatch(self):
        """Test an error is raised if coordinate names are different for wind
        speed and direction cubes"""
        self.wind_speed_cube.coord(axis='x').rename('longitude')
        self.wind_speed_cube.coord(axis='y').rename('latitude')
        msg = 'Wind speed and direction cubes have unmatched coordinates'
        with self.assertRaisesRegexp(ValueError, msg):
            _, _ = self.plugin.process(
                self.wind_speed_cube, self.wind_direction_cube)

    def test_multiple_realizations(self):
        """Test a cube with more than one realization is correctly processed"""
        wind_speed_3d = expand_realizations(self.wind_speed_cube, 3)
        wind_direction_3d = expand_realizations(self.wind_direction_cube, 3)
        ucube, vcube = self.plugin.process(wind_speed_3d, wind_direction_3d)
        self.assertSequenceEqual(ucube.shape, (3, 3, 4))
        self.assertArrayAlmostEqual(ucube[1].data, self.expected_u)
        self.assertArrayAlmostEqual(vcube[2].data, self.expected_v)


if __name__ == '__main__':
    unittest.main()
