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
""" Unit tests for the optical_flow.AdvectField plugin """

import datetime
import numpy as np
import unittest

import iris
from iris.coords import DimCoord
from iris.exceptions import InvalidCubeError
from iris.tests import IrisTest

from improver.utilities.temporal import iris_time_to_datetime
from improver.optical_flow.optical_flow import AdvectField


def set_up_xy_velocity_cube(name, coord_points_y=None, val_units='m s-1'):
    """Set up a 3x4 cube of simple velocities (no convergence / divergence)"""
    data = np.ones(shape=(4, 3))
    coord_points_x = 0.6*np.arange(3)
    if coord_points_y is None:
        coord_points_y = 0.6*np.arange(4)
    x_coord = DimCoord(coord_points_x, 'projection_x_coordinate', units='km')
    y_coord = DimCoord(coord_points_y, 'projection_y_coordinate', units='km')
    cube = iris.cube.Cube(data, long_name=name, units=val_units,
                          dim_coords_and_dims=[(y_coord, 0), (x_coord, 1)])
    return cube


class Test__init__(IrisTest):
    """Test class initialisation"""

    def test_basic(self):
        """Test for cubes and coordinates in class instance"""
        vel_x = set_up_xy_velocity_cube("advection_velocity_x")
        vel_y = set_up_xy_velocity_cube("advection_velocity_y")
        plugin = AdvectField(vel_x, vel_y)
        self.assertEqual(plugin.x_coord.name(), "projection_x_coordinate")
        self.assertIsInstance(plugin.vel_y, iris.cube.Cube)

    def test_units(self):
        """Test velocity fields are converted to m/s"""
        vel_x = set_up_xy_velocity_cube("advection_velocity_x",
                                        val_units="km h-1")
        expected_vel_x = vel_x.data / 3.6
        plugin = AdvectField(vel_x, vel_x)
        self.assertArrayAlmostEqual(plugin.vel_x.data, expected_vel_x)

    def test_raises_grid_mismatch_error(self):
        """Test error is raised if x- and y- velocity grids are mismatched"""
        vel_x = set_up_xy_velocity_cube("advection_velocity_x")
        vel_y = set_up_xy_velocity_cube("advection_velocity_y",
                                        coord_points_y=2*np.arange(4))
        msg = "Velocity cubes on unmatched grids"
        with self.assertRaisesRegexp(InvalidCubeError, msg):
            plugin = AdvectField(vel_x, vel_y)


class Test__check_input_coords(IrisTest):
    """Test cubes with inappropriate coordinates are rejected"""

    def setUp(self):
        """Set up dummy cube and plugin instance"""
        self.valid = set_up_xy_velocity_cube("advection_velocity_x")
        self.plugin = AdvectField(self.valid, self.valid)

    def test_missing_spatial_dimension(self):
        """Test rejects cube missing y axis"""
        invalid_1d = self.valid[0]
        invalid_1d.remove_coord("projection_y_coordinate")

        with self.assertRaises(InvalidCubeError):
            self.plugin._check_input_coords(invalid_1d)

    def test_spatial_dimensions(self):
        """Test rejects cube with multiple realizations"""
        v1 = self.valid.copy()
        v1.add_aux_coord(DimCoord(1, standard_name="realization"))
        v2 = self.valid.copy()
        v2.add_aux_coord(DimCoord(2, standard_name="realization"))
        invalid_3d, = (iris.cube.CubeList([v1, v2])).merge()
        msg = "Cube has 3"
        with self.assertRaisesRegexp(InvalidCubeError, msg):
            self.plugin._check_input_coords(invalid_3d)

    def test_time(self):
        """Test rejects cube without time coord"""
        msg = "Input cube has no time coordinate"
        with self.assertRaisesRegexp(InvalidCubeError, msg):
            self.plugin._check_input_coords(self.valid, require_time=True)


class Test__advect_field(IrisTest):
    """Test dimensionless gridded data is correctly advected"""

    def setUp(self):
        """Set up dimensionless velocity arrays and gridded data"""
        vel_x = set_up_xy_velocity_cube("advection_velocity_x")
        vel_y = vel_x.copy(data=2.*np.ones(shape=(4, 3)))
        self.dummy_plugin = AdvectField(vel_x, vel_y)

        self.grid_vel_x = 0.5*vel_x.data
        self.grid_vel_y = 0.5*vel_y.data
        self.data = np.array([[2., 3., 4.],
                              [1., 2., 3.],
                              [0., 1., 2.],
                              [0., 0., 1.]])

    def test_basic(self):
        """Test data is advected correctly (by 1 and 2 grid points along the x
        and y axes respectively)"""
        expected_output = np.array([[0., 0., 0.],
                                    [0., 0., 0.],
                                    [0., 2., 3.],
                                    [0., 1., 2.]])
        result = self.dummy_plugin._advect_field(
            self.data, self.grid_vel_x, self.grid_vel_y, 2., 0.)
        self.assertIsInstance(result, np.ndarray)
        self.assertArrayAlmostEqual(result, expected_output)

    def test_partial(self):
        """Test advection by a quarter of a grid point in the x direction and
        one grid point in the y direction"""
        expected_output = np.array([[0., 0., 0.],
                                    [0., 2.75, 3.75],
                                    [0., 1.75, 2.75],
                                    [0., 0.75, 1.75]])
        result = self.dummy_plugin._advect_field(
            self.data, self.grid_vel_x, 2.*self.grid_vel_y, 0.5, 0.)
        self.assertIsInstance(result, np.ndarray)
        self.assertArrayAlmostEqual(result, expected_output)


class Test_process(IrisTest):
    """Test dimensioned cube data is correctly advected"""

    def setUp(self):
        """Set up plugin instance and a cube to advect"""
        vel_x = set_up_xy_velocity_cube("advection_velocity_x")
        vel_y = vel_x.copy(data=2.*np.ones(shape=(4, 3)))
        vel_y.rename("advection_velocity_y")
        self.plugin = AdvectField(vel_x, vel_y)
        data = np.array([[2., 3., 4.],
                         [1., 2., 3.],
                         [0., 1., 2.],
                         [0., 0., 1.]])
        self.cube = iris.cube.Cube(
            data, standard_name='rainfall_rate', units='mm h-1',
            dim_coords_and_dims=[(self.plugin.y_coord, 0),
                                 (self.plugin.x_coord, 1)])

        # input time: [datetime.datetime(2018, 2, 20, 4, 0)]
        self.time_coord = DimCoord(1519099200, standard_name="time",
                                   units='seconds since 1970-01-01 00:00:00')
        self.cube.add_aux_coord(self.time_coord)

        self.timestep = datetime.timedelta(seconds=600)

    def test_basic(self):
        """Test output cube data is as expected"""
        expected_output = np.array([[0., 0., 0.],
                                    [0., 0., 0.],
                                    [0., 2., 3.],
                                    [0., 1., 2.]])
        result = self.plugin.process(self.cube, self.timestep)
        self.assertArrayAlmostEqual(result.data, expected_output)

    def test_background_values(self):
        """Test output cube data is padded as expected where source grid
        points are out of bounds"""
        expected_output = np.array([[-1., -1., -1.],
                                    [-1., -1., -1.],
                                    [-1., 2., 3.],
                                    [-1., 1., 2.]])
        result = self.plugin.process(self.cube, self.timestep, bgd=-1.)
        self.assertArrayAlmostEqual(result.data, expected_output)

    def test_time_step(self):
        """Test outputs are OK for a time step with non-second components"""
        expected_output = np.zeros(shape=(4, 3))
        result = self.plugin.process(self.cube, datetime.timedelta(hours=1))
        self.assertArrayAlmostEqual(result.data, expected_output)

    def test_raises_grid_mismatch_error(self):
        """Test error is raised if cube grid does not match velocity grids"""
        x_coord = DimCoord(np.arange(5), 'projection_x_coordinate', units='km')
        y_coord = DimCoord(np.arange(4), 'projection_y_coordinate', units='km')
        cube = iris.cube.Cube(np.zeros(shape=(4, 5)),
                              standard_name='rainfall_rate', units='mm h-1',
                              dim_coords_and_dims=[(y_coord, 0), (x_coord, 1)])
        cube.add_aux_coord(self.time_coord)

        msg = "Input data grid does not match advection velocities"
        with self.assertRaisesRegexp(InvalidCubeError, msg):
            self.plugin.process(cube, self.timestep)

    def test_validity_time(self):
        """Test output cube time is correctly updated"""
        result = self.plugin.process(self.cube, self.timestep)
        output_cube_time, = iris_time_to_datetime(result.coord("time"))
        self.assertEqual(output_cube_time.year, 2018)
        self.assertEqual(output_cube_time.month, 02)
        self.assertEqual(output_cube_time.day, 20)
        self.assertEqual(output_cube_time.hour, 4)
        self.assertEqual(output_cube_time.minute, 10)


if __name__ == '__main__':
    unittest.main()
