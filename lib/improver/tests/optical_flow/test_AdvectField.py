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
from iris.tests import IrisTest

from improver.optical_flow.optical_flow import AdvectField

def set_up_xy_velocity_cube(name, coord_points=None, val_units='m s-1'):
    """Set up a 3x3 cube of simple velocities (no convergence / divergence)"""
    data = np.ones(shape=(3, 3))
    if coord_points is None:
        coord_points = 0.6*np.arange(3)
    x_coord = DimCoord(coord_points, 'projection_x_coordinate', units='km')
    y_coord = DimCoord(coord_points, 'projection_y_coordinate', units='km')
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
                                        coord_points=2*np.arange(3))
        msg = "Velocity cubes on unmatched grids"
        with self.assertRaisesRegexp(ValueError, msg):
            plugin = AdvectField(vel_x, vel_y)


class Test_process(IrisTest):

    """Test cube data is correctly advected"""

    def setUp(self):
        """Set up plugin instance and a cube to advect"""
        vel_x = set_up_xy_velocity_cube("advection_velocity_x")
        vel_y = set_up_xy_velocity_cube("advection_velocity_y")
        self.plugin = AdvectField(vel_x, vel_y)
       
        data = np.array([[1., 2., 3.],
                         [0., 1., 2.],
                         [0., 0., 1.]])
        self.cube = iris.cube.Cube(data, standard_name='rainfall_rate',
                                   units='mm h-1', dim_coords_and_dims=[
                                   (self.plugin.y_coord, 0),
                                   (self.plugin.x_coord, 1)])
        # TODO add time coordinate

        self.timestep = datetime.timedelta(seconds=600)

    def test_values(self):
        """Test output cube data is as expected"""
        expected_output = np.array([[0., 0., 0.],
                                    [0., 1., 2.],
                                    [0., 0., 1.]])
        result = self.plugin.process(self.cube, self.timestep)
        self.assertArrayAlmostEqual(result.data, expected_output)

    def test_validity_time(self):
        """Test output cube time is correctly updated"""
        pass


if __name__ == '__main__':
    unittest.main()
