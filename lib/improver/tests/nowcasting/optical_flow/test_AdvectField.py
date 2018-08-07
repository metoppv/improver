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
""" Unit tests for the nowcasting.AdvectField plugin """

import datetime
import unittest
import numpy as np

import iris
from iris.coords import DimCoord
from iris.exceptions import InvalidCubeError
from iris.tests import IrisTest

from improver.nowcasting.optical_flow import AdvectField


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
        with self.assertRaisesRegex(InvalidCubeError, msg):
            _ = AdvectField(vel_x, vel_y)


class Test__repr__(IrisTest):
    """Test class representation"""

    def test_basic(self):
        """Test string representation"""
        vel_x = set_up_xy_velocity_cube("advection_velocity_x")
        vel_y = set_up_xy_velocity_cube("advection_velocity_y")
        result = str(AdvectField(vel_x, vel_y))
        self.assertEqual(result, '<AdvectField>')


class Test__increment_output_array(IrisTest):
    """Tests for the _increment_output_array method"""

    def setUp(self):
        """Create input arrays"""
        vel_x = set_up_xy_velocity_cube("advection_velocity_x")
        vel_y = vel_x.copy(data=2.*np.ones(shape=(4, 3)))
        self.dummy_plugin = AdvectField(vel_x, vel_y)

        self.data = np.array([[2., 3., 4.],
                              [1., 2., 3.],
                              [0., 1., 2.],
                              [0., 0., 1.]])
        (self.xgrid, self.ygrid) = np.meshgrid(np.arange(3),
                                               np.arange(4))

    def test_basic(self):
        """Test one increment from the points negative x-wards and positive
        y-wards on the source grid, with different directional weightings"""
        xsrc = np.array([[-1, 0, 1],
                         [-1, 0, 1],
                         [-1, 0, 1],
                         [-1, 0, 1]])
        ysrc = np.array([[1, 1, 1],
                         [2, 2, 2],
                         [3, 3, 3],
                         [4, 4, 4]])
        cond = np.array([[False, True, True],
                         [False, True, True],
                         [False, True, True],
                         [False, False, False]])
        xfrac = np.full((4, 3), 0.5)
        yfrac = np.full((4, 3), 0.75)
        outdata = np.zeros(shape=(4, 3))

        expected_output = np.array([[0., 0.375, 0.750],
                                    [0., 0.000, 0.375],
                                    [0., 0.000, 0.000],
                                    [0., 0.000, 0.000]])

        self.dummy_plugin._increment_output_array(
            self.data, outdata, cond, self.xgrid, self.ygrid,
            xsrc, ysrc, xfrac, yfrac)

        self.assertIsInstance(outdata, np.ndarray)
        self.assertArrayAlmostEqual(outdata, expected_output)


class Test__advect_field(IrisTest):
    """Tests for the _advect_field method"""

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
        self.timestep = 2.

    def test_basic(self):
        """Test function returns an array"""
        result = self.dummy_plugin._advect_field(
            self.data, self.grid_vel_x, self.grid_vel_y, self.timestep, 0.)
        self.assertIsInstance(result, np.ndarray)

    def test_advect_integer_grid_point(self):
        """Test data is advected correctly (by 1 and 2 grid points along the x
        and y axes respectively)"""
        expected_output = np.array([[0., 0., 0.],
                                    [0., 0., 0.],
                                    [0., 2., 3.],
                                    [0., 1., 2.]])
        result = self.dummy_plugin._advect_field(
            self.data, self.grid_vel_x, self.grid_vel_y, self.timestep, 0.)
        self.assertArrayAlmostEqual(result, expected_output)

    def test_advect_partial_grid_point(self):
        """Test advection by a quarter of a grid point in the x direction and
        one grid point in the y direction"""
        expected_output = np.array([[0., 0., 0.],
                                    [0., 2.75, 3.75],
                                    [0., 1.75, 2.75],
                                    [0., 0.75, 1.75]])
        result = self.dummy_plugin._advect_field(
            self.data, self.grid_vel_x, 2.*self.grid_vel_y, 0.5, 0.)
        self.assertArrayAlmostEqual(result, expected_output)

    def test_negative_advection_velocities(self):
        """Test data is advected correctly in the negative x direction"""
        expected_output = np.array([[0., 0., 0.],
                                    [0., 0., 0.],
                                    [3., 4., 0.],
                                    [2., 3., 0.]])
        self.grid_vel_x *= -1.
        result = self.dummy_plugin._advect_field(
            self.data, self.grid_vel_x, self.grid_vel_y, self.timestep, 0.)
        self.assertArrayAlmostEqual(result, expected_output)

    def test_masked_input(self):
        """Test masked data is correctly advected"""
        mask = np.array([[True, True, True],
                         [True, False, False],
                         [False, False, False],
                         [False, False, False]])
        masked_data = np.ma.MaskedArray(self.data, mask=mask)
        expected_data = np.array([[0., 0., 0.],
                                  [0., np.nan, np.nan],
                                  [0., np.nan, 2.75],
                                  [0., 0.75, 1.75]])
        expected_mask = np.where(np.isfinite(expected_data), False, True)
        expected_data = np.ma.MaskedArray(expected_data, mask=expected_mask)
        result = self.dummy_plugin._advect_field(
            masked_data, self.grid_vel_x, 2*self.grid_vel_y, 0.5, 0.)
        self.assertIsInstance(result, np.ma.MaskedArray)
        self.assertArrayAlmostEqual(result[~result.mask],
                                    expected_data[~result.mask])
        self.assertArrayEqual(result.mask, expected_mask)


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
        """Test plugin returns a cube"""
        result = self.plugin.process(self.cube, self.timestep)
        self.assertIsInstance(result, iris.cube.Cube)

    def test_advected_values(self):
        """Test output cube data is as expected"""
        expected_output = np.array([[0., 0., 0.],
                                    [0., 0., 0.],
                                    [0., 2., 3.],
                                    [0., 1., 2.]])
        result = self.plugin.process(self.cube, self.timestep)
        self.assertArrayAlmostEqual(result.data, expected_output)

    def test_fill_values(self):
        """Test output cube data is padded as expected where source grid
        points are out of bounds"""
        expected_output = np.array([[-1., -1., -1.],
                                    [-1., -1., -1.],
                                    [-1., 2., 3.],
                                    [-1., 1., 2.]])
        result = self.plugin.process(self.cube, self.timestep, fill_value=-1.)
        self.assertArrayAlmostEqual(result.data, expected_output)

    def test_masked_input(self):
        """Test input mask is correctly advected"""
        mask = np.array([[True, True, True],
                         [False, False, False],
                         [False, False, False],
                         [False, False, False]])
        masked_data = np.ma.MaskedArray(self.cube.data, mask=mask)
        masked_cube = self.cube.copy(masked_data)

        expected_data = np.array([[0., 0., 0.],
                                  [0., 0., 0.],
                                  [0., np.nan, np.nan],
                                  [0., 1., 2.]])
        expected_mask = ~np.isfinite(expected_data)
        expected_data = np.ma.MaskedArray(expected_data, mask=expected_mask)

        result = self.plugin.process(masked_cube, self.timestep)
        self.assertIsInstance(result.data, np.ma.MaskedArray)
        self.assertArrayAlmostEqual(result.data[~result.data.mask],
                                    expected_data[~result.data.mask])
        self.assertArrayEqual(result.data.mask, expected_mask)

    def test_mask_creation(self):
        """Test a mask is added if the fill value is NaN"""
        expected_output = np.array([[np.nan, np.nan, np.nan],
                                    [np.nan, np.nan, np.nan],
                                    [np.nan, 2., 3.],
                                    [np.nan, 1., 2.]])
        result = self.plugin.process(self.cube, self.timestep,
                                     fill_value=np.nan)
        self.assertIsInstance(result.data, np.ma.MaskedArray)
        self.assertArrayAlmostEqual(result.data[~result.data.mask],
                                    expected_output[~result.data.mask])

    def test_time_step(self):
        """Test outputs are OK for a time step with non-second components"""
        self.plugin.vel_x.data = self.plugin.vel_x.data / 6.
        self.plugin.vel_y.data = self.plugin.vel_y.data / 6.
        expected_output = np.array([[0., 0., 0.],
                                    [0., 0., 0.],
                                    [0., 2., 3.],
                                    [0., 1., 2.]])
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
        with self.assertRaisesRegex(InvalidCubeError, msg):
            self.plugin.process(cube, self.timestep)

    def test_validity_time(self):
        """Test output cube time is correctly updated"""
        result = self.plugin.process(self.cube, self.timestep)
        output_cube_time, = \
            (result.coord("time").units).num2date(result.coord("time").points)
        self.assertEqual(output_cube_time.year, 2018)
        self.assertEqual(output_cube_time.month, 2)
        self.assertEqual(output_cube_time.day, 20)
        self.assertEqual(output_cube_time.hour, 4)
        self.assertEqual(output_cube_time.minute, 10)

    def test_lead_time(self):
        """Test output cube has a forecast_period coordinate with the correct
        value and units"""
        result = self.plugin.process(self.cube, self.timestep)
        result.coord("forecast_period").convert_units("s")
        lead_time = result.coord("forecast_period").points
        self.assertEqual(len(lead_time), 1)
        self.assertEqual(lead_time[0], self.timestep.total_seconds())


if __name__ == '__main__':
    unittest.main()
