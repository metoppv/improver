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
"""Unit tests for the build_spotdata_cube function"""

import unittest
from datetime import datetime

import iris
import numpy as np
from iris.tests import IrisTest

from improver.spotdata.build_spotdata_cube import build_spotdata_cube

from ...set_up_test_cubes import construct_scalar_time_coords


class Test_build_spotdata_cube(IrisTest):
    """Tests for the build_spotdata_cube function"""

    def setUp(self):
        """Set up some auxiliary coordinate points for re-use"""
        self.altitude = np.array([256.5, 359.1, 301.8, 406.2])
        self.latitude = np.linspace(58.0, 59.5, 4)
        self.longitude = np.linspace(-0.25, 0.5, 4)
        self.wmo_id = ['03854', '03962', '03142', '03331']

        self.neighbour_methods = ['nearest', 'nearest_land']
        self.grid_attributes = ['x_index', 'y_index', 'dz']

    def test_scalar(self):
        """Test output for a single site"""
        result = build_spotdata_cube(
            1.6, 'air_temperature', 'degC', 10., 59.5, 1.3, '03854')

        # check result type
        self.assertIsInstance(result, iris.cube.Cube)

        # check data
        self.assertArrayAlmostEqual(result.data, np.array([1.6]))
        self.assertEqual(result.name(), 'air_temperature')
        self.assertEqual(result.units, 'degC')

        # check coordinate values and units
        self.assertEqual(result.coord('spot_index').points[0], 0)
        self.assertAlmostEqual(result.coord('altitude').points[0], 10.)
        self.assertEqual(result.coord('altitude').units, 'm')
        self.assertAlmostEqual(result.coord('latitude').points[0], 59.5)
        self.assertEqual(result.coord('latitude').units, 'degrees')
        self.assertAlmostEqual(result.coord('longitude').points[0], 1.3)
        self.assertEqual(result.coord('longitude').units, 'degrees')
        self.assertEqual(result.coord('wmo_id').points[0], '03854')

    def test_site_list(self):
        """Test output for a list of sites"""
        data = np.array([1.6, 1.3, 1.4, 1.1])
        result = build_spotdata_cube(
            data, 'air_temperature', 'degC', self.altitude, self.latitude,
            self.longitude, self.wmo_id)

        self.assertArrayAlmostEqual(result.data, data)
        self.assertArrayAlmostEqual(
            result.coord('altitude').points, self.altitude)
        self.assertArrayAlmostEqual(
            result.coord('latitude').points, self.latitude)
        self.assertArrayAlmostEqual(
            result.coord('longitude').points, self.longitude)
        self.assertArrayEqual(result.coord('wmo_id').points, self.wmo_id)

    def test_neighbour_method(self):
        """Test output where neighbour_methods is populated"""
        data = np.array([[1.6, 1.7], [1.3, 1.5], [1.4, 1.4], [1.1, 1.3]])

        result = build_spotdata_cube(
            data, 'air_temperature', 'degC', self.altitude, self.latitude,
            self.longitude, self.wmo_id,
            neighbour_methods=self.neighbour_methods)

        self.assertArrayAlmostEqual(result.data, data)
        self.assertEqual(result.coord_dims('neighbour_selection_method')[0], 1)
        self.assertArrayEqual(
            result.coord('neighbour_selection_method').points, np.arange(2))
        self.assertArrayEqual(
            result.coord('neighbour_selection_method_name').points,
            self.neighbour_methods)

    def test_grid_attributes(self):
        """Test output where grid_attributes is populated"""
        data = np.array([[1.6, 1.7, 1.8], [1.3, 1.5, 1.5],
                         [1.4, 1.4, 1.5], [1.1, 1.3, 1.4]])

        result = build_spotdata_cube(
            data, 'air_temperature', 'degC', self.altitude, self.latitude,
            self.longitude, self.wmo_id, grid_attributes=self.grid_attributes,
            grid_attributes_dim=1)

        self.assertArrayAlmostEqual(result.data, data)
        self.assertEqual(result.coord_dims('grid_attributes')[0], 1)
        self.assertArrayEqual(
            result.coord('grid_attributes').points, np.arange(3))
        self.assertArrayEqual(
            result.coord('grid_attributes_key').points, self.grid_attributes)

    def test_3d_spot_cube(self):
        """Test output with two extra dimensions"""
        data = np.ones((4, 2, 3), dtype=np.float32)
        result = build_spotdata_cube(
            data, 'air_temperature', 'degC', self.altitude, self.latitude,
            self.longitude, self.wmo_id,
            neighbour_methods=self.neighbour_methods,
            grid_attributes=self.grid_attributes)

        self.assertArrayAlmostEqual(result.data, data)
        self.assertEqual(result.coord_dims('neighbour_selection_method')[0], 1)
        self.assertEqual(result.coord_dims('grid_attributes')[0], 2)

    def test_3d_spot_cube_with_unequal_length_coordinates(self):
        """Test error is raised if coordinates lengths do not match data
        dimensions."""

        data = np.ones((4, 2, 2), dtype=np.float32)

        msg = "Unequal lengths"
        with self.assertRaisesRegex(ValueError, msg):
            build_spotdata_cube(
                data, 'air_temperature', 'degC', self.altitude, self.latitude,
                self.longitude, self.wmo_id,
                neighbour_methods=self.neighbour_methods,
                grid_attributes=self.grid_attributes)

    def test_scalar_coords(self):
        """Test additional scalar coordinates"""
        [(time_coord, _), (frt_coord, _), (fp_coord, _)] = (
            construct_scalar_time_coords(
                datetime(2015, 11, 23, 4, 30), None,
                datetime(2015, 11, 22, 22, 30)))

        data = np.ones((4, 2), dtype=np.float32)
        result = build_spotdata_cube(
            data, 'air_temperature', 'degC', self.altitude, self.latitude,
            self.longitude, self.wmo_id,
            scalar_coords=[time_coord, frt_coord, fp_coord],
            neighbour_methods=self.neighbour_methods)

        self.assertEqual(result.coord('time').points[0], time_coord.points[0])
        self.assertEqual(
            result.coord('forecast_reference_time').points[0],
            frt_coord.points[0])
        self.assertEqual(
            result.coord('forecast_period').points[0], fp_coord.points[0])

    def test_renaming_to_set_standard_name(self):
        """Test that CF standard names are set as such in the returned cube,
        whilst non-standard names remain as the long_name."""
        standard_name_cube = build_spotdata_cube(
            1.6, 'air_temperature', 'degC', 10., 59.5, 1.3, '03854')
        non_standard_name_cube = build_spotdata_cube(
            1.6, 'toast_temperature', 'degC', 10., 59.5, 1.3, '03854')

        self.assertEqual(standard_name_cube.standard_name, 'air_temperature')
        self.assertEqual(standard_name_cube.long_name, None)
        self.assertEqual(non_standard_name_cube.standard_name, None)
        self.assertEqual(non_standard_name_cube.long_name, 'toast_temperature')


if __name__ == '__main__':
    unittest.main()
