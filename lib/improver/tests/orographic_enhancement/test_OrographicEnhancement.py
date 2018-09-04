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
"""Unit tests for the OrographicEnhancement plugin."""

import unittest
import numpy as np

import iris
from iris.coords import DimCoord
from iris.coord_systems import GeogCS, TransverseMercator
from iris.tests import IrisTest

from improver.utilities.cube_manipulation import sort_coord_in_cube
from improver.orographic_enhancement import OrographicEnhancement


# define UKPP projection
TMercCS = TransverseMercator(
    latitude_of_projection_origin=49.0, longitude_of_central_meridian=-2.0,
    false_easting=400000.0, false_northing=-100000.0,
    scale_factor_at_central_meridian=0.9996013045310974,
    ellipsoid=GeogCS(semi_major_axis=6377563.396, semi_minor_axis=6356256.91))


def set_up_variable_cube(data, name="temperature", units="degC",
                         xo=400000., yo=0.):
    """
    Set up cube containing diagnostic variable data for regridding tests.
    Data are on a 2 km Transverse Mercator grid with an inverted y-axis,
    located in the UK.
    """
    y_points = 2000.*(data.shape[0] - np.arange(data.shape[0])) + yo
    x_points = 2000.*np.arange(data.shape[1]) + xo

    y_coord = DimCoord(
        y_points, 'projection_y_coordinate', units='m', coord_system=TMercCS)
    x_coord = DimCoord(
        x_points, 'projection_x_coordinate', units='m', coord_system=TMercCS)

    cube = iris.cube.Cube(data, long_name=name, units=units,
                          dim_coords_and_dims=[(y_coord, 0), (x_coord, 1)])
    return cube


def set_up_orography_cube(data, xo=400000., yo=0.):
    """
    Set up cube containing high resolution UK orography data
    """
    y_points = 1000.*(data.shape[0] - np.arange(data.shape[0])) + yo
    x_points = 1000.*np.arange(data.shape[1]) + xo

    y_coord = DimCoord(
        y_points, 'projection_y_coordinate', units='m', coord_system=TMercCS)
    x_coord = DimCoord(
        x_points, 'projection_x_coordinate', units='m', coord_system=TMercCS)

    cube = iris.cube.Cube(data, long_name="topography", units="m",
                          dim_coords_and_dims=[(y_coord, 0), (x_coord, 1)])
    return cube


class Test__init__(IrisTest):
    """Test the __init__ method"""

    def test_basic(self):
        """Test initialisation with no arguments"""
        plugin = OrographicEnhancement()
        self.assertAlmostEqual(plugin.orog_thresh_m, 20.)
        self.assertAlmostEqual(plugin.rh_thresh_ratio, 0.8)
        self.assertAlmostEqual(plugin.vgradz_thresh_ms, 0.0005)
        self.assertAlmostEqual(plugin.upstream_range_of_influence_km, 15.)
        self.assertAlmostEqual(plugin.efficiency_factor, 0.23265)
        self.assertAlmostEqual(plugin.cloud_lifetime_s, 102.)


class Test__repr__(IrisTest):
    """Test the __repr__ method"""

    def test_basic(self):
        """Test string representation of plugin"""
        expected = ('OrographicEnhancement() instance with orography '
                    'threshold 20.0 m, relative humidity threshold 0.8, '
                    'v.gradz threshold 0.0005 m/s, maximum upstream influence '
                    '15.0 km, upstream efficiency factor 0.23265, cloud '
                    'lifetime 102.0 s')
        plugin = OrographicEnhancement()
        self.assertEqual(str(plugin), expected)


class Test__orography_gradients(IrisTest):
    """Test the _orography_gradients method"""

    def setUp(self):
        """Set up an input cube"""
        self.plugin = OrographicEnhancement()
        data = np.array([[200., 450., 850.],
                         [320., 500., 1000.],
                         [230., 600., 900.]])
        x_coord = DimCoord(np.arange(3), 'projection_x_coordinate',
                           units='km')
        y_coord = DimCoord(np.arange(3), 'projection_y_coordinate',
                           units='km')
        self.plugin.topography = iris.cube.Cube(
            data, long_name="topography", units="m",
            dim_coords_and_dims=[(y_coord, 0), (x_coord, 1)])

    def test_basic(self):
        """Test outputs are cubes"""
        gradx, grady = self.plugin._orography_gradients()
        self.assertIsInstance(gradx, iris.cube.Cube)
        self.assertIsInstance(grady, iris.cube.Cube)

    def test_values(self):
        """Test output values and units"""
        expected_gradx = np.array([[0.12333333, 0.33, 0.53666667],
                                   [0.2, 0.33333333, 0.46666667],
                                   [0.27666667, 0.33666667, 0.39666667]])

        expected_grady = np.array([[0.15833333, 0.175, 0.19166667],
                                   [0.035, 0.03833333, 0.04166667],
                                   [-0.08833333, -0.09833333, -0.10833333]])
        gradx, grady = self.plugin._orography_gradients()
        self.assertArrayAlmostEqual(gradx.data, expected_gradx)
        self.assertArrayAlmostEqual(grady.data, expected_grady)
        for cube in [gradx, grady]:
            self.assertEqual(cube.units, '1')


class Test__regrid_variable(IrisTest):
    """Test the _regrid_variable method"""

    def setUp(self):
        """Set up input cubes"""
        temperature = np.arange(6).reshape(2, 3)
        self.temperature_cube = set_up_variable_cube(temperature)
        orography = np.array([[20., 30., 40., 30., 25., 25.],
                              [30., 50., 80., 60., 50., 45.],
                              [50., 65., 90., 70., 60., 50.],
                              [45., 60., 85., 65., 55., 45.]])
        orography_cube = set_up_orography_cube(orography)
        self.plugin = OrographicEnhancement()
        self.plugin.topography = sort_coord_in_cube(
            orography_cube, orography_cube.coord(axis='y'))

    def test_basic(self):
        """Test cube of the correct shape is returned"""
        expected_data = np.array([[4.5, 5., 5.5, 6., 6.5, 7.],
                                  [3., 3.5, 4., 4.5, 5., 5.5],
                                  [1.5, 2., 2.5, 3., 3.5, 4.],
                                  [0., 0.5, 1., 1.5, 2., 2.5]])
        result = self.plugin._regrid_variable(self.temperature_cube, "degC")
        self.assertIsInstance(result, iris.cube.Cube)
        self.assertArrayAlmostEqual(result.data, expected_data)

    def test_axis_inversion(self):
        """Test axes are output in ascending order"""
        result = self.plugin._regrid_variable(self.temperature_cube, "degC")
        x_points = result.coord(axis='x').points
        y_points = result.coord(axis='y').points
        self.assertTrue(x_points[1] > x_points[0])
        self.assertTrue(y_points[1] > y_points[0])

    def test_unit_conversion(self):
        """Test units are correctly converted"""
        expected_data = np.array([
            [277.65, 278.15, 278.65, 279.15, 279.65, 280.15],
            [276.15, 276.65, 277.15, 277.65, 278.15, 278.65],
            [274.65, 275.15, 275.65, 276.15, 276.65, 277.15],
            [273.15, 273.65, 274.15, 274.65, 275.15, 275.65]])
        result = self.plugin._regrid_variable(self.temperature_cube, "kelvin")
        self.assertEqual(result.units, 'kelvin')
        self.assertArrayAlmostEqual(result.data, expected_data)


class Test__regrid_and_populate(IrisTest):
    """Test the _regrid_and_populate method"""

    def setUp(self):
        temperature = np.arange(6).reshape(2, 3)
        self.temperature = set_up_variable_cube(temperature)
        humidity = np.arange(0.75, 0.86, 0.02).reshape(2, 3)
        self.humidity = set_up_variable_cube(humidity, 'relhumidity', '1')
        pressure = np.arange(820, 921, 20).reshape(2, 3)
        self.pressure = set_up_variable_cube(pressure, 'pressure', 'hPa')
        uwind = np.ones((2, 3), dtype=np.float32)
        self.uwind = set_up_variable_cube(uwind, 'wind-u', 'knots')
        vwind = np.ones((2, 3), dtype=np.float32)
        self.vwind = set_up_variable_cube(vwind, 'wind-v', 'knots')

        orography = np.array([[20., 30., 40., 30., 25., 25.],
                              [30., 50., 80., 60., 50., 45.],
                              [50., 65., 90., 70., 60., 50.],
                              [45., 60., 85., 65., 55., 45.]])
        self.orography_cube = set_up_orography_cube(orography)

    def test_basic(self):
        """Test function populates class instance"""
        plugin = OrographicEnhancement()
        plugin._regrid_and_populate(
            self.temperature, self.humidity, self.pressure,
            self.uwind, self.vwind, self.orography_cube)
        self.assertIsInstance(plugin.temperature, iris.cube.Cube)
        self.assertIsInstance(plugin.humidity, iris.cube.Cube)
        self.assertIsInstance(plugin.pressure, iris.cube.Cube)
        self.assertIsInstance(plugin.uwind, iris.cube.Cube)
        self.assertIsInstance(plugin.vwind, iris.cube.Cube)
        self.assertIsInstance(plugin.topography, iris.cube.Cube)
        self.assertIsInstance(plugin.vgradz, np.ndarray)

    def test_variables(self):
        """Test variable values are sensible"""
        expected_temperature = np.array([
            [277.65, 278.15, 278.65, 279.15, 279.65, 280.15],
            [276.15, 276.65, 277.15, 277.65, 278.15, 278.65],
            [274.65, 275.15, 275.65, 276.15, 276.65, 277.15],
            [273.15, 273.65, 274.15, 274.65, 275.15, 275.65]])

        expected_humidity = np.array([[0.84, 0.85, 0.86, 0.87, 0.88, 0.89],
                                      [0.81, 0.82, 0.83, 0.84, 0.85, 0.86],
                                      [0.78, 0.79, 0.80, 0.81, 0.82, 0.83],
                                      [0.75, 0.76, 0.77, 0.78, 0.79, 0.80]])

        expected_pressure = np.array([
            [91000., 92000., 93000., 94000., 95000., 96000.],
            [88000., 89000., 90000., 91000., 92000., 93000.],
            [85000., 86000., 87000., 88000., 89000., 90000.],
            [82000., 83000., 84000., 85000., 86000., 87000.]])

        expected_wind = np.full((4, 6), 0.51444447, dtype=np.float32)

        plugin = OrographicEnhancement()
        plugin._regrid_and_populate(
            self.temperature, self.humidity, self.pressure,
            self.uwind, self.vwind, self.orography_cube)

        for cube, array in zip(
                [plugin.temperature, plugin.humidity, plugin.pressure,
                 plugin.uwind, plugin.vwind],
                [expected_temperature, expected_humidity, expected_pressure,
                 expected_wind, expected_wind]):
            self.assertArrayAlmostEqual(cube.data, array)

        self.assertArrayAlmostEqual(
            plugin.topography.data, np.flipud(self.orography_cube.data))

    def test_vgradz(self):
        """Test values of vgradz are sensible"""
        expected_vgradz = np.array([
            [0.01371852, 0.01800556, 0.00814537, -0.00128611,
             0.00085741, 0.0004287],
            [0.00257222, 0.00857407, 0., -0.00900278,
             -0.00557315, -0.00428704],
            [-0.00214352, -0.0004287, -0.00943148, -0.01714815,
             -0.0120037, -0.00900278],
            [0.0004287, -0.00643055, -0.01929167, -0.02700833,
             -0.01929167, -0.01457593]])

        plugin = OrographicEnhancement()
        plugin._regrid_and_populate(
            self.temperature, self.humidity, self.pressure,
            self.uwind, self.vwind, self.orography_cube)
        self.assertArrayAlmostEqual(plugin.vgradz, expected_vgradz)


class SVPTest(IrisTest):
    """Shared setUp method for saturation vapour pressure functions"""

    def setUp(self):
        """Set up and populate a plugin instance"""
        x_coord = DimCoord(np.arange(3), 'projection_x_coordinate',
                           units='km')
        y_coord = DimCoord(np.arange(3), 'projection_y_coordinate',
                           units='km')

        temperature = np.array([[277.1, 278.2, 277.7],
                                [278.6, 278.4, 278.9],
                                [278.9, 279.0, 279.6]])
        humidity = np.array([[0.74, 0.85, 0.94],
                             [0.81, 0.82, 0.91],
                             [0.86, 0.93, 0.97]])
        pressure = np.array([[100000., 80000., 90000.],
                             [90000., 85000., 89000.],
                             [88000., 84000., 88000.]])

        self.plugin = OrographicEnhancement()
        self.plugin.temperature = iris.cube.Cube(
            temperature, long_name="temperature", units="kelvin",
            dim_coords_and_dims=[(y_coord, 0), (x_coord, 1)])
        self.plugin.humidity = iris.cube.Cube(
            humidity, long_name="relhumidity", units="1",
            dim_coords_and_dims=[(y_coord, 0), (x_coord, 1)])
        self.plugin.pressure = iris.cube.Cube(
            pressure, long_name="pressure", units="Pa",
            dim_coords_and_dims=[(y_coord, 0), (x_coord, 1)])


class Test__calculate_steps_svp_millibars(SVPTest):
    """Test the _calculate_steps_svp_millibars method"""

    def test_basic(self):
        """Test output is an array"""
        result = self.plugin._calculate_steps_svp_millibars(
            self.plugin.temperature.data)
        self.assertIsInstance(result, np.ndarray)

    def test_values(self):
        """Test output values"""
        expected_result = np.array([[8.1005659, 8.7493667, 8.4489765],
                                    [8.9964323, 8.8721398, 9.1857549],
                                    [9.1857549, 9.2496397, 9.6412525]])
        result = self.plugin._calculate_steps_svp_millibars(
            self.plugin.temperature.data)
        self.assertArrayAlmostEqual(result, expected_result)


class Test__calculate_svp(SVPTest):
    """Test the _calculate_svp method"""

    def test_basic(self):
        """Test output is a cube"""
        self.plugin._calculate_svp(method='IMPROVER')
        self.assertIsInstance(self.plugin.svp, iris.cube.Cube)

    def test_values(self):
        """Test output values from the IMPROVER method"""
        expected_data = np.array([
            [813.645300, 878.022313, 848.258948],
            [903.222415, 890.543221, 922.188935],
            [922.147421, 928.393655, 967.875963]])
        self.plugin._calculate_svp(method='IMPROVER')
        self.assertArrayAlmostEqual(self.plugin.svp.data, expected_data)

    def test_values_steps(self):
        """Test output values from the STEPS method"""
        expected_data = np.array([
            [810.056593, 874.936670, 844.897647],
            [899.643230, 887.213984, 918.575494],
            [918.575494, 924.963966, 964.125249]])
        self.plugin._calculate_svp(method='STEPS')
        self.assertArrayAlmostEqual(self.plugin.svp.data, expected_data)


class Test__generate_mask(IrisTest):
    """Test the _generate_mask method"""

    def setUp(self):
        """Set up and populate a plugin instance"""
        self.plugin = OrographicEnhancement()
        x_coord = DimCoord(np.arange(5), 'projection_x_coordinate',
                           units='km')
        y_coord = DimCoord(np.arange(5), 'projection_y_coordinate',
                           units='km')

        # this is neighbourhood-processed as part of mask generation
        topography_data = np.array([[0., 10., 20., 50., 100.],
                                    [10., 20., 50., 100., 200.],
                                    [25., 60., 80., 160., 220.],
                                    [50., 80., 100., 200., 250.],
                                    [50., 80., 100., 200., 250.]])
        self.plugin.topography = iris.cube.Cube(
            topography_data, long_name="topography", units="m",
            dim_coords_and_dims=[(y_coord, 0), (x_coord, 1)])

        humidity_data = np.full((5, 5), 0.9)
        humidity_data[1, 3] = 0.5
        self.plugin.humidity = iris.cube.Cube(
            humidity_data, long_name="relhumidity", units="1",
            dim_coords_and_dims=[(y_coord, 0), (x_coord, 1)])

        self.plugin.vgradz = np.full((5, 5), 0.01)
        self.plugin.vgradz[3:, :] = 0.

    def test_basic(self):
        """Test output is array"""
        result = self.plugin._generate_mask()
        self.assertIsInstance(result, np.ndarray)

    def test_values(self):
        """Test output mask is correct"""
        expected_output = np.array([[True, True, False, False, False],
                                    [False, False, False, True, False],
                                    [False, False, False, False, False],
                                    [True, True, True, True, True],
                                    [True, True, True, True, True]])
        result = self.plugin._generate_mask()
        self.assertArrayEqual(result, expected_output)


class Test__site_orogenh(IrisTest):
    """Test the _site_orogenh method"""

    def setUp(self):
        """Set up and populate a plugin instance"""
        x_coord = DimCoord(np.arange(3), 'projection_x_coordinate',
                           units='km')
        y_coord = DimCoord(np.arange(3), 'projection_y_coordinate',
                           units='km')

        temperature = np.array([[277.1, 278.2, 277.7],
                                [278.6, 278.4, 278.9],
                                [278.9, 279.0, 279.6]])
        humidity = np.array([[0.74, 0.85, 0.94],
                             [0.81, 0.82, 0.91],
                             [0.86, 0.93, 0.97]])
        svp = np.array([[813.6, 878.0, 848.3],
                        [903.2, 890.5, 922.2],
                        [922.1, 928.4, 967.9]])

        self.plugin = OrographicEnhancement()
        self.plugin.temperature = iris.cube.Cube(
            temperature, long_name="temperature", units="kelvin",
            dim_coords_and_dims=[(y_coord, 0), (x_coord, 1)])
        self.plugin.humidity = iris.cube.Cube(
            humidity, long_name="relhumidity", units="1",
            dim_coords_and_dims=[(y_coord, 0), (x_coord, 1)])
        self.plugin.svp = iris.cube.Cube(
            svp, long_name="saturation_vapour_pressure", units="Pa",
            dim_coords_and_dims=[(y_coord, 0), (x_coord, 1)])

        self.plugin.vgradz = np.array([[0.02, 0.08, 0.2],
                                       [0.06, 0.12, 0.22],
                                       [0.08, 0.16, 0.23]])

        topography_data = np.full((3, 3), 50., dtype=np.float32)
        self.plugin.topography = iris.cube.Cube(
            topography_data, long_name="topography", units="m",
            dim_coords_and_dims=[(y_coord, 0), (x_coord, 1)])

    def test_basic(self):
        """Test output is an array"""
        result = self.plugin._site_orogenh()
        self.assertIsInstance(result, np.ndarray)

    def test_values(self):
        """Test output values are as expected"""
        expected_values = np.array([
            [0., 1.67372072, 4.47886658],
            [1.22878468, 2.45468903, 5.1627059],
            [1.77400422, 3.86162901, 6.02323198]])
        result = self.plugin._site_orogenh()
        self.assertArrayAlmostEqual(result, expected_values)


class Test__get_distance_weights(IrisTest):
    """Test the _get_distance_weights function"""
    pass  # TODO


class Test__locate_source_points(IrisTest):
    """Test the _locate_source_points method"""

    def setUp(self):
        """Define matrices"""
        self.wind_speed = np.ones((3, 4), dtype=np.float32)
        self.sin_wind_dir = np.full((3, 4), 0.4, dtype=np.float32)
        self.cos_wind_dir = np.full((3, 4), np.sqrt(0.84), dtype=np.float32)
        self.plugin = OrographicEnhancement()
        self.plugin.grid_spacing_km = 3.
        self.distance_weight = self.plugin._get_distance_weights(
            self.wind_speed, self.cos_wind_dir)

    def test_basic(self):
        """Test location of source points"""
        xsrc, ysrc = self.plugin._locate_source_points(
            self.wind_speed, self.distance_weight,
            self.sin_wind_dir, self.cos_wind_dir)

        expected_xsrc = np.array([[[0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3]],
                                  [[0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3]],
                                  [[0, 0, 1, 2], [0, 0, 1, 2], [0, 0, 1, 2]],
                                  [[0, 0, 1, 2], [0, 0, 1, 2], [0, 0, 1, 2]]])

        expected_ysrc = np.array([[[0, 0, 0, 0], [1, 1, 1, 1], [2, 2, 2, 2]],
                                  [[0, 0, 0, 0], [0, 0, 0, 0], [1, 1, 1, 1]],
                                  [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                                  [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]])

        self.assertArrayEqual(xsrc, expected_xsrc)
        self.assertArrayEqual(ysrc, expected_ysrc)


class Test__compute_weighted_values(IrisTest):
    """Test the _compute_weighted_values method"""

    def setUp(self):
        """Set up plugin and some inputs"""
        self.plugin = OrographicEnhancement()
        self.plugin.grid_spacing_km = 3.

        self.site_orogenh = np.array([[4.1, 4.6, 5.6, 6.8, 5.5],
                                      [4.4, 4.6, 5.8, 6.2, 5.5],
                                      [5.2, 3.0, 3.4, 5.1, 3.3],
                                      [0.6, 2.0, 1.8, 4.2, 2.5],
                                      [0.0, 0.0, 0.2, 3.2, 1.8]])

        self.wind_speed = np.full((5, 5), 25., dtype=np.float32)
        sin_wind_dir = np.full((5, 5), 0.4, dtype=np.float32)
        cos_wind_dir = np.full((5, 5), np.sqrt(0.84), dtype=np.float32)
        self.distance_weight = self.plugin._get_distance_weights(
            self.wind_speed, cos_wind_dir)
        self.xsrc, self.ysrc = self.plugin._locate_source_points(
            self.wind_speed, self.distance_weight, sin_wind_dir, cos_wind_dir)

    def test_basic(self):
        """Test output is two arrays"""
        orogenh, weights = self.plugin._compute_weighted_values(
            self.site_orogenh, self.xsrc, self.ysrc,
            self.distance_weight, self.wind_speed)
        self.assertIsInstance(orogenh, np.ndarray)
        self.assertIsInstance(weights, np.ndarray)

    def test_values(self):
        """Test values are as expected"""
        expected_orogenh = np.array([
            [6.0531969, 6.7725644, 8.2301264, 9.9942646, 8.1690931],
            [6.3531971, 6.7725644, 8.4301271, 9.3942642, 8.1690931],
            [7.2848172, 5.1725645, 6.1178742, 8.0310230, 5.9690924],
            [3.0469213, 3.4817038, 3.4649093, 6.6558237, 4.1816435],
            [0.4585612, 1.0727906, 1.1036499, 5.1721582, 3.0895371]])
        expected_weights = np.full((5, 5), 1.4763895, dtype=np.float32)
        orogenh, weights = self.plugin._compute_weighted_values(
            self.site_orogenh, self.xsrc, self.ysrc,
            self.distance_weight, self.wind_speed)
        self.assertArrayAlmostEqual(orogenh, expected_orogenh)
        self.assertArrayAlmostEqual(weights, expected_weights)


class Test__add_upstream_component(IrisTest):
    """Test the _add_upstream_component method"""

    def setUp(self):
        """Set up a plugin with wind components"""
        x_coord = DimCoord(3.*np.arange(5), 'projection_x_coordinate',
                           units='km')
        y_coord = DimCoord(3.*np.arange(5), 'projection_y_coordinate',
                           units='km')
        uwind = np.full((5, 5), 20., dtype=np.float32)
        vwind = np.full((5, 5), 12., dtype=np.float32)

        self.plugin = OrographicEnhancement()
        self.plugin.uwind = iris.cube.Cube(
            uwind, long_name="grid_eastward_wind", units="m s-1",
            dim_coords_and_dims=[(y_coord, 0), (x_coord, 1)])
        self.plugin.vwind = iris.cube.Cube(
            vwind, long_name="grid_northward_wind", units="m s-1",
            dim_coords_and_dims=[(y_coord, 0), (x_coord, 1)])
        self.plugin.grid_spacing_km = 3.

        self.site_orogenh = np.array([[4.1, 4.6, 5.6, 6.8, 5.5],
                                      [4.4, 4.6, 5.8, 6.2, 5.5],
                                      [5.2, 3.0, 3.4, 5.1, 3.3],
                                      [0.6, 2.0, 1.8, 4.2, 2.5],
                                      [0.0, 0.0, 0.2, 3.2, 1.8]])

    def test_basic(self):
        """Test output is an array"""
        result = self.plugin._add_upstream_component(self.site_orogenh)
        self.assertIsInstance(result, np.ndarray)

    def test_values(self):
        """Test output values are sensible"""
        expected_values = np.array([
            [0.953865, 1.039876, 1.241070, 1.506976, 1.355637],
            [1.005472, 1.039876, 1.275474, 1.403762, 1.355637],
            [1.161275, 0.782825, 0.863303, 1.226206, 0.942638],
            [0.418468, 0.659300, 0.496544, 0.927728, 0.735382],
            [0.036423, 0.036423, 0.152506, 0.660092, 0.558801]])

        result = self.plugin._add_upstream_component(self.site_orogenh)
        self.assertArrayAlmostEqual(result, expected_values)


class Test__create_output_cubes(IrisTest):
    """Test the _create_output_cubes method"""
    pass  # TODO


class Test_process(IrisTest):
    """Test the process method"""
    pass  # TODO


if __name__ == '__main__':
    unittest.main()
