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
from iris.tests import IrisTest

from improver.orographic_enhancement import OrographicEnhancement


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


class Test__smooth_data(IrisTest):
    """Test the _smooth_data method"""

    def setUp(self):
        """Set up an input array"""
        self.plugin = OrographicEnhancement()
        self.data = np.array([[200., 450., 850.],
                              [320., 500., 1000.],
                              [230., 600., 900.]])

    def test_basic(self):
        """Test output is np.array"""
        result = self.plugin._smooth_data(self.data, axis=0)
        self.assertIsInstance(result, np.ndarray)

    def test_axis_zero(self):
        """Test smoothing along first axis"""
        expected_result = np.array([[240., 466.66666667, 900.],
                                    [250., 516.66666667, 916.66666667],
                                    [260., 566.66666667, 933.33333333]])
        result = self.plugin._smooth_data(self.data, axis=0)
        self.assertArrayAlmostEqual(result, expected_result)

    def test_axis_one(self):
        """Test smoothing along second axis"""
        expected_result = np.array([[283.33333333, 500., 716.66666667],
                                    [380., 606.66666667, 833.33333333],
                                    [353.33333333, 576.66666667, 800.]])
        result = self.plugin._smooth_data(self.data, axis=1)
        self.assertArrayAlmostEqual(result, expected_result)


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
        self.topography = iris.cube.Cube(
            data, long_name="topography", units="m",
            dim_coords_and_dims=[(y_coord, 0), (x_coord, 1)])

    def test_basic(self):
        """Test outputs are cubes"""
        gradx, grady = self.plugin._orography_gradients(self.topography)
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
        gradx, grady = self.plugin._orography_gradients(self.topography)
        self.assertArrayAlmostEqual(gradx.data, expected_gradx)
        self.assertArrayAlmostEqual(grady.data, expected_grady)
        for cube in [gradx, grady]:
            self.assertEqual(cube.units, '1')


class Test__regrid_and_populate(IrisTest):
    """Test the _regrid_and_populate method"""
    pass # TODO


class Test__calculate_svp(IrisTest):
    """Test the _calculate_svp method"""

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

    def test_basic(self):
        """Test output is a cube"""
        self.plugin._calculate_svp(method='IMPROVER')
        self.assertIsInstance(self.plugin.svp, iris.cube.Cube)

    def test_values(self):
        """Test output values from the IMPROVER method"""
        expected_data = np.array([
            [813.64530012, 878.02231292, 848.25894782],
            [903.22241567, 890.54322113, 922.18893543],
            [922.14742099, 928.39365513, 967.87596347]])
        self.plugin._calculate_svp(method='IMPROVER')
        self.assertArrayAlmostEqual(self.plugin.svp.data, expected_data)


class Test__generate_mask(IrisTest):
    """Test the _generate_mask method"""

    def setUp(self):
        """Set up and populate a plugin instance"""
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
        self.topography = iris.cube.Cube(
            topography_data, long_name="topography", units="m",
            dim_coords_and_dims=[(y_coord, 0), (x_coord, 1)])

        self.plugin = OrographicEnhancement()

        humidity_data = np.full((5, 5), 0.9)
        humidity_data[1, 3] = 0.5
        self.plugin.humidity = iris.cube.Cube(
            humidity_data, long_name="relhumidity", units="1",
            dim_coords_and_dims=[(y_coord, 0), (x_coord, 1)])

        self.plugin.vgradz = np.full((5, 5), 0.01)
        self.plugin.vgradz[3:, :] = 0.

    def test_basic(self):
        """Test output is array"""
        result = self.plugin._generate_mask(self.topography)
        self.assertIsInstance(result, np.ndarray)

    def test_values(self):
        """Test output mask is correct"""
        expected_output = np.array([[True, True, False, False, False],
                                    [False, False, False, True, False],
                                    [False, False, False, False, False],
                                    [True, True, True, True, True],
                                    [True, True, True, True, True]])
        result = self.plugin._generate_mask(self.topography)
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

        self.plugin.mask = np.full((3, 3), False, dtype=bool)
        self.plugin.mask[0, 0] = True
        self.plugin.vgradz = np.array([[0.02, 0.08, 0.2],
                                       [0.06, 0.12, 0.22],
                                       [0.08, 0.16, 0.23]])

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


class Test__add_upstream_component(IrisTest):
    """Test the _add_upstream_component method"""

    def setUp(self):
        """Set up a plugin with wind components"""
        x_coord = DimCoord(3.*np.arange(5), 'projection_x_coordinate',
                           units='km')
        y_coord = DimCoord(3.*np.arange(5), 'projection_y_coordinate',
                           units='km')
        uwind = 20.*np.ones((5, 5))
        vwind = 12.*np.ones((5, 5))

        self.plugin = OrographicEnhancement()
        self.plugin.uwind = iris.cube.Cube(
            uwind, long_name="grid_eastward_wind", units="m s-1",
            dim_coords_and_dims=[(y_coord, 0), (x_coord, 1)])
        self.plugin.vwind = iris.cube.Cube(
            vwind, long_name="grid_northward_wind", units="m s-1",
            dim_coords_and_dims=[(y_coord, 0), (x_coord, 1)])

        self.site_orogenh = np.array([[4.1, 4.6, 5.6, 6.8, 5.5],
                                      [4.4, 4.6, 5.8, 6.2, 5.5],
                                      [5.2, 3.0, 3.4, 5.1, 3.3],
                                      [0.6, 2.0, 1.8, 4.2, 2.5],
                                      [0.0, 0.0, 0.2, 3.2, 1.8]])

    def test_basic(self):
        """Test output is an array"""
        result = self.plugin._add_upstream_component(
            self.site_orogenh, grid_spacing=3.)
        self.assertIsInstance(result, np.ndarray)

    def test_values(self):
        """Test output values are sensible"""
        expected_values = np.array([
            [0.953865, 1.039876, 1.241069, 1.506976, 1.355637],
            [1.022974, 1.057379, 1.275474, 1.415431, 1.320632],
            [1.207951, 0.829502, 0.769960, 1.086190, 0.878462],
            [0.150106, 0.390938, 0.438211, 0.834390, 0.682859],
            [0.001372, 0.001372, 0.035776, 0.566698, 0.500450]])
        result = self.plugin._add_upstream_component(
            self.site_orogenh, grid_spacing=3.)
        self.assertArrayAlmostEqual(result, expected_values)


class Test_process(IrisTest):
    """Test the process method"""
    pass  # TODO



if __name__ == '__main__':
    unittest.main()
