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
from cf_units import Unit

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
        self.assertAlmostEqual(plugin.vgradz_thresh, 0.0005)
        self.assertAlmostEqual(plugin.upstream_range_of_influence_km, 15.)
        self.assertAlmostEqual(plugin.efficiency_factor, 0.23265)
        self.assertAlmostEqual(plugin.cloud_lifetime_s, 102.)


class Test__repr__(IrisTest):
    """Test the __repr__ method"""

    def test_basic(self):
        """Test string representation of plugin"""
        expected = ('OrographicEnhancement() instance with orography '
                    'threshold 20.0 m, relative humidity threshold 0.8, '
                    'v.gradz threshold 0.0005, maximum upstream influence '
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
                         [300., 600., 1000.],
                         [250., 600., 900.]])
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
        expected_gradx = np.array([[0.175, 0.325, 0.475],
                                   [0.250, 0.350, 0.450],
                                   [0.375, 0.325, 0.275]])
        expected_grady = np.array([[0.175, 0.225, 0.275],
                                   [0.025, 0.075, 0.025],
                                   [-0.125, -0.075, -0.225]])
        gradx, grady = self.plugin._orography_gradients(self.topography)
        self.assertArrayAlmostEqual(gradx.data, expected_gradx)
        self.assertArrayAlmostEqual(grady.data, expected_grady)
        for cube in [gradx, grady]:
            self.assertEqual(cube.units, '1')


class Test__regrid_and_populate(IrisTest):
    """Test the _regrid_and_populate method"""
    pass # TODO


class Test__generate_mask(IrisTest):
    """Test the _generate_mask method"""

    def setUp(self):
        """Set up plugin instance with data cubes"""
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
    pass # TODO


class Test__add_upstream_component(IrisTest):
    """Test the _add_upstream_component method"""
    pass # TODO


if __name__ == '__main__':
    unittest.main()
