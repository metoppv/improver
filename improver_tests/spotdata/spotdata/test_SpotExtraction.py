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
"""Unit tests for SpotExtraction class"""

import unittest

import iris
import numpy as np
from iris.tests import IrisTest

from improver.metadata.constants.mo_attributes import MOSG_GRID_ATTRIBUTES
from improver.metadata.utilities import create_coordinate_hash
from improver.spotdata.build_spotdata_cube import build_spotdata_cube
from improver.spotdata.spot_extraction import SpotExtraction


class Test_SpotExtraction(IrisTest):

    """Test class for the SpotExtraction tests, setting up inputs."""

    def setUp(self):
        """
        Set up cubes and sitelists for use in testing SpotExtraction.
        The envisaged scenario is an island (1) surrounded by water (0).

          Land-sea       Orography      Diagnostic

          0 0 0 0 0      0 0 0 0 0       0  1  2  3  4
          0 1 1 1 0      0 1 2 1 0       5  6  7  8  9
          0 1 1 1 0      0 2 3 2 0      10 11 12 13 14
          0 1 1 1 0      0 1 2 1 0      15 16 17 18 19
          0 0 0 0 0      0 0 0 0 0      20 21 22 23 24

        """
        # Set up diagnostic data cube and neighbour cubes.
        diagnostic_data = np.arange(25).reshape(5, 5)

        xcoord = iris.coords.DimCoord(
            np.linspace(0, 40, 5), standard_name='longitude', units='degrees')
        ycoord = iris.coords.DimCoord(
            np.linspace(0, 40, 5), standard_name='latitude', units='degrees')

        # Grid attributes must be included in diagnostic cubes so their removal
        # can be tested
        attributes = {
            'mosg__grid_domain': 'global',
            'mosg__grid_type': 'standard'}

        diagnostic_cube_xy = iris.cube.Cube(
            diagnostic_data, standard_name="air_temperature", units='K',
            dim_coords_and_dims=[(ycoord, 1), (xcoord, 0)],
            attributes=attributes)
        diagnostic_cube_yx = iris.cube.Cube(
            diagnostic_data.T, standard_name="air_temperature", units='K',
            dim_coords_and_dims=[(ycoord, 0), (xcoord, 1)],
            attributes=attributes)

        diagnostic_cube_hash = create_coordinate_hash(diagnostic_cube_yx)

        # neighbours, each group is for a point under two methods, e.g.
        # [ 0.  0.  0.] is the nearest point to the first spot site, whilst
        # [ 1.  1. -1.] is the nearest land point to the same site.
        neighbours = np.array([[[0., 0., 0.],
                                [1., 1., -1.]],
                               [[0., 0., -1.],
                                [1., 1., 0.]],
                               [[2., 2., 0.],
                                [2., 2., 0.]],
                               [[2., 2., 1.],
                                [2., 2., 1.]]])
        altitudes = np.array([0, 1, 3, 2])
        latitudes = np.array([10, 10, 20, 20])
        longitudes = np.array([10, 10, 20, 20])
        wmo_ids = np.arange(4)
        grid_attributes = ['x_index', 'y_index', 'vertical_displacement']
        neighbour_methods = ['nearest', 'nearest_land']
        neighbour_cube = build_spotdata_cube(
            neighbours, 'grid_neighbours', 1, altitudes, latitudes,
            longitudes, wmo_ids, grid_attributes=grid_attributes,
            neighbour_methods=neighbour_methods)
        neighbour_cube.attributes['model_grid_hash'] = diagnostic_cube_hash

        coordinate_cube = neighbour_cube.extract(
            iris.Constraint(neighbour_selection_method_name='nearest') &
            iris.Constraint(grid_attributes_key=['x_index', 'y_index']))
        coordinate_cube.data = np.rint(coordinate_cube.data).astype(int)

        self.latitudes = latitudes
        self.longitudes = longitudes
        self.diagnostic_cube_xy = diagnostic_cube_xy
        self.diagnostic_cube_yx = diagnostic_cube_yx
        self.neighbours = neighbours
        self.neighbour_cube = neighbour_cube
        self.coordinate_cube = coordinate_cube

        self.expected_attributes = self.diagnostic_cube_xy.attributes
        for attr in MOSG_GRID_ATTRIBUTES:
            self.expected_attributes.pop(attr, None)
        self.expected_attributes["title"] = "unknown"
        self.expected_attributes["model_grid_hash"] = (
            self.neighbour_cube.attributes['model_grid_hash'])


class Test__repr__(IrisTest):

    """Tests the class __repr__ function."""

    def test_basic(self):
        """Test that the __repr__ returns the expected string with defaults."""
        plugin = SpotExtraction()
        result = str(plugin)
        msg = '<SpotExtraction: neighbour_selection_method: nearest>'
        self.assertEqual(result, msg)

    def test_non_default(self):
        """Test that the __repr__ returns the expected string with non-default
        options."""
        plugin = SpotExtraction(neighbour_selection_method='nearest_land')
        result = str(plugin)
        msg = '<SpotExtraction: neighbour_selection_method: nearest_land>'
        self.assertEqual(result, msg)


class Test_extract_coordinates(Test_SpotExtraction):

    """Test the extraction of x and y coordinate indices from a neighbour
    cube for a given neighbour_selection_method."""

    def test_nearest(self):
        """Test extraction of nearest neighbour x and y indices."""
        plugin = SpotExtraction(neighbour_selection_method='nearest')
        expected = self.neighbours[:, 0, 0:2].astype(int)
        result = plugin.extract_coordinates(self.neighbour_cube)
        self.assertArrayEqual(result.data, expected)

    def test_nearest_land(self):
        """Test extraction of nearest land neighbour x and y indices."""
        plugin = SpotExtraction(neighbour_selection_method='nearest_land')
        expected = self.neighbours[:, 1, 0:2].astype(int)
        result = plugin.extract_coordinates(self.neighbour_cube)
        self.assertArrayEqual(result.data, expected)

    def test_invalid_method(self):
        """Test attempt to extract neighbours found with a method that is not
        available within the neighbour cube. Raises an exception."""
        plugin = SpotExtraction(neighbour_selection_method='furthest')
        msg = 'The requested neighbour_selection_method "furthest" is not'
        with self.assertRaisesRegex(ValueError, msg):
            plugin.extract_coordinates(self.neighbour_cube)


class Test_extract_diagnostic_data(Test_SpotExtraction):

    """Test the extraction of data from the provided coordinates."""

    def test_xy_ordered_cube(self):
        """Test extraction of diagnostic data that is natively ordered xy."""
        plugin = SpotExtraction()
        expected = [0, 0, 12, 12]
        result = plugin.extract_diagnostic_data(self.coordinate_cube,
                                                self.diagnostic_cube_xy)
        self.assertArrayEqual(result, expected)

    def test_yx_ordered_cube(self):
        """Test extraction of diagnostic data that is natively ordered yx.
        This will be reordered before extraction to become xy."""
        plugin = SpotExtraction()
        expected = [0, 0, 12, 12]
        result = plugin.extract_diagnostic_data(self.coordinate_cube,
                                                self.diagnostic_cube_yx)
        self.assertArrayEqual(result, expected)


class Test_build_diagnostic_cube(Test_SpotExtraction):

    """Test the building of a spot data cube with given inputs."""

    def test_building_cube(self):
        """Test that a cube is built as expected."""
        plugin = SpotExtraction()
        spot_values = [0, 0, 12, 12]
        result = plugin.build_diagnostic_cube(self.neighbour_cube,
                                              self.diagnostic_cube_xy,
                                              spot_values)
        self.assertArrayEqual(result.coord('latitude').points, self.latitudes)
        self.assertArrayEqual(result.coord('longitude').points,
                              self.longitudes)
        self.assertArrayEqual(result.data, spot_values)


class Test_process(Test_SpotExtraction):

    """Test the process method which extracts data and builds cubes with
    metadata added."""

    def test_unmatched_cube_error(self):
        """Test that an error is raised if the neighbour cube and diagnostic
        cube do not have matching grids."""
        self.neighbour_cube.attributes['model_grid_hash'] = '123'
        plugin = SpotExtraction()
        msg = ("Cubes do not share or originate from the same grid, so cannot "
               "be used together.")
        with self.assertRaisesRegex(ValueError, msg):
            plugin.process(self.neighbour_cube, self.diagnostic_cube_xy)

    def test_returned_cube_nearest(self):
        """Test that data within the returned cube is as expected for the
        nearest neigbours."""
        plugin = SpotExtraction()
        expected = [0, 0, 12, 12]
        result = plugin.process(self.neighbour_cube, self.diagnostic_cube_xy)
        self.assertArrayEqual(result.data, expected)
        self.assertEqual(result.name(), self.diagnostic_cube_xy.name())
        self.assertEqual(result.units, self.diagnostic_cube_xy.units)
        self.assertArrayEqual(result.coord('latitude').points, self.latitudes)
        self.assertArrayEqual(result.coord('longitude').points,
                              self.longitudes)
        self.assertDictEqual(result.attributes, self.expected_attributes)

    def test_returned_cube_nearest_land(self):
        """Test that data within the returned cube is as expected for the
        nearest land neighbours."""
        plugin = SpotExtraction(neighbour_selection_method='nearest_land')
        expected = [6, 6, 12, 12]
        result = plugin.process(self.neighbour_cube, self.diagnostic_cube_xy)
        self.assertArrayEqual(result.data, expected)
        self.assertEqual(result.name(), self.diagnostic_cube_xy.name())
        self.assertEqual(result.units, self.diagnostic_cube_xy.units)
        self.assertArrayEqual(result.coord('latitude').points, self.latitudes)
        self.assertArrayEqual(result.coord('longitude').points,
                              self.longitudes)
        self.assertDictEqual(result.attributes, self.expected_attributes)

    def test_new_title(self):
        """Test title is updated as expected"""
        expected_attributes = self.expected_attributes
        expected_attributes["title"] = "IMPROVER Spot Forecast"
        plugin = SpotExtraction(neighbour_selection_method='nearest_land')
        result = plugin.process(self.neighbour_cube, self.diagnostic_cube_xy,
                                new_title="IMPROVER Spot Forecast")
        self.assertDictEqual(result.attributes, expected_attributes)

    def test_cube_with_leading_dimensions(self):
        """Test that a cube with a leading dimension such as realization or
        probability results in a spotdata cube with the same leading
        dimension."""
        realization0 = iris.coords.DimCoord(
            [0], standard_name='realization', units=1)
        realization1 = iris.coords.DimCoord(
            [1], standard_name='realization', units=1)

        cube0 = self.diagnostic_cube_xy.copy()
        cube1 = self.diagnostic_cube_xy.copy()
        cube0.add_aux_coord(realization0)
        cube1.add_aux_coord(realization1)
        cubes = iris.cube.CubeList([cube0, cube1])
        cube = cubes.merge_cube()

        plugin = SpotExtraction()
        expected = [[0, 0, 12, 12], [0, 0, 12, 12]]
        expected_coord = iris.coords.DimCoord(
            [0, 1], standard_name='realization', units=1)
        result = plugin.process(self.neighbour_cube, cube)
        self.assertArrayEqual(result.data, expected)
        self.assertEqual(result.name(), cube.name())
        self.assertEqual(result.units, cube.units)
        self.assertArrayEqual(result.coord('latitude').points, self.latitudes)
        self.assertArrayEqual(result.coord('longitude').points,
                              self.longitudes)
        self.assertEqual(result.coord('realization'), expected_coord)
        self.assertDictEqual(result.attributes, self.expected_attributes)


if __name__ == '__main__':
    unittest.main()
