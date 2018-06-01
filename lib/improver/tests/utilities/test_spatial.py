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
"""Unit tests for the convert_distance_into_number_of_grid_cells function from
 spatial.py."""

import unittest
import numpy as np

from iris.tests import IrisTest
from iris.coords import AuxCoord
from iris import coord_systems
from iris.cube import Cube
import cartopy.crs as ccrs

from improver.tests.nbhood.nbhood.test_BaseNeighbourhoodProcessing import (
    set_up_cube, set_up_cube_lat_long)
from improver.utilities.spatial import (
    check_if_grid_is_equal_area, convert_distance_into_number_of_grid_cells,
    lat_lon_determine, lat_lon_transform, transform_grid_to_lat_lon,
    get_nearest_coords)
from improver.tests.spotdata.spotdata.test_common_functions import (
    Test_common_functions)


class Test_convert_distance_into_number_of_grid_cells(IrisTest):

    """Test conversion of distance in metres into number of grid cells."""

    def setUp(self):
        """Set up the cube."""
        self.DISTANCE = 6100
        self.MAX_DISTANCE_IN_GRID_CELLS = 500
        self.cube = set_up_cube()

    def test_basic_distance_to_grid_cells(self):
        """Test the distance in metres to grid cell conversion."""
        result = convert_distance_into_number_of_grid_cells(
            self.cube, self.DISTANCE, self.MAX_DISTANCE_IN_GRID_CELLS)
        self.assertEqual(result, (3, 3))

    def test_basic_distance_to_grid_cells_different_max_distance(self):
        """
        Test the distance in metres to grid cell conversion for an alternative
        max distance in grid cells.
        """
        max_distance_in_grid_cells = 50
        result = convert_distance_into_number_of_grid_cells(
            self.cube, self.DISTANCE, max_distance_in_grid_cells)
        self.assertEqual(result, (3, 3))

    def test_basic_distance_to_grid_cells_km_grid(self):
        """Test the distance-to-grid-cell conversion, grid in km."""
        self.cube.coord("projection_x_coordinate").convert_units("kilometres")
        self.cube.coord("projection_y_coordinate").convert_units("kilometres")
        result = convert_distance_into_number_of_grid_cells(
            self.cube, self.DISTANCE, self.MAX_DISTANCE_IN_GRID_CELLS)
        self.assertEqual(result, (3, 3))

    def test_single_point_lat_long(self):
        """Test behaviour for a single grid cell on lat long grid."""
        cube = set_up_cube_lat_long()
        msg = "Invalid grid: projection_x/y coords required"
        with self.assertRaisesRegex(ValueError, msg):
            convert_distance_into_number_of_grid_cells(
                cube, self.DISTANCE, self.MAX_DISTANCE_IN_GRID_CELLS)

    def test_single_point_range_negative(self):
        """Test behaviour with a non-zero point with negative range."""
        distance = -1.0 * self.DISTANCE
        msg = "distance of -6100.0m gives a negative cell extent"
        with self.assertRaisesRegex(ValueError, msg):
            convert_distance_into_number_of_grid_cells(
                self.cube, distance, self.MAX_DISTANCE_IN_GRID_CELLS)

    def test_single_point_range_0(self):
        """Test behaviour with a non-zero point with zero range."""
        distance = 5
        msg = "Distance of 5m gives zero cell extent"
        with self.assertRaisesRegex(ValueError, msg):
            convert_distance_into_number_of_grid_cells(
                self.cube, distance, self.MAX_DISTANCE_IN_GRID_CELLS)

    def test_single_point_range_lots(self):
        """Test behaviour with a non-zero point with unhandleable range."""
        distance = 40000.0
        max_distance_in_grid_cells = 10
        msg = "distance of 40000.0m exceeds maximum grid cell extent"
        with self.assertRaisesRegex(ValueError, msg):
            convert_distance_into_number_of_grid_cells(
                self.cube, distance, max_distance_in_grid_cells)

    def test_single_point_range_greater_than_domain(self):
        """Test correct exception raised when the distance is larger than the
           corner-to-corner distance of the domain."""
        distance = 42500.0
        msg = "Distance of 42500.0m exceeds max domain distance of "
        with self.assertRaisesRegex(ValueError, msg):
            convert_distance_into_number_of_grid_cells(
                self.cube, distance, self.MAX_DISTANCE_IN_GRID_CELLS)


class Test_check_if_grid_is_equal_area(IrisTest):

    """Test that the grid is an equal area grid."""

    def test_equal_area(self):
        """Test an that no exception is raised if the x and y coordinates
        are on an equal area grid."""
        cube = set_up_cube()
        self.assertEqual(check_if_grid_is_equal_area(cube), None)

    def test_wrong_coordinate(self):
        """Test an exception is raised if the x and y coordinates are not
        projection_x_coordinate or projection_y_coordinate."""
        cube = set_up_cube_lat_long()
        msg = "Invalid grid"
        with self.assertRaisesRegex(ValueError, msg):
            check_if_grid_is_equal_area(cube)

    def test_allow_negative_stride(self):
        """Test no errors raised if cube has negative stride in x and y axes"""
        cube = set_up_cube()
        coord_points_x = np.arange(-20000, -52000., -2000)
        coord_points_y = np.arange(30000., -2000, -2000)
        cube.coord("projection_x_coordinate").points = coord_points_x
        cube.coord("projection_y_coordinate").points = coord_points_y
        self.assertEqual(check_if_grid_is_equal_area(cube), None)

    def non_equal_intervals_along_axis(self):
        """Test that the cube has equal intervals along the x or y axis."""
        cube = set_up_cube()
        msg = "Intervals between points along the "
        with self.assertRaisesRegex(ValueError, msg):
            check_if_grid_is_equal_area(cube)

    def non_equal_area_grid(self):
        """Test that the cubes have an equal areas grid."""
        cube = set_up_cube()
        msg = "The size of the intervals along the x and y axis"
        with self.assertRaisesRegex(ValueError, msg):
            check_if_grid_is_equal_area(cube)


class Test_lat_lon_determine(Test_common_functions):
    """Test function that tests projections used in diagnostic cubes."""

    def test_projection_test(self):
        """Test identification of non-lat/lon projections."""
        src_crs = ccrs.PlateCarree()
        trg_crs = ccrs.LambertConformal(central_longitude=50,
                                        central_latitude=10)
        trg_crs_iris = coord_systems.LambertConformal(
            central_lon=50, central_lat=10)
        lons = self.cube.coord('longitude').points
        lats = self.cube.coord('latitude').points
        xvals, yvals = [], []
        for lon, lat in zip(lons, lats):
            x_trg, y_trg = trg_crs.transform_point(lon, lat, src_crs)
            xvals.append(x_trg)
            yvals.append(y_trg)

        new_x = AuxCoord(xvals, standard_name='projection_x_coordinate',
                         units='m', coord_system=trg_crs_iris)
        new_y = AuxCoord(yvals, standard_name='projection_y_coordinate',
                         units='m', coord_system=trg_crs_iris)

        cube = Cube(self.cube.data,
                    long_name="air_temperature",
                    dim_coords_and_dims=[(self.cube.coord('time'), 0)],
                    aux_coords_and_dims=[(new_y, 1), (new_x, 2)],
                    units="K")

        plugin = lat_lon_determine
        expected = trg_crs
        result = plugin(cube)
        self.assertEqual(expected, result)


class Test_lat_lon_transform(Test_common_functions):
    """
    Test function that transforms the lookup latitude and longitude into the
    projection used in a diagnostic cube.

    """
    def test_projection_transform(self):
        """
        Test transformation of lookup coordinates to the projections in
        which the diagnostic is provided.

        """
        trg_crs = ccrs.LambertConformal(central_longitude=50,
                                        central_latitude=10)

        plugin = lat_lon_transform
        expected_x, expected_y = 0., 0.
        result_x, result_y = plugin(trg_crs, 10, 50)
        self.assertAlmostEqual(expected_x, result_x)
        self.assertAlmostEqual(expected_y, result_y)


class Test_transform_grid_to_lat_lon(IrisTest):
    """
    Test function that transforms the points in the cube
    into grid of latitudes and longitudes

    """

    def setUp(self):
        """Set up the cube."""
        self.cube = set_up_cube(zero_point_indices=((0, 0, 1, 1),),
                                num_grid_points=2)
        self.cube.coord(axis='x').points = np.array([-1158000.0, 924000.0])
        self.cube.coord(axis='y').points = np.array([-1036000.0, 902000.0])

    def test_transform_grid(self):
        """
        Test transformation of grid
        """
        expected_lons = np.array([-17.11712928, 9.21255933,
                                  -24.5099247, 15.27976922]).reshape(2, 2)
        expected_lats = np.array([44.51715281, 44.899873,
                                  61.31885886, 61.9206868]).reshape(2, 2)
        plugin = transform_grid_to_lat_lon
        result_lats, result_lons = plugin(self.cube)
        self.assertIsInstance(result_lats, np.ndarray)
        self.assertIsInstance(result_lons, np.ndarray)
        self.assertArrayAlmostEqual(result_lons, expected_lons)
        self.assertArrayAlmostEqual(result_lats, expected_lats)


class Test_get_nearest_coords(Test_common_functions):
    """Test wrapper for iris.cube.Cube.nearest_neighbour_index."""

    def test_nearest_coords(self):
        """Test correct indices are returned."""
        plugin = get_nearest_coords
        longitude = 80
        latitude = -25
        expected = (4, 8)
        result = plugin(self.cube, latitude, longitude,
                        'latitude', 'longitude')
        self.assertEqual(expected, result)


if __name__ == '__main__':
    unittest.main()
