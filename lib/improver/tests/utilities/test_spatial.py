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
"""Unit tests for the convert_distance_into_number_of_grid_cells function from
 spatial.py."""

import unittest
from datetime import datetime as dt

import cartopy.crs as ccrs
import cf_units
import numpy as np
from iris import Constraint
from iris import coord_systems
from iris.coord_systems import GeogCS
from iris.coords import AuxCoord, DimCoord
from iris.cube import Cube, CubeList
from iris.tests import IrisTest
from iris.time import PartialDateTime

from improver.tests.nbhood.nbhood.test_BaseNeighbourhoodProcessing import (
    set_up_cube, set_up_cube_lat_long)
from improver.utilities.spatial import (
    check_if_grid_is_equal_area, convert_distance_into_number_of_grid_cells,
    convert_number_of_grid_cells_into_distance,
    lat_lon_determine, lat_lon_transform, transform_grid_to_lat_lon,
    get_nearest_coords)


class Test_common_functions(IrisTest):

    """A class originally written for testing spot-data functionality that no
    longer exists. It was also used in this set of tests, so upon deletion of
    the old spot-data code, the class was moved here."""

    def setUp(self):
        """
        Create a cube containing a regular lat-lon grid.

        Data is striped horizontally,
        e.g.
              1 1 1 1 1 1
              1 1 1 1 1 1
              2 2 2 2 2 2
              2 2 2 2 2 2
              3 3 3 3 3 3
              3 3 3 3 3 3
        """
        data = np.ones((12, 12))
        data[0:4, :] = 1
        data[4:8, :] = 2
        data[8:, :] = 3

        latitudes = np.linspace(-90, 90, 12)
        longitudes = np.linspace(-180, 180, 12)
        latitude = DimCoord(latitudes, standard_name='latitude',
                            units='degrees', coord_system=GeogCS(6371229.0))
        longitude = DimCoord(longitudes, standard_name='longitude',
                             units='degrees', coord_system=GeogCS(6371229.0),
                             circular=True)

        # Use time of 2017-02-17 06:00:00
        time = DimCoord(
            [1487311200], standard_name='time',
            units=cf_units.Unit('seconds since 1970-01-01 00:00:00',
                                calendar='gregorian'))
        long_time_coord = DimCoord(
            list(range(1487311200, 1487397600, 3600)),
            standard_name='time',
            units=cf_units.Unit('seconds since 1970-01-01 00:00:00',
                                calendar='gregorian'))

        time_dt = dt(2017, 2, 17, 6, 0)
        time_extract = Constraint(
            time=lambda cell: cell.point == PartialDateTime(
                time_dt.year, time_dt.month, time_dt.day, time_dt.hour))

        cube = Cube(data.reshape((1, 12, 12)),
                    long_name="air_temperature",
                    dim_coords_and_dims=[(time, 0),
                                         (latitude, 1),
                                         (longitude, 2)],
                    units="K")

        long_cube = Cube(np.arange(3456).reshape(24, 12, 12),
                         long_name="air_temperature",
                         dim_coords_and_dims=[(long_time_coord, 0),
                                              (latitude, 1),
                                              (longitude, 2)],
                         units="K")

        orography = Cube(np.ones((12, 12)),
                         long_name="surface_altitude",
                         dim_coords_and_dims=[(latitude, 0),
                                              (longitude, 1)],
                         units="m")

        # Western half of grid at altitude 0, eastern half at 10.
        # Note that the pressure_on_height_levels data is left unchanged,
        # so it is as if there is a sharp front running up the grid with
        # differing pressures on either side at equivalent heights above
        # the surface (e.g. east 1000hPa at 0m AMSL, west 1000hPa at 10m AMSL).
        # So there is higher pressure in the west.
        orography.data[0:10] = 0
        orography.data[10:] = 10
        ancillary_data = {}
        ancillary_data['orography'] = orography

        additional_data = {}
        adlist = CubeList()
        adlist.append(cube)
        additional_data['air_temperature'] = adlist

        data_indices = [list(data.nonzero()[0]),
                        list(data.nonzero()[1])]

        self.cube = cube
        self.long_cube = long_cube
        self.data = data
        self.time_dt = time_dt
        self.time_extract = time_extract
        self.data_indices = data_indices
        self.ancillary_data = ancillary_data
        self.additional_data = additional_data


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
            self.cube, self.DISTANCE,
            max_distance_in_grid_cells=self.MAX_DISTANCE_IN_GRID_CELLS)
        self.assertEqual(result, (3, 3))

    def test_basic_distance_to_grid_cells_float(self):
        """Test the distance in metres to grid cell conversion."""
        result = convert_distance_into_number_of_grid_cells(
            self.cube, self.DISTANCE,
            max_distance_in_grid_cells=self.MAX_DISTANCE_IN_GRID_CELLS,
            int_grid_cells=False)
        self.assertEqual(result, (3.05, 3.05))

    def test_basic_no_limit(self):
        """Test the distance in metres to grid cell conversion still works when
        the maximum distance limit is not explicitly set."""
        result = convert_distance_into_number_of_grid_cells(
            self.cube, self.DISTANCE)
        self.assertEqual(result, (3, 3))

    def test_basic_distance_to_grid_cells_different_max_distance(self):
        """
        Test the distance in metres to grid cell conversion for an alternative
        max distance in grid cells.
        """
        max_distance_in_grid_cells = 50
        result = convert_distance_into_number_of_grid_cells(
            self.cube, self.DISTANCE,
            max_distance_in_grid_cells=max_distance_in_grid_cells)
        self.assertEqual(result, (3, 3))

    def test_basic_distance_to_grid_cells_km_grid(self):
        """Test the distance-to-grid-cell conversion, grid in km."""
        self.cube.coord("projection_x_coordinate").convert_units("kilometres")
        self.cube.coord("projection_y_coordinate").convert_units("kilometres")
        result = convert_distance_into_number_of_grid_cells(
            self.cube, self.DISTANCE)
        self.assertEqual(result, (3, 3))

    def test_single_point_lat_long(self):
        """Test behaviour for a single grid cell on lat long grid."""
        cube = set_up_cube_lat_long()
        msg = "Invalid grid: projection_x/y coords required"
        with self.assertRaisesRegex(ValueError, msg):
            convert_distance_into_number_of_grid_cells(
                cube, self.DISTANCE)

    def test_single_point_range_negative(self):
        """Test behaviour with a non-zero point with negative range."""
        distance = -1.0 * self.DISTANCE
        msg = "Distance of -6100.0m gives a negative cell extent"
        with self.assertRaisesRegex(ValueError, msg):
            convert_distance_into_number_of_grid_cells(
                self.cube, distance)

    def test_single_point_range_0(self):
        """Test behaviour with a non-zero point with zero range."""
        distance = 5
        msg = "Distance of 5m gives zero cell extent"
        with self.assertRaisesRegex(ValueError, msg):
            convert_distance_into_number_of_grid_cells(
                self.cube, distance)

    def test_single_point_range_lots(self):
        """Test behaviour with a non-zero point with unhandleable range."""
        distance = 40000.0
        max_distance_in_grid_cells = 10
        msg = "Distance of 40000.0m exceeds maximum permitted"
        with self.assertRaisesRegex(ValueError, msg):
            convert_distance_into_number_of_grid_cells(
                self.cube, distance,
                max_distance_in_grid_cells=max_distance_in_grid_cells)

    def test_single_point_range_greater_than_domain(self):
        """Test correct exception raised when the distance is larger than the
           corner-to-corner distance of the domain."""
        distance = 42500.0
        msg = "Distance of 42500.0m exceeds max domain distance of "
        with self.assertRaisesRegex(ValueError, msg):
            convert_distance_into_number_of_grid_cells(
                self.cube, distance)


class Test_convert_number_of_grid_cells_into_distance(IrisTest):

    """Test the convert_number_of_grid_cells_into_distance method"""

    def setUp(self):
        """Set up a cube with x and y coordinates"""
        data = np.ones((3, 4))
        self.cube = Cube(data, standard_name="air_temperature",)
        self.cube.add_dim_coord(
            DimCoord(np.linspace(2000.0, 6000.0, 3),
                     'projection_x_coordinate', units='m'), 0)
        self.cube.add_dim_coord(
            DimCoord(np.linspace(2000.0, 8000.0, 4),
                     "projection_y_coordinate", units='m'), 1)

    def test_basic(self):
        """Test the function does what it's meant to in a simple case."""
        result_radius = convert_number_of_grid_cells_into_distance(
            self.cube, 2)
        expected_result = 4000.0
        self.assertAlmostEqual(result_radius, expected_result)
        self.assertIs(type(expected_result), float)

    def test_check_input_in_km(self):
        """
        Test that the output is still in metres when the input coordinates
        are in a different unit.
        """
        result_radius = convert_number_of_grid_cells_into_distance(
            self.cube, 2)
        for coord in self.cube.coords():
            coord.convert_units("km")
        expected_result = 4000.0
        self.assertAlmostEqual(result_radius, expected_result)
        self.assertIs(type(expected_result), float)

    def test_not_equal_areas(self):
        """
        Check it raises an error when the input is not an equal areas grid.
        """

        self.cube.remove_coord("projection_x_coordinate")
        self.cube.add_dim_coord(
            DimCoord(np.linspace(200.0, 600.0, 3),
                     'projection_x_coordinate', units='m'), 0)
        with self.assertRaisesRegex(
                ValueError,
                "The size of the intervals along the x and y axis"
                " should be equal."):
            convert_number_of_grid_cells_into_distance(self.cube, 2)

    def test_check_different_input_radius(self):
        """Check it works for different input values."""
        result_radius = convert_number_of_grid_cells_into_distance(
            self.cube, 5)
        expected_result = 10000.0
        self.assertAlmostEqual(result_radius, expected_result)
        self.assertIs(type(expected_result), float)


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
