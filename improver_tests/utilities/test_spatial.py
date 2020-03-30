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
"""Unit tests for the distance_to_number_of_grid_cells function from
 spatial.py."""

from datetime import datetime as dt

import cartopy.crs as ccrs
import cf_units
import numpy as np
from iris import Constraint, coord_systems
from iris.coord_systems import GeogCS
from iris.coords import AuxCoord, DimCoord
from iris.cube import Cube, CubeList
from iris.tests import IrisTest
from iris.time import PartialDateTime

from improver.utilities.spatial import (
    calculate_grid_spacing, check_if_grid_is_equal_area,
    distance_to_number_of_grid_cells,
    number_of_grid_cells_to_distance, lat_lon_determine,
    transform_grid_to_lat_lon)

from ..nbhood.nbhood.test_BaseNeighbourhoodProcessing import set_up_cube
from ..set_up_test_cubes import set_up_variable_cube


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


class Test_calculate_grid_spacing(IrisTest):
    """Test the calculate_grid_spacing function"""

    def setUp(self):
        """Set up an equal area cube"""
        self.cube = set_up_variable_cube(
            np.ones((5, 5), dtype=np.float32), spatial_grid='equalarea')
        self.spacing = 200000.0
        self.unit = 'metres'
        self.lat_lon_cube = set_up_variable_cube(
            np.ones((5, 5), dtype=np.float32))

    def test_basic(self):
        """Test correct answer is returned from an equal area grid"""
        result = calculate_grid_spacing(self.cube, self.unit)
        self.assertAlmostEqual(result, self.spacing)

    def test_units(self):
        """Test correct answer is returned for coordinates in km"""
        for axis in ['x', 'y']:
            self.cube.coord(axis=axis).convert_units('km')
        result = calculate_grid_spacing(self.cube, self.unit)
        self.assertAlmostEqual(result, self.spacing)
        for axis in ['x', 'y']:
            self.assertEqual(self.cube.coord(axis=axis).units, 'km')

    def test_axis_keyword(self):
        """Test using the other axis"""
        self.cube.coord(axis='y').points = 2*(self.cube.coord(axis='y').points)
        result = calculate_grid_spacing(self.cube, self.unit, axis='y')
        self.assertAlmostEqual(result, 2*self.spacing)

    def test_lat_lon_equal_spacing(self):
        """Test outputs with lat-lon grid in degrees"""
        result = calculate_grid_spacing(self.lat_lon_cube, 'degrees')
        self.assertAlmostEqual(result, 10.0)

    def test_incorrect_units(self):
        """Test ValueError for incorrect units"""
        msg = "Unable to convert from"
        with self.assertRaisesRegex(ValueError, msg):
            calculate_grid_spacing(self.lat_lon_cube, self.unit)


class Test_convert_distance_into_number_of_grid_cells(IrisTest):

    """Test conversion of distance in metres into number of grid cells."""

    def setUp(self):
        """Set up the cube."""
        self.DISTANCE = 6100
        self.cube = set_up_cube()

    def test_basic_distance_to_grid_cells(self):
        """Test the distance in metres to grid cell conversion along the
        x-axis (default)."""
        result = distance_to_number_of_grid_cells(self.cube, self.DISTANCE)
        self.assertEqual(result, 3)

    def test_distance_to_grid_cells_other_axis(self):
        """Test the distance in metres to grid cell conversion along the
        y-axis."""
        self.cube.coord(axis='y').points = 0.5*self.cube.coord(axis='y').points
        result = distance_to_number_of_grid_cells(self.cube, self.DISTANCE,
                                                  axis='y')
        self.assertEqual(result, 6)

    def test_basic_distance_to_grid_cells_float(self):
        """Test the distance in metres to grid cell conversion."""
        result = distance_to_number_of_grid_cells(self.cube, self.DISTANCE,
                                                  return_int=False)
        self.assertEqual(result, 3.05)

    def test_max_distance(self):
        """
        Test the distance in metres to grid cell conversion within a maximum
        distance in grid cells.
        """
        result = distance_to_number_of_grid_cells(self.cube, self.DISTANCE)
        self.assertEqual(result, 3)

    def test_basic_distance_to_grid_cells_km_grid(self):
        """Test the distance-to-grid-cell conversion, grid in km."""
        self.cube.coord("projection_x_coordinate").convert_units("kilometres")
        self.cube.coord("projection_y_coordinate").convert_units("kilometres")
        result = distance_to_number_of_grid_cells(self.cube, self.DISTANCE)
        self.assertEqual(result, 3)

    def test_error_negative_distance(self):
        """Test behaviour with a non-zero point with negative range."""
        distance = -1.0 * self.DISTANCE
        msg = "Please specify a positive distance in metres"
        with self.assertRaisesRegex(ValueError, msg):
            distance_to_number_of_grid_cells(self.cube, distance)

    def test_error_zero_grid_cell_range(self):
        """Test behaviour with a non-zero point with zero range."""
        distance = 5
        msg = "Distance of 5m gives zero cell extent"
        with self.assertRaisesRegex(ValueError, msg):
            distance_to_number_of_grid_cells(self.cube, distance)

    def test_single_point_range_0(self):
        """Test behaviour with zero range."""
        cube = self.cube
        radius = 0.
        msg = "Please specify a positive distance in metres"
        with self.assertRaisesRegex(ValueError, msg):
            distance_to_number_of_grid_cells(cube, radius)


class Test_number_of_grid_cells_to_distance(IrisTest):

    """Test the number_of_grid_cells_to_distance method"""

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
        result_radius = number_of_grid_cells_to_distance(self.cube, 2)
        expected_result = 4000.0
        self.assertAlmostEqual(result_radius, expected_result)
        self.assertIs(type(expected_result), float)

    def test_check_input_in_km(self):
        """
        Test that the output is still in metres when the input coordinates
        are in a different unit.
        """
        result_radius = number_of_grid_cells_to_distance(self.cube, 2)
        for coord in self.cube.coords():
            coord.convert_units("km")
        expected_result = 4000.0
        self.assertAlmostEqual(result_radius, expected_result)
        self.assertIs(type(expected_result), float)

    def test_check_different_input_radius(self):
        """Check it works for different input values."""
        result_radius = number_of_grid_cells_to_distance(self.cube, 5)
        expected_result = 10000.0
        self.assertAlmostEqual(result_radius, expected_result)
        self.assertIs(type(expected_result), float)


class Test_check_if_grid_is_equal_area(IrisTest):

    """Test that the grid is an equal area grid."""

    def setUp(self):
        """Set up an equal area cube"""
        self.cube = set_up_variable_cube(
            np.ones((16, 16), dtype=np.float32), spatial_grid='equalarea')
        self.lat_lon_cube = set_up_variable_cube(
            np.ones((5, 5), dtype=np.float32))

    def test_equal_area(self):
        """Test an that no exception is raised if the x and y coordinates
        are on an equal area grid"""
        self.assertIsNone(check_if_grid_is_equal_area(self.cube))

    def test_allow_negative_stride(self):
        """Test no errors raised if cube has negative stride in x and y axes"""
        coord_points_x = np.arange(-20000, -52000., -2000)
        coord_points_y = np.arange(30000., -2000, -2000)
        self.cube.coord("projection_x_coordinate").points = coord_points_x
        self.cube.coord("projection_y_coordinate").points = coord_points_y
        self.assertIsNone(check_if_grid_is_equal_area(self.cube))

    def test_lat_lon_failure(self):
        """Test that a lat/lon cube fails"""
        msg = "Unable to convert from"
        with self.assertRaisesRegex(ValueError, msg):
            check_if_grid_is_equal_area(self.lat_lon_cube)

    def test_lat_lon_failure_with_override(self):
        """Test that a lat/lon cube still fails when 'require_equal_xy_spacing'
        is set to False"""
        msg = "Unable to convert from"
        with self.assertRaisesRegex(ValueError, msg):
            check_if_grid_is_equal_area(
                self.lat_lon_cube, require_equal_xy_spacing=False)

    def test_non_equal_xy_spacing(self):
        """Test that the cubes have an equal areas grid"""
        self.cube.coord(axis='x').points = 2*self.cube.coord(axis='x').points
        msg = "Grid does not have equal spacing in x and y"
        with self.assertRaisesRegex(ValueError, msg):
            check_if_grid_is_equal_area(self.cube)

    def test_non_equal_xy_spacing_override(self):
        """Test that the requirement for equal x and y spacing can be
        overridden"""
        self.cube.coord(axis='x').points = 2*self.cube.coord(axis='x').points
        self.assertIsNone(
            check_if_grid_is_equal_area(
                self.cube, require_equal_xy_spacing=False))


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
