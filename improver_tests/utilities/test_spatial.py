# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown copyright. The Met Office.
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
from copy import copy
from datetime import datetime as dt

import cartopy.crs as ccrs
import cf_units
import numpy as np
import pytest
from iris import Constraint, coord_systems
from iris.coord_systems import GeogCS
from iris.coords import AuxCoord, CellMethod, DimCoord
from iris.cube import Cube, CubeList
from iris.tests import IrisTest
from iris.time import PartialDateTime
from numpy.testing import assert_almost_equal

from improver.synthetic_data.set_up_test_cubes import (
    set_up_probability_cube,
    set_up_variable_cube,
)
from improver.utilities.spatial import (
    calculate_grid_spacing,
    check_if_grid_is_equal_area,
    distance_to_number_of_grid_cells,
    get_grid_y_x_values,
    lat_lon_determine,
    number_of_grid_cells_to_distance,
    transform_grid_to_lat_lon,
    update_name_and_vicinity_coord,
)


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
        latitude = DimCoord(
            latitudes,
            standard_name="latitude",
            units="degrees",
            coord_system=GeogCS(6371229.0),
        )
        longitude = DimCoord(
            longitudes,
            standard_name="longitude",
            units="degrees",
            coord_system=GeogCS(6371229.0),
            circular=True,
        )

        # Use time of 2017-02-17 06:00:00
        time = DimCoord(
            [1487311200],
            standard_name="time",
            units=cf_units.Unit(
                "seconds since 1970-01-01 00:00:00", calendar="gregorian"
            ),
        )
        long_time_coord = DimCoord(
            list(range(1487311200, 1487397600, 3600)),
            standard_name="time",
            units=cf_units.Unit(
                "seconds since 1970-01-01 00:00:00", calendar="gregorian"
            ),
        )

        time_dt = dt(2017, 2, 17, 6, 0)
        time_extract = Constraint(
            time=lambda cell: cell.point
            == PartialDateTime(time_dt.year, time_dt.month, time_dt.day, time_dt.hour)
        )

        cube = Cube(
            data.reshape((1, 12, 12)),
            long_name="air_temperature",
            dim_coords_and_dims=[(time, 0), (latitude, 1), (longitude, 2)],
            units="K",
        )

        long_cube = Cube(
            np.arange(3456).reshape(24, 12, 12),
            long_name="air_temperature",
            dim_coords_and_dims=[(long_time_coord, 0), (latitude, 1), (longitude, 2)],
            units="K",
        )

        orography = Cube(
            np.ones((12, 12)),
            long_name="surface_altitude",
            dim_coords_and_dims=[(latitude, 0), (longitude, 1)],
            units="m",
        )

        # Western half of grid at altitude 0, eastern half at 10.
        # Note that the pressure_on_height_levels data is left unchanged,
        # so it is as if there is a sharp front running up the grid with
        # differing pressures on either side at equivalent heights above
        # the surface (e.g. east 1000hPa at 0m AMSL, west 1000hPa at 10m AMSL).
        # So there is higher pressure in the west.
        orography.data[0:10] = 0
        orography.data[10:] = 10
        ancillary_data = {}
        ancillary_data["orography"] = orography

        additional_data = {}
        adlist = CubeList()
        adlist.append(cube)
        additional_data["air_temperature"] = adlist

        data_indices = [list(data.nonzero()[0]), list(data.nonzero()[1])]

        self.cube = cube
        self.long_cube = long_cube
        self.data = data
        self.time_dt = time_dt
        self.time_extract = time_extract
        self.data_indices = data_indices
        self.ancillary_data = ancillary_data
        self.additional_data = additional_data


class GridSpacingTest(IrisTest):
    """Base class for testing the calculate_grid_spacing function"""

    def setUp(self):
        """Set up an equal area cube"""
        self.cube = set_up_variable_cube(
            np.ones((5, 5), dtype=np.float32), spatial_grid="equalarea"
        )
        self.spacing = 2000.0
        self.unit = "metres"
        self.lat_lon_cube = set_up_variable_cube(np.ones((5, 5), dtype=np.float32))


class Test_calculate_grid_spacing_without_tolerance(GridSpacingTest):
    """Test the calculate_grid_spacing function without tolerance"""

    def test_basic(self):
        """Test correct answer is returned from an equal area grid"""
        result = calculate_grid_spacing(self.cube, self.unit)
        self.assertAlmostEqual(result, self.spacing)

    def test_negative_x(self):
        """Test positive answer is returned from a negative-striding x-axis"""
        result = calculate_grid_spacing(self.cube[..., ::-1], self.unit, axis="x")
        self.assertAlmostEqual(result, self.spacing)

    def test_negative_y(self):
        """Test positive answer is returned from a negative-striding y-axis"""
        result = calculate_grid_spacing(self.cube[..., ::-1, :], self.unit, axis="y")
        self.assertAlmostEqual(result, self.spacing)

    def test_units(self):
        """Test correct answer is returned for coordinates in km"""
        for axis in ["x", "y"]:
            self.cube.coord(axis=axis).convert_units("km")
        result = calculate_grid_spacing(self.cube, self.unit)
        self.assertAlmostEqual(result, self.spacing)
        for axis in ["x", "y"]:
            self.assertEqual(self.cube.coord(axis=axis).units, "km")

    def test_axis_keyword(self):
        """Test using the other axis"""
        self.cube.coord(axis="y").points = 2 * (self.cube.coord(axis="y").points)
        result = calculate_grid_spacing(self.cube, self.unit, axis="y")
        self.assertAlmostEqual(result, 2 * self.spacing)

    def test_lat_lon_equal_spacing(self):
        """Test grid spacing outputs with lat-lon grid in degrees"""
        result = calculate_grid_spacing(self.lat_lon_cube, "degrees")
        self.assertAlmostEqual(result, 10.0)

    def test_incorrect_units(self):
        """Test ValueError for incorrect units"""
        msg = "Unable to convert from"
        with self.assertRaisesRegex(ValueError, msg):
            calculate_grid_spacing(self.lat_lon_cube, self.unit)


class Test_calculate_grid_spacing_with_tolerance(GridSpacingTest):
    """Test the calculate_grid_spacing function with tolerance settings"""

    def setUp(self):
        """Set up an equal area cube"""
        super().setUp()
        self.longitude_points = np.array(
            [-19.99999, -10.0, 0.0, 10.0, 20.00001], dtype=np.float32
        )
        self.longitude_points_thirds = np.array(
            [160.0, 160.33333, 160.66667, 161.0, 161.33333], dtype=np.float32
        )
        self.rtol = 1.0e-5
        self.expected = 10.0
        self.expected_thirds = 0.33333
        self.rtol_thirds = 4.0e-5

    def test_lat_lon_equal_spacing(self):
        """Test grid spacing outputs with lat-lon grid with tolerance"""
        self.lat_lon_cube.coord("longitude").points = self.longitude_points
        result = calculate_grid_spacing(self.lat_lon_cube, "degrees", rtol=self.rtol)
        self.assertAlmostEqual(result, self.expected)

    def test_lat_lon_negative_spacing(self):
        """Test negative-striding axes grid spacing is positive with lat-lon grid in degrees"""
        for axis in "yx":
            self.lat_lon_cube.coord(axis=axis).points = self.lat_lon_cube.coord(
                axis=axis
            ).points[::-1]
            result = calculate_grid_spacing(
                self.lat_lon_cube, "degrees", rtol=self.rtol, axis=axis
            )
            self.assertAlmostEqual(result, self.expected)

    def test_lat_lon_not_equal_spacing(self):
        """Test outputs with lat-lon grid in degrees"""
        points = self.longitude_points
        points[0] = -19.998
        self.lat_lon_cube.coord("longitude").points = points
        msg = "Coordinate longitude points are not equally spaced"
        with self.assertRaisesRegex(ValueError, msg):
            calculate_grid_spacing(self.lat_lon_cube, "degrees", rtol=self.rtol)

    def test_lat_lon_equal_spacing_recurring_decimal_spacing_fails(self):
        """Test grid spacing with lat-lon grid with with 1/3 degree
        intervals with tolerance of 1.0e-5"""
        self.lat_lon_cube.coord("longitude").points = self.longitude_points_thirds
        msg = "Coordinate longitude points are not equally spaced"
        with self.assertRaisesRegex(ValueError, msg):
            calculate_grid_spacing(self.lat_lon_cube, "degrees", rtol=self.rtol)

    def test_lat_lon_equal_spacing_recurring_decimal_spacing_passes(self):
        """Test grid spacing outputs with lat-lon grid with 1/3 degree
        intervals with tolerance of 4.0e-5"""
        self.lat_lon_cube.coord("longitude").points = self.longitude_points_thirds
        result = calculate_grid_spacing(
            self.lat_lon_cube, "degrees", rtol=self.rtol_thirds
        )
        self.assertAlmostEqual(result, self.expected_thirds, places=5)


class Test_convert_distance_into_number_of_grid_cells(IrisTest):

    """Test conversion of distance in metres into number of grid cells."""

    def setUp(self):
        """Set up the cube."""
        self.DISTANCE = 6100
        data = np.ones((1, 16, 16), dtype=np.float32)
        data[:, 7, 7] = 0.0
        self.cube = set_up_variable_cube(
            data, "precipitation_amount", "kg m^-2", "equalarea",
        )

    def test_basic_distance_to_grid_cells(self):
        """Test the distance in metres to grid cell conversion along the
        x-axis (default)."""
        result = distance_to_number_of_grid_cells(self.cube, self.DISTANCE)
        self.assertEqual(result, 3)

    def test_distance_to_grid_cells_other_axis(self):
        """Test the distance in metres to grid cell conversion along the
        y-axis."""
        self.cube.coord(axis="y").points = 0.5 * self.cube.coord(axis="y").points
        result = distance_to_number_of_grid_cells(self.cube, self.DISTANCE, axis="y")
        self.assertEqual(result, 6)

    def test_basic_distance_to_grid_cells_float(self):
        """Test the distance in metres to grid cell conversion."""
        result = distance_to_number_of_grid_cells(
            self.cube, self.DISTANCE, return_int=False
        )
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
        radius = 0.0
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
            DimCoord(
                np.linspace(2000.0, 6000.0, 3), "projection_x_coordinate", units="m"
            ),
            0,
        )
        self.cube.add_dim_coord(
            DimCoord(
                np.linspace(2000.0, 8000.0, 4), "projection_y_coordinate", units="m"
            ),
            1,
        )

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
            np.ones((16, 16), dtype=np.float32), spatial_grid="equalarea"
        )
        self.lat_lon_cube = set_up_variable_cube(np.ones((5, 5), dtype=np.float32))

    def test_equal_area(self):
        """Test an that no exception is raised if the x and y coordinates
        are on an equal area grid"""
        self.assertIsNone(check_if_grid_is_equal_area(self.cube))

    def test_allow_negative_stride(self):
        """Test no errors raised if cube has negative stride in x and y axes"""
        coord_points_x = np.arange(-20000, -52000.0, -2000)
        coord_points_y = np.arange(30000.0, -2000, -2000)
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
                self.lat_lon_cube, require_equal_xy_spacing=False
            )

    def test_non_equal_xy_spacing(self):
        """Test that the cubes have an equal areas grid"""
        self.cube.coord(axis="x").points = 2 * self.cube.coord(axis="x").points
        msg = "Grid does not have equal spacing in x and y"
        with self.assertRaisesRegex(ValueError, msg):
            check_if_grid_is_equal_area(self.cube)

    def test_non_equal_xy_spacing_override(self):
        """Test that the requirement for equal x and y spacing can be
        overridden"""
        self.cube.coord(axis="x").points = 2 * self.cube.coord(axis="x").points
        self.assertIsNone(
            check_if_grid_is_equal_area(self.cube, require_equal_xy_spacing=False)
        )


class Test_lat_lon_determine(Test_common_functions):
    """Test function that tests projections used in diagnostic cubes."""

    @pytest.mark.xfail
    def test_projection_test(self):
        """Test identification of non-lat/lon projections."""
        src_crs = ccrs.PlateCarree()
        trg_crs = ccrs.LambertConformal(central_longitude=50, central_latitude=10)
        trg_crs_iris = coord_systems.LambertConformal(central_lon=50, central_lat=10)
        lons = self.cube.coord("longitude").points
        lats = self.cube.coord("latitude").points
        xvals, yvals = [], []
        for lon, lat in zip(lons, lats):
            x_trg, y_trg = trg_crs.transform_point(lon, lat, src_crs)
            xvals.append(x_trg)
            yvals.append(y_trg)

        new_x = AuxCoord(
            xvals,
            standard_name="projection_x_coordinate",
            units="m",
            coord_system=trg_crs_iris,
        )
        new_y = AuxCoord(
            yvals,
            standard_name="projection_y_coordinate",
            units="m",
            coord_system=trg_crs_iris,
        )

        cube = Cube(
            self.cube.data,
            long_name="air_temperature",
            dim_coords_and_dims=[(self.cube.coord("time"), 0)],
            aux_coords_and_dims=[(new_y, 1), (new_x, 2)],
            units="K",
        )

        plugin = lat_lon_determine
        expected = trg_crs
        result = plugin(cube)
        self.assertEqual(expected, result)


class Test_get_grid_y_x_values(IrisTest):
    """Test function that extract the points in the cube into
    grid of y and x coordinate values."""

    def setUp(self):
        """Set up the cube."""
        data = np.ones((1, 2, 4), dtype=np.float32)
        self.latlon_cube = set_up_variable_cube(
            data, "precipitation_amount", "kg m^-2",
        )
        self.expected_lons = np.array([-15, -5, 5, 15, -15, -5, 5, 15]).reshape(2, 4)
        self.expected_lats = np.array([-5, -5, -5, -5, 5, 5, 5, 5]).reshape(2, 4)
        self.equalarea_cube = set_up_variable_cube(
            data,
            "precipitation_amount",
            "kg m^-2",
            "equalarea",
            grid_spacing=2000,
            domain_corner=(-1000, -1000),
        )
        self.expected_proj_x = np.array(
            [-1000, 1000, 3000, 5000, -1000, 1000, 3000, 5000]
        ).reshape(2, 4)
        self.expected_proj_y = np.array(
            [-1000, -1000, -1000, -1000, 1000, 1000, 1000, 1000]
        ).reshape(2, 4)

    def test_lat_lon_grid(self):
        """Test extraction of over lat/lon grid."""
        result_y_points, result_x_points = get_grid_y_x_values(self.latlon_cube)
        self.assertIsInstance(result_y_points, np.ndarray)
        self.assertIsInstance(result_x_points, np.ndarray)
        assert_almost_equal(result_x_points, self.expected_lons)
        assert_almost_equal(result_y_points, self.expected_lats)

    def test_equal_area_grid(self):
        """Test extraction of over lat/lon grid."""
        result_y_points, result_x_points = get_grid_y_x_values(self.equalarea_cube)
        self.assertIsInstance(result_y_points, np.ndarray)
        self.assertIsInstance(result_x_points, np.ndarray)
        assert_almost_equal(result_x_points, self.expected_proj_x)
        assert_almost_equal(result_y_points, self.expected_proj_y)


class Test_transform_grid_to_lat_lon(IrisTest):
    """Test function that transforms the points in the cube
    into grid of latitudes and longitudes."""

    def setUp(self):
        """Set up the cube."""
        data = np.ones((1, 2, 2), dtype=np.float32)
        self.cube = set_up_variable_cube(
            data,
            "precipitation_amount",
            "kg m^-2",
            "equalarea",
            grid_spacing=2000000,
            domain_corner=(-1000000, -1000000),
        )
        self.expected_lons = np.array(
            [-15.23434831, 10.23434831, -22.21952954, 17.21952954]
        ).reshape(2, 2)
        self.expected_lats = np.array(
            [45.10433161, 45.10433161, 62.57973333, 62.57973333]
        ).reshape(2, 2)

    def test_transform_grid(self):
        """Test transformation of grid for equal area grid with spatial
        coordinates defined in metres."""
        result_lats, result_lons = transform_grid_to_lat_lon(self.cube)
        self.assertIsInstance(result_lats, np.ndarray)
        self.assertIsInstance(result_lons, np.ndarray)
        assert_almost_equal(result_lons, self.expected_lons)
        assert_almost_equal(result_lats, self.expected_lats)

    def test_non_metre_input(self):
        """Test transformation of grid for equal area grid with spatial
        coordinates defined in kilometres."""

        self.cube.coord(axis="x").convert_units("km")
        self.cube.coord(axis="y").convert_units("km")

        result_lats, result_lons = transform_grid_to_lat_lon(self.cube)
        self.assertIsInstance(result_lats, np.ndarray)
        self.assertIsInstance(result_lons, np.ndarray)
        assert_almost_equal(result_lons, self.expected_lons)
        assert_almost_equal(result_lats, self.expected_lats)

    def test_exception_for_degrees_input(self):
        """Test that an exception is raised if the input cube has spatial
        coordinates with units that cannot be converted to the default unit of
        the projection."""
        self.cube.coord(axis="x").units = "degrees"
        msg = "Cube passed to transform_grid_to_lat_lon does not have an x coordinate"
        with self.assertRaisesRegex(ValueError, msg):
            transform_grid_to_lat_lon(self.cube)


@pytest.mark.parametrize("input_has_coord", (True, False))
@pytest.mark.parametrize("vicinity_radius", (2000.0, 10000.0, 20000.0))
def test_update_name_and_vicinity_coord(vicinity_radius, input_has_coord):
    """Test that the vicinity_radius correctly updates the right coord.
    The update_diagnostic_name method already has tests covering its
    functionality."""
    input_name_suffix = "_in_vicinity"
    source_name = "lwe_thickness_of_precipitation_amount"
    kwargs = {}
    if input_has_coord:
        kwargs["include_scalar_coords"] = [
            DimCoord(
                np.array([10000.0], dtype=np.float32),
                long_name="radius_of_vicinity",
                units="m",
            )
        ]
    cube = set_up_probability_cube(
        np.zeros((2, 2, 2), dtype=np.float32),
        [0, 1],
        f"{source_name}{input_name_suffix}",
        "mm",
        spatial_grid="equalarea",
        **kwargs,
    )
    cube.add_cell_method(CellMethod("mean", "time", comments=f"of {source_name}"))
    new_base_name = "lwe_thickness_of_precipitation_amount"
    new_name = f"{new_base_name}_in_variable_vicinity"
    expected_value = copy(vicinity_radius)
    update_name_and_vicinity_coord(cube, new_name, vicinity_radius)
    assert np.allclose(cube.coord("radius_of_vicinity").points, expected_value)
    coord_comment = cube.coord("radius_of_vicinity").attributes.get("comment")
    if input_has_coord:
        assert coord_comment is None
    else:
        assert coord_comment == "Maximum"
