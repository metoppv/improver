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
"""
Unit tests for cube setup functions
"""

import unittest
from datetime import datetime

import iris
import numpy as np
from iris.tests import IrisTest

from improver.grids import GLOBAL_GRID_CCRS, STANDARD_GRID_CCRS
from improver.metadata.check_datatypes import check_mandatory_standards
from improver.metadata.constants.time_types import TIME_COORDS
from improver.metadata.probabilistic import find_threshold_coordinate
from improver.synthetic_data.set_up_test_cubes import (
    add_coordinate,
    construct_scalar_time_coords,
    construct_yx_coords,
    set_up_percentile_cube,
    set_up_probability_cube,
    set_up_variable_cube,
)
from improver.utilities.cube_manipulation import get_dim_coord_names
from improver.utilities.temporal import iris_time_to_datetime


class Test_construct_yx_coords(IrisTest):
    """Test the construct_yx_coords method"""

    def test_lat_lon(self):
        """Test coordinates created for a lat-lon grid"""
        y_coord, x_coord = construct_yx_coords(4, 3, "latlon")
        self.assertEqual(y_coord.name(), "latitude")
        self.assertEqual(x_coord.name(), "longitude")
        for crd in [y_coord, x_coord]:
            self.assertEqual(crd.units, "degrees")
            self.assertEqual(crd.dtype, np.float32)
            self.assertEqual(crd.coord_system, GLOBAL_GRID_CCRS)
        self.assertEqual(len(y_coord.points), 4)
        self.assertEqual(len(x_coord.points), 3)

    def test_lat_lon_values(self):
        """Test latitude and longitude point values are as expected"""
        y_coord, x_coord = construct_yx_coords(3, 3, "latlon")
        self.assertArrayAlmostEqual(x_coord.points, [-10.0, 0.0, 10.0])
        self.assertArrayAlmostEqual(y_coord.points, [-10.0, 0.0, 10.0])

    def test_lat_lon_grid_spacing(self):
        """Test latitude and longitude point values created around 0,0 with
        provided grid spacing"""
        y_coord, x_coord = construct_yx_coords(3, 3, "latlon", grid_spacing=1)
        self.assertArrayEqual(x_coord.points, [-1.0, 0.0, 1.0])
        self.assertArrayEqual(y_coord.points, [-1.0, 0.0, 1.0])

        y_coord, x_coord = construct_yx_coords(4, 4, "latlon", grid_spacing=1)
        self.assertArrayEqual(x_coord.points, [-1.5, -0.5, 0.5, 1.5])
        self.assertArrayEqual(y_coord.points, [-1.5, -0.5, 0.5, 1.5])

    def test_lat_lon_grid_spacing_domain_corner(self):
        """Test latitude and longitude point values start at domain corner
        with provided grid spacing"""
        y_coord, x_coord = construct_yx_coords(
            3, 3, "latlon", grid_spacing=2, domain_corner=(15, 12)
        )
        self.assertArrayEqual(x_coord.points, [12.0, 14.0, 16.0])
        self.assertArrayEqual(y_coord.points, [15.0, 17.0, 19.0])

    def test_lat_lon_domain_corner(self):
        """Test grid points generated with default grid spacing if domain corner
        provided and grid spacing not provided"""
        y_coord, x_coord = construct_yx_coords(3, 3, "latlon", domain_corner=(0, 0))
        self.assertArrayEqual(x_coord.points, [0.0, 10.0, 20.0])
        self.assertArrayEqual(y_coord.points, [0.0, 10.0, 20.0])

    def test_proj_xy(self):
        """Test coordinates created for an equal area grid"""
        y_coord, x_coord = construct_yx_coords(4, 3, "equalarea")
        self.assertEqual(y_coord.name(), "projection_y_coordinate")
        self.assertEqual(x_coord.name(), "projection_x_coordinate")
        for crd in [y_coord, x_coord]:
            self.assertEqual(crd.units, "metres")
            self.assertEqual(crd.dtype, np.float32)
            self.assertEqual(crd.coord_system, STANDARD_GRID_CCRS)
        self.assertEqual(len(y_coord.points), 4)
        self.assertEqual(len(x_coord.points), 3)

    def test_equal_area_grid_spacing(self):
        """Test projection_y_coordinate and projection_x_coordinate point
        values created around 0,0 with provided grid spacing"""
        y_coord, x_coord = construct_yx_coords(3, 3, "equalarea", grid_spacing=1)
        self.assertArrayEqual(x_coord.points, [-1.0, 0.0, 1.0])
        self.assertArrayEqual(y_coord.points, [-1.0, 0.0, 1.0])

        y_coord, x_coord = construct_yx_coords(4, 4, "equalarea", grid_spacing=1)
        self.assertArrayEqual(x_coord.points, [-1.5, -0.5, 0.5, 1.5])
        self.assertArrayEqual(y_coord.points, [-1.5, -0.5, 0.5, 1.5])

    def test_equal_area_grid_spacing_domain_corner(self):
        """Test projection_y_coordinate and projection_x_coordinate point values
        start at domain corner with provided grid spacing"""
        y_coord, x_coord = construct_yx_coords(
            3, 3, "equalarea", grid_spacing=2, domain_corner=(15, 12)
        )
        self.assertArrayEqual(x_coord.points, [12.0, 14.0, 16.0])
        self.assertArrayEqual(y_coord.points, [15.0, 17.0, 19.0])

    def test_equal_area_domain_corner(self):
        """Test grid points generated with default grid spacing if domain
        corner provided and grid spacing not provided"""
        y_coord, x_coord = construct_yx_coords(3, 3, "equalarea", domain_corner=(0, 0))
        self.assertArrayEqual(x_coord.points, [0.0, 2000.0, 4000.0])
        self.assertArrayEqual(y_coord.points, [0.0, 2000.0, 4000.0])

    def test_unknown_spatial_grid(self):
        """Test error raised if spatial_grid unknown"""
        spatial_grid = "unknown"
        msg = "Grid type {} not recognised".format(spatial_grid)
        with self.assertRaisesRegex(ValueError, msg):
            construct_yx_coords(3, 3, spatial_grid, domain_corner=(0, 0))


class Test_construct_scalar_time_coords(IrisTest):
    """Test the construct_scalar_time_coords method"""

    def basic_test(
        self, ref_time_coord="forecast_reference_time", ref_time_kword="frt"
    ):
        """Common method for test_basic and test_blend_time"""
        coord_dims = construct_scalar_time_coords(
            datetime(2017, 12, 1, 14, 0),
            None,
            **{ref_time_kword: datetime(2017, 12, 1, 9, 0)},
        )
        time_coords = [item[0] for item in coord_dims]

        for crd in time_coords:
            self.assertIsInstance(crd, iris.coords.DimCoord)

        self.assertEqual(time_coords[0].name(), "time")
        self.assertEqual(
            iris_time_to_datetime(time_coords[0])[0], datetime(2017, 12, 1, 14, 0)
        )
        self.assertEqual(time_coords[1].name(), ref_time_coord)
        self.assertEqual(
            iris_time_to_datetime(time_coords[1])[0], datetime(2017, 12, 1, 9, 0)
        )
        self.assertEqual(time_coords[2].name(), "forecast_period")
        self.assertEqual(time_coords[2].points[0], 3600 * 5)

        for crd in time_coords[:2]:
            self.assertEqual(crd.dtype, np.int64)
            self.assertEqual(crd.units, "seconds since 1970-01-01 00:00:00")
        self.assertEqual(time_coords[2].units, "seconds")
        self.assertEqual(time_coords[2].dtype, np.int32)

    def test_basic(self):
        """Test times can be set"""
        self.basic_test()

    def test_blend_time(self):
        """Test if blend_time is supplied instead of frt"""
        self.basic_test(ref_time_coord="blend_time", ref_time_kword="blend_time")

    def test_error_frt_and_blend_time_differ(self):
        """Test error is raised if both frt and blend_time are supplied but with different values"""
        msg = (
            "Refusing to create cube with different values for forecast_reference_time and "
            "blend_time"
        )
        with self.assertRaisesRegex(ValueError, msg):
            construct_scalar_time_coords(
                time=datetime(2018, 3, 1, 12, 0),
                frt=datetime(2018, 3, 1, 9, 0),
                blend_time=datetime(2018, 3, 1, 8, 0),
            )

    def test_error_negative_fp(self):
        """Test an error is raised if the calculated forecast period is
        negative"""
        msg = "Cannot set up cube with negative forecast period"
        with self.assertRaisesRegex(ValueError, msg):
            _ = construct_scalar_time_coords(
                datetime(2017, 12, 1, 14, 0), None, datetime(2017, 12, 1, 16, 0)
            )

    def test_error_no_reference_time(self):
        """Test an error is raised if neither a forecast reference time nor blend time are supplied
        """
        msg = (
            "Cannot create forecast_period without either a forecast reference time "
            "or a blend time."
        )
        with self.assertRaisesRegex(ValueError, msg):
            construct_scalar_time_coords(datetime(2017, 12, 1, 14, 0), None)

    def test_time_bounds(self):
        """Test creation of time coordinate with bounds"""
        coord_dims = construct_scalar_time_coords(
            datetime(2017, 12, 1, 14, 0),
            (datetime(2017, 12, 1, 13, 0), datetime(2017, 12, 1, 14, 0)),
            datetime(2017, 12, 1, 9, 0),
        )
        time_coord = coord_dims[0][0]
        self.assertEqual(
            iris_time_to_datetime(time_coord)[0], datetime(2017, 12, 1, 14, 0)
        )
        self.assertEqual(time_coord.bounds[0][0], time_coord.points[0] - 3600)
        self.assertEqual(time_coord.bounds[0][1], time_coord.points[0])

    def test_time_bounds_wrong_order(self):
        """Test time bounds are correctly applied even if supplied in the wrong
        order"""
        coord_dims = construct_scalar_time_coords(
            datetime(2017, 12, 1, 14, 0),
            (datetime(2017, 12, 1, 14, 0), datetime(2017, 12, 1, 13, 0)),
            datetime(2017, 12, 1, 9, 0),
        )
        time_coord = coord_dims[0][0]
        self.assertEqual(
            iris_time_to_datetime(time_coord)[0], datetime(2017, 12, 1, 14, 0)
        )
        self.assertEqual(time_coord.bounds[0][0], time_coord.points[0] - 3600)
        self.assertEqual(time_coord.bounds[0][1], time_coord.points[0])

    def test_error_invalid_time_bounds(self):
        """Test an error is raised if the time point is not between the
        specified bounds"""
        msg = "not within bounds"
        with self.assertRaisesRegex(ValueError, msg):
            _ = construct_scalar_time_coords(
                datetime(2017, 11, 10, 4, 0),
                (datetime(2017, 12, 1, 13, 0), datetime(2017, 12, 1, 14, 0)),
                datetime(2017, 11, 10, 0, 0),
            )


class Test_set_up_variable_cube(IrisTest):
    """Test the set_up_variable_cube base function"""

    def setUp(self):
        """Set up simple temperature data array"""
        self.data = np.linspace(275.0, 284.0, 12).reshape(3, 4).astype(np.float32)
        self.data_3d = np.array([self.data, self.data, self.data])

    def test_defaults(self):
        """Test default arguments produce cube with expected dimensions
        and metadata"""
        result = set_up_variable_cube(self.data)

        # check type, data and attributes
        self.assertIsInstance(result, iris.cube.Cube)
        self.assertEqual(result.standard_name, "air_temperature")
        self.assertEqual(result.name(), "air_temperature")
        self.assertEqual(result.units, "K")
        self.assertArrayAlmostEqual(result.data, self.data)
        self.assertEqual(result.attributes, {})

        # check dimension coordinates
        self.assertEqual(result.coord_dims("latitude"), (0,))
        self.assertEqual(result.coord_dims("longitude"), (1,))

        # check scalar time coordinates
        for time_coord in ["time", "forecast_reference_time"]:
            self.assertEqual(result.coord(time_coord).dtype, np.int64)
        self.assertEqual(result.coord("forecast_period").dtype, np.int32)

        expected_time = datetime(2017, 11, 10, 4, 0)
        time_point = iris_time_to_datetime(result.coord("time"))[0]
        self.assertEqual(time_point, expected_time)

        expected_frt = datetime(2017, 11, 10, 0, 0)
        frt_point = iris_time_to_datetime(result.coord("forecast_reference_time"))[0]
        self.assertEqual(frt_point, expected_frt)

        self.assertEqual(result.coord("forecast_period").units, "seconds")
        self.assertEqual(result.coord("forecast_period").points[0], 14400)

        check_mandatory_standards(result)

    def test_non_standard_name(self):
        """Test non CF standard cube naming"""
        result = set_up_variable_cube(self.data, name="temp_in_the_air")
        self.assertEqual(result.name(), "temp_in_the_air")

    def test_name_and_units(self):
        """Test ability to set data name and units"""
        result = set_up_variable_cube(
            self.data - 273.15, name="wet_bulb_temperature", units="degC"
        )
        self.assertArrayAlmostEqual(result.data, self.data - 273.15)
        self.assertEqual(result.name(), "wet_bulb_temperature")
        self.assertEqual(result.units, "degC")

    def test_attributes(self):
        """Test ability to set attributes"""
        attributes = {"source": "IMPROVER"}
        result = set_up_variable_cube(self.data, attributes=attributes)
        self.assertEqual(result.attributes, attributes)

    def test_spatial_grid(self):
        """Test ability to set up non lat-lon grid"""
        result = set_up_variable_cube(self.data, spatial_grid="equalarea")
        self.assertEqual(result.coord_dims("projection_y_coordinate"), (0,))
        self.assertEqual(result.coord_dims("projection_x_coordinate"), (1,))

    def test_time_points(self):
        """Test ability to configure time and forecast reference time"""
        expected_time = datetime(2018, 3, 1, 12, 0)
        expected_frt = datetime(2018, 3, 1, 9, 0)
        result = set_up_variable_cube(self.data, time=expected_time, frt=expected_frt)
        time_point = iris_time_to_datetime(result.coord("time"))[0]
        self.assertEqual(time_point, expected_time)
        frt_point = iris_time_to_datetime(result.coord("forecast_reference_time"))[0]
        self.assertEqual(frt_point, expected_frt)
        self.assertEqual(result.coord("forecast_period").points[0], 10800)
        self.assertFalse(result.coords("time", dim_coords=True))

    def test_blend_time(self):
        """Test use of blend_time instead of forecast reference time"""
        expected_time = datetime(2018, 3, 1, 12, 0)
        expected_frt = datetime(2018, 3, 1, 9, 0)
        result = set_up_variable_cube(
            self.data, time=expected_time, blend_time=expected_frt
        )
        time_point = iris_time_to_datetime(result.coord("time"))[0]
        self.assertEqual(time_point, expected_time)
        frt_point = iris_time_to_datetime(result.coord("blend_time"))[0]
        self.assertEqual(frt_point, expected_frt)
        self.assertEqual(result.coord("forecast_period").points[0], 10800)
        self.assertFalse(result.coords("time", dim_coords=True))

    def test_height_levels(self):
        """Test height coordinate is added"""
        height_levels = [1.5, 3.0, 4.5]
        expected_units = "m"
        expected_attributes = {"positive": "up"}
        result = set_up_variable_cube(self.data_3d, height_levels=height_levels)
        self.assertArrayAlmostEqual(result.data, self.data_3d)
        self.assertEqual(result.coord_dims("height"), (0,))
        self.assertArrayEqual(result.coord("height").points, np.array(height_levels))
        self.assertEqual(result.coord("height").units, expected_units)
        self.assertEqual(result.coord("height").attributes, expected_attributes)
        self.assertEqual(result.coord_dims("latitude"), (1,))
        self.assertEqual(result.coord_dims("longitude"), (2,))

    def test_pressure_levels(self):
        """Test pressure coordinate is added"""
        height_levels = [90000, 70000, 3000]
        expected_units = "Pa"
        pressure = True
        expected_attributes = {"positive": "down"}
        result = set_up_variable_cube(
            self.data_3d, height_levels=height_levels, pressure=pressure
        )
        self.assertArrayAlmostEqual(result.data, self.data_3d)
        self.assertEqual(result.coord_dims("pressure"), (0,))
        self.assertArrayEqual(result.coord("pressure").points, np.array(height_levels))
        self.assertEqual(result.coord("pressure").units, expected_units)
        self.assertEqual(result.coord("pressure").attributes, expected_attributes)
        self.assertEqual(result.coord_dims("latitude"), (1,))
        self.assertEqual(result.coord_dims("longitude"), (2,))

    def test_realizations_from_data(self):
        """Test realization coordinate is added for 3D data"""
        result = set_up_variable_cube(self.data_3d)
        self.assertArrayAlmostEqual(result.data, self.data_3d)
        self.assertEqual(result.coord_dims("realization"), (0,))
        self.assertArrayEqual(result.coord("realization").points, np.array([0, 1, 2]))
        self.assertEqual(result.coord_dims("latitude"), (1,))
        self.assertEqual(result.coord_dims("longitude"), (2,))

    def test_realizations(self):
        """Test specific realization values"""
        result = set_up_variable_cube(self.data_3d, realizations=np.array([0, 3, 4]))
        self.assertArrayEqual(result.coord("realization").points, np.array([0, 3, 4]))

    def test_error_unmatched_realizations(self):
        """Test error is raised if the realizations provided do not match the
        data dimensions"""
        realizations_len = 4
        data_len = len(self.data_3d[0])
        msg = "Cannot generate {} realizations with data of length {}".format(
            realizations_len, data_len
        )
        with self.assertRaisesRegex(ValueError, msg):
            _ = set_up_variable_cube(
                self.data_3d, realizations=np.arange(realizations_len)
            )

    def test_error_unmatched_height_levels(self):
        """Test error is raised if the heights provided do not match the
        data dimensions"""
        height_levels_len = 4
        data_len = len(self.data_3d[0])
        msg = "Cannot generate {} heights with data of length {}".format(
            height_levels_len, data_len
        )
        with self.assertRaisesRegex(ValueError, msg):
            _ = set_up_variable_cube(
                self.data_3d, height_levels=np.arange(height_levels_len)
            )

    def test_realizations_from_data_height_levels(self):
        """ Tests realizations from data and height coordinates added """
        height_levels = [1.5, 3.0, 4.5]
        data_4d = np.array([self.data_3d, self.data_3d])
        result = set_up_variable_cube(data_4d, height_levels=height_levels)
        self.assertArrayAlmostEqual(result.data, data_4d)
        self.assertEqual(result.coord_dims("realization"), (0,))
        self.assertArrayEqual(result.coord("realization").points, np.array([0, 1]))
        self.assertEqual(result.coord_dims("height"), (1,))
        self.assertArrayEqual(result.coord("height").points, np.array(height_levels))
        self.assertEqual(result.coord_dims("latitude"), (2,))
        self.assertEqual(result.coord_dims("longitude"), (3,))

    def test_realizations_height_levels(self):
        """ Tests realizations and height coordinates added """
        realizations = [0, 3]
        height_levels = [1.5, 3.0, 4.5]
        data_4d = np.array([self.data_3d, self.data_3d])
        result = set_up_variable_cube(
            data_4d, realizations=realizations, height_levels=height_levels
        )
        self.assertArrayAlmostEqual(result.data, data_4d)
        self.assertEqual(result.coord_dims("realization"), (0,))
        self.assertArrayEqual(
            result.coord("realization").points, np.array(realizations)
        )
        self.assertEqual(result.coord_dims("height"), (1,))
        self.assertArrayEqual(result.coord("height").points, np.array(height_levels))
        self.assertEqual(result.coord_dims("latitude"), (2,))
        self.assertEqual(result.coord_dims("longitude"), (3,))

    def test_error_no_height_levels_4d_data(self):
        """ Tests error is raised if 4d data provided but not height_levels """
        data_4d = np.array([self.data_3d, self.data_3d])
        msg = "Height levels must be provided if data has 4 dimensions."
        with self.assertRaisesRegex(ValueError, msg):
            _ = set_up_variable_cube(data_4d)

    def test_error_too_many_dimensions(self):
        """Test error is raised if input cube has more than 4 dimensions"""
        data_5d = np.array([[self.data_3d, self.data_3d], [self.data_3d, self.data_3d]])
        msg = "Expected 2 to 4 dimensions on input data: got 5"
        with self.assertRaisesRegex(ValueError, msg):
            _ = set_up_variable_cube(data_5d)

    def test_error_not_enough_dimensions(self):
        """Test error is raised if 3D input cube and both realizations and heights provided"""
        realizations = [0, 3, 4]
        height_levels = [1.5, 3.0, 4.5]
        msg = (
            "Input data must have 4 dimensions to add both realization "
            "and height coordinates: got 3"
        )
        with self.assertRaisesRegex(ValueError, msg):
            _ = set_up_variable_cube(
                self.data_3d, realizations=realizations, height_levels=height_levels
            )

    def test_standard_grid_metadata_uk(self):
        """Test standard grid metadata is added if specified"""
        result = set_up_variable_cube(self.data, standard_grid_metadata="uk_det")
        self.assertEqual(result.attributes["mosg__grid_type"], "standard")
        self.assertEqual(result.attributes["mosg__grid_version"], "1.3.0")
        self.assertEqual(result.attributes["mosg__grid_domain"], "uk_extended")
        self.assertEqual(result.attributes["mosg__model_configuration"], "uk_det")

    def test_standard_grid_metadata_global(self):
        """Test standard grid metadata is added if specified"""
        result = set_up_variable_cube(self.data_3d, standard_grid_metadata="gl_ens")
        self.assertEqual(result.attributes["mosg__grid_type"], "standard")
        self.assertEqual(result.attributes["mosg__grid_version"], "1.3.0")
        self.assertEqual(result.attributes["mosg__grid_domain"], "global")
        self.assertEqual(result.attributes["mosg__model_configuration"], "gl_ens")

    def test_latlon_grid_spacing(self):
        """Test ability to set up lat-lon grid around 0,0 with specified grid spacing"""
        grid_spacing = 1
        result = set_up_variable_cube(
            self.data, spatial_grid="latlon", grid_spacing=grid_spacing
        )

        self.assertEqual(result.coord_dims("latitude"), (0,))
        self.assertEqual(result.coord_dims("longitude"), (1,))

        lat_spacing = abs(
            result.coord("latitude").points[0] - result.coord("latitude").points[1]
        )
        lon_spacing = abs(
            result.coord("longitude").points[0] - result.coord("longitude").points[1]
        )
        self.assertEqual(lat_spacing, grid_spacing)
        self.assertEqual(lon_spacing, grid_spacing)
        self.assertEqual(
            abs(result.coord("latitude").points[0]),
            abs(result.coord("latitude").points[-1]),
        )
        self.assertEqual(
            abs(result.coord("longitude").points[0]),
            abs(result.coord("longitude").points[-1]),
        )

    def test_equalarea_grid_spacing(self):
        """Test ability to set up equalarea grid around 0,0 with specified grid spacing"""
        grid_spacing = 1
        result = set_up_variable_cube(
            self.data, spatial_grid="equalarea", grid_spacing=grid_spacing
        )
        self.assertEqual(result.coord_dims("projection_y_coordinate"), (0,))
        self.assertEqual(result.coord_dims("projection_x_coordinate"), (1,))

        y_spacing = abs(
            result.coord("projection_y_coordinate").points[0]
            - result.coord("projection_y_coordinate").points[1]
        )
        x_spacing = abs(
            result.coord("projection_x_coordinate").points[0]
            - result.coord("projection_x_coordinate").points[1]
        )
        self.assertEqual(y_spacing, grid_spacing)
        self.assertEqual(x_spacing, grid_spacing)
        self.assertEqual(
            abs(result.coord("projection_y_coordinate").points[0]),
            abs(result.coord("projection_y_coordinate").points[-1]),
        )
        self.assertEqual(
            abs(result.coord("projection_x_coordinate").points[0]),
            abs(result.coord("projection_x_coordinate").points[-1]),
        )

    def test_latlon_domain_corner_grid_spacing(self):
        """Test ability to set up lat-lon grid from domain corner with grid spacing"""
        grid_spacing = 1
        domain_corner = (-17, -10)
        result = set_up_variable_cube(
            self.data,
            spatial_grid="latlon",
            grid_spacing=grid_spacing,
            domain_corner=domain_corner,
        )

        self.assertEqual(result.coord_dims("latitude"), (0,))
        self.assertEqual(result.coord_dims("longitude"), (1,))

        lat_spacing = abs(
            result.coord("latitude").points[0] - result.coord("latitude").points[1]
        )
        lon_spacing = abs(
            result.coord("longitude").points[0] - result.coord("longitude").points[1]
        )
        self.assertEqual(lat_spacing, grid_spacing)
        self.assertEqual(lon_spacing, grid_spacing)
        self.assertEqual(result.coord("latitude").points[0], domain_corner[0])
        self.assertEqual(result.coord("longitude").points[0], domain_corner[1])

    def test_equalarea_domain_corner_grid_spacing(self):
        """Test ability to set up equalarea grid from domain corner with grid spacing"""
        grid_spacing = 1
        domain_corner = (1100, 300)
        result = set_up_variable_cube(
            self.data,
            spatial_grid="equalarea",
            grid_spacing=grid_spacing,
            domain_corner=domain_corner,
        )

        self.assertEqual(result.coord_dims("projection_y_coordinate"), (0,))
        self.assertEqual(result.coord_dims("projection_x_coordinate"), (1,))

        y_spacing = abs(
            result.coord("projection_y_coordinate").points[0]
            - result.coord("projection_y_coordinate").points[1]
        )
        x_spacing = abs(
            result.coord("projection_x_coordinate").points[0]
            - result.coord("projection_x_coordinate").points[1]
        )
        self.assertEqual(y_spacing, grid_spacing)
        self.assertEqual(x_spacing, grid_spacing)
        self.assertEqual(
            result.coord("projection_y_coordinate").points[0], domain_corner[0]
        )
        self.assertEqual(
            result.coord("projection_x_coordinate").points[0], domain_corner[1]
        )

    def test_latlon_domain_corner(self):
        """Test grid points generated with default grid spacing if domain
        corner provided and grid spacing not provided"""
        domain_corner = (-17, -10)
        result = set_up_variable_cube(
            self.data, spatial_grid="latlon", domain_corner=domain_corner
        )
        self.assertArrayEqual(result.coord("latitude").points, [-17.0, -7.0, 3.0])
        self.assertArrayEqual(
            result.coord("longitude").points, [-10.0, 0.0, 10.0, 20.0]
        )

    def test_equalarea_domain_corner(self):
        """Test grid points generated with default grid spacing if domain
        corner provided and grid spacing not provided"""
        domain_corner = (1100, 300)
        result = set_up_variable_cube(
            self.data, spatial_grid="equalarea", domain_corner=domain_corner
        )
        self.assertArrayEqual(
            result.coord("projection_y_coordinate").points, [1100.0, 3100.0, 5100.0]
        )
        self.assertArrayEqual(
            result.coord("projection_x_coordinate").points,
            [300.0, 2300.0, 4300.0, 6300.0],
        )


class Test_set_up_percentile_cube(IrisTest):
    """Test the set_up_percentile_cube function"""

    def setUp(self):
        """Set up simple array of percentile-type data"""
        self.data = np.array(
            [
                [[273.5, 275.1, 274.9], [274.2, 274.8, 274.1]],
                [[274.2, 276.4, 275.5], [275.1, 276.8, 274.6]],
                [[275.6, 278.1, 277.2], [276.4, 277.5, 275.3]],
            ],
            dtype=np.float32,
        )
        self.percentiles = np.array([20, 50, 80])

    def test_defaults(self):
        """Test default arguments produce cube with expected dimensions
        and metadata"""
        result = set_up_percentile_cube(self.data, self.percentiles)
        perc_coord = result.coord("percentile")
        self.assertArrayEqual(perc_coord.points, self.percentiles)
        self.assertEqual(perc_coord.units, "%")
        check_mandatory_standards(result)

    def test_standard_grid_metadata(self):
        """Test standard grid metadata"""
        result = set_up_percentile_cube(
            self.data, self.percentiles, standard_grid_metadata="uk_ens"
        )
        self.assertEqual(result.attributes["mosg__grid_type"], "standard")
        self.assertEqual(result.attributes["mosg__grid_version"], "1.3.0")
        self.assertEqual(result.attributes["mosg__grid_domain"], "uk_extended")
        self.assertEqual(result.attributes["mosg__model_configuration"], "uk_ens")

    def test_single_percentile(self):
        """Test a cube with one percentile correctly stores this as a scalar
        coordinate"""
        result = set_up_percentile_cube(self.data[1:2], self.percentiles[1:2])
        dim_coords = get_dim_coord_names(result)
        self.assertNotIn("percentile", dim_coords)


class Test_set_up_probability_cube(IrisTest):
    """Test the set_up_probability_cube function"""

    def setUp(self):
        """Set up array of exceedance probabilities"""
        self.data = np.array(
            [
                [[1.0, 1.0, 0.9], [0.9, 0.9, 0.8]],
                [[0.8, 0.8, 0.7], [0.7, 0.6, 0.4]],
                [[0.6, 0.4, 0.3], [0.3, 0.2, 0.1]],
                [[0.2, 0.1, 0.0], [0.1, 0.0, 0.0]],
            ],
            dtype=np.float32,
        )
        self.thresholds = np.array([275.0, 275.5, 276.0, 276.5], dtype=np.float32)

    def test_defaults(self):
        """Test default arguments produce cube with expected dimensions
        and metadata"""
        result = set_up_probability_cube(self.data, self.thresholds)
        thresh_coord = find_threshold_coordinate(result)
        self.assertEqual(
            result.name(), "probability_of_air_temperature_above_threshold"
        )
        self.assertEqual(result.units, "1")
        self.assertArrayEqual(thresh_coord.points, self.thresholds)
        self.assertEqual(thresh_coord.name(), "air_temperature")
        self.assertEqual(thresh_coord.var_name, "threshold")
        self.assertEqual(thresh_coord.units, "K")
        self.assertEqual(len(thresh_coord.attributes), 1)
        self.assertEqual(
            thresh_coord.attributes["spp__relative_to_threshold"], "greater_than",
        )
        check_mandatory_standards(result)

    def test_relative_to_threshold(self):
        """Test ability to reset the "spp__relative_to_threshold" attribute"""
        data = np.flipud(self.data)
        result = set_up_probability_cube(
            data, self.thresholds, spp__relative_to_threshold="less_than"
        )
        self.assertEqual(len(result.coord(var_name="threshold").attributes), 1)
        self.assertEqual(
            result.coord(var_name="threshold").attributes["spp__relative_to_threshold"],
            "less_than",
        )

    def test_relative_to_threshold_set(self):
        """Test that an error is raised if the "spp__relative_to_threshold"
        attribute has not been set when setting up a probability cube"""
        msg = "The spp__relative_to_threshold attribute MUST be set"
        with self.assertRaisesRegex(ValueError, msg):
            set_up_probability_cube(
                self.data, self.thresholds, spp__relative_to_threshold=None
            )

    def test_standard_grid_metadata(self):
        """Test standard grid metadata"""
        result = set_up_probability_cube(
            self.data, self.thresholds, standard_grid_metadata="uk_ens"
        )
        self.assertEqual(result.attributes["mosg__grid_type"], "standard")
        self.assertEqual(result.attributes["mosg__grid_version"], "1.3.0")
        self.assertEqual(result.attributes["mosg__grid_domain"], "uk_extended")
        self.assertEqual(result.attributes["mosg__model_configuration"], "uk_ens")

    def test_single_threshold(self):
        """Test a cube with one threshold correctly stores this as a scalar
        coordinate"""
        result = set_up_probability_cube(self.data[1:2], self.thresholds[1:2])
        dim_coords = get_dim_coord_names(result)
        self.assertNotIn("air_temperature", dim_coords)

    def test_vicinity_cube(self):
        """Test an in-vicinity cube gets the correct name and threshold coordinate"""
        result = set_up_probability_cube(
            self.data, self.thresholds, variable_name="air_temperature_in_vicinity",
        )
        thresh_coord = find_threshold_coordinate(result)
        self.assertEqual(
            result.name(), "probability_of_air_temperature_in_vicinity_above_threshold"
        )
        self.assertEqual(thresh_coord.name(), "air_temperature")
        self.assertEqual(thresh_coord.var_name, "threshold")


class Test_add_coordinate(IrisTest):
    """Test the add_coordinate utility"""

    def setUp(self):
        """Set up new coordinate descriptors"""
        self.height_points = np.arange(100.0, 1001.0, 100.0)
        self.height_unit = "metres"
        self.input_cube = set_up_variable_cube(
            np.ones((3, 4), dtype=np.float32),
            time=datetime(2017, 10, 10, 1, 0),
            frt=datetime(2017, 10, 9, 21, 0),
        )

    def test_basic(self):
        """Test addition of a leading height coordinate"""
        result = add_coordinate(
            self.input_cube, self.height_points, "height", coord_units=self.height_unit
        )
        self.assertIsInstance(result, iris.cube.Cube)
        self.assertSequenceEqual(result.shape, (10, 3, 4))
        self.assertEqual(result.coord_dims("height"), (0,))
        self.assertArrayAlmostEqual(result.coord("height").points, self.height_points)
        self.assertEqual(result.coord("height").dtype, np.float32)
        self.assertEqual(result.coord("height").units, self.height_unit)
        check_mandatory_standards(result)

    def test_adding_coordinate_with_attribute(self):
        """Test addition of a leading height coordinate with an appropriate
        attribute."""
        height_attribute = {"positive": "up"}
        result = add_coordinate(
            self.input_cube,
            self.height_points,
            "height",
            coord_units=self.height_unit,
            attributes=height_attribute,
        )
        self.assertIsInstance(result, iris.cube.Cube)
        self.assertEqual(result.coord_dims("height"), (0,))
        self.assertEqual(result.coord("height").attributes, height_attribute)

    def test_reorder(self):
        """Test new coordinate can be placed in different positions"""
        input_cube = set_up_variable_cube(np.ones((4, 3, 4), dtype=np.float32))
        result = add_coordinate(
            input_cube,
            self.height_points,
            "height",
            coord_units=self.height_unit,
            order=[1, 0, 2, 3],
        )
        self.assertSequenceEqual(result.shape, (4, 10, 3, 4))
        self.assertEqual(result.coord_dims("height"), (1,))

    def test_datatype(self):
        """Test coordinate datatype"""
        result = add_coordinate(
            self.input_cube,
            self.height_points,
            "height",
            coord_units=self.height_unit,
            dtype=np.int32,
        )
        self.assertEqual(result.coord("height").dtype, np.int32)

    def test_datetime(self):
        """Test a leading time coordinate can be added successfully"""
        datetime_points = [datetime(2017, 10, 10, 3, 0), datetime(2017, 10, 10, 4, 0)]
        result = add_coordinate(
            self.input_cube, datetime_points, "time", is_datetime=True
        )
        # check time is now the leading dimension
        self.assertEqual(result.coord_dims("time"), (0,))
        self.assertEqual(len(result.coord("time").points), 2)
        # check forecast period has been updated
        expected_fp_points = 3600 * np.array([6, 7], dtype=np.int64)
        self.assertArrayAlmostEqual(
            result.coord("forecast_period").points, expected_fp_points
        )

    def test_datetime_no_fp(self):
        """Test a leading time coordinate can be added successfully when there
        is no forecast period on the input cube"""
        self.input_cube.remove_coord("forecast_period")
        datetime_points = [datetime(2017, 10, 10, 3, 0), datetime(2017, 10, 10, 4, 0)]
        result = add_coordinate(
            self.input_cube, datetime_points, "time", is_datetime=True
        )
        # check a forecast period coordinate has been added
        expected_fp_points = 3600 * np.array([6, 7], dtype=np.int64)
        self.assertArrayAlmostEqual(
            result.coord("forecast_period").points, expected_fp_points
        )

    def test_time_points(self):
        """Test a time coordinate can be added using integer points rather
        than datetimes, and that forecast period is correctly re-calculated"""
        time_val = self.input_cube.coord("time").points[0]
        time_points = np.array([time_val + 3600, time_val + 7200])
        fp_val = self.input_cube.coord("forecast_period").points[0]
        expected_fp_points = np.array([fp_val + 3600, fp_val + 7200])
        result = add_coordinate(
            self.input_cube,
            time_points,
            "time",
            coord_units=TIME_COORDS["time"].units,
            dtype=TIME_COORDS["time"].dtype,
        )
        self.assertArrayEqual(result.coord("time").points, time_points)
        self.assertArrayEqual(
            result.coord("forecast_period").points, expected_fp_points
        )


if __name__ == "__main__":
    unittest.main()
