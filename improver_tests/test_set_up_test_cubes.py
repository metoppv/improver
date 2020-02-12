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
"""
Unit tests for cube setup functions
"""

import unittest
from datetime import datetime

import iris
import numpy as np
from iris.tests import IrisTest

from improver.grids import GLOBAL_GRID_CCRS, STANDARD_GRID_CCRS
from improver.metadata.constants.time_types import TIME_COORDS
from improver.metadata.probabilistic import find_threshold_coordinate
from improver.metadata.check_datatypes import check_mandatory_standards
from improver.utilities.temporal import iris_time_to_datetime

from .set_up_test_cubes import (
    add_coordinate, construct_scalar_time_coords, construct_xy_coords,
    set_up_percentile_cube, set_up_probability_cube, set_up_variable_cube)


class test_construct_xy_coords(IrisTest):
    """Test the construct_xy_coords method"""

    def test_lat_lon(self):
        """Test coordinates created for a lat-lon grid"""
        y_coord, x_coord = construct_xy_coords(4, 3, 'latlon')
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
        y_coord, x_coord = construct_xy_coords(3, 3, 'latlon')
        self.assertArrayAlmostEqual(x_coord.points, [-20., 0., 20.])
        self.assertArrayAlmostEqual(y_coord.points, [40., 60., 80.])

    def test_proj_xy(self):
        """Test coordinates created for an equal area grid"""
        y_coord, x_coord = construct_xy_coords(4, 3, 'equalarea')
        self.assertEqual(y_coord.name(), "projection_y_coordinate")
        self.assertEqual(x_coord.name(), "projection_x_coordinate")
        for crd in [y_coord, x_coord]:
            self.assertEqual(crd.units, "metres")
            self.assertEqual(crd.dtype, np.float32)
            self.assertEqual(crd.coord_system, STANDARD_GRID_CCRS)
        self.assertEqual(len(y_coord.points), 4)
        self.assertEqual(len(x_coord.points), 3)


class test_construct_scalar_time_coords(IrisTest):
    """Test the construct_scalar_time_coords method"""

    def test_basic(self):
        """Test times can be set"""
        coord_dims = construct_scalar_time_coords(
            datetime(2017, 12, 1, 14, 0), None, datetime(2017, 12, 1, 9, 0))
        time_coords = [item[0] for item in coord_dims]

        for crd in time_coords:
            self.assertIsInstance(crd, iris.coords.DimCoord)

        self.assertEqual(time_coords[0].name(), "time")
        self.assertEqual(iris_time_to_datetime(time_coords[0])[0],
                         datetime(2017, 12, 1, 14, 0))
        self.assertEqual(time_coords[1].name(), "forecast_reference_time")
        self.assertEqual(iris_time_to_datetime(time_coords[1])[0],
                         datetime(2017, 12, 1, 9, 0))
        self.assertEqual(time_coords[2].name(), "forecast_period")
        self.assertEqual(time_coords[2].points[0], 3600*5)

        for crd in time_coords[:2]:
            self.assertEqual(crd.dtype, np.int64)
            self.assertEqual(
                crd.units, "seconds since 1970-01-01 00:00:00")
        self.assertEqual(time_coords[2].units, "seconds")
        self.assertEqual(time_coords[2].dtype, np.int32)

    def test_error_negative_fp(self):
        """Test an error is raised if the calculated forecast period is
        negative"""
        msg = 'Cannot set up cube with negative forecast period'
        with self.assertRaisesRegex(ValueError, msg):
            _ = construct_scalar_time_coords(
                datetime(2017, 12, 1, 14, 0), None,
                datetime(2017, 12, 1, 16, 0))

    def test_time_bounds(self):
        """Test creation of time coordinate with bounds"""
        coord_dims = construct_scalar_time_coords(
            datetime(2017, 12, 1, 14, 0), (datetime(2017, 12, 1, 13, 0),
                                           datetime(2017, 12, 1, 14, 0)),
            datetime(2017, 12, 1, 9, 0))
        time_coord = coord_dims[0][0]
        self.assertEqual(iris_time_to_datetime(time_coord)[0],
                         datetime(2017, 12, 1, 14, 0))
        self.assertEqual(time_coord.bounds[0][0], time_coord.points[0] - 3600)
        self.assertEqual(time_coord.bounds[0][1], time_coord.points[0])

    def test_time_bounds_wrong_order(self):
        """Test time bounds are correctly applied even if supplied in the wrong
        order"""
        coord_dims = construct_scalar_time_coords(
            datetime(2017, 12, 1, 14, 0), (datetime(2017, 12, 1, 14, 0),
                                           datetime(2017, 12, 1, 13, 0)),
            datetime(2017, 12, 1, 9, 0))
        time_coord = coord_dims[0][0]
        self.assertEqual(iris_time_to_datetime(time_coord)[0],
                         datetime(2017, 12, 1, 14, 0))
        self.assertEqual(time_coord.bounds[0][0], time_coord.points[0] - 3600)
        self.assertEqual(time_coord.bounds[0][1], time_coord.points[0])

    def test_error_invalid_time_bounds(self):
        """Test an error is raised if the time point is not between the
        specified bounds"""
        msg = 'not within bounds'
        with self.assertRaisesRegex(ValueError, msg):
            _ = construct_scalar_time_coords(
                datetime(2017, 11, 10, 4, 0), (datetime(2017, 12, 1, 13, 0),
                                               datetime(2017, 12, 1, 14, 0)),
                datetime(2017, 11, 10, 0, 0))


class test_set_up_variable_cube(IrisTest):
    """Test the set_up_variable_cube base function"""

    def setUp(self):
        """Set up simple temperature data array"""
        self.data = (
            np.linspace(275.0, 284.0, 12).reshape(3, 4).astype(np.float32))
        self.data_3d = np.array([self.data, self.data, self.data])

    def test_defaults(self):
        """Test default arguments produce cube with expected dimensions
        and metadata"""
        result = set_up_variable_cube(self.data)

        # check type, data and attributes
        self.assertIsInstance(result, iris.cube.Cube)
        self.assertEqual(result.standard_name, 'air_temperature')
        self.assertEqual(result.name(), 'air_temperature')
        self.assertEqual(result.units, 'K')
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
        frt_point = iris_time_to_datetime(
            result.coord("forecast_reference_time"))[0]
        self.assertEqual(frt_point, expected_frt)

        self.assertEqual(result.coord("forecast_period").units, "seconds")
        self.assertEqual(result.coord("forecast_period").points[0], 14400)

        check_mandatory_standards(result)

    def test_non_standard_name(self):
        """Test non CF standard cube naming"""
        result = set_up_variable_cube(self.data, name="temp_in_the_air")
        self.assertEqual(result.name(), "temp_in_the_air")

    def test_var_name(self):
        """Test ability to set data name and units"""
        result = set_up_variable_cube(
            self.data-273.15, name='wet_bulb_temperature', units='degC')
        self.assertArrayAlmostEqual(result.data, self.data-273.15)
        self.assertEqual(result.name(), 'wet_bulb_temperature')
        self.assertEqual(result.units, 'degC')

    def test_attributes(self):
        """Test ability to set attributes"""
        attributes = {"source": "IMPROVER"}
        result = set_up_variable_cube(self.data, attributes=attributes)
        self.assertEqual(result.attributes, attributes)

    def test_spatial_grid(self):
        """Test ability to set up non lat-lon grid"""
        result = set_up_variable_cube(self.data, spatial_grid='equalarea')
        self.assertEqual(result.coord_dims('projection_y_coordinate'), (0,))
        self.assertEqual(result.coord_dims('projection_x_coordinate'), (1,))

    def test_time_points(self):
        """Test ability to configure time and forecast reference time"""
        expected_time = datetime(2018, 3, 1, 12, 0)
        expected_frt = datetime(2018, 3, 1, 9, 0)
        result = set_up_variable_cube(self.data, time=expected_time,
                                      frt=expected_frt)
        time_point = iris_time_to_datetime(result.coord("time"))[0]
        self.assertEqual(time_point, expected_time)
        frt_point = iris_time_to_datetime(
            result.coord("forecast_reference_time"))[0]
        self.assertEqual(frt_point, expected_frt)
        self.assertEqual(result.coord("forecast_period").points[0], 10800)
        self.assertFalse(result.coords('time', dim_coords=True))

    def test_realizations_from_data(self):
        """Test realization coordinate is added for 3D data"""
        result = set_up_variable_cube(self.data_3d)
        self.assertArrayAlmostEqual(result.data, self.data_3d)
        self.assertEqual(result.coord_dims("realization"), (0,))
        self.assertArrayEqual(
            result.coord("realization").points, np.array([0, 1, 2]))
        self.assertEqual(result.coord_dims("latitude"), (1,))
        self.assertEqual(result.coord_dims("longitude"), (2,))

    def test_realizations(self):
        """Test specific realization values"""
        result = set_up_variable_cube(
            self.data_3d, realizations=np.array([0, 3, 4]))
        self.assertArrayEqual(
            result.coord("realization").points, np.array([0, 3, 4]))

    def test_error_unmatched_realizations(self):
        """Test error is raised if the realizations provided do not match the
        data dimensions"""
        msg = 'Cannot generate 4 realizations'
        with self.assertRaisesRegex(ValueError, msg):
            _ = set_up_variable_cube(self.data_3d, realizations=np.arange(4))

    def test_error_too_many_dimensions(self):
        """Test error is raised if input cube has more than 3 dimensions"""
        data_4d = np.array([self.data_3d, self.data_3d])
        msg = 'Expected 2 or 3 dimensions on input data: got 4'
        with self.assertRaisesRegex(ValueError, msg):
            _ = set_up_variable_cube(data_4d)

    def test_standard_grid_metadata_uk(self):
        """Test standard grid metadata is added if specified"""
        result = set_up_variable_cube(
            self.data, standard_grid_metadata='uk_det')
        self.assertEqual(result.attributes['mosg__grid_type'], 'standard')
        self.assertEqual(result.attributes['mosg__grid_version'], '1.3.0')
        self.assertEqual(result.attributes['mosg__grid_domain'], 'uk_extended')
        self.assertEqual(
            result.attributes['mosg__model_configuration'], 'uk_det')

    def test_standard_grid_metadata_global(self):
        """Test standard grid metadata is added if specified"""
        result = set_up_variable_cube(
            self.data_3d, standard_grid_metadata='gl_ens')
        self.assertEqual(result.attributes['mosg__grid_type'], 'standard')
        self.assertEqual(result.attributes['mosg__grid_version'], '1.3.0')
        self.assertEqual(result.attributes['mosg__grid_domain'], 'global')
        self.assertEqual(
            result.attributes['mosg__model_configuration'], 'gl_ens')


class test_set_up_percentile_cube(IrisTest):
    """Test the set_up_percentile_cube function"""

    def setUp(self):
        """Set up simple array of percentile-type data"""
        self.data = np.array([
            [[273.5, 275.1, 274.9], [274.2, 274.8, 274.1]],
            [[274.2, 276.4, 275.5], [275.1, 276.8, 274.6]],
            [[275.6, 278.1, 277.2], [276.4, 277.5, 275.3]]], dtype=np.float32)
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
        result = set_up_percentile_cube(self.data, self.percentiles,
                                        standard_grid_metadata='uk_ens')
        self.assertEqual(result.attributes['mosg__grid_type'], 'standard')
        self.assertEqual(result.attributes['mosg__grid_version'], '1.3.0')
        self.assertEqual(
            result.attributes['mosg__grid_domain'], 'uk_extended')
        self.assertEqual(
            result.attributes['mosg__model_configuration'], 'uk_ens')


class test_set_up_probability_cube(IrisTest):
    """Test the set_up_probability_cube function"""

    def setUp(self):
        """Set up array of exceedance probabilities"""
        self.data = np.array([
            [[1.0, 1.0, 0.9], [0.9, 0.9, 0.8]],
            [[0.8, 0.8, 0.7], [0.7, 0.6, 0.4]],
            [[0.6, 0.4, 0.3], [0.3, 0.2, 0.1]],
            [[0.2, 0.1, 0.0], [0.1, 0.0, 0.0]]], dtype=np.float32)
        self.thresholds = np.array(
            [275., 275.5, 276., 276.5], dtype=np.float32)

    def test_defaults(self):
        """Test default arguments produce cube with expected dimensions
        and metadata"""
        result = set_up_probability_cube(self.data, self.thresholds)
        thresh_coord = find_threshold_coordinate(result)
        self.assertEqual(
            result.name(), 'probability_of_air_temperature_above_threshold')
        self.assertEqual(result.units, '1')
        self.assertArrayEqual(thresh_coord.points, self.thresholds)
        self.assertEqual(thresh_coord.name(), 'air_temperature')
        self.assertEqual(thresh_coord.var_name, 'threshold')
        self.assertEqual(thresh_coord.units, 'K')
        self.assertEqual(len(thresh_coord.attributes), 1)
        self.assertEqual(
            thresh_coord.attributes['spp__relative_to_threshold'], 'above')
        check_mandatory_standards(result)

    def test_relative_to_threshold(self):
        """Test ability to reset the "spp__relative_to_threshold" attribute"""
        data = np.flipud(self.data)
        result = set_up_probability_cube(data, self.thresholds,
                                         spp__relative_to_threshold='below')
        self.assertEqual(len(result.coord(var_name="threshold").attributes), 1)
        self.assertEqual(
            result.coord(var_name="threshold"
                         ).attributes['spp__relative_to_threshold'], 'below')

    def test_relative_to_threshold_set(self):
        """Test that an error is raised if the "spp__relative_to_threshold"
        attribute has not been set when setting up a probability cube"""
        msg = 'The spp__relative_to_threshold attribute MUST be set'
        with self.assertRaisesRegex(ValueError, msg):
            set_up_probability_cube(self.data, self.thresholds,
                                    spp__relative_to_threshold=None)

    def test_standard_grid_metadata(self):
        """Test standard grid metadata"""
        result = set_up_probability_cube(self.data, self.thresholds,
                                         standard_grid_metadata='uk_ens')
        self.assertEqual(result.attributes['mosg__grid_type'], 'standard')
        self.assertEqual(result.attributes['mosg__grid_version'], '1.3.0')
        self.assertEqual(
            result.attributes['mosg__grid_domain'], 'uk_extended')
        self.assertEqual(
            result.attributes['mosg__model_configuration'], 'uk_ens')


class test_add_coordinate(IrisTest):
    """Test the add_coordinate utility"""

    def setUp(self):
        """Set up new coordinate descriptors"""
        self.height_points = np.arange(100., 1001., 100.)
        self.height_unit = "metres"
        self.input_cube = set_up_variable_cube(
            np.ones((3, 4), dtype=np.float32),
            time=datetime(2017, 10, 10, 1, 0),
            frt=datetime(2017, 10, 9, 21, 0))

    def test_basic(self):
        """Test addition of a leading height coordinate"""
        result = add_coordinate(
            self.input_cube, self.height_points, 'height',
            coord_units=self.height_unit)
        self.assertIsInstance(result, iris.cube.Cube)
        self.assertSequenceEqual(result.shape, (10, 3, 4))
        self.assertEqual(result.coord_dims('height'), (0,))
        self.assertArrayAlmostEqual(
            result.coord('height').points, self.height_points)
        self.assertEqual(result.coord('height').dtype, np.float32)
        self.assertEqual(result.coord('height').units, self.height_unit)
        check_mandatory_standards(result)

    def test_adding_coordinate_with_attribute(self):
        """Test addition of a leading height coordinate with an appropriate
        attribute."""
        height_attribute = {"positive": "up"}
        result = add_coordinate(
            self.input_cube, self.height_points, 'height',
            coord_units=self.height_unit, attributes=height_attribute)
        self.assertIsInstance(result, iris.cube.Cube)
        self.assertEqual(result.coord_dims('height'), (0,))
        self.assertEqual(result.coord('height').attributes, height_attribute)

    def test_reorder(self):
        """Test new coordinate can be placed in different positions"""
        input_cube = set_up_variable_cube(np.ones((4, 3, 4), dtype=np.float32))
        result = add_coordinate(
            input_cube, self.height_points, 'height',
            coord_units=self.height_unit, order=[1, 0, 2, 3])
        self.assertSequenceEqual(result.shape, (4, 10, 3, 4))
        self.assertEqual(result.coord_dims('height'), (1,))

    def test_datatype(self):
        """Test coordinate datatype"""
        result = add_coordinate(
            self.input_cube, self.height_points, 'height',
            coord_units=self.height_unit, dtype=np.int32)
        self.assertEqual(result.coord('height').dtype, np.int32)

    def test_datetime(self):
        """Test a leading time coordinate can be added successfully"""
        datetime_points = [
            datetime(2017, 10, 10, 3, 0), datetime(2017, 10, 10, 4, 0)]
        result = add_coordinate(
            self.input_cube, datetime_points, "time", is_datetime=True)
        # check time is now the leading dimension
        self.assertEqual(result.coord_dims("time"), (0,))
        self.assertEqual(len(result.coord("time").points), 2)
        # check forecast period has been updated
        expected_fp_points = 3600*np.array([6, 7], dtype=np.int64)
        self.assertArrayAlmostEqual(
            result.coord("forecast_period").points, expected_fp_points)

    def test_datetime_no_fp(self):
        """Test a leading time coordinate can be added successfully when there
        is no forecast period on the input cube"""
        self.input_cube.remove_coord("forecast_period")
        datetime_points = [
            datetime(2017, 10, 10, 3, 0), datetime(2017, 10, 10, 4, 0)]
        result = add_coordinate(
            self.input_cube, datetime_points, "time", is_datetime=True)
        # check a forecast period coordinate has been added
        expected_fp_points = 3600*np.array([6, 7], dtype=np.int64)
        self.assertArrayAlmostEqual(
            result.coord("forecast_period").points, expected_fp_points)

    def test_time_points(self):
        """Test a time coordinate can be added using integer points rather
        than datetimes, and that forecast period is correctly re-calculated"""
        time_val = self.input_cube.coord("time").points[0]
        time_points = np.array([time_val + 3600, time_val + 7200])
        fp_val = self.input_cube.coord("forecast_period").points[0]
        expected_fp_points = np.array([fp_val + 3600, fp_val + 7200])
        result = add_coordinate(
            self.input_cube, time_points, "time",
            coord_units=TIME_COORDS["time"].units,
            dtype=TIME_COORDS["time"].dtype)
        self.assertArrayEqual(result.coord("time").points, time_points)
        self.assertArrayEqual(
            result.coord("forecast_period").points, expected_fp_points)


if __name__ == '__main__':
    unittest.main()
