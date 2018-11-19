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
"""
Unit tests for cube setup functions
"""

import unittest
from datetime import datetime
import numpy as np

import iris
from iris.tests import IrisTest

from improver.utilities.temporal import iris_time_to_datetime
from improver.tests.set_up_test_cubes import (
    construct_xy_coords, construct_scalar_time_coords, set_up_variable_cube,
    set_up_percentile_cube, set_up_probability_cube, add_coordinate)


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
        self.assertEqual(len(y_coord.points), 4)
        self.assertEqual(len(x_coord.points), 3)

    def test_proj_xy(self):
        """Test coordinates created for an equal area grid"""
        y_coord, x_coord = construct_xy_coords(4, 3, 'equalarea')
        self.assertEqual(y_coord.name(), "projection_y_coordinate")
        self.assertEqual(x_coord.name(), "projection_x_coordinate")
        for crd in [y_coord, x_coord]:
            self.assertEqual(crd.units, "metres")
            self.assertEqual(crd.dtype, np.float32)
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
            self.assertEqual(crd.dtype, np.int64)

        self.assertEqual(time_coords[0].name(), "time")
        self.assertEqual(iris_time_to_datetime(time_coords[0])[0],
                         datetime(2017, 12, 1, 14, 0))
        self.assertEqual(time_coords[1].name(), "forecast_reference_time")
        self.assertEqual(iris_time_to_datetime(time_coords[1])[0],
                         datetime(2017, 12, 1, 9, 0))
        self.assertEqual(time_coords[2].name(), "forecast_period")
        self.assertEqual(time_coords[2].points[0], 3600*5)

        for crd in time_coords[:2]:
            self.assertEqual(
                crd.units, "seconds since 1970-01-01 00:00:00")
        self.assertEqual(time_coords[2].units, "seconds")

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
        self.assertEqual(result.name(), 'air_temperature')
        self.assertEqual(result.units, 'K')
        self.assertArrayAlmostEqual(result.data, self.data)
        self.assertEqual(result.attributes, {})

        # check dimension coordinates
        self.assertEqual(result.coord_dims("latitude"), (0,))
        self.assertEqual(result.coord_dims("longitude"), (1,))

        # check scalar time coordinates
        for time_coord in ["time", "forecast_reference_time",
                           "forecast_period"]:
            self.assertEqual(result.coord(time_coord).dtype, np.int64)

        expected_time = datetime(2017, 11, 10, 4, 0)
        time_point = iris_time_to_datetime(result.coord("time"))[0]
        self.assertEqual(time_point, expected_time)

        expected_frt = datetime(2017, 11, 10, 0, 0)
        frt_point = iris_time_to_datetime(
            result.coord("forecast_reference_time"))[0]
        self.assertEqual(frt_point, expected_frt)

        self.assertEqual(result.coord("forecast_period").units, "seconds")
        self.assertEqual(result.coord("forecast_period").points[0], 14400)

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
        result = set_up_variable_cube(self.data, spatial_grid='equal_area')
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
        perc_coord = result.coord("percentile_over_realization")
        self.assertArrayEqual(perc_coord.points, self.percentiles)
        self.assertEqual(perc_coord.units, "%")

    def test_percentile_coord_name(self):
        """Test ability to set a different name"""
        result = set_up_percentile_cube(self.data, self.percentiles,
                                        percentile_dim_name="percentile")
        dim_coords = [coord.name() for coord in result.coords(dim_coords=True)]
        self.assertIn("percentile", dim_coords)


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
        thresh_coord = result.coord("threshold")
        self.assertEqual(result.name(), 'probability_of_air_temperature')
        self.assertEqual(result.units, '1')
        self.assertArrayEqual(thresh_coord.points, self.thresholds)
        self.assertEqual(thresh_coord.units, 'K')
        self.assertEqual(len(result.attributes), 1)
        self.assertEqual(result.attributes['relative_to_threshold'], 'above')

    def test_probability_of_name(self):
        """Test a name with "probability" at the start is correctly parsed"""
        result = set_up_probability_cube(
            self.data, self.thresholds,
            variable_name='probability_of_air_temperature')
        self.assertEqual(result.name(), 'probability_of_air_temperature')

    def test_relative_to_threshold(self):
        """Test ability to reset the "relative_to_threshold" attribute"""
        data = np.flipud(self.data)
        result = set_up_probability_cube(data, self.thresholds,
                                         relative_to_threshold='below')
        self.assertEqual(len(result.attributes), 1)
        self.assertEqual(result.attributes['relative_to_threshold'], 'below')


class test_add_coordinate(IrisTest):

    def setUp(self):
        """Set up new coordinate descriptors"""
        self.height_points = np.arange(100., 1001., 100.)
        self.height_unit = "metres"

    def test_basic(self):
        """Test addition of a leading height coordinate"""
        input_cube = set_up_variable_cube(np.ones((3, 4), dtype=np.float32))
        result = add_coordinate(
            input_cube, self.height_points, 'height',
            coord_units=self.height_unit)
        self.assertIsInstance(result, iris.cube.Cube)
        self.assertSequenceEqual(result.shape, (10, 3, 4))
        self.assertEqual(result.coord_dims('height'), (0,))
        self.assertArrayAlmostEqual(
            result.coord('height').points, self.height_points)
        self.assertEqual(result.coord('height').dtype, np.float32)
        self.assertEqual(result.coord('height').units, self.height_unit)

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
        input_cube = set_up_variable_cube(np.ones((3, 4), dtype=np.float32))
        result = add_coordinate(
            input_cube, self.height_points, 'height',
            coord_units=self.height_unit, dtype=np.int32)
        self.assertEqual(result.coord('height').dtype, np.int32)


if __name__ == '__main__':
    unittest.main()
