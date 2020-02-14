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
"""Unit tests for forecast time coordinate utilities"""

import unittest
from datetime import datetime, timedelta

import iris
import numpy as np
from iris.exceptions import CoordinateNotFoundError
from iris.tests import IrisTest

from improver.metadata.forecast_times import (
    _calculate_forecast_period, find_latest_cycletime, forecast_period_coord,
    rebadge_forecasts_as_latest_cycle, unify_cycletime)
from improver.utilities.warnings_handler import ManageWarnings
from improver.metadata.constants.time_types import TimeSpec

from ..set_up_test_cubes import add_coordinate, set_up_variable_cube


class Test_forecast_period_coord(IrisTest):
    """Test the forecast_period_coord function"""

    def setUp(self):
        """Set up a test cube with a forecast period scalar coordinate"""
        self.cube = set_up_variable_cube(np.ones((1, 3, 3), dtype=np.float32))

    def test_basic(self):
        """Test that an iris.coords.DimCoord is returned from a cube with an
        existing forecast period"""
        result = forecast_period_coord(self.cube)
        self.assertIsInstance(result, iris.coords.DimCoord)

    def test_no_forecast_period(self):
        """Test that an iris.coords.AuxCoord is returned from a cube with no
        forecast period"""
        self.cube.remove_coord('forecast_period')
        result = forecast_period_coord(
            self.cube, force_lead_time_calculation=True)
        self.assertIsInstance(result, iris.coords.AuxCoord)

    def test_values(self):
        """Test that the data within the coord is as expected with the
        expected units, when the input cube has a forecast_period coordinate.
        """
        fp_coord = self.cube.coord("forecast_period").copy()
        result = forecast_period_coord(self.cube)
        self.assertArrayEqual(result.points, fp_coord.points)
        self.assertEqual(result.units, fp_coord.units)
        self.assertEqual(result.dtype, fp_coord.dtype)

    def test_values_force_lead_time_calculation(self):
        """Test that the data within the coord is as expected with the
        expected units, when the input cube has a forecast_period coordinate.
        """
        fp_coord = self.cube.coord("forecast_period").copy()
        # put incorrect data into the existing coordinate so we can test it is
        # correctly recalculated
        self.cube.coord("forecast_period").points = np.array([-3600],
                                                             dtype=np.int32)
        result = forecast_period_coord(
            self.cube, force_lead_time_calculation=True)
        self.assertArrayEqual(result.points, fp_coord.points)
        self.assertEqual(result.units, fp_coord.units)
        self.assertEqual(result.dtype, fp_coord.dtype)

    def test_exception_insufficient_data(self):
        """Test that a CoordinateNotFoundError exception is raised if forecast
        period cannot be calculated from the available coordinates
        """
        self.cube.remove_coord("forecast_reference_time")
        self.cube.remove_coord("forecast_period")
        msg = "The forecast period coordinate is not available"
        with self.assertRaisesRegex(CoordinateNotFoundError, msg):
            forecast_period_coord(self.cube)


class Test__calculate_forecast_period(IrisTest):
    """Test the _calculate_forecast_period function"""

    def setUp(self):
        """Set up test inputs (4 hour forecast period)"""
        cube = set_up_variable_cube(np.ones((1, 3, 3), dtype=np.float32))
        self.time_coord = cube.coord("time")
        self.frt_coord = cube.coord("forecast_reference_time")
        self.fp_coord = cube.coord("forecast_period")

    def test_basic(self):
        """Test correct coordinate type is returned"""
        result = _calculate_forecast_period(self.time_coord, self.frt_coord)
        self.assertIsInstance(result, iris.coords.AuxCoord)

    def test_dim_coord(self):
        """Test it is possible to create a dimension coordinate"""
        result = _calculate_forecast_period(
            self.time_coord, self.frt_coord, dim_coord=True)
        self.assertIsInstance(result, iris.coords.DimCoord)

    def test_values(self):
        """Test correct values are returned"""
        result = _calculate_forecast_period(self.time_coord, self.frt_coord)
        self.assertArrayAlmostEqual(result.points, self.fp_coord.points)
        self.assertEqual(result.units, self.fp_coord.units)
        self.assertEqual(result.dtype, self.fp_coord.dtype)

    def test_changing_mandatory_types(self):
        """Test that the data within the coord is as expected with the
        expected units, when mandatory standards for the forecast_period
        coordinate are changed.
        """
        local_spec = TimeSpec(calendar=None, dtype=np.float64, units="hours")

        result = _calculate_forecast_period(self.time_coord, self.frt_coord,
                                            coord_spec=local_spec)
        self.assertEqual(result.units, 'hours')
        self.assertArrayAlmostEqual(result.points * 3600.,
                                    self.fp_coord.points)
        self.assertEqual(result.dtype, np.float64)

    def test_bounds(self):
        """Test that the forecast_period coord has bounds where appropriate"""
        time_point = self.time_coord.points[0]
        self.time_coord.bounds = [[time_point - 3600, time_point]]
        fp_point = self.fp_coord.points[0]
        expected_fp_bounds = [[fp_point - 3600, fp_point]]
        result = _calculate_forecast_period(self.time_coord, self.frt_coord)
        self.assertArrayAlmostEqual(result.points, [fp_point])
        self.assertArrayAlmostEqual(result.bounds, expected_fp_bounds)

    def test_multiple_time_points(self):
        """Test a multi-valued forecast period coordinate can be created"""
        time_point = self.time_coord.points[0]
        new_time_points = [time_point, time_point + 3600, time_point + 7200]
        new_time_coord = self.time_coord.copy(new_time_points)
        fp_point = self.fp_coord.points[0]
        expected_fp_points = [fp_point, fp_point + 3600, fp_point + 7200]
        result = _calculate_forecast_period(new_time_coord, self.frt_coord)
        self.assertArrayAlmostEqual(result.points, expected_fp_points)

    def test_check_time_unit_conversion(self):
        """Test correct values and units are returned when the input time and
        forecast reference time coordinates are in different units
        """
        self.time_coord.convert_units("seconds since 1970-01-01 00:00:00")
        self.frt_coord.convert_units("hours since 1970-01-01 00:00:00")
        result = _calculate_forecast_period(self.time_coord, self.frt_coord)
        self.assertEqual(result, self.fp_coord)

    @ManageWarnings(record=True)
    def test_negative_forecast_period(self, warning_list=None):
        """Test a warning is raised if the calculated forecast period is
        negative"""
        # default cube has a 4 hour forecast period, so add 5 hours to frt
        self.frt_coord.points = self.frt_coord.points + 5*3600
        result = _calculate_forecast_period(self.time_coord, self.frt_coord)
        warning_msg = "The values for the time"
        result = _calculate_forecast_period(self.time_coord, self.frt_coord)
        self.assertTrue(any(item.category == UserWarning
                            for item in warning_list))
        self.assertTrue(any(warning_msg in str(item)
                            for item in warning_list))
        self.assertEqual(result.points, [-3600])


class Test_rebadge_forecasts_as_latest_cycle(IrisTest):
    """Test the rebadge_forecasts_as_latest_cycle function"""

    def setUp(self):
        """Set up some cubes with different cycle times"""
        self.cycletime = '20190711T1200Z'
        validity_time = datetime(2019, 7, 11, 14)
        self.cube_early = set_up_variable_cube(
            np.full((4, 4), 273.15, dtype=np.float32),
            time=validity_time, frt=datetime(2019, 7, 11, 9))
        self.cube_late = set_up_variable_cube(
            np.full((4, 4), 273.15, dtype=np.float32),
            time=validity_time, frt=datetime(2019, 7, 11, 10))

    def test_cubelist(self):
        """Test a list of cubes is returned with the latest frt"""
        expected = self.cube_late.copy()
        result = rebadge_forecasts_as_latest_cycle(
            [self.cube_early, self.cube_late])
        self.assertIsInstance(result, iris.cube.CubeList)
        self.assertEqual(len(result), 2)
        for cube in result:
            for coord in ["forecast_reference_time", "forecast_period"]:
                self.assertEqual(cube.coord(coord), expected.coord(coord))

    def test_cycletime(self):
        """Test a list of cubes using the cycletime argument"""
        expected_frt_point = (
            self.cube_late.coord("forecast_reference_time").points[0] + 2*3600)
        expected_fp_point = (
            self.cube_late.coord("forecast_period").points[0] - 2*3600)
        result = rebadge_forecasts_as_latest_cycle(
            [self.cube_early, self.cube_late], cycletime=self.cycletime)
        for cube in result:
            self.assertEqual(cube.coord("forecast_reference_time").points[0],
                             expected_frt_point)
            self.assertEqual(cube.coord("forecast_period").points[0],
                             expected_fp_point)

    def test_single_cube(self):
        """Test a single cube is returned unchanged if the cycletime argument
        is not set"""
        expected = self.cube_early.copy()
        result, = rebadge_forecasts_as_latest_cycle([self.cube_early])
        for coord in ["forecast_reference_time", "forecast_period"]:
            self.assertEqual(result.coord(coord), expected.coord(coord))

    def test_single_cube_with_cycletime(self):
        """Test a single cube has its forecast reference time and period
        updated if cycletime is specified"""
        expected_frt_point = (
            self.cube_late.coord("forecast_reference_time").points[0] + 2*3600)
        expected_fp_point = (
            self.cube_late.coord("forecast_period").points[0] - 2*3600)
        result, = rebadge_forecasts_as_latest_cycle(
            [self.cube_late], cycletime=self.cycletime)
        self.assertEqual(result.coord("forecast_reference_time").points[0],
                         expected_frt_point)
        self.assertEqual(result.coord("forecast_period").points[0],
                         expected_fp_point)


class Test_unify_cycletime(IrisTest):

    """Test the unify_cycletime function."""

    def setUp(self):
        """Set up a UK deterministic cube for testing."""
        self.cycletime = datetime(2017, 1, 10, 6)
        cube_uk_det = set_up_variable_cube(
            np.full((4, 4), 273.15, dtype=np.float32),
            time=self.cycletime, frt=datetime(2017, 1, 10, 3))

        # set up forecast periods of 6, 8 and 10 hours
        time_points = [1484038800, 1484046000, 1484053200]
        cube_uk_det = add_coordinate(
            cube_uk_det, time_points, "time", dtype=np.int64,
            coord_units="seconds since 1970-01-01 00:00:00")

        self.cube_uk_det = add_coordinate(cube_uk_det, [1000], "model_id")
        self.cube_uk_det.add_aux_coord(
            iris.coords.AuxCoord(["uk_det"], long_name="model_configuration"))

    def test_cubelist_input(self):
        """Test when supplying a cubelist as input containing cubes
        representing UK deterministic and UK ensemble model configuration
        and unifying the forecast_reference_time, so that both model
        configurations have a common forecast_reference_time."""
        cube_uk_ens = set_up_variable_cube(
            np.full((3, 4, 4), 273.15, dtype=np.float32),
            time=self.cycletime, frt=datetime(2017, 1, 10, 4))

        # set up forecast periods of 5, 7 and 9 hours
        time_points = [1484031600, 1484038800, 1484046000]
        cube_uk_ens = add_coordinate(
            cube_uk_ens, time_points, "time", dtype=np.int64,
            coord_units="seconds since 1970-01-01 00:00:00")

        expected_uk_det = self.cube_uk_det.copy()
        frt_units = expected_uk_det.coord('forecast_reference_time').units
        frt_points = [
            np.round(frt_units.date2num(self.cycletime)).astype(np.int64)]
        expected_uk_det.coord("forecast_reference_time").points = frt_points
        expected_uk_det.coord("forecast_period").points = (
            np.array([3, 5, 7]) * 3600)
        expected_uk_ens = cube_uk_ens.copy()
        expected_uk_ens.coord("forecast_reference_time").points = frt_points
        expected_uk_ens.coord("forecast_period").points = (
            np.array([1, 3, 5]) * 3600)
        expected = iris.cube.CubeList([expected_uk_det, expected_uk_ens])

        cubes = iris.cube.CubeList([self.cube_uk_det, cube_uk_ens])
        result = unify_cycletime(cubes, self.cycletime)

        self.assertIsInstance(result, iris.cube.CubeList)
        self.assertEqual(result, expected)

    def test_single_item_cubelist_input(self):
        """Test when supplying a cube representing a UK deterministic model
        configuration only. This effectively updates the
        forecast_reference_time on the cube to the specified cycletime."""
        expected_uk_det = self.cube_uk_det.copy()
        frt_units = expected_uk_det.coord('forecast_reference_time').units
        frt_points = [
            np.round(frt_units.date2num(self.cycletime)).astype(np.int64)]
        expected_uk_det.coord("forecast_reference_time").points = frt_points
        expected_uk_det.coord("forecast_period").points = (
            np.array([3, 5, 7]) * 3600)
        result = unify_cycletime([self.cube_uk_det], self.cycletime)
        self.assertIsInstance(result, iris.cube.CubeList)
        self.assertEqual(result[0], expected_uk_det)

    def test_input_no_forecast_period_coordinate(self):
        """Test when supplying a cube representing a UK deterministic model
        configuration only. This forces a forecast_period coordinate to be
        created from a forecast_reference_time coordinate and a time
        coordinate."""
        expected_uk_det = self.cube_uk_det.copy()
        frt_units = expected_uk_det.coord('forecast_reference_time').units
        frt_points = [
            np.round(frt_units.date2num(self.cycletime)).astype(np.int64)]
        expected_uk_det.coord("forecast_reference_time").points = frt_points
        expected_uk_det.coord("forecast_period").points = (
            np.array([3, 5, 7]) * 3600)
        cube_uk_det = self.cube_uk_det.copy()
        cube_uk_det.remove_coord("forecast_period")
        result = unify_cycletime([cube_uk_det], self.cycletime)
        self.assertIsInstance(result, iris.cube.CubeList)
        self.assertEqual(result[0], expected_uk_det)


class Test_find_latest_cycletime(IrisTest):

    """Test the find_latest_cycletime function."""

    def setUp(self):
        """Set up a template cubes with scalar time, forecast_reference_time
           and forecast_period coordinates"""
        self.input_cube = set_up_variable_cube(
            np.full((7, 7), 273.15, dtype=np.float32),
            time=datetime(2015, 11, 23, 6), frt=datetime(2015, 11, 23, 3))
        self.input_cube2 = self.input_cube.copy()
        self.input_cube2.coord("forecast_reference_time").points = np.array(
            self.input_cube2.coord("forecast_reference_time").points[0] + 3600)
        self.input_cubelist = iris.cube.CubeList(
            [self.input_cube, self.input_cube2])

    def test_basic(self):
        """Test the type of the output and that the input is unchanged."""
        original_cubelist = iris.cube.CubeList(
            [self.input_cube.copy(), self.input_cube2.copy()])
        cycletime = find_latest_cycletime(self.input_cubelist)
        self.assertEqual(self.input_cubelist[0], original_cubelist[0])
        self.assertEqual(self.input_cubelist[1], original_cubelist[1])
        self.assertIsInstance(cycletime, datetime)

    def test_returns_latest(self):
        """Test the returned cycle time is the latest in the input cubelist."""
        cycletime = find_latest_cycletime(self.input_cubelist)
        expected_datetime = datetime(2015, 11, 23, 4)
        self.assertEqual(timedelta(hours=0, seconds=0),
                         cycletime - expected_datetime)

    def test_two_cubes_same_reference_time(self):
        """Test the a cycletime is still found when two cubes have the same
           cycletime."""
        input_cubelist = iris.cube.CubeList(
            [self.input_cube, self.input_cube.copy()])
        cycletime = find_latest_cycletime(input_cubelist)
        expected_datetime = datetime(2015, 11, 23, 3)
        self.assertEqual(timedelta(hours=0, seconds=0),
                         cycletime - expected_datetime)

    def test_one_input_cube(self):
        """Test the a cycletime is still found when only one input cube."""
        input_cubelist = iris.cube.CubeList([self.input_cube])
        cycletime = find_latest_cycletime(input_cubelist)
        expected_datetime = datetime(2015, 11, 23, 3)
        self.assertEqual(timedelta(hours=0, seconds=0),
                         cycletime - expected_datetime)

    def test_different_units(self):
        """Test the right cycletime is still returned if the coords have
        different units."""
        self.input_cube2.coord("forecast_reference_time").convert_units(
            'minutes since 1970-01-01 00:00:00')
        cycletime = find_latest_cycletime(self.input_cubelist)
        expected_datetime = datetime(2015, 11, 23, 4)
        self.assertEqual(timedelta(hours=0, seconds=0),
                         cycletime - expected_datetime)

    def test_raises_error(self):
        """Test the error is raised if time is dimensional"""
        input_cube2 = iris.util.new_axis(
            self.input_cube2, "forecast_reference_time")
        input_cubelist = iris.cube.CubeList([self.input_cube, input_cube2])
        msg = "Expecting scalar forecast_reference_time for each input cube"
        with self.assertRaisesRegex(ValueError, msg):
            find_latest_cycletime(input_cubelist)


if __name__ == '__main__':
    unittest.main()
