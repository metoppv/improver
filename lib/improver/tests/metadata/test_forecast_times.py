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
    forecast_period_coord, rebadge_forecasts_as_latest_cycle,
    unify_cycletime, find_latest_cycletime)
from improver.tests.set_up_test_cubes import (
    set_up_variable_cube, add_coordinate)
from improver.utilities.warnings_handler import ManageWarnings


class Test_forecast_period_coord(IrisTest):

    """Test determining of the lead times present within the input cube."""

    def setUp(self):
        """Set up a test cube with a forecast period scalar coordinate"""
        self.cube = set_up_variable_cube(np.ones((1, 3, 3), dtype=np.float32))

    def test_basic(self):
        """Test that an iris.coords.DimCoord is returned."""
        result = forecast_period_coord(self.cube)
        self.assertIsInstance(result, iris.coords.DimCoord)

    def test_basic_AuxCoord(self):
        """Test that an iris.coords.AuxCoord is returned."""
        self.cube.remove_coord('forecast_period')
        result = forecast_period_coord(
            self.cube, force_lead_time_calculation=True)
        self.assertIsInstance(result, iris.coords.AuxCoord)

    def test_check_coordinate(self):
        """Test that the data within the coord is as expected with the
        expected units, when the input cube has a forecast_period coordinate.
        """
        fp_coord = self.cube.coord("forecast_period").copy()
        expected_points = fp_coord.points
        expected_units = str(fp_coord.units)
        result = forecast_period_coord(self.cube)
        self.assertArrayEqual(result.points, expected_points)
        self.assertEqual(str(result.units), expected_units)

    def test_check_coordinate_force_lead_time_calculation(self):
        """Test that the data within the coord is as expected with the
        expected units, when the input cube has a forecast_period coordinate.
        """
        fp_coord = self.cube.coord("forecast_period").copy()
        expected_points = fp_coord.points
        expected_units = str(fp_coord.units)
        result = forecast_period_coord(
            self.cube, force_lead_time_calculation=True)
        self.assertArrayEqual(result.points, expected_points)
        self.assertEqual(result.units, expected_units)

    def test_check_coordinate_in_hours_force_lead_time_calculation(self):
        """Test that the data within the coord is as expected with the
        expected units, when the input cube has a forecast_period coordinate.
        """
        fp_coord = self.cube.coord("forecast_period").copy()
        fp_coord.convert_units("hours")
        expected_points = fp_coord.points
        expected_units = str(fp_coord.units)
        result = forecast_period_coord(
            self.cube, force_lead_time_calculation=True,
            result_units=fp_coord.units)
        self.assertArrayEqual(result.points, expected_points)
        self.assertEqual(result.units, expected_units)

    def test_check_coordinate_without_forecast_period(self):
        """Test that the data within the coord is as expected with the
        expected units, when the input cube has a time coordinate and a
        forecast_reference_time coordinate.
        """
        fp_coord = self.cube.coord("forecast_period").copy()
        expected_result = fp_coord
        self.cube.remove_coord("forecast_period")
        result = forecast_period_coord(self.cube)
        self.assertEqual(result, expected_result)

    def test_check_time_unit_conversion(self):
        """Test that the data within the coord is as expected with the
        expected units, when the input cube time and forecast reference time
        coordinates are in different units.
        """
        expected_result = self.cube.coord("forecast_period")
        self.cube.coord("time").convert_units(
            "seconds since 1970-01-01 00:00:00")
        self.cube.coord("forecast_reference_time").convert_units(
            "hours since 1970-01-01 00:00:00")
        result = forecast_period_coord(
            self.cube, force_lead_time_calculation=True)
        self.assertEqual(result, expected_result)

    def test_check_time_unit_has_bounds(self):
        """Test that the forecast_period coord has bounds if time has bounds.
        """
        cube = set_up_variable_cube(
            np.ones((3, 3), dtype=np.float32),
            time=datetime(2018, 3, 12, 20), frt=datetime(2018, 3, 12, 15),
            time_bounds=[datetime(2018, 3, 12, 19), datetime(2018, 3, 12, 20)])
        expected_result = cube.coord("forecast_period").copy()
        expected_result.bounds = [[14400, 18000]]
        result = forecast_period_coord(cube, force_lead_time_calculation=True)
        self.assertEqual(result, expected_result)

    @ManageWarnings(record=True)
    def test_negative_forecast_periods_warning(self, warning_list=None):
        """Test that a warning is raised if the point within the
        time coordinate is prior to the point within the
        forecast_reference_time, and therefore the forecast_period values that
        have been generated are negative.
        """
        cube = set_up_variable_cube(np.ones((3, 3), dtype=np.float32))
        cube.remove_coord("forecast_period")
        # default cube has a 4 hour forecast period, so add 5 hours to frt
        cube.coord("forecast_reference_time").points = (
            cube.coord("forecast_reference_time").points + 5*3600)
        warning_msg = "The values for the time"
        forecast_period_coord(cube)
        self.assertTrue(any(item.category == UserWarning
                            for item in warning_list))
        self.assertTrue(any(warning_msg in str(item)
                            for item in warning_list))

    def test_exception_raised(self):
        """Test that a CoordinateNotFoundError exception is raised if the
        forecast_period, or the time and forecast_reference_time,
        are not present.
        """
        self.cube.remove_coord("forecast_reference_time")
        self.cube.remove_coord("forecast_period")
        msg = "The forecast period coordinate is not available"
        with self.assertRaisesRegex(CoordinateNotFoundError, msg):
            forecast_period_coord(self.cube)


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

        cube_uk_det.remove_coord("forecast_period")
        # set up forecast periods of 6, 8 and 10 hours
        time_points = [1484038800, 1484046000, 1484053200]
        cube_uk_det = add_coordinate(
            cube_uk_det, time_points, "time", dtype=np.int64,
            coord_units="seconds since 1970-01-01 00:00:00")
        fp_coord = forecast_period_coord(cube_uk_det)
        cube_uk_det.add_aux_coord(fp_coord, data_dims=0)

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

        cube_uk_ens.remove_coord("forecast_period")
        # set up forecast periods of 5, 7 and 9 hours
        time_points = [1484031600, 1484038800, 1484046000]
        cube_uk_ens = add_coordinate(
            cube_uk_ens, time_points, "time", dtype=np.int64,
            coord_units="seconds since 1970-01-01 00:00:00")
        fp_coord = forecast_period_coord(cube_uk_ens)
        cube_uk_ens.add_aux_coord(fp_coord, data_dims=0)

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
