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
""" Unit tests for the nowcasting.Accumulation plugin """

import datetime
import unittest

import iris
import numpy as np
from cf_units import Unit
from iris.tests import IrisTest

from improver.nowcasting.accumulation import Accumulation
from improver.utilities.warnings_handler import ManageWarnings

from ...set_up_test_cubes import set_up_variable_cube


class rate_cube_set_up(IrisTest):
    """Set up a sequence of precipitation rates cubes for use in testing the
    accumulation plugin functionality."""

    def setUp(self):
        """Set up 11 precipitation rate cubes offset by 1 minute (spanning 10
        minutes), with a shower moving across the array from left to right
        (west to east). A uniform rate of precipitation covers the eastern half
        of the domain. Lighter precipitation covers the north-west quadrant,
        and there is no precipitation in the south-western quadrant. This
        shower is advected eastwards by the optical flow at a uniform rate over
        the period considered. Beyond the western boundary there is no
        precipitation, so precipitation stops in the west as the shower is
        advected east.

        A mask covers two cells towards the centre of the domain to simulate a
        radar quality mask.

        Accumulations are greatest in the right-hand array columns and smallest
        in the left. All accumulations to which a masked cell has contributed
        are returned as masked; note that in the arrays of expected values in
        the tests below those cells that are expected to be masked are given
        a value of np.nan.
        """
        ncells = 10
        # Rates equivalent to 5.4 and 1.8 mm/hr
        rates = np.ones((ncells)) * 5.4
        rates[0:ncells//2] = 1.8
        rates = rates / 3600.

        datalist = []
        for i in range(ncells):
            data = np.vstack([rates] * 4)
            data = np.roll(data, i, axis=1)
            try:
                data[0:2, :i] = 0
                data[2:, :i+ncells//2] = 0
            except IndexError:
                pass
            mask = np.zeros((4, ncells))
            mask[1:3, ncells//2+i:ncells//2 + i + 1] = 1
            data = np.ma.MaskedArray(data, mask=mask, dtype=np.float32)
            datalist.append(data)

        datalist.append(np.ma.MaskedArray(np.zeros((4, ncells)),
                                          mask=np.zeros((4, ncells)),
                                          dtype=np.float32))

        name = "lwe_precipitation_rate"
        units = "mm s-1"
        self.cubes = iris.cube.CubeList()
        for index, data in enumerate(datalist):
            cube = set_up_variable_cube(
                data, name=name, units=units, spatial_grid="equalarea",
                time=datetime.datetime(2017, 11, 10, 4, index),
                frt=datetime.datetime(2017, 11, 10, 4, 0))
            self.cubes.append(cube)
        return self.cubes


class Test__init__(IrisTest):
    """Test class initialisation"""

    def test_default(self):
        """Test the default accumulation_units are set when not specified."""
        plugin = Accumulation()
        self.assertEqual(plugin.accumulation_units, "m")
        self.assertEqual(plugin.accumulation_period, None)

    def test_units_set(self):
        """Test the accumulation_units are set when specified."""
        plugin = Accumulation(accumulation_units="cm")
        self.assertEqual(plugin.accumulation_units, "cm")

    def test_accumulation_period_set(self):
        """Test the accumulation_period is set when specified."""
        plugin = Accumulation(accumulation_period=180)
        self.assertEqual(plugin.accumulation_period, 180)

    def test_forecast_period_set(self):
        """Test the forecast_period is set when specified."""
        plugin = Accumulation(forecast_periods=[60, 120])
        self.assertListEqual(plugin.forecast_periods, [60, 120])


class Test__repr__(IrisTest):
    """Test class representation"""

    def test_basic(self):
        """Test string representation"""
        result = str(Accumulation(accumulation_units="cm",
                                  accumulation_period=60,
                                  forecast_periods=[60, 120]))
        expected_result = ("<Accumulation: accumulation_units=cm, "
                           "accumulation_period=60s, "
                           "forecast_periods=[60, 120]s>")
        self.assertEqual(result, expected_result)


class Test_sort_cubes_by_time(rate_cube_set_up):
    """Tests the input cubes are sorted in time ascending order."""

    def test_returns_cubelist(self):
        """Test function returns a cubelist."""

        result, _ = Accumulation.sort_cubes_by_time(self.cubes)
        self.assertIsInstance(result, iris.cube.CubeList)

    def test_reorders(self):
        """Test function reorders a cubelist that is not time ordered."""

        expected = [cube.coord('time').points[0] for cube in self.cubes]
        self.cubes = self.cubes[::-1]
        reordered = [cube.coord('time').points[0] for cube in self.cubes]

        result, _ = Accumulation.sort_cubes_by_time(self.cubes)
        result_times = [cube.coord('time').points[0] for cube in result]

        self.assertIsInstance(result, iris.cube.CubeList)
        self.assertEqual(result_times, expected)
        self.assertNotEqual(result_times, reordered)

    def test_times(self):
        """Test function returns the correct times for the sorted cubes."""

        expected = [1510286400, 1510286460, 1510286520, 1510286580,
                    1510286640, 1510286700, 1510286760, 1510286820,
                    1510286880, 1510286940, 1510287000]
        _, times = Accumulation.sort_cubes_by_time(self.cubes)

        self.assertArrayEqual(times, expected)


class Test__check_inputs(rate_cube_set_up):

    """Test the _check_inputs method."""

    def test_basic(self):
        """Test that the expected time_interval is returned and that the
        returned list of cubes has the expected units."""
        expected_time_interval = 60
        expected_cubes = self.cubes.copy()
        for cube in expected_cubes:
            cube.convert_units("m/s")
        cubes, time_interval = Accumulation()._check_inputs(self.cubes)
        self.assertEqual(cubes, expected_cubes)
        self.assertEqual(time_interval, expected_time_interval)

    def test_specify_accumulation_period(self):
        """Test that the expected time interval is returned when the
        accumulation period is specified. Also test that the returned list of
        cubes has the expected units."""
        expected_time_interval = 60
        expected_cubes = self.cubes.copy()
        for cube in expected_cubes:
            cube.convert_units("m/s")
        accumulation_period = 60*60
        plugin = Accumulation(accumulation_period=accumulation_period)
        cubes, time_interval = plugin._check_inputs(self.cubes)
        self.assertEqual(cubes, expected_cubes)
        self.assertEqual(time_interval, expected_time_interval)
        self.assertEqual(plugin.accumulation_period, accumulation_period)

    def test_specify_forecast_period(self):
        """Test that the expected time interval is returned when the forecast
        periods are specified. Also test that the returned list of cubes has
        the expected units."""
        expected_time_interval = 60
        expected_cubes = self.cubes.copy()
        for cube in expected_cubes:
            cube.convert_units("m/s")
        forecast_periods = [600]
        plugin = Accumulation(forecast_periods=forecast_periods)
        cubes, time_interval = plugin._check_inputs(self.cubes)
        self.assertEqual(cubes, expected_cubes)
        self.assertEqual(time_interval, expected_time_interval)
        self.assertEqual(plugin.forecast_periods, forecast_periods)

    def test_specify_accumulation_period_and_forecast_period(self):
        """Test that the expected time interval is returned when the
        accumulation period and forecast periods are specified. Also test that
        the returned list of cubes has the expected units."""
        expected_time_interval = 60
        expected_cubes = self.cubes.copy()
        for cube in expected_cubes:
            cube.convert_units("m/s")
        accumulation_period = 20*60
        forecast_periods = np.array([15])*60
        plugin = Accumulation(accumulation_period=accumulation_period,
                              forecast_periods=forecast_periods)
        cubes, time_interval = plugin._check_inputs(self.cubes)
        self.assertEqual(cubes, expected_cubes)
        self.assertEqual(time_interval, expected_time_interval)

    def test_raises_exception_for_unevenly_spaced_cubes(self):
        """Test function raises an exception if the input cubes are not
        spaced equally in time."""

        last_time = self.cubes[-1].coord('time').points
        self.cubes[-1].coord('time').points = last_time + 60

        msg = ("Accumulation is designed to work with rates "
               "cubes at regular time intervals.")
        plugin = Accumulation(accumulation_period=120)

        with self.assertRaisesRegex(ValueError, msg):
            plugin.process(self.cubes)

    def test_raises_exception_for_small_accumulation_period(self):
        """Test that if the forecast period of the upper bound cube is
        not within the list of requested forecast periods, then the
        subset of cubes returned is equal to None."""
        msg = (
            "The accumulation_period is less than the time interval "
            "between the rates cubes. The rates cubes provided are "
            "therefore insufficient for computing the accumulation period "
            "requested.")
        reduced_cubelist = iris.cube.CubeList([self.cubes[0], self.cubes[-1]])
        plugin = Accumulation(
            accumulation_period=5*60,
            forecast_periods=np.array([5])*60)
        with self.assertRaisesRegex(ValueError, msg):
            plugin.process(reduced_cubelist)

    def test_raises_exception_for_impossible_aggregation(self):
        """Test function raises an exception when attempting to create an
        accumulation_period that cannot be created from the input cubes."""

        plugin = Accumulation(accumulation_period=119)
        msg = "The specified accumulation period "

        with self.assertRaisesRegex(ValueError, msg):
            plugin._check_inputs(self.cubes)


class Test__get_cube_subsets(rate_cube_set_up):

    """Test the _get_cube_subsets method."""

    def test_basic(self):
        """Test that the subset of cubes that are within the accumulation
        period are correctly identified. In this case, the subset of cubes
        used for each accumulation period is expected to consist of 6 cubes."""
        expected_cube_subset = self.cubes[:6]
        upper_bound_fp, = self.cubes[5].coord("forecast_period").points
        plugin = Accumulation(
            accumulation_period=5*60,
            forecast_periods=np.array([5])*60)
        result = plugin._get_cube_subsets(self.cubes, upper_bound_fp)
        self.assertEqual(expected_cube_subset, result)


class Test__calculate_accumulation(rate_cube_set_up):

    """Test the _calculate_accumulation method."""

    def test_basic(self):
        """Check the calculations of the accumulations, where an accumulation
        is computed by finding the mean rate between each adjacent pair of
        cubes within the cube_subset and multiplying this mean rate by the
        time_interval, in order to compute an accumulation. In this case,
        as the cube_subset only contains a pair of cubes, then the
        accumulation from this pair will be the same as the total accumulation.
        """
        expected_t0 = np.array([
            [0.015, 0.03, 0.03, 0.03, 0.03, 0.06, 0.09, 0.09, 0.09, 0.09],
            [0.015, 0.03, 0.03, 0.03, 0.03, np.nan, np.nan, 0.09, 0.09, 0.09],
            [0., 0., 0., 0., 0., np.nan, np.nan, 0.09, 0.09, 0.09],
            [0., 0., 0., 0., 0., 0.045, 0.09, 0.09, 0.09, 0.09]])

        expected_mask_t0 = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
        time_interval = 60
        result = Accumulation()._calculate_accumulation(
            self.cubes[:2], time_interval)
        self.assertArrayAlmostEqual(result, expected_t0)
        self.assertArrayAlmostEqual(result.mask, expected_mask_t0)


class Test__set_metadata(rate_cube_set_up):

    """Test the _set_metadata method."""

    def test_basic(self):
        """Check that the metadata is set as expected."""
        expected_name = "lwe_thickness_of_precipitation_amount"
        expected_units = Unit("m")
        expected_time_point = [datetime.datetime(2017, 11, 10, 4, 10)]
        expected_time_bounds = [(datetime.datetime(2017, 11, 10, 4, 0),
                                 datetime.datetime(2017, 11, 10, 4, 10))]
        expected_fp_point = 600
        expected_fp_bounds = [[0, 600]]
        result = Accumulation()._set_metadata(self.cubes)
        self.assertEqual(result.name(), expected_name)
        self.assertEqual(result.units, expected_units)
        points = [value.point for value in result.coord("time").cells()]
        bounds = [value.bound for value in result.coord("time").cells()]
        self.assertEqual(points, expected_time_point)
        self.assertArrayAlmostEqual(
            result.coord("forecast_period").points, expected_fp_point)
        self.assertEqual(bounds, expected_time_bounds)
        self.assertArrayAlmostEqual(
            result.coord("forecast_period").bounds, expected_fp_bounds)


class Test_process(rate_cube_set_up):
    """Tests the process method results in the expected outputs."""

    def setUp(self):
        """Set up forecast periods used for testing."""
        super().setUp()
        self.forecast_periods = [
            cube.coord("forecast_period").points for cube in self.cubes[1:]]

    def test_returns_cubelist(self):
        """Test function returns a cubelist."""

        plugin = Accumulation(
            accumulation_period=60, forecast_periods=self.forecast_periods)
        result = plugin.process(self.cubes)
        self.assertIsInstance(result, iris.cube.CubeList)

    def test_accumulation_length(self):
        """Test to check that the length of the accumulation period is
        consistent across all output cubes. Only complete periods are
        required."""

        accumulation_length = 120
        plugin = Accumulation(
            accumulation_period=accumulation_length,
            forecast_periods=self.forecast_periods)
        result = plugin.process(self.cubes)
        for cube in result:
            self.assertEqual(np.diff(cube.coord("forecast_period").bounds),
                             accumulation_length)

    def test_returns_masked_cubes(self):
        """Test function returns a list of masked cubes for masked input
        data."""

        result = Accumulation(
            forecast_periods=[600]).process(self.cubes)
        self.assertIsInstance(result[0].data, np.ma.MaskedArray)

    def test_default_output_units(self):
        """Test the function returns accumulations in the default units if no
        units are explicitly set, where the default is metres."""

        # Multiply the rates in mm/s by 60 to get accumulation over 1 minute
        # and divide by 1000 to get into metres.
        expected = self.cubes[0].copy(
            data=(0.5 * (self.cubes[0].data + self.cubes[1].data) * 60 / 1000))

        plugin = Accumulation(
            accumulation_period=60, forecast_periods=self.forecast_periods)
        result = plugin.process(self.cubes)

        self.assertEqual(result[0].units, 'm')
        self.assertArrayAlmostEqual(result[0].data, expected.data)

    def test_default_altered_output_units(self):
        """Test the function returns accumulations in the specified units if
        they are explicitly set. Here the units are set to mm."""

        # Multiply the rates in mm/s by 60 to get accumulation over 1 minute
        expected = self.cubes[0].copy(
            data=(0.5 * (self.cubes[0].data + self.cubes[1].data) * 60))

        plugin = Accumulation(accumulation_units='mm', accumulation_period=60,
                              forecast_periods=self.forecast_periods)
        result = plugin.process(self.cubes)
        self.assertEqual(result[0].units, 'mm')
        self.assertArrayAlmostEqual(result[0].data, expected.data)

    @ManageWarnings(ignored_messages=["The provided cubes result in a"],
                    warning_types=[UserWarning])
    def test_does_not_use_incomplete_period_data(self):
        """Test function returns only 2 accumulation periods when a 4 minute
        aggregation period is used with 10 minutes of input data. The trailing
        2 cubes are insufficient to create another period and so are discarded.
        A warning is raised by the chunking function and has been tested above,
        so is ignored here.
        """

        plugin = Accumulation(accumulation_period=240,
                              forecast_periods=[240, 480])
        result = plugin.process(self.cubes)
        self.assertEqual(len(result), 2)

    def test_returns_expected_values_5_minutes(self):
        """Test function returns the expected accumulations over a 5 minute
        aggregation period. These are written out long hand to make the
        comparison easy. Check that the number of accumulation cubes returned
        is the expected number."""

        expected_t0 = np.array([
            [0.015, 0.045, 0.075, 0.105, 0.135, 0.18, 0.24, 0.3, 0.36, 0.42],
            [0.015, 0.045, 0.075, 0.105, 0.135,
             np.nan, np.nan, np.nan, np.nan, np.nan],
            [0., 0., 0., 0., 0., np.nan, np.nan, np.nan, np.nan, np.nan],
            [0., 0., 0., 0., 0., 0.045, 0.135, 0.225, 0.315, 0.405]])

        expected_t1 = np.array([
            [0., 0., 0., 0., 0., 0.015, 0.045, 0.075, 0.105, 0.135],
            [0., 0., 0., 0., 0., 0.015, 0.045, 0.075, 0.105, 0.135],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]])

        expected_mask_t0 = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

        expected_mask_t1 = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

        plugin = Accumulation(accumulation_period=300, accumulation_units='mm',
                              forecast_periods=[300, 600])
        result = plugin.process(self.cubes)

        self.assertArrayAlmostEqual(result[0].data, expected_t0)
        self.assertArrayAlmostEqual(result[1].data, expected_t1)
        self.assertArrayAlmostEqual(result[0].data.mask, expected_mask_t0)
        self.assertArrayAlmostEqual(result[1].data.mask, expected_mask_t1)
        self.assertEqual(len(result), 2)

    def test_returns_expected_values_10_minutes(self):
        """Test function returns the expected accumulations over the complete
        10 minute aggregation period. These are written out long hand to make
        the comparison easy. Note that the test have been constructed such that
        only the top row is expected to show a difference by including the last
        5 minutes of the accumulation, all the other results are the same as
        for the 5 minute test above. Check that the number of accumulation
        cubes returned is the expected number."""

        expected_t0 = np.array([
            [0.015, 0.045, 0.075, 0.105, 0.135,
             0.195, 0.285, 0.375, 0.465, 0.555],
            [0.015, 0.045, 0.075, 0.105, 0.135,
             np.nan, np.nan, np.nan, np.nan, np.nan],
            [0., 0., 0., 0., 0.,
             np.nan, np.nan, np.nan, np.nan, np.nan],
            [0., 0., 0., 0., 0.,
             0.045, 0.135, 0.225, 0.315, 0.405]])

        expected_mask_t0 = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

        plugin = Accumulation(accumulation_period=600, accumulation_units='mm',
                              forecast_periods=[600])
        result = plugin.process(self.cubes)

        self.assertArrayAlmostEqual(result[0].data, expected_t0)
        self.assertArrayAlmostEqual(result[0].data.mask, expected_mask_t0)
        self.assertEqual(len(result), 1)

    @ManageWarnings(ignored_messages=["The provided cubes result in a"],
                    warning_types=[UserWarning])
    def test_returns_total_accumulation_if_no_period_specified(self):
        """Test function returns a list containing a single accumulation cube
        that is the accumulation over the whole period specified by the rates
        cubes. The results are the same as the 10 minute test above as that is
        the total span of the input rates cubes. Check that the number of
        accumulation cubes returned is the expected number."""

        expected_t0 = np.array([
            [0.015, 0.045, 0.075, 0.105, 0.135,
             0.195, 0.285, 0.375, 0.465, 0.555],
            [0.015, 0.045, 0.075, 0.105, 0.135,
             np.nan, np.nan, np.nan, np.nan, np.nan],
            [0., 0., 0., 0., 0.,
             np.nan, np.nan, np.nan, np.nan, np.nan],
            [0., 0., 0., 0., 0.,
             0.045, 0.135, 0.225, 0.315, 0.405]])

        expected_mask_t0 = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

        plugin = Accumulation(accumulation_units='mm')
        result = plugin.process(self.cubes)

        self.assertArrayAlmostEqual(result[0].data, expected_t0)
        self.assertArrayAlmostEqual(result[0].data.mask, expected_mask_t0)
        self.assertEqual(len(result), 1)

    def test_returns_expected_values_1_minute(self):
        """Test function returns the expected accumulations over a 1 minute
        aggregation period. Check that the number of accumulation cubes
        returned is the expected number."""

        expected_t0 = np.array([
            [0.015, 0.03, 0.03, 0.03, 0.03, 0.06, 0.09, 0.09, 0.09, 0.09],
            [0.015, 0.03, 0.03, 0.03, 0.03, np.nan, np.nan, 0.09, 0.09, 0.09],
            [0., 0., 0., 0., 0., np.nan, np.nan, 0.09, 0.09, 0.09],
            [0., 0., 0., 0., 0., 0.045, 0.09, 0.09, 0.09, 0.09]])

        expected_t7 = np.array([
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.015, 0.03, 0.03],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.015, 0.03, 0.03],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])

        expected_mask_t0 = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

        expected_mask_t7 = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

        plugin = Accumulation(accumulation_period=60, accumulation_units='mm',
                              forecast_periods=self.forecast_periods)
        result = plugin.process(self.cubes)

        self.assertArrayAlmostEqual(result[0].data, expected_t0)
        self.assertArrayAlmostEqual(result[7].data, expected_t7)
        self.assertArrayAlmostEqual(result[0].data.mask, expected_mask_t0)
        self.assertArrayAlmostEqual(result[7].data.mask, expected_mask_t7)
        self.assertEqual(len(result), 10)


if __name__ == '__main__':
    unittest.main()
