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
import numpy as np

import iris
from iris.tests import IrisTest

from improver.utilities.warnings_handler import ManageWarnings
from improver.nowcasting.accumulation import Accumulation
from improver.tests.set_up_test_cubes import set_up_variable_cube


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

    def test_period_set(self):
        """Test the accumulation_period is set when specified."""
        plugin = Accumulation(accumulation_period=180)
        self.assertEqual(plugin.accumulation_period, 180)


class Test__repr__(IrisTest):
    """Test class representation"""

    def test_basic(self):
        """Test string representation"""
        result = str(Accumulation(accumulation_units="cm",
                                  accumulation_period=60))
        expected_result = ("<Accumulation: accumulation_units=cm, "
                           "accumulation_period=60>")
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


class Test_get_period_sets(rate_cube_set_up):
    """Tests the requested accumulation_period can be constructed from the
    input cubes and return lists of the required inputs to construct each
    such period."""

    def test_returns_correct_type(self):
        """Test function returns the expected list."""

        time_interval = 60
        plugin = Accumulation(accumulation_period=120)
        result = plugin.get_period_sets(time_interval, self.cubes)
        self.assertIsInstance(result, list)

    def test_returns_all_cubes_if_period_unspecified(self):
        """Test function returns a list containing the original cube list if
        the accumulation_period is not set."""

        time_interval = 60
        plugin = Accumulation()
        result = plugin.get_period_sets(time_interval, self.cubes)
        self.assertSequenceEqual(result, [self.cubes])

    @ManageWarnings(record=True)
    def test_raises_warning_for_unused_cubes(self, warning_list=None):
        """Test function raises a warning when there are insufficient cubes to
        complete the last period."""

        time_interval = 60
        warning_msg = (
            "The provided cubes result in a partial period given the specified"
            " accumulation_period, i.e. the number of cubes is insufficient to"
            " give a set of complete periods. Only complete periods will be"
            " returned.")

        expected = [self.cubes[0:4], self.cubes[3:7], self.cubes[6:10]]

        plugin = Accumulation(accumulation_period=180)
        result = plugin.get_period_sets(time_interval, self.cubes)

        for index, sublist in enumerate(result):
            self.assertSequenceEqual(sublist, expected[index])
        self.assertTrue(any(item.category == UserWarning
                            for item in warning_list))
        self.assertTrue(any(warning_msg in str(item)
                            for item in warning_list))

    def test_raises_exception_for_impossible_aggregation(self):
        """Test function raises an exception when attempting to create an
        accumulation_period that cannot be created from the input cubes."""

        time_interval = 61
        plugin = Accumulation(accumulation_period=120)
        msg = "The specified accumulation period "

        with self.assertRaisesRegex(ValueError, msg):
            plugin.get_period_sets(time_interval, self.cubes)


class Test_process(rate_cube_set_up):
    """Tests the process method results in the expected outputs."""

    def test_returns_cubelist(self):
        """Test function returns a cubelist containing one less entry than the
        input cube list."""

        result = Accumulation(accumulation_period=60).process(self.cubes)
        self.assertIsInstance(result, iris.cube.CubeList)
        self.assertEqual(len(self.cubes) - 1, len(result))

    def test_returns_masked_cubes(self):
        """Test function returns a list of masked cubes for masked input
        data."""

        result = Accumulation().process(self.cubes)
        self.assertIsInstance(result[0].data, np.ma.MaskedArray)

    def test_default_output_units(self):
        """Test the function returns accumulations in the default units if no
        units are explicitly set, where the default is metres."""

        # Multiply the rates in mm/s by 60 to get accumulation over 1 minute
        # and divide by 1000 to get into metres.
        expected = self.cubes[0].copy(
            data=(0.5 * (self.cubes[0].data + self.cubes[1].data) * 60 / 1000))

        plugin = Accumulation(accumulation_period=60)
        result = plugin.process(self.cubes)

        self.assertEqual(result[0].units, 'm')
        self.assertArrayAlmostEqual(result[0].data, expected.data)

    def test_default_altered_output_units(self):
        """Test the function returns accumulations in the specified units if
        they are explicitly set. Here the units are set to mm."""

        # Multiply the rates in mm/s by 60 to get accumulation over 1 minute
        expected = self.cubes[0].copy(
            data=(0.5 * (self.cubes[0].data + self.cubes[1].data) * 60))

        plugin = Accumulation(accumulation_units='mm', accumulation_period=60)
        result = plugin.process(self.cubes)

        self.assertEqual(result[0].units, 'mm')
        self.assertArrayAlmostEqual(result[0].data, expected.data)

    def test_raises_exception_for_unevenly_spaced_cubes(self):
        """Test function raises an exception if the input cubes are not
        spaced equally in time."""

        last_time = self.cubes[-1].coord('time').points
        self.cubes[-1].coord('time').points = last_time + 60

        msg = ("Accumulation is designed to work with rates "
               "cubes at regualar time intervals.")
        plugin = Accumulation(accumulation_period=120)

        with self.assertRaisesRegex(ValueError, msg):
            plugin.process(self.cubes)

    @ManageWarnings(ignored_messages=["The provided cubes result in a"],
                    warning_types=[UserWarning])
    def test_does_not_use_incomplete_period_data(self):
        """Test function returns only 2 accumulation periods when a 4 minute
        aggregation period is used with 10 minutes of input data. The trailing
        2 cubes are insufficient to create another period and so are disgarded.
        A warning is raised by the chunking function and has been tested above,
        so is ignored here.
        """

        plugin = Accumulation(accumulation_period=240)
        result = plugin.process(self.cubes)
        self.assertEqual(len(result), 2)

    def test_returns_expected_values_5_minutes(self):
        """Test function returns the expected accumulations over a 5 minute
        aggregation period. These are written out long hand to make the
        comparison easy."""

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

        plugin = Accumulation(accumulation_period=300, accumulation_units='mm')
        result = plugin.process(self.cubes)

        self.assertArrayAlmostEqual(result[0].data, expected_t0)
        self.assertArrayAlmostEqual(result[1].data, expected_t1)
        self.assertArrayAlmostEqual(result[0].data.mask, expected_mask_t0)
        self.assertArrayAlmostEqual(result[1].data.mask, expected_mask_t1)

    def test_returns_expected_values_10_minutes(self):
        """Test function returns the expected accumulations over the complete
        10 minute aggregation period. These are written out long hand to make
        the comparison easy. Note that the test have been constructed such that
        only the top row is expected to show a difference by including the last
        5 minutes of the accumulation, all the other results are the same as
        for the 5 minute test above."""

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

        plugin = Accumulation(accumulation_period=600, accumulation_units='mm')
        result = plugin.process(self.cubes)

        self.assertArrayAlmostEqual(result[0].data, expected_t0)
        self.assertArrayAlmostEqual(result[0].data.mask, expected_mask_t0)

    def test_returns_total_accumulation_if_no_period_specified(self):
        """Test function returns a list containing a single accumulation cube
        that is the accumulation over the whole period specified by the rates
        cubes. The results are the same as the 10 minute test above as that is
        the total span of the input rates cubes."""

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

    def test_returns_expected_values_1_minute(self):
        """Test function returns the expected accumulations over a 1 minute
        aggregation period. In this case there is no aggregation, so the input
        cube should be returned as the output cube."""

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

        plugin = Accumulation(accumulation_period=60, accumulation_units='mm')
        result = plugin.process(self.cubes)

        self.assertArrayAlmostEqual(result[0].data, expected_t0)
        self.assertArrayAlmostEqual(result[7].data, expected_t7)
        self.assertArrayAlmostEqual(result[0].data.mask, expected_mask_t0)
        self.assertArrayAlmostEqual(result[7].data.mask, expected_mask_t7)


if __name__ == '__main__':
    unittest.main()
