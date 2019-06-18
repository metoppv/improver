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
""" Unit tests for the nowcasting.AccumulationAggregator plugin """

import types
import unittest
import numpy as np

import iris
from iris.tests import IrisTest

from improver.nowcasting.accumulation import (Accumulation,
                                              AccumulationAggregator)
from improver.utilities.warnings_handler import ManageWarnings
from improver.tests.nowcasting.accumulation.test_Accumulation import (
    rate_cube_set_up)


class accumulation_cube_set_up(IrisTest):
    """Set up a sequence of precipitation accumulation cubes for use in testing
    the accumulation aggregation plugin functionality."""

    def setUp(self):
        """Set up test cubes."""
        cubes = rate_cube_set_up().setUp()
        self.cubes = Accumulation(accumulation_units='mm').process(cubes)


class Test__init__(IrisTest):
    """Test class initialisation"""

    def test_default(self):
        """Test the default accumulation_period is set to None when not
        specified."""
        plugin = AccumulationAggregator()
        self.assertEqual(plugin.accumulation_period, None)

    def test_period_set(self):
        """Test the accumulation_period is set when specified."""
        plugin = AccumulationAggregator(accumulation_period=120)
        self.assertEqual(plugin.accumulation_period, 120)


class Test__repr__(IrisTest):
    """Test class representation"""

    def test_basic(self):
        """Test string representation"""
        result = str(AccumulationAggregator(accumulation_period=60))
        expected_result = "<AccumulationAggregator: accumulation_period=60>"
        self.assertEqual(result, expected_result)


class Test_check_accumulation_period(IrisTest):
    """Tests the requested accumulation_period can be constructed from the
    input cubes."""

    def test_returns_correct_integer(self):
        """Test function returns the expected integer."""

        ncubes = 10
        time_interval = 60
        plugin = AccumulationAggregator(accumulation_period=120)
        result = plugin.check_accumulation_period(time_interval, ncubes)
        self.assertIsInstance(result, int)
        self.assertEqual(result, 2)

    def test_returns_ncubes_for_unspecified_accumulation_period(self):
        """Test function returns ncubes if the accumulation_period is not
        specified."""

        ncubes = 10
        time_interval = 60
        plugin = AccumulationAggregator()
        result = plugin.check_accumulation_period(time_interval, ncubes)
        self.assertIsInstance(result, int)
        self.assertEqual(result, ncubes)

    @ManageWarnings(record=True)
    def test_raises_warning_for_unused_cubes(self, warning_list=None):
        """Test function raises a warning when there are insufficient cubes to
        complete the last period."""

        ncubes = 11
        time_interval = 60
        warning_msg = (
            "The provided cubes result in a partial period given the specified"
            " accumulation_period, i.e. the number of cubes is insufficient to"
            " give a set of complete periods. Only complete periods will be"
            " returned.")

        plugin = AccumulationAggregator(accumulation_period=120)
        result = plugin.check_accumulation_period(time_interval, ncubes)
        self.assertIsInstance(result, int)
        self.assertEqual(result, 2)
        self.assertTrue(any(item.category == UserWarning
                            for item in warning_list))
        self.assertTrue(any(warning_msg in str(item)
                            for item in warning_list))

    def test_raises_exception_for_impossible_aggregation(self):
        """Test function raises an exception when attempting to create an
        accumulation_period that cannot be created from the input cubes."""

        ncubes = 10
        time_interval = 61
        plugin = AccumulationAggregator(accumulation_period=120)
        msg = "The specified accumulation period "

        with self.assertRaisesRegex(ValueError, msg):
            plugin.check_accumulation_period(time_interval, ncubes)


class Test_chunk_list(accumulation_cube_set_up):
    """Tests the function chunks the input cube list as expected."""

    def test_returns_a_generator(self):
        """Test function returns a generator."""

        cube_subset = 2
        plugin = AccumulationAggregator(accumulation_period=120)
        result = plugin.chunk_list(self.cubes, cube_subset)
        self.assertIsInstance(result, types.GeneratorType)

    def test_expected_chunks_are_created(self):
        """Test function returns a generator that produces the expected
        subsets of the original list."""

        cube_subset = 5
        plugin = AccumulationAggregator(accumulation_period=120)
        result = plugin.chunk_list(self.cubes, cube_subset)
        self.assertArrayEqual(next(result), self.cubes[0:5])
        self.assertArrayEqual(next(result), self.cubes[5:])

    def test_short_chunks_returned_for_incomplete_periods(self):
        """Test function returns a generator that produces the expected
        subsets of the original list, including any trailing incomplete
        period lists."""

        cube_subset = 4
        plugin = AccumulationAggregator(accumulation_period=120)
        result = plugin.chunk_list(self.cubes, cube_subset)
        self.assertArrayEqual(next(result), self.cubes[0:4])
        self.assertArrayEqual(next(result), self.cubes[4:8])
        self.assertArrayEqual(next(result), self.cubes[8:])


class Test_process(accumulation_cube_set_up):
    """Tests the function creates aggregated accumulations as expected."""

    def test_returns_a_cubelist(self):
        """Test function returns a cubelist."""

        plugin = AccumulationAggregator(accumulation_period=120)
        result = plugin.process(self.cubes)
        self.assertIsInstance(result, iris.cube.CubeList)

    def test_raises_exception_for_unevenly_spaced_cubes(self):
        """Test function raises an exception if the input cubes are not
        spaced equally in time."""

        last_time = self.cubes[-1].coord('time').points
        self.cubes[-1].coord('time').points = last_time + 60

        msg = ("AccumulationAggregator is designed to work with accumulation "
               "cubes at regualar time intervals.")
        plugin = AccumulationAggregator(accumulation_period=120)

        with self.assertRaisesRegex(ValueError, msg):
            plugin.process(self.cubes)

    @ManageWarnings(record=False)
    def test_does_not_use_incomplete_period_data(self):
        """Test function returns only 2 accumulation periods when a 4 minute
        aggregation period is used with 10 minutes of input data. The trailing
        2 cubes are insufficient to create another period and so are disgarded.
        A warning is raised by the chunking function and has been tested above,
        so is ignored here.
        """

        plugin = AccumulationAggregator(accumulation_period=240)
        result = plugin.process(self.cubes)
        self.assertEqual(len(result), 2)

    def test_returns_expected_values_5_minutes(self):
        """Test function returns the expected accumulations over a 5 minute
        aggregation period. These are written out long hand to make the
        comparison easy."""

        expected_1st_row_t0 = np.array([
            0.03, 0.06, 0.09, 0.12, 0.15, 0.21, 0.27, 0.33, 0.39, 0.45])
        expected_2nd_row_t0 = np.array([
            0.03, 0.06, 0.09, 0.12, 0.15,
            np.inf, np.inf, np.inf, np.inf, np.inf])
        expected_3rd_row_t0 = np.array([
            0., 0., 0., 0., 0., np.inf, np.inf, np.inf, np.inf, np.inf])
        expected_4th_row_t0 = np.array([
            0., 0., 0., 0., 0., 0.09, 0.18, 0.27, 0.36, 0.45])

        expected_1st_row_t1 = np.array([
            0., 0., 0., 0., 0., 0.03, 0.06, 0.09, 0.12, 0.15])
        expected_2nd_row_t1 = np.array([
            0., 0., 0., 0., 0., 0.03, 0.06, 0.09, 0.12, 0.15])
        expected_3rd_row_t1 = np.zeros((10))
        expected_4th_row_t1 = np.zeros((10))

        plugin = AccumulationAggregator(accumulation_period=300)
        result = plugin.process(self.cubes)

        self.assertArrayAlmostEqual(result[0].data[0, :],
                                    expected_1st_row_t0)
        self.assertArrayAlmostEqual(result[0].data[1, :],
                                    expected_2nd_row_t0)
        self.assertArrayAlmostEqual(result[0].data[2, :],
                                    expected_3rd_row_t0)
        self.assertArrayAlmostEqual(result[0].data[3, :],
                                    expected_4th_row_t0)

        self.assertArrayAlmostEqual(result[1].data[0, :],
                                    expected_1st_row_t1)
        self.assertArrayAlmostEqual(result[1].data[1, :],
                                    expected_2nd_row_t1)
        self.assertArrayAlmostEqual(result[1].data[2, :],
                                    expected_3rd_row_t1)
        self.assertArrayAlmostEqual(result[1].data[3, :],
                                    expected_4th_row_t1)

    def test_returns_expected_values_10_minutes(self):
        """Test function returns the expected accumulations over the complete
        10 minute aggregation period. These are written out long hand to make
        the comparison easy. Note that the test have been constructed such that
        only the top row is expected to show a difference by including the last
        5 minutes of the accumulation, all the other results are the same as
        for the 5 minute test above."""

        expected_1st_row = np.array([
            0.03, 0.06, 0.09, 0.12, 0.15, 0.24, 0.33, 0.42, 0.51, 0.6])
        expected_2nd_row = np.array([
            0.03, 0.06, 0.09, 0.12, 0.15,
            np.inf, np.inf, np.inf, np.inf, np.inf])
        expected_3rd_row = np.array([
            0., 0., 0., 0., 0., np.inf, np.inf, np.inf, np.inf, np.inf])
        expected_4th_row = np.array([
            0., 0., 0., 0., 0., 0.09, 0.18, 0.27, 0.36, 0.45])

        plugin = AccumulationAggregator()
        result = plugin.process(self.cubes)

        self.assertArrayAlmostEqual(result[0].data[0, :],
                                    expected_1st_row)
        self.assertArrayAlmostEqual(result[0].data[1, :],
                                    expected_2nd_row)
        self.assertArrayAlmostEqual(result[0].data[2, :],
                                    expected_3rd_row)
        self.assertArrayAlmostEqual(result[0].data[3, :],
                                    expected_4th_row)


if __name__ == '__main__':
    unittest.main()
