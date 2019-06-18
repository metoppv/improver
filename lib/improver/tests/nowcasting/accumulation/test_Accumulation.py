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

import copy
import datetime
import unittest
import numpy as np

import iris
from iris.coords import DimCoord
from iris.tests import IrisTest

from improver.nowcasting.accumulation import Accumulation
from improver.tests.set_up_test_cubes import set_up_variable_cube


class rate_cube_set_up(IrisTest):
    """Set up a sequence of precipitation rates cubes for use in testing the
    accumulation plugin functionality."""

    def setUp(self):
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
            except:
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

    def test_units_set(self):
        """Test the accumulation_units are set when specified."""
        plugin = Accumulation(accumulation_units="cm")
        self.assertEqual(plugin.accumulation_units, "cm")


class Test__repr__(IrisTest):
    """Test class representation"""

    def test_basic(self):
        """Test string representation"""
        result = str(Accumulation(accumulation_units="cm"))
        expected_result = "<Accumulation: accumulation_units=cm>"
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


class Test_create_accumulation_cube(rate_cube_set_up):
    """Tests the creation of accumulation cubes."""

    def test_returns_cube(self):
        """Test function returns a cube."""

        cube = self.cubes[0]
        time_interval = 60

        result = Accumulation.create_accumulation_cube(cube, time_interval)

        self.assertIsInstance(result, iris.cube.Cube)

    def test_cube_attributes(self):
        """Test the cube created by the function has the expected
        attributes."""

        cube = self.cubes[0]
        time_interval = 60

        expected_name = 'lwe_thickness_of_precipitation_amount'
        expected_units = 'm'

        result = Accumulation.create_accumulation_cube(cube, time_interval)

        self.assertEqual(result.name(), expected_name)
        self.assertEqual(result.units, expected_units)

    def test_cube_coordinates(self):
        """Test the cube created by the function has the expected
        coordinates."""

        cube = self.cubes[1]
        time_interval = 60

        expected_time_coord = iris.coords.DimCoord(
            1510286460, 'time', units="seconds since 1970-01-01 00:00:00",
            bounds=[1510286460 - time_interval, 1510286460])
        expected_fp_coord = iris.coords.DimCoord(
            60, 'forecast_period', units="seconds",
            bounds=[0,time_interval])

        result = Accumulation.create_accumulation_cube(cube, time_interval)

        self.assertEqual(result.coord('time'), expected_time_coord)
        self.assertEqual(result.coord('forecast_period'), expected_fp_coord)


class Test_determine_masking(IrisTest):
    """Tests the determination of data masking."""

    def setUp(self):
        data = np.ones((3, 3), dtype=np.float32)
        self.unmasked = set_up_variable_cube(data)
        self.masked_no_mask = set_up_variable_cube(np.ma.MaskedArray(data))
        self.masked = set_up_variable_cube(np.ma.MaskedArray(data, mask=data))

    def test_unmasked_data(self):
        """Test function returns an empty dictionary and an unmasked numpy
        array type for unmasked input data."""

        expected = {}
        array_type, mask = Accumulation.determine_masking(self.unmasked)

        self.assertIsInstance(array_type(1), np.ndarray)
        self.assertEqual(mask, expected)

    def test_masked_data_no_mask(self):
        """Test function returns a dictionary containing mask=False when
        a masked numpy array type without a set mask is used as input data. The
        returned array type should be np.ma.MaskedArray."""

        expected = {'mask': False}
        array_type, mask = Accumulation.determine_masking(self.masked_no_mask)

        self.assertIsInstance(array_type(1), np.ma.MaskedArray)
        self.assertEqual(mask, expected)

    def test_masked_data(self):
        """Test function returns a dictionary containing the required mask when
        a masked numpy array type with a mask is used as input data. The
        returned array type should be np.ma.MaskedArray."""

        expected = {'mask': np.ones((3, 3)).astype(bool)}
        array_type, mask = Accumulation.determine_masking(self.masked)

        self.assertIsInstance(array_type(1), np.ma.MaskedArray)
        self.assertEqual(mask.keys(), expected.keys())
        self.assertArrayEqual(mask['mask'], expected['mask'])


class Test_process(rate_cube_set_up):
    """Tests the process method results in the expected outputs."""

    def test_returns_cubelist(self):
        """Test function returns a cubelist containing one less entry than the
        input cube list."""

        result = Accumulation().process(self.cubes)
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
        expected = self.cubes[0].copy(data=self.cubes[0].data * 60 / 1000)

        result = Accumulation().process(self.cubes)

        self.assertEqual(result[0].units, 'm')
        self.assertArrayAlmostEqual(result[0].data, expected.data)

    def test_default_altered_output_units(self):
        """Test the function returns accumulations in the specified units if
        they are explicitly set. Here the units are set to mm."""

        # Multiply the rates in mm/s by 60 to get accumulation over 1 minute
        expected = self.cubes[0].copy(data=self.cubes[0].data * 60)

        result = Accumulation(accumulation_units='mm').process(self.cubes)

        self.assertEqual(result[0].units, 'mm')
        self.assertArrayAlmostEqual(result[0].data, expected.data)


if __name__ == '__main__':
    unittest.main()
