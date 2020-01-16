#!/usr/bin/env python
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
"""Module with tests for the ApplyOrographicEnhancement plugin."""

import unittest
from datetime import datetime

import iris
import numpy as np
from cf_units import Unit
from iris.tests import IrisTest

from improver.nowcasting.utilities import ApplyOrographicEnhancement

from ...set_up_test_cubes import set_up_variable_cube

MIN_PRECIP_RATE_MMH = ApplyOrographicEnhancement("add").min_precip_rate_mmh


def set_up_precipitation_rate_cubelist():
    """Create a cubelist containing cubes with metadata and values suitable
    for precipitation rate."""
    data = np.array([[[0., 1., 2.],
                      [1., 2., 3.],
                      [0., 2., 2.]]], dtype=np.float32)
    cube1 = set_up_variable_cube(
        data, name="lwe_precipitation_rate", units="mm h-1",
        time=datetime(2017, 1, 10, 3), frt=datetime(2017, 1, 10, 3))
    cube1.convert_units("m s-1")

    data = np.array([[[4., 4., 1.],
                      [4., 4., 1.],
                      [4., 4., 1.]]], dtype=np.float32)
    cube2 = set_up_variable_cube(
        data, name="lwe_precipitation_rate", units="mm h-1",
        time=datetime(2017, 1, 10, 4), frt=datetime(2017, 1, 10, 4))
    cube2.convert_units("m s-1")

    return iris.cube.CubeList([cube1, cube2])


def set_up_orographic_enhancement_cube():
    """Create a cube with metadata and values suitable for
    precipitation rate."""
    data = np.array([[[0., 0., 0.],
                      [0., 0., 4.],
                      [1., 1., 2.]]], dtype=np.float32)
    cube1 = set_up_variable_cube(
        data, name="orographic_enhancement", units="mm h-1",
        time=datetime(2017, 1, 10, 3), frt=datetime(2017, 1, 10, 3))
    cube1.convert_units("m s-1")

    data = np.array([[[5., 5., 5.],
                      [2., 1., 0.],
                      [2., 1., 0.]]], dtype=np.float32)
    cube2 = set_up_variable_cube(
        data, name="orographic_enhancement", units="mm h-1",
        time=datetime(2017, 1, 10, 4), frt=datetime(2017, 1, 10, 4))
    cube2.convert_units("m s-1")

    return iris.cube.CubeList([cube1, cube2]).merge_cube()


class Test__init__(IrisTest):

    """Test the __init__ method."""

    def test_basic(self):
        """Test that the plugin can be initialised as required."""
        plugin = ApplyOrographicEnhancement("add")
        self.assertEqual(plugin.operation, "add")


class Test__repr__(IrisTest):

    """Test the __repr__ method."""

    def test_basic(self):
        """Test that the __repr__ returns the expected string."""
        result = str(ApplyOrographicEnhancement("add"))
        msg = ('<ApplyOrographicEnhancement: operation: add>')
        self.assertEqual(result, msg)


class Test__select_orographic_enhancement_cube(IrisTest):

    """Test the _select_orographic_enhancement method."""

    def setUp(self):
        """Set up cubes for testing."""
        self.precip_cube = set_up_precipitation_rate_cubelist()[0]
        self.oe_cube = set_up_orographic_enhancement_cube()
        self.first_slice = self.oe_cube[0]
        self.second_slice = self.oe_cube[1]

    def test_basic(self):
        """Test extracting a time coordinate from the orographic enhancement
        cube."""
        plugin = ApplyOrographicEnhancement("add")
        result = plugin._select_orographic_enhancement_cube(
            self.precip_cube, self.oe_cube)
        self.assertIsInstance(result, iris.cube.Cube)
        self.assertEqual(result.metadata, self.first_slice.metadata)
        self.assertEqual(result, self.first_slice)
        self.assertEqual(result.coord("time"), self.first_slice.coord("time"))

    def test_alternative_time_quarter_past(self):
        """Test extracting a time coordinate from the orographic enhancement
        cube at quarter past an hour."""
        plugin = ApplyOrographicEnhancement("add")
        self.precip_cube.coord("time").points = (
            self.precip_cube.coord("time").points + 15*60)  # add 15 mins
        result = plugin._select_orographic_enhancement_cube(
            self.precip_cube, self.oe_cube)
        self.assertIsInstance(result, iris.cube.Cube)
        self.assertEqual(result.metadata, self.first_slice.metadata)
        self.assertEqual(result, self.first_slice)
        self.assertEqual(result.coord("time"), self.first_slice.coord("time"))

    def test_alternative_time_half_past(self):
        """Test extracting a time coordinate from the orographic enhancement
        cube at half past an hour. Note that the time point will round down
        from the midpoint."""
        plugin = ApplyOrographicEnhancement("add")
        self.precip_cube.coord("time").points = (
            self.precip_cube.coord("time").points + 30*60)  # add 30 mins
        result = plugin._select_orographic_enhancement_cube(
            self.precip_cube, self.oe_cube)
        self.assertIsInstance(result, iris.cube.Cube)
        self.assertEqual(result.metadata, self.first_slice.metadata)
        self.assertEqual(result, self.first_slice)
        self.assertEqual(result.coord("time"), self.first_slice.coord("time"))

    def test_alternative_time_quarter_to(self):
        """Test extracting a time coordinate from the orographic enhancement
        cube at quarter to an hour."""
        plugin = ApplyOrographicEnhancement("add")
        self.precip_cube.coord("time").points = (
            self.precip_cube.coord("time").points + 45*60)  # add 45 mins
        result = plugin._select_orographic_enhancement_cube(
            self.precip_cube, self.oe_cube)
        self.assertIsInstance(result, iris.cube.Cube)
        self.assertEqual(result.metadata, self.second_slice.metadata)
        self.assertEqual(result, self.second_slice)
        self.assertEqual(result.coord("time"), self.second_slice.coord("time"))


class Test__apply_orographic_enhancement(IrisTest):

    """Test the _apply_orographic_enhancement method."""

    def setUp(self):
        """Set up cubes for testing."""
        self.precip_cube = set_up_precipitation_rate_cubelist()[0]
        self.oe_cube = set_up_orographic_enhancement_cube()
        self.sliced_oe_cube = self.oe_cube[0]

    def test_check_expected_values_add(self):
        """Test the expected values are returned when cubes are added.
        First check."""
        expected = np.array([[[0., 1., 2.],
                              [1., 2., 7.],
                              [0., 3., 4.]]])
        plugin = ApplyOrographicEnhancement("add")
        result = plugin._apply_orographic_enhancement(
            self.precip_cube, self.sliced_oe_cube)
        self.assertIsInstance(result, iris.cube.Cube)
        self.assertEqual(result.metadata, self.precip_cube.metadata)
        result.convert_units("mm/hr")
        self.assertArrayAlmostEqual(result.data, expected)

    def test_check_expected_values_subtract(self):
        """Test the expected values are returned when one cube is subtracted
        from another."""
        expected = np.array([[[0., 1., 2.],
                              [1., 2., -1.],
                              [0., 1., 0.]]])
        plugin = ApplyOrographicEnhancement("subtract")
        result = plugin._apply_orographic_enhancement(
            self.precip_cube, self.sliced_oe_cube)
        self.assertIsInstance(result, iris.cube.Cube)
        self.assertEqual(result.metadata, self.precip_cube.metadata)
        result.convert_units("mm/hr")
        self.assertArrayAlmostEqual(result.data, expected)

    def test_check_expected_values_for_different_units(self):
        """Test the expected values are returned when cubes are combined when
        the orographic enhancement cube is in different units to the
        precipitation rate cube."""
        expected = np.array([[[0., 1., 2.],
                              [1., 2., 7.],
                              [0., 3., 4.]]])
        oe_cube = self.sliced_oe_cube
        oe_cube.convert_units("m/hr")
        plugin = ApplyOrographicEnhancement("add")
        result = plugin._apply_orographic_enhancement(
            self.precip_cube, oe_cube)
        self.assertIsInstance(result, iris.cube.Cube)
        self.assertEqual(result.metadata, self.precip_cube.metadata)
        result.convert_units("mm/hr")
        self.assertArrayAlmostEqual(result.data, expected)

    def test_check_unchanged_oe_cube_for_subtract(self):
        """Test the expected values are returned when one cube is subtracted
        from another."""
        orig_oe_cube = self.sliced_oe_cube.copy()
        oe_cube = self.sliced_oe_cube.copy()
        oe_cube.convert_units("m/hr")
        plugin = ApplyOrographicEnhancement("subtract")
        plugin._apply_orographic_enhancement(self.precip_cube, oe_cube)
        self.assertEqual(orig_oe_cube, self.sliced_oe_cube)


class Test__apply_minimum_precip_rate(IrisTest):

    """Test the _apply_minimum_precip_rate method."""

    def setUp(self):
        """Set up cubes for testing. This includes a 'subtracted_cube'
        containing some negative precipitation values that should be
        set to a minimum precipitation rate threshold."""
        self.precip_cube = set_up_precipitation_rate_cubelist()[0]
        oe_cube = set_up_orographic_enhancement_cube()[0]
        # Cap orographic enhancement to be zero where there is a precipitation
        # rate of zero.
        original_units = Unit("mm/hr")
        threshold_in_cube_units = (
            original_units.convert(MIN_PRECIP_RATE_MMH,
                                   self.precip_cube.units))
        oe_cube.data[self.precip_cube.data < threshold_in_cube_units] = 0.
        self.oe_cube = oe_cube
        self.added_cube = self.precip_cube + oe_cube
        self.subtracted_cube = self.precip_cube - oe_cube

    def test_basic_add(self):
        """Test a minimum precipitation rate is applied, when the orographic
        enhancement causes the precipitation rate to become negative"""
        expected = np.array([[[0., 1., 2.],
                              [1., 2., 7.],
                              [0., 3., 4.]]])
        precip_cube = self.precip_cube.copy()
        added_cube = self.added_cube.copy()
        plugin = ApplyOrographicEnhancement("add")
        result = plugin._apply_minimum_precip_rate(precip_cube, added_cube)
        self.assertIsInstance(result, iris.cube.Cube)
        self.assertEqual(result.units, Unit("m/s"))
        self.assertEqual(result.metadata, added_cube.metadata)
        result.convert_units("mm/hr")
        self.assertArrayAlmostEqual(result.data, expected)

    def test_basic_subtract(self):
        """Test a minimum precipitation rate is applied, when the orographic
        enhancement causes the precipitation rate to become negative"""
        expected = np.array([[[0., 1., 2.],
                              [1., 2., MIN_PRECIP_RATE_MMH],
                              [0., 1., MIN_PRECIP_RATE_MMH]]])
        precip_cube = self.precip_cube.copy()
        subtracted_cube = self.subtracted_cube.copy()
        plugin = ApplyOrographicEnhancement("subtract")
        result = (
            plugin._apply_minimum_precip_rate(precip_cube, subtracted_cube))
        self.assertIsInstance(result, iris.cube.Cube)
        self.assertEqual(result.units, Unit("m/s"))
        self.assertEqual(result.metadata, subtracted_cube.metadata)
        result.convert_units("mm/hr")
        self.assertArrayAlmostEqual(result.data, expected)

    def test_no_min_precip_rate_applied_no_input_precip(self):
        """Test no minimum precipitation rate is applied, when the input
        precipitation rate to always below the orographic enhancement."""
        expected = np.array([[[0., 0., 0.],
                              [0., 0., 1.],
                              [0., 1., 1.]]])
        precip_cube = self.precip_cube.copy()
        precip_cube.convert_units("mm/hr")
        precip_cube.data = np.array([[[0., 0., 0.],
                                      [0., 0., 5.],
                                      [0., 2., 3.]]])
        precip_cube.convert_units("m/s")
        subtracted_cube = precip_cube - self.oe_cube
        plugin = ApplyOrographicEnhancement("subtract")
        result = (
            plugin._apply_minimum_precip_rate(precip_cube, subtracted_cube))
        self.assertIsInstance(result, iris.cube.Cube)
        self.assertEqual(result.units, Unit("m/s"))
        self.assertEqual(result.metadata, subtracted_cube.metadata)
        result.convert_units("mm/hr")
        self.assertArrayAlmostEqual(result.data, expected)

    def test_no_min_precip_rate_applied_no_negative_rates(self):
        """Test no minimum precipitation rate is applied, when the cube
        calculated by subtracting the orographic enhancement from the input
        precipitation is always positive, so there are no negative values
        that require the minimum precipitation rate to be applied."""
        expected = np.array([[[1., 1., 1.],
                              [1., 2., 3.],
                              [3., 3., 4.]]])
        precip_cube = self.precip_cube.copy()
        subtracted_cube = self.subtracted_cube.copy()
        subtracted_cube.convert_units("mm/hr")
        subtracted_cube.data = np.array([[[1., 1., 1.],
                                          [1., 2., 3.],
                                          [3., 3., 4.]]])
        subtracted_cube.convert_units("m/s")
        plugin = ApplyOrographicEnhancement("subtract")
        result = (
            plugin._apply_minimum_precip_rate(precip_cube, subtracted_cube))
        self.assertIsInstance(result, iris.cube.Cube)
        self.assertEqual(result.units, Unit("m/s"))
        self.assertEqual(result.metadata, subtracted_cube.metadata)
        result.convert_units("mm/hr")
        self.assertArrayAlmostEqual(result.data, expected)

    def test_no_unit_conversion(self):
        """Test that the minimum precipitation rate is applied correctly,
        when the units of the input cube do not require conversion to mm/hr."""
        expected = np.array([[[0., 1., 2.],
                              [1., 2., MIN_PRECIP_RATE_MMH],
                              [0., 1., MIN_PRECIP_RATE_MMH]]])
        precip_cube = self.precip_cube.copy()
        subtracted_cube = self.subtracted_cube.copy()
        precip_cube.convert_units("mm/hr")
        subtracted_cube.convert_units("mm/hr")
        plugin = ApplyOrographicEnhancement("subtract")
        result = (
            plugin._apply_minimum_precip_rate(precip_cube, subtracted_cube))
        self.assertIsInstance(result, iris.cube.Cube)
        self.assertEqual(result.units, Unit("mm/hr"))
        self.assertEqual(result.metadata, subtracted_cube.metadata)
        self.assertArrayAlmostEqual(result.data, expected)

    def test_differing_units(self):
        """Test that the minimum precipitation rate is applied correctly,
        when the units of the input precipitation cube and the cube
        computed using the input precipitation cube and the orographic
        enhancement cube are handled correctly, even if the units differ."""
        expected = np.array([[[0., 1., 2.],
                              [1., 2., MIN_PRECIP_RATE_MMH],
                              [0., 1., MIN_PRECIP_RATE_MMH]]])
        precip_cube = self.precip_cube.copy()
        subtracted_cube = self.subtracted_cube.copy()
        precip_cube.convert_units("km/hr")
        subtracted_cube.convert_units("ft/s")
        plugin = ApplyOrographicEnhancement("subtract")
        result = (
            plugin._apply_minimum_precip_rate(precip_cube, subtracted_cube))
        self.assertIsInstance(result, iris.cube.Cube)
        self.assertEqual(result.units, Unit("ft/s"))
        self.assertEqual(result.metadata, subtracted_cube.metadata)
        result.convert_units("mm/hr")
        self.assertArrayAlmostEqual(result.data, expected)

    def test_NaN_values(self):
        """Test that NaN values are preserved when they are contained within
        the input when applying a minimum precipitation rate."""
        expected = np.array([[[0., 1., 2.],
                              [np.NaN, np.NaN, np.NaN],
                              [0., 1., MIN_PRECIP_RATE_MMH]]])
        precip_cube = self.precip_cube.copy()
        subtracted_cube = self.subtracted_cube.copy()
        subtracted_cube.convert_units("mm/hr")
        subtracted_cube.data = np.array([[[0., 1., 2.],
                                          [np.NaN, np.NaN, np.NaN],
                                          [0., 1., 0.]]])
        subtracted_cube.convert_units("m/s")
        plugin = ApplyOrographicEnhancement("subtract")
        result = (
            plugin._apply_minimum_precip_rate(precip_cube, subtracted_cube))
        self.assertIsInstance(result, iris.cube.Cube)
        self.assertEqual(result.units, Unit("m/s"))
        self.assertEqual(result.metadata, subtracted_cube.metadata)
        result.convert_units("mm/hr")
        self.assertArrayAlmostEqual(result.data, expected)


class Test_process(IrisTest):

    """Test the apply_orographic_enhancement method."""

    def setUp(self):
        """Set up cubes for testing."""
        self.precip_cubes = set_up_precipitation_rate_cubelist()
        self.oe_cube = set_up_orographic_enhancement_cube()

    def test_basic_add(self):
        """Test the addition of a precipitation rate cubelist and an
        orographic enhancement cube with multiple times."""
        expected0 = np.array([[[0., 1., 2.],
                               [1., 2., 7.],
                               [0., 3., 4.]]])
        expected1 = np.array([[[9., 9., 6.],
                               [6., 5., 1.],
                               [6., 5., 1.]]])
        plugin = ApplyOrographicEnhancement("add")
        result = plugin.process(self.precip_cubes, self.oe_cube)
        self.assertIsInstance(result, iris.cube.CubeList)
        for aresult, precip_cube in zip(result, self.precip_cubes):
            self.assertEqual(
                aresult.metadata, precip_cube.metadata)
        for cube in result:
            cube.convert_units("mm/hr")
        self.assertArrayAlmostEqual(result[0].data, expected0)
        self.assertArrayAlmostEqual(result[1].data, expected1)

    def test_basic_subtract(self):
        """Test the subtraction of a cube of orographic
        enhancement with multiple times from cubes of precipitation rate."""
        expected0 = np.array([[[0., 1., 2.],
                               [1., 2., MIN_PRECIP_RATE_MMH],
                               [0., 1., MIN_PRECIP_RATE_MMH]]])
        expected1 = np.array(
            [[[MIN_PRECIP_RATE_MMH, MIN_PRECIP_RATE_MMH, MIN_PRECIP_RATE_MMH],
              [2., 3., 1.],
              [2., 3., 1.]]])
        plugin = ApplyOrographicEnhancement("subtract")
        result = plugin.process(self.precip_cubes, self.oe_cube)
        self.assertIsInstance(result, iris.cube.CubeList)
        for aresult, precip_cube in zip(result, self.precip_cubes):
            self.assertEqual(
                aresult.metadata, precip_cube.metadata)
        for cube in result:
            cube.convert_units("mm/hr")
        self.assertArrayAlmostEqual(result[0].data, expected0)
        self.assertArrayAlmostEqual(result[1].data, expected1)

    def test_exception(self):
        """Test that an exception is raised if the operation requested is
        not a valid choice."""
        plugin = ApplyOrographicEnhancement("invalid")
        msg = "Operation 'invalid' not supported for"
        with self.assertRaisesRegex(ValueError, msg):
            plugin.process(self.precip_cubes, self.oe_cube)

    def test_add_with_mask(self):
        """Test the addition of cubelists containing cubes of
        precipitation rate and orographic enhancement, where a mask has
        been applied. Orographic enhancement is not added to the masked
        points (where precip rate <= 1 mm/hr)."""
        expected0 = np.array([[[0., 1., 2.],
                               [1., 2., 7.],
                               [0., 3., 4.]]])
        expected1 = np.array([[[9., 9., 1.],
                               [6., 5., 1.],
                               [6., 5., 1.]]])

        precip_cubes = self.precip_cubes.copy()

        # Mask values within the input precipitation cube that are equal to,
        # or below 1.
        new_precip_cubes = iris.cube.CubeList([])
        for precip_cube in precip_cubes:
            precip_cube.convert_units("mm/hr")
            masked = np.ma.masked_where(
                precip_cube.data <= 1, precip_cube.data)
            precip_cube.data = masked
            precip_cube.convert_units("m/s")
            new_precip_cubes.append(precip_cube)

        plugin = ApplyOrographicEnhancement("add")
        result = plugin.process(self.precip_cubes, self.oe_cube)

        self.assertIsInstance(result, iris.cube.CubeList)
        for aresult, precip_cube in zip(result, self.precip_cubes):
            self.assertIsInstance(aresult.data, np.ma.MaskedArray)
            self.assertEqual(
                aresult.metadata, precip_cube.metadata)
        for cube in result:
            cube.convert_units("mm/hr")
        self.assertArrayAlmostEqual(result[0].data.data, expected0)
        self.assertArrayAlmostEqual(result[1].data.data, expected1)

    def test_subtract_with_mask(self):
        """Test the subtraction of cubelists containing cubes of orographic
        enhancement from cubes of precipitation rate, where a mask has been
        applied. Orographic enhancement is not added to the masked points
        (where precip rate <= 1 mm/hr)."""
        expected0 = np.array([[[0., 1., 2.],
                               [1., 2., MIN_PRECIP_RATE_MMH],
                               [0., 1., MIN_PRECIP_RATE_MMH]]])
        expected1 = np.array([[[MIN_PRECIP_RATE_MMH, MIN_PRECIP_RATE_MMH, 1.],
                               [2., 3., 1.],
                               [2., 3., 1.]]])

        # Mask values within the input precipitation cube that are equal to,
        # or below 1.
        precip_cubes = self.precip_cubes.copy()
        new_precip_cubes = iris.cube.CubeList([])
        for precip_cube in precip_cubes:
            precip_cube.convert_units("mm/hr")
            masked = np.ma.masked_where(
                precip_cube.data <= 1, precip_cube.data)
            precip_cube.data = masked
            precip_cube.convert_units("m/s")
            new_precip_cubes.append(precip_cube)

        plugin = ApplyOrographicEnhancement("subtract")
        result = plugin.process(self.precip_cubes, self.oe_cube)
        self.assertIsInstance(result, iris.cube.CubeList)
        for aresult, precip_cube in zip(result, self.precip_cubes):
            self.assertIsInstance(aresult.data, np.ma.MaskedArray)
            self.assertEqual(
                aresult.metadata, precip_cube.metadata)
        for cube in result:
            cube.convert_units("mm/hr")
        self.assertArrayAlmostEqual(result[0].data.data, expected0)
        self.assertArrayAlmostEqual(result[1].data.data, expected1)

    def test_one_input_cube(self):
        """Test the addition of precipitation rate and orographic enhancement,
        where a single precipitation rate cube is provided."""
        expected = np.array([[[0., 1., 2.],
                              [1., 2., 7.],
                              [0., 3., 4.]]])
        plugin = ApplyOrographicEnhancement("add")
        result = plugin.process(self.precip_cubes[0], self.oe_cube)
        self.assertIsInstance(result, iris.cube.CubeList)
        for aresult, precip_cube in zip(result, self.precip_cubes):
            self.assertEqual(
                aresult.metadata, precip_cube.metadata)
        for cube in result:
            cube.convert_units("mm/hr")
        self.assertArrayAlmostEqual(result[0].data, expected)

    def test_only_one_orographic_enhancement_cube(self):
        """Test where is an orographic enhancement cube with a single time
        point is supplied, so that multiple input precipitation fields are
        adjusted by the same orographic enhancement."""
        expected0 = np.array([[[0., 1., 2.],
                               [1., 2., MIN_PRECIP_RATE_MMH],
                               [0., 1., MIN_PRECIP_RATE_MMH]]])
        expected1 = np.array(
            [[[4., 4., 1.],
              [4., 4., MIN_PRECIP_RATE_MMH],
              [3., 3., MIN_PRECIP_RATE_MMH]]])
        sliced_oe_cube = self.oe_cube[0]
        # Change the time of precip to be within 30 mins of oe_cube.
        self.precip_cubes[1].coord('time').points = (
                self.precip_cubes[1].coord('time').points - 2700)
        plugin = ApplyOrographicEnhancement("subtract")
        result = plugin.process(self.precip_cubes, sliced_oe_cube)
        self.assertIsInstance(result, iris.cube.CubeList)
        for aresult, precip_cube in zip(result, self.precip_cubes):
            self.assertEqual(
                aresult.metadata, precip_cube.metadata)
        for cube in result:
            cube.convert_units("mm/hr")
        self.assertArrayAlmostEqual(result[0].data, expected0)
        self.assertArrayAlmostEqual(result[1].data, expected1)


if __name__ == '__main__':
    unittest.main()
