#!/usr/bin/env python
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
"""Module with tests for the ApplyOrographicEnhancement plugin."""

import unittest

from cf_units import Unit
import iris
from iris.tests import IrisTest
import numpy as np

from improver.nowcasting.nowcasting import ApplyOrographicEnhancement
from improver.tests.ensemble_calibration.ensemble_calibration.\
    helper_functions import set_up_cube


def set_up_precipitation_rate_cube():
    """Create a cube with metadata and values suitable for
    precipitation rate."""
    data = np.array([[[[0., 1., 2.],
                       [1., 2., 3.],
                       [2., 2., 2.]]]])
    cube1 = set_up_cube(data, "lwe_precipitation_rate", "mm/hr",
                        realizations=np.array([0]), timesteps=1)
    cube1.coord("time").points = [412227.0]
    cube1.convert_units("m s-1")

    data = np.array([[[[4., 4., 1.],
                       [4., 4., 1.],
                       [4., 4., 1.]]]])
    cube2 = set_up_cube(data, "lwe_precipitation_rate", "mm/hr",
                        realizations=np.array([0]), timesteps=1)
    cube2.coord("time").points = [412228.0]
    cube2.convert_units("m s-1")

    return iris.cube.CubeList([cube1, cube2])


def set_up_orographic_enhancement_cube():
    """Create a cube with metadata and values suitable for
    precipitation rate."""
    data = np.array([[[[0., 0., 0.],
                       [0., 0., 0.],
                       [1., 1., 2.]]]])
    cube1 = set_up_cube(data, "orographic_enhancement", "mm/hr",
                        realizations=np.array([0]), timesteps=1)
    cube1.coord("time").points = [412227.0]
    cube1.convert_units("m s-1")

    data = np.array([[[[2., 1., 0.],
                       [2., 1., 0.],
                       [2., 1., 0.]]]])
    cube2 = set_up_cube(data, "orographic_enhancement", "mm/hr",
                        realizations=np.array([0]), timesteps=1)
    cube2.coord("time").points = [412228.0]
    cube2.convert_units("m s-1")

    return iris.cube.CubeList([cube1, cube2])


class Test__init__(IrisTest):

    """Test the __init__ method."""

    def setUp(self):
        """Set up cubes for testing."""
        self.precip_cubes = set_up_precipitation_rate_cube()
        self.oe_cubes = set_up_orographic_enhancement_cube()

    def test_basic(self):
        plugin = ApplyOrographicEnhancement("add")
        self.assertEqual(plugin.operation, "add")

    def test_exception(self):
        msg = "Operation 'multiply' not supported for"
        with self.assertRaisesRegex(ValueError, msg):
            ApplyOrographicEnhancement("multiply")


class Test__repr__(IrisTest):

    """Test the __repr__ method."""

    def test_basic(self):
        """Test that the __repr__ returns the expected string."""
        result = str(ApplyOrographicEnhancement("add"))
        msg = ('<ApplyOrographicEnhancement: operation: add>')
        self.assertEqual(result, msg)


class Test__extract_orographic_enhancement_cube(IrisTest):

    """Test the _extract_orographic_enhancement method."""

    def setUp(self):
        """Set up cubes for testing."""
        self.precip_cubes = set_up_precipitation_rate_cube()
        self.oe_cube = set_up_orographic_enhancement_cube().concatenate_cube()
        self.first_slice = self.oe_cube[:, 0, :, :]
        self.second_slice = self.oe_cube[:, 1, :, :]

    def test_basic(self):
        """Test extracting a time coordinate from the orographic enhancement
        cube."""
        expected0 = np.array([[[[0., 1., 2.],
                                [1., 2., 3.],
                                [3., 3., 4.]]]])
        plugin = ApplyOrographicEnhancement("add")
        result = plugin._extract_orographic_enhancement_cube(
            self.precip_cubes[0], self.first_slice)
        self.assertIsInstance(result, iris.cube.Cube)
        self.assertEqual(result.metadata, self.first_slice.metadata)
        self.assertEqual(result, self.first_slice)
        self.assertEqual(result.coord("time"), self.first_slice.coord("time"))

    def test_alternative_time_quarter_past(self):
        """Test extracting a time coordinate from the orographic enhancement
        cube."""
        expected0 = np.array([[[[0., 1., 2.],
                                [1., 2., 3.],
                                [3., 3., 4.]]]])
        plugin = ApplyOrographicEnhancement("add")
        self.precip_cubes[0].coord("time").points = 412227.25
        print("self.precip_cubes[0] = ", self.precip_cubes[0])
        print("self.precip_cubes[0] = ", self.precip_cubes[0].coord("time"))
        print("self.oe_cube = ", self.oe_cube)
        result = plugin._extract_orographic_enhancement_cube(
            self.precip_cubes[0], self.oe_cube)
        self.assertIsInstance(result, iris.cube.Cube)
        self.assertEqual(result.metadata, self.first_slice.metadata)
        self.assertEqual(result, self.first_slice)
        self.assertEqual(result.coord("time"), self.first_slice.coord("time"))

    def test_alternative_time_half_past(self):
        """Test extracting a time coordinate from the orographic enhancement
        cube."""
        expected0 = np.array([[[[0., 1., 2.],
                                [1., 2., 3.],
                                [3., 3., 4.]]]])
        plugin = ApplyOrographicEnhancement("add")
        self.precip_cubes[0].coord("time").points = 412227.51
        print("self.precip_cubes[0] = ", self.precip_cubes[0])
        print("self.precip_cubes[0] = ", self.precip_cubes[0].coord("time"))
        print("self.oe_cube = ", self.oe_cube)
        result = plugin._extract_orographic_enhancement_cube(
            self.precip_cubes[0], self.oe_cube)
        self.assertIsInstance(result, iris.cube.Cube)
        self.assertEqual(result.metadata, self.second_slice.metadata)
        self.assertEqual(result, self.second_slice)
        self.assertEqual(result.coord("time"), self.second_slice.coord("time"))

    def test_alternative_time_quarter_to(self):
        """Test extracting a time coordinate from the orographic enhancement
        cube."""
        expected0 = np.array([[[[0., 1., 2.],
                                [1., 2., 3.],
                                [3., 3., 4.]]]])
        plugin = ApplyOrographicEnhancement("add")
        self.precip_cubes[0].coord("time").points = 412227.75
        print("self.precip_cubes[0] = ", self.precip_cubes[0])
        print("self.precip_cubes[0] = ", self.precip_cubes[0].coord("time"))
        print("self.oe_cube = ", self.oe_cube)
        result = plugin._extract_orographic_enhancement_cube(
            self.precip_cubes[0], self.oe_cube)
        self.assertIsInstance(result, iris.cube.Cube)
        self.assertEqual(result.metadata, self.second_slice.metadata)
        self.assertEqual(result, self.second_slice)
        self.assertEqual(result.coord("time"), self.second_slice.coord("time"))


class Test__apply_cube_combiner(IrisTest):

    """Test the __apply_cube_combiner method."""

    def setUp(self):
        """Set up cubes for testing."""
        self.precip_cubes = set_up_precipitation_rate_cube()
        self.oe_cubes = set_up_orographic_enhancement_cube()

    def test_check_expected_values_first(self):
        """Test the expected values are returned when cubes are combined.
        First check."""
        expected0 = np.array([[[[0., 1., 2.],
                                [1., 2., 3.],
                                [3., 3., 4.]]]])
        plugin = ApplyOrographicEnhancement("add")
        result = plugin._apply_cube_combiner(
            self.precip_cubes[0], self.oe_cubes[0])
        self.assertIsInstance(result, iris.cube.Cube)
        self.assertEqual(result.metadata, self.precip_cubes[0].metadata)
        result.convert_units("mm/hr")
        self.assertArrayAlmostEqual(result.data, expected0)

    def test_check_expected_values_second(self):
        """Test the expected values are returned when cubes are combined.
        Second check."""
        expected1 = np.array([[[[6., 5., 1.],
                                [6., 5., 1.],
                                [6., 5., 1.]]]])
        plugin = ApplyOrographicEnhancement("add")
        result = plugin._apply_cube_combiner(
            self.precip_cubes[1], self.oe_cubes[1])
        self.assertIsInstance(result, iris.cube.Cube)
        self.assertEqual(result.metadata, self.precip_cubes[1].metadata)
        result.convert_units("mm/hr")
        self.assertArrayAlmostEqual(result.data, expected1)

    def test_check_scalar_time_dimensions(self):
        """Test the expected values are returned when cubes are combined when
        the dimensions of the input cubes mismatch."""
        expected0 = np.array([[[[0., 1., 2.],
                                [1., 2., 3.],
                                [3., 3., 4.]]]])
        oe_cube = self.oe_cubes[0][:, 0, :, :]
        plugin = ApplyOrographicEnhancement("add")
        result = plugin._apply_cube_combiner(self.precip_cubes[0], oe_cube)
        self.assertIsInstance(result, iris.cube.Cube)
        self.assertEqual(result.metadata, self.precip_cubes[0].metadata)
        result.convert_units("mm/hr")
        self.assertArrayAlmostEqual(result.data, expected0)


class Test_apply_minimum_precip_rate(IrisTest):

    """Test the __apply_minimum_precip_rate method."""

    def setUp(self):
        """Set up cubes for testing."""
        self.precip_cube = set_up_precipitation_rate_cube()[0]

    def test_basic(self):
        """Test a minimum precipitation rate is applied, when the orographic
        enhancement causes the precipitation rate to become negative"""
        expected0 = np.array([[[[1/32., 1/32., 0.],
                                [1., 2., 3.],
                                [3., 3., 4.]]]])
        precip_cube = self.precip_cube.copy()
        precip_cube.convert_units("mm/hr")
        precip_cube.data = np.array([[[[-1., -1., 0.],
                                       [1., 2., 3.],
                                       [3., 3., 4.]]]])
        precip_cube.convert_units("m/s")
        plugin = ApplyOrographicEnhancement("add")
        result = plugin.apply_minimum_precip_rate(precip_cube)
        self.assertIsInstance(result, iris.cube.Cube)
        self.assertEqual(result.units, Unit("m/s"))
        self.assertEqual(result.metadata, precip_cube.metadata)
        result.convert_units("mm/hr")
        self.assertArrayAlmostEqual(result.data, expected0)

    def test_no_rate_below_zero(self):
        """Test no minimum precipitation rate is applied, when the orographic
        enhancement does not cause the precipitation rate to become
        negative."""
        expected0 = np.array([[[[1/32., 1/32., 0.],
                                [1., 2., 3.],
                                [3., 3., 4.]]]])
        precip_cube = self.precip_cube.copy()
        precip_cube.convert_units("mm/hr")
        precip_cube.data = np.array([[[[-1., -1., 0.],
                                       [1., 2., 3.],
                                       [3., 3., 4.]]]])
        precip_cube.convert_units("m/s")
        plugin = ApplyOrographicEnhancement("add")
        result = plugin.apply_minimum_precip_rate(precip_cube)
        self.assertIsInstance(result, iris.cube.Cube)
        self.assertEqual(result.units, Unit("m/s"))
        self.assertEqual(result.metadata, precip_cube.metadata)
        result.convert_units("mm/hr")
        self.assertArrayAlmostEqual(result.data, expected0)

    def test_no_unit_conversion(self):
        """Test that the minimum precipitation rate is applied correctly,
        when the units of the input cube do not require conversion to mm/hr."""
        expected0 = np.array([[[[1/32., 1/32., 0.],
                                [1., 2., 3.],
                                [3., 3., 4.]]]])
        precip_cube = self.precip_cube.copy()
        precip_cube.convert_units("mm/hr")
        precip_cube.data = np.array([[[[-1., -1., 0.],
                                       [1., 2., 3.],
                                       [3., 3., 4.]]]])
        self.precip_cube.convert_units("mm/hr")
        plugin = ApplyOrographicEnhancement("add")
        result = plugin.apply_minimum_precip_rate(precip_cube)
        self.assertIsInstance(result, iris.cube.Cube)
        self.assertEqual(result.units, Unit("mm/hr"))
        self.assertEqual(result.metadata, precip_cube.metadata)
        self.assertArrayAlmostEqual(result.data, expected0)

    def test_tolerance(self):
        """Test that the minimum precipitation rate is applied correctly,
        when the allowed tolerance is altered."""
        expected0 = np.array([[[[1/32., 1/32., 1/32.],
                                [1/32., 1/32., 1/32.],
                                [1/32., 1/32., 1/32.]]]])
        precip_cube = self.precip_cube.copy()
        precip_cube.convert_units("mm/hr")
        precip_cube.data = np.array([[[[-1., -1., 0.],
                                       [1., 2., 3.],
                                       [3., 3., 4.]]]])
        self.precip_cube.convert_units("mm/hr")
        plugin = ApplyOrographicEnhancement("add")
        result = plugin.apply_minimum_precip_rate(precip_cube, tolerance=10)
        self.assertIsInstance(result, iris.cube.Cube)
        self.assertEqual(result.units, Unit("mm/hr"))
        self.assertEqual(result.metadata, precip_cube.metadata)
        self.assertArrayAlmostEqual(result.data, expected0)

    def test_NaN_values(self):
        """Test a minimum precipitation rate is applied, when NaN values are
        present within the input."""
        expected0 = np.array([[[[1/32., 1/32., np.NaN],
                                [1., 2., np.NaN],
                                [3., 3., 4.]]]])
        precip_cube = self.precip_cube.copy()
        precip_cube.convert_units("mm/hr")
        precip_cube.data = np.array([[[[-1., -1., np.NaN],
                                       [1., 2., np.NaN],
                                       [3., 3., 4.]]]])
        precip_cube.convert_units("m/s")
        plugin = ApplyOrographicEnhancement("add")
        result = plugin.apply_minimum_precip_rate(precip_cube)
        self.assertIsInstance(result, iris.cube.Cube)
        self.assertEqual(result.units, Unit("m/s"))
        self.assertEqual(result.metadata, precip_cube.metadata)
        result.convert_units("mm/hr")
        self.assertArrayAlmostEqual(result.data, expected0)


class Test_process(IrisTest):

    """Test the apply_orographic_enhancement method."""

    def setUp(self):
        """Set up cubes for testing."""
        self.precip_cubes = set_up_precipitation_rate_cube()
        self.oe_cubes = set_up_orographic_enhancement_cube()

    def test_basic_add(self):
        """Test the addition of cubelists containing cubes of
        precipitation rate and orographic enhancement."""
        expected0 = np.array([[[[0., 1., 2.],
                                [1., 2., 3.],
                                [3., 3., 4.]]]])
        expected1 = np.array([[[[6., 5., 1.],
                                [6., 5., 1.],
                                [6., 5., 1.]]]])
        plugin = ApplyOrographicEnhancement("add")
        result = plugin.process(self.precip_cubes, self.oe_cubes)
        print("result = ", result)
        self.assertIsInstance(result, iris.cube.CubeList)
        for aresult, precip_cube in zip(result, self.precip_cubes):
            self.assertEqual(
                aresult.metadata, precip_cube.metadata)
        for cube in result:
            cube.convert_units("mm/hr")
        self.assertArrayAlmostEqual(result[0].data, expected0)
        self.assertArrayAlmostEqual(result[1].data, expected1)

    def test_basic_subtract(self):
        """Test the subtraction of cubelists containing cubes of orographic
        enhancement from cubes of precipitation rate."""
        expected0 = np.array([[[[0., 1., 2.],
                                [1., 2., 3.],
                                [1., 1., 0.]]]])
        expected1 = np.array([[[[2., 3., 1.],
                                [2., 3., 1.],
                                [2., 3., 1.]]]])
        plugin = ApplyOrographicEnhancement("subtract")
        result = plugin.process(self.precip_cubes, self.oe_cubes)
        self.assertIsInstance(result, iris.cube.CubeList)
        for aresult, precip_cube in zip(result, self.precip_cubes):
            self.assertEqual(
                aresult.metadata, precip_cube.metadata)
        for cube in result:
            cube.convert_units("mm/hr")
        self.assertArrayAlmostEqual(result[0].data, expected0)
        self.assertArrayAlmostEqual(result[1].data, expected1)

    def test_one_input_cube(self):
        """Test the addition of precipitation rate and orographic enhancement,
        where a single precipitation rate cube is provided."""
        expected0 = np.array([[[[0., 1., 2.],
                                [1., 2., 3.],
                                [3., 3., 4.]]]])
        plugin = ApplyOrographicEnhancement("add")
        result = plugin.process(self.precip_cubes[0], self.oe_cubes)
        self.assertIsInstance(result, iris.cube.CubeList)
        for aresult, precip_cube in zip(result, self.precip_cubes):
            self.assertEqual(
                aresult.metadata, precip_cube.metadata)
        for cube in result:
            cube.convert_units("mm/hr")
        self.assertArrayAlmostEqual(result[0].data, expected0)

    def test_inputs_and_orographic_enhancements_as_cubes(self):
        """Test the addition of precipitation rate and orographic enhancement,
        where a single precipitation rate cube and a single orographic
        enhancement cube is provided."""
        expected0 = np.array([[[[0., 1., 2.],
                                [1., 2., 3.],
                                [3., 3., 4.]]]])
        plugin = ApplyOrographicEnhancement("add")
        result = plugin.process(self.precip_cubes[0], self.oe_cubes[0])
        self.assertIsInstance(result, iris.cube.CubeList)
        for aresult, precip_cube in zip(result, self.precip_cubes):
            self.assertEqual(
                aresult.metadata, precip_cube.metadata)
        for cube in result:
            cube.convert_units("mm/hr")
        self.assertArrayAlmostEqual(result[0].data, expected0)


if __name__ == '__main__':
    unittest.main()
