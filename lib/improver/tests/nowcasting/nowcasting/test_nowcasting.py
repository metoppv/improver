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
"""Module with tests of the utilities required for nowcasting."""

import unittest

import iris
from iris.tests import IrisTest
import numpy as np

from improver.nowcasting.nowcasting import apply_orographic_enhancement
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
    cube1 = set_up_cube(data, None, "mm/hr",
                        long_name="orographic_enhancement",
                        realizations=np.array([0]), timesteps=1)
    cube1.coord("time").points = [412227.0]
    cube1.convert_units("m s-1")

    data = np.array([[[[2., 1., 0.],
                       [2., 1., 0.],
                       [2., 1., 0.]]]])
    cube2 = set_up_cube(data, None, "mm/hr",
                        long_name="orographic_enhancement",
                        realizations=np.array([0]), timesteps=1)
    cube2.coord("time").points = [412228.0]
    cube2.convert_units("m s-1")

    return iris.cube.CubeList([cube1, cube2])


class Test_apply_orographic_enhancement(IrisTest):

    """Test the apply_orographic_enhancement function."""

    def setUp(self):
        """Set up cubes for testing."""
        self.precip_cubes = set_up_precipitation_rate_cube()
        self.oe_cubes = set_up_orographic_enhancement_cube()

    def test_basic_add(self):
        """Test the addition of cubelists containing cubes of
        precipitation rate and the orographic enhancement."""
        expected0 = np.array([[[[0., 1., 2.],
                                [1., 2., 3.],
                                [3., 3., 4.]]]])
        expected1 = np.array([[[[6., 5., 1.],
                                [6., 5., 1.],
                                [6., 5., 1.]]]])
        result = apply_orographic_enhancement(
            self.precip_cubes, self.oe_cubes, "add")
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
        result = apply_orographic_enhancement(
            self.precip_cubes, self.oe_cubes, "subtract")
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
        result = apply_orographic_enhancement(
            self.precip_cubes[0], self.oe_cubes, "add")
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
        result = apply_orographic_enhancement(
            self.precip_cubes[0], self.oe_cubes[0], "add")
        self.assertIsInstance(result, iris.cube.CubeList)
        for aresult, precip_cube in zip(result, self.precip_cubes):
            self.assertEqual(
                aresult.metadata, precip_cube.metadata)
        for cube in result:
            cube.convert_units("mm/hr")
        self.assertArrayAlmostEqual(result[0].data, expected0)

    def test_orographic_enhancement_not_available(self):
        """Test where an orographic enhancement is not available for the
        time point required."""
        msg = "There is no orographic enhancement available for"
        with self.assertRaisesRegex(ValueError, msg):
            apply_orographic_enhancement(
                self.precip_cubes, self.oe_cubes[0], "add")


if __name__ == '__main__':
    unittest.main()
