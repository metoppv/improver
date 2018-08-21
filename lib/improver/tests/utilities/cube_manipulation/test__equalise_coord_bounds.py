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
Unit tests for the function "cube_manipulation._equalise_coord_bounds".
"""

import unittest

import iris
from iris.tests import IrisTest

from improver.utilities.cube_manipulation import _equalise_coord_bounds

from improver.tests.ensemble_calibration.ensemble_calibration.\
    helper_functions import set_up_temperature_cube

from improver.utilities.warnings_handler import ManageWarnings


class Test__equalise_coord_bounds(IrisTest):

    """Test the_equalise_cube_coords utility."""

    def setUp(self):
        """Use temperature cube to test with."""
        self.cube = set_up_temperature_cube()

    def test_basic(self):
        """Test that the input remains an iris.cube.CubeList."""
        result = iris.cube.CubeList([self.cube])
        _equalise_coord_bounds(result)
        self.assertIsInstance(result, iris.cube.CubeList)

    def test_basic_twocubes(self):
        """Test that the input remains an iris.cube.CubeList when two cubes
        are present."""
        result = iris.cube.CubeList([self.cube, self.cube])
        _equalise_coord_bounds(result)
        self.assertIsInstance(result, iris.cube.CubeList)

    def test_basic_threecubes(self):
        """Test that the input remains an iris.cube.CubeList when three cubes
        are present."""
        result = iris.cube.CubeList([self.cube, self.cube, self.cube])
        _equalise_coord_bounds(result)
        self.assertIsInstance(result, iris.cube.CubeList)

    def test_with_bounds(self):
        """Test that the inputs are unchanged when bounds match."""
        cubeA = self.cube.copy()
        cubeB = self.cube.copy()
        bounds = [0., 1.]
        cubeA.coord('time').bounds = bounds
        cubeB.coord('time').bounds = bounds
        result = iris.cube.CubeList([cubeA, cubeB])
        _equalise_coord_bounds(result)
        self.assertArrayAlmostEqual(result[0].coord('time').bounds, [bounds])
        self.assertArrayAlmostEqual(result[1].coord('time').bounds, [bounds])

    def test_with_bounds_threecubes(self):
        """Test that the inputs are unchanged when bounds match."""
        cubeA = self.cube.copy()
        cubeB = self.cube.copy()
        cubeC = self.cube.copy()
        bounds = [0., 1.]
        cubeA.coord('time').bounds = bounds
        cubeB.coord('time').bounds = bounds
        cubeC.coord('time').bounds = bounds
        result = iris.cube.CubeList([cubeA, cubeB, cubeC])
        _equalise_coord_bounds(result)
        for cube in result:
            self.assertArrayAlmostEqual(cube.coord('time').bounds, [bounds])

    @ManageWarnings(record=True)
    def test_remove_bounds(self, warning_list=None):
        """Test that the first input cube has bounds removed and warning is
        raised."""
        cube_with_bounds = self.cube.copy()
        cube_with_bounds.coord('time').bounds = [0., 1.]
        result = iris.cube.CubeList([cube_with_bounds, self.cube])
        _equalise_coord_bounds(result)
        warning_msg = ("Removing mismatched bounds from cube")
        self.assertTrue(any(warning_msg in str(item)
                            for item in warning_list))
        for cube in result:
            self.assertIsNone(cube.coord('time').bounds)

    @ManageWarnings(record=True)
    def test_remove_bounds_threecubes(self, warning_list=None):
        """Test that the second of three input cube has bounds removed and
        warning is raised."""
        cube_with_bounds = self.cube.copy()
        cube_with_bounds.coord('time').bounds = [0., 1.]
        result = iris.cube.CubeList([self.cube, cube_with_bounds, self.cube])
        _equalise_coord_bounds(result)
        warning_msg = ("Removing mismatched bounds from cube")
        self.assertTrue(any(warning_msg in str(item)
                            for item in warning_list))
        for cube in result:
            self.assertIsNone(cube.coord('time').bounds)

    def test_mismatched_bounds(self):
        """Test for error when input cubes has mismatched bounds."""
        cube_with_bounds = self.cube.copy()
        cube_with_bounds.coord('time').bounds = [0., 1.]
        cube_diff_bounds = self.cube.copy()
        cube_diff_bounds.coord('time').bounds = [3., 4.]
        result = iris.cube.CubeList([cube_with_bounds, cube_diff_bounds])
        msg = "Cubes with mismatching bounds are not compatible"
        with self.assertRaisesRegex(ValueError, msg):
            _equalise_coord_bounds(result)


if __name__ == '__main__':
    unittest.main()
