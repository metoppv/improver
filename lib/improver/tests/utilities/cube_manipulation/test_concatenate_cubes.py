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
"""
Unit tests for the function "cube_manipulation.concatenate_cubes".
"""

import unittest

from cf_units import Unit
import iris
from iris.coords import DimCoord
from iris.cube import Cube
from iris.exceptions import ConcatenateError
from iris.tests import IrisTest
import numpy as np

from improver.utilities.cube_manipulation import concatenate_cubes

from improver.tests.ensemble_calibration.ensemble_calibration.\
    helper_functions import set_up_temperature_cube


class Test_concatenate_cubes(IrisTest):

    """Test the concatenate_cubes utility."""

    def setUp(self):
        """Use temperature cube to test with."""
        self.cube = set_up_temperature_cube()

    def test_basic(self):
        """Test that the utility returns an iris.cube.Cube."""
        result = concatenate_cubes(self.cube)
        self.assertIsInstance(result, Cube)

    def test_identical_cubes(self):
        """
        Test that the utility returns the expected error message,
        if an attempt is made to concatenate identical cubes.
        """
        cubes = iris.cube.CubeList([self.cube, self.cube])
        msg = "An unexpected problem prevented concatenation."
        with self.assertRaisesRegex(ConcatenateError, msg):
            concatenate_cubes(cubes)

    def test_cubelist_type_and_data(self):
        """
        Test that the utility returns an iris.cube.Cube with the expected
        resulting data, if a CubeList containing non-identical cubes
        (different values for the time coordinate) is passed in as the input.
        """
        cube1 = self.cube.copy()
        cube2 = self.cube.copy()

        cube3 = self.cube.copy()
        cube3.transpose([1, 0, 2, 3])
        expected_result = np.vstack([cube3.data, cube3.data])

        cube2.coord("time").points = np.array([412230.0], dtype=np.float64)

        cubelist = iris.cube.CubeList([cube1, cube2])

        result = concatenate_cubes(cubelist)
        self.assertIsInstance(result, Cube)
        self.assertArrayAlmostEqual(expected_result, result.data)

    def test_cubelist_different_number_of_realizations(self):
        """
        Test that the utility returns an iris.cube.Cube with the expected
        realizations, if a CubeList containing cubes with different numbers
        of realizations are passed in as the input.
        """
        cube1 = self.cube.copy()

        cube3 = iris.cube.CubeList([])
        for cube in cube1.slices_over("realization"):
            if cube.coord("realization").points == 0:
                cube2 = cube
            elif cube.coord("realization").points in [1, 2]:
                cube3.append(cube)
        cube3 = cube3.merge_cube()

        cubelist = iris.cube.CubeList([cube2, cube3])

        result = concatenate_cubes(cubelist)
        self.assertIsInstance(result, Cube)
        self.assertArrayAlmostEqual(
            result.coord("realization").points, [0, 1, 2])

    def test_cubelist_different_number_of_realizations_time(self):
        """
        Test that the utility returns the expected error message, if a
        CubeList containing cubes with different numbers of realizations are
        passed in as the input, and the slicing done, in order to help the
        concatenation is only done over time.
        """
        cube1 = self.cube.copy()

        cube3 = iris.cube.CubeList([])
        for cube in cube1.slices_over("realization"):
            if cube.coord("realization").points == 0:
                cube2 = cube
            elif cube.coord("realization").points in [1, 2]:
                cube3.append(cube)
        cube3 = cube3.merge_cube()

        cubelist = iris.cube.CubeList([cube2, cube3])
        msg = "failed to concatenate into a single cube"
        with self.assertRaisesRegex(ConcatenateError, msg):
            concatenate_cubes(cubelist, coords_to_slice_over=["time"])

    def test_cubelist_slice_over_time_only(self):
        """
        Test that the utility returns an iris.cube.Cube with the expected
        time coordinate, if a CubeList containing cubes with different
        timesteps is passed in as the input.
        """
        cube1 = self.cube.copy()
        cube2 = self.cube.copy()

        cube2.coord("time").points = np.array([412230.0], dtype=np.float64)

        cubelist = iris.cube.CubeList([cube1, cube2])

        result = concatenate_cubes(
            cubelist, coords_to_slice_over=["time"])
        self.assertIsInstance(result, Cube)
        self.assertArrayAlmostEqual(
            result.coord("time").points, [412227.0, 412230.0])

    def test_cubelist_slice_over_realization_only(self):
        """
        Test that the utility returns an iris.cube.Cube with the expected
        realization coordinate, if a CubeList containing cubes with different
        realizations is passed in as the input.
        """
        cube1 = self.cube.copy()
        cube2 = self.cube.copy()

        cube2.coord("time").points = np.float64(412230.0)

        cubelist = iris.cube.CubeList([cube1, cube2])

        result = concatenate_cubes(
            cubelist, coords_to_slice_over=["realization"])
        self.assertIsInstance(result, Cube)
        self.assertArrayAlmostEqual(
            result.coord("realization").points, [0, 1, 2])

    def test_cubelist_with_forecast_reference_time_only(self):
        """
        Test that the utility returns an iris.cube.Cube with the expected
        resulting data, if a CubeList containing cubes with different
        forecast_reference_time coordinates is passed in as the input.
        This makes sure that the forecast_reference_time from the input cubes
        is maintained within the output cube, after concatenation.
        """
        cube1 = self.cube.copy()
        cube2 = self.cube.copy()

        cube2.coord("time").points = np.array([412230.0], dtype=np.float64)
        time_origin = "hours since 1970-01-01 00:00:00"
        calendar = "gregorian"
        tunit = Unit(time_origin, calendar)
        cube1.add_aux_coord(
            DimCoord([412227.0], "forecast_reference_time", units=tunit))
        cube2.add_aux_coord(
            DimCoord([412230.0], "forecast_reference_time", units=tunit))
        cubelist = iris.cube.CubeList([cube1, cube2])

        result = concatenate_cubes(
            cubelist, coordinates_for_association=["forecast_reference_time"])
        self.assertArrayAlmostEqual(
            result.coord("forecast_reference_time").points,
            [412227.0, 412230.0])

    def test_cubelist_different_var_names(self):
        """
        Test that the utility returns an iris.cube.Cube, if a CubeList
        containing non-identical cubes is passed in as the input.
        """
        cube1 = self.cube.copy()
        cube2 = self.cube.copy()
        cube2.coord("time").points = np.float64(412230.0)

        cube1.coord("time").var_name = "time_0"
        cube2.coord("time").var_name = "time_1"

        cubelist = iris.cube.CubeList([cube1, cube2])

        result = concatenate_cubes(cubelist)
        self.assertIsInstance(result, Cube)


if __name__ == '__main__':
    unittest.main()
