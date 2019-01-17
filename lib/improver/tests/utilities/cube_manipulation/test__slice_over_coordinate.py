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
Unit tests for the function "cube_manipulation._slice_over_coordinate".
"""

import unittest

from cf_units import Unit
import iris
from iris.coords import DimCoord
from iris.tests import IrisTest
import numpy as np

from improver.utilities.cube_manipulation import _slice_over_coordinate

from improver.tests.ensemble_calibration.ensemble_calibration.\
    helper_functions import set_up_temperature_cube


class Test__slice_over_coordinate(IrisTest):

    """Test the _slice_over_coordinate utility."""

    def setUp(self):
        """Use temperature cube to test with."""
        self.cube = set_up_temperature_cube()

    def test_basic(self):
        """Test that the utility returns an iris.cube.CubeList."""
        cubelist = iris.cube.CubeList([self.cube])
        result = _slice_over_coordinate(cubelist, "time")
        self.assertIsInstance(result, iris.cube.CubeList)

    def test_basic_cube(self):
        """Test that the utility returns an iris.cube.CubeList."""
        result = _slice_over_coordinate(self.cube, "time")
        self.assertIsInstance(result, iris.cube.CubeList)

    def test_time_first_dimension(self):
        """
        Test that the first dimension of the output cube within the
        output cubelist has time as the first dimension.
        """
        cubelist = iris.cube.CubeList([self.cube])
        result = _slice_over_coordinate(cubelist, "time")
        dim_coord_names = []
        for cube in result:
            for dim_coord in cube.dim_coords:
                dim_coord_names.append(dim_coord.name())
        self.assertEqual(dim_coord_names[0], "time")

    def test_number_of_slices(self):
        """
        Test that the number of cubes returned, after slicing over the
        given coordinate is as expected.
        """
        cube1 = self.cube.copy()
        cube2 = self.cube.copy()
        cube2.coord("time").points = 402195.5
        time_origin = "hours since 1970-01-01 00:00:00"
        calendar = "gregorian"
        tunit = Unit(time_origin, calendar)
        cube1.add_aux_coord(
            DimCoord([402192.5], "forecast_reference_time", units=tunit))
        cube1.add_aux_coord(
            DimCoord([0], "forecast_period", units="hours"))
        cube2.add_aux_coord(
            DimCoord([402195.5], "forecast_reference_time", units=tunit))
        cube2.add_aux_coord(
            DimCoord([3], "forecast_period", units="hours"))

        cubelist = iris.cube.CubeList([cube1, cube2])

        result = _slice_over_coordinate(cubelist, "forecast_period")
        self.assertEqual(len(result), 2)

    def test_number_of_slices_from_one_cube(self):
        """
        Test that the number of cubes returned, after slicing over the
        given coordinate is as expected.
        """
        cube1 = self.cube.copy()
        cube2 = self.cube.copy()
        cube2.coord("time").points = np.float64(402195.5)
        time_origin = "hours since 1970-01-01 00:00:00"
        calendar = "gregorian"
        tunit = Unit(time_origin, calendar)
        cube1.add_aux_coord(
            DimCoord(np.array([402192.5], dtype=np.float64),
                     "forecast_reference_time", units=tunit),
            data_dims=1)
        cube1.add_aux_coord(
            DimCoord([0], "forecast_period", units="hours"), data_dims=1)
        cube2.add_aux_coord(
            DimCoord(np.array([402195.5], dtype=np.float64),
                     "forecast_reference_time", units=tunit),
            data_dims=1)
        cube2.add_aux_coord(
            DimCoord([3], "forecast_period", units="hours"), data_dims=1)

        cubelist = iris.cube.CubeList([cube1, cube2])

        cubelist = cubelist.concatenate_cube()

        result = _slice_over_coordinate(cubelist, "forecast_period")
        self.assertEqual(len(result), 2)


if __name__ == '__main__':
    unittest.main()
