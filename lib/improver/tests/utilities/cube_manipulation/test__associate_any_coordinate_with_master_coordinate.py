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
Unit tests for the utilities within the
"cube_manipulation._associate_any_coordinate_with_master_coordinate" module.
"""

import unittest

from cf_units import Unit
import iris
from iris.coords import DimCoord
from iris.tests import IrisTest

from improver.utilities.cube_manipulation import (
    _associate_any_coordinate_with_master_coordinate)

from improver.tests.ensemble_calibration.ensemble_calibration.\
    helper_functions import set_up_temperature_cube

from improver.tests.utilities.cube_manipulation.\
    helper_functions import check_coord_type


class Test__associate_any_coordinate_with_master_coordinate(IrisTest):

    """Test the _associate_any_coordinate_with_master_coordinate utility."""

    def setUp(self):
        """Use temperature cube to test with."""
        self.cube = set_up_temperature_cube()

    def test_cube_with_forecast_period_and_forecast_reference_time(self):
        """
        Test that the utility returns an iris.cube.Cube with the
        expected values for the forecast_reference_time and forecast_period
        coordinates. This checks that the auxiliary coordinates that were
        added to the cube are still present.

        """
        cube = self.cube

        time_origin = "hours since 1970-01-01 00:00:00"
        calendar = "gregorian"
        tunit = Unit(time_origin, calendar)
        cube.add_aux_coord(
            DimCoord([402192.5], "forecast_reference_time", units=tunit))
        cube.add_aux_coord(
            DimCoord([0], "forecast_period", units="hours"))

        result = _associate_any_coordinate_with_master_coordinate(
            cube, coordinates=["forecast_reference_time", "forecast_period"])
        self.assertArrayAlmostEqual(
            result.coord("forecast_reference_time").points, [402192.5])
        self.assertArrayAlmostEqual(
            result.coord("forecast_period").points, [0])

    def test_cube_check_coord_type(self):
        """
        Test that the utility returns an iris.cube.Cube with the
        expected values for the forecast_reference_time and forecast_period
        coordinates. This checks that the auxiliary coordinates that were
        added to the cube are still present.

        """
        cube = self.cube

        time_origin = "hours since 1970-01-01 00:00:00"
        calendar = "gregorian"
        tunit = Unit(time_origin, calendar)
        cube.add_aux_coord(
            DimCoord([402192.5], "forecast_reference_time", units=tunit))
        cube.add_aux_coord(
            DimCoord([0], "forecast_period", units="hours"))
        result = _associate_any_coordinate_with_master_coordinate(
            cube, coordinates=["forecast_reference_time", "forecast_period"])
        scalar, aux = check_coord_type(result, 'forecast_period')
        self.assertFalse(scalar)
        self.assertTrue(aux)
        scalar, aux = check_coord_type(result, 'forecast_reference_time')
        self.assertFalse(scalar)
        self.assertTrue(aux)

    def test_cube_with_latitude_and_height(self):
        """
        Test that the utility returns an iris.cube.Cube with a height
        coordinate, if this coordinate is added to the input cube. This checks
        that the height coordinate points, which were added to the cube
        are the same as the after applying the utility.
        """
        cube = self.cube
        for latitude_slice in cube.slices_over("latitude"):
            cube = iris.util.new_axis(latitude_slice, "latitude")

        cube.add_aux_coord(
            DimCoord([10], "height", units="m"))

        result = _associate_any_coordinate_with_master_coordinate(
            cube, master_coord="latitude", coordinates=["height"])
        self.assertArrayAlmostEqual(
            result.coord("height").points, [10])

    def test_coordinate_not_on_cube(self):
        """
        Test that the utility returns an iris.cube.Cube without
        forecast_reference_time and forecast_period coordinates, if these
        have not been added to the cube.
        """
        cube = self.cube

        result = _associate_any_coordinate_with_master_coordinate(
            cube, coordinates=["forecast_reference_time", "forecast_period"])
        self.assertFalse(result.coords("forecast_reference_time"))
        self.assertFalse(result.coords("forecast_period"))

    def test_no_time_dimension(self):
        """
        Test that the plugin returns the expected error message,
        if the input cubes do not contain a time coordinate.
        """
        cube1 = self.cube.copy()
        cube1.remove_coord("time")

        time_origin = "hours since 1970-01-01 00:00:00"
        calendar = "gregorian"
        tunit = Unit(time_origin, calendar)
        cube1.add_aux_coord(
            DimCoord([402192.5], "forecast_reference_time", units=tunit))
        cube1.add_aux_coord(
            DimCoord([0], "forecast_period", units="hours"))

        msg = "The master coordinate for associating other coordinates"
        with self.assertRaisesRegex(ValueError, msg):
            _associate_any_coordinate_with_master_coordinate(
                cube1, master_coord="time",
                coordinates=["forecast_reference_time", "forecast_period"])

    def test_scalar_time_coordinate(self):
        """Test that the output cube retains scalar coordinates for the time,
        forecast_period and forecast_reference_time coordinates, if these
        coordinates are scalar within the input cube."""
        cube = self.cube
        time_origin = "hours since 1970-01-01 00:00:00"
        calendar = "gregorian"
        tunit = Unit(time_origin, calendar)
        cube.add_aux_coord(
            DimCoord([402192.5], "forecast_reference_time", units=tunit))
        cube.add_aux_coord(
            DimCoord([0], "forecast_period", units="hours"))
        cube = cube[:, 0, ...]
        result = _associate_any_coordinate_with_master_coordinate(
            cube, coordinates=["forecast_reference_time", "forecast_period"])
        self.assertTrue(result.coords("time", dimensions=[]))
        self.assertTrue(result.coords("forecast_period", dimensions=[]))
        self.assertTrue(
            result.coords("forecast_reference_time", dimensions=[]))


if __name__ == '__main__':
    unittest.main()
