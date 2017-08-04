# -*- coding: iso-8859-1 -*-
# -----------------------------------------------------------------------------
# (C) British Crown Copyright 2017 Met Office.
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
Unit tests for the utilities within the "cube_manipulation" module.

"""
import unittest

from cf_units import Unit
import iris
from iris.coords import DimCoord
from iris.cube import Cube
from iris.exceptions import ConcatenateError
from iris.tests import IrisTest
import numpy as np

from improver.utilities.cube_manipulation import (
    concatenate_cubes, _strip_var_names, _slice_over_coordinate,
    _associate_any_coordinate_with_master_coordinate)
from improver.tests.ensemble_calibration.ensemble_calibration.\
    helper_functions import set_up_temperature_cube


def _check_coord_type(cube, coord):
    '''Function to test whether coord is classified
       as scalar or auxiliary coordinate.
       Parameters
       ----------
       cube: cube
           Iris cube containing coordinates to be checked
       coord: coordinate
           Cube coordinate to check
    '''
    coord_scalar = False
    coord_aux = False
    cube_summary = cube.summary()
    aux_ind = cube_summary.find("Auxiliary")
    if coord in cube_summary[aux_ind:]:
        coord_scalar = False
        coord_aux = True
    return coord_scalar, coord_aux


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
        scalar, aux = _check_coord_type(result, 'forecast_period')
        self.assertFalse(scalar)
        self.assertTrue(aux)
        scalar, aux = _check_coord_type(result, 'forecast_reference_time')
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
        with self.assertRaisesRegexp(ValueError, msg):
            _associate_any_coordinate_with_master_coordinate(
                cube1, master_coord="time",
                coordinates=["forecast_reference_time", "forecast_period"])


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
        cube2.coord("time").points = 402195.5
        time_origin = "hours since 1970-01-01 00:00:00"
        calendar = "gregorian"
        tunit = Unit(time_origin, calendar)
        cube1.add_aux_coord(
            DimCoord([402192.5], "forecast_reference_time", units=tunit),
            data_dims=1)
        cube1.add_aux_coord(
            DimCoord([0], "forecast_period", units="hours"), data_dims=1)
        cube2.add_aux_coord(
            DimCoord([402195.5], "forecast_reference_time", units=tunit),
            data_dims=1)
        cube2.add_aux_coord(
            DimCoord([3], "forecast_period", units="hours"), data_dims=1)

        cubelist = iris.cube.CubeList([cube1, cube2])

        cubelist = cubelist.concatenate_cube()

        result = _slice_over_coordinate(cubelist, "forecast_period")
        self.assertEqual(len(result), 2)

    def test_cubelist_history_removal(self):
        """
        Test that the utility returns an iris.cube.Cube without a
        history attribute, given that the utility will try to remove the
        history attribute, if it exists.
        """
        cube1 = self.cube.copy()
        cube2 = self.cube.copy()
        cube2.coord("time").points = 402195.0
        cube1.attributes["history"] = "2017-01-18T08:59:53: StaGE Decoupler"
        cube2.attributes["history"] = "2017-01-19T08:59:53: StaGE Decoupler"

        cubelist = iris.cube.CubeList([cube1, cube2])

        result = _slice_over_coordinate(cubelist, "time")
        self.assertNotIn("history", result[0].attributes.keys())
        self.assertNotIn("history", result[1].attributes.keys())

    def test_cubelist_no_history_removal(self):
        """
        Test that the utility returns an iris.cube.Cube with a
        history attribute and with the remove_history keyword argument
        set to True.
        """
        cube1 = self.cube.copy()
        cube2 = self.cube.copy()
        cube2.coord("time").points = 402195.0
        cube1.attributes["history"] = "2017-01-18T08:59:53: StaGE Decoupler"
        cube2.attributes["history"] = "2017-01-18T08:59:53: StaGE Decoupler"

        cubelist = iris.cube.CubeList([cube1, cube2])

        result = _slice_over_coordinate(
            cubelist, "time", remove_history=False)
        self.assertIn("history", result[0].attributes.keys())
        self.assertIn("history", result[1].attributes.keys())


class Test__strip_var_names(IrisTest):

    """Test the _slice_var_names utility."""

    def setUp(self):
        """Use temperature cube to test with."""
        self.cube = set_up_temperature_cube()

    def test_basic(self):
        """Test that the utility returns an iris.cube.CubeList."""
        self.cube.var_name = "air_temperature"
        result = _strip_var_names(self.cube)
        self.assertIsInstance(result, iris.cube.CubeList)

    def test_cube_var_name_is_none(self):
        """
        Test that the utility returns an iris.cube.Cube with a
        var_name of None.
        """
        self.cube.var_name = "air_temperature"
        result = _strip_var_names(self.cube)
        self.assertIsNone(result[0].var_name, None)

    def test_cube_coord_var_name_is_none(self):
        """
        Test that the coordinates have var_names of None.
        """
        self.cube.coord("time").var_name = "time"
        self.cube.coord("latitude").var_name = "latitude"
        self.cube.coord("longitude").var_name = "longitude"
        result = _strip_var_names(self.cube)
        for cube in result:
            for coord in cube.coords():
                self.assertIsNone(coord.var_name, None)

    def test_cubelist(self):
        """Test that the utility returns an iris.cube.CubeList."""
        cube1 = self.cube
        cube2 = self.cube
        cubes = iris.cube.CubeList([cube1, cube2])
        self.cube.var_name = "air_temperature"
        result = _strip_var_names(cubes)
        self.assertIsInstance(result, iris.cube.CubeList)
        for cube in result:
            for coord in cube.coords():
                self.assertIsNone(coord.var_name, None)


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
        with self.assertRaisesRegexp(ConcatenateError, msg):
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

        cube2.coord("time").points = 402195.0

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
        with self.assertRaisesRegexp(ConcatenateError, msg):
            concatenate_cubes(cubelist, coords_to_slice_over=["time"])

    def test_cubelist_slice_over_time_only(self):
        """
        Test that the utility returns an iris.cube.Cube with the expected
        time coordinate, if a CubeList containing cubes with different
        timesteps is passed in as the input.
        """
        cube1 = self.cube.copy()
        cube2 = self.cube.copy()

        cube2.coord("time").points = 402195.0

        cubelist = iris.cube.CubeList([cube1, cube2])

        result = concatenate_cubes(
            cubelist, coords_to_slice_over=["time"])
        self.assertIsInstance(result, Cube)
        self.assertArrayAlmostEqual(
            result.coord("time").points, [402192.5, 402195.0])

    def test_cubelist_slice_over_realization_only(self):
        """
        Test that the utility returns an iris.cube.Cube with the expected
        realization coordinate, if a CubeList containing cubes with different
        realizations is passed in as the input.
        """
        cube1 = self.cube.copy()
        cube2 = self.cube.copy()

        cube2.coord("time").points = 402195.0

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
        cube2.coord("time").points = 402195.5
        time_origin = "hours since 1970-01-01 00:00:00"
        calendar = "gregorian"
        tunit = Unit(time_origin, calendar)
        cube1.add_aux_coord(
            DimCoord([402192.5], "forecast_reference_time", units=tunit))
        cube2.add_aux_coord(
            DimCoord([402195.5], "forecast_reference_time", units=tunit))

        cubelist = iris.cube.CubeList([cube1, cube2])

        result = concatenate_cubes(cubelist)
        self.assertArrayAlmostEqual(
            result.coord("forecast_reference_time").points,
            [402192.5, 402195.5])

    def test_cubelist_different_var_names(self):
        """
        Test that the utility returns an iris.cube.Cube, if a CubeList
        containing non-identical cubes is passed in as the input.
        """
        cube1 = self.cube.copy()
        cube2 = self.cube.copy()
        cube2.coord("time").points = 402195.5

        cube1.coord("time").var_name = "time_0"
        cube2.coord("time").var_name = "time_1"

        cubelist = iris.cube.CubeList([cube1, cube2])

        result = concatenate_cubes(cubelist)
        self.assertIsInstance(result, Cube)


if __name__ == '__main__':
    unittest.main()
