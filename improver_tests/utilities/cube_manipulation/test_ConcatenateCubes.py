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
Unit tests for the "cube_manipulation.ConcatenateCubes" plugin.
"""

import unittest
from datetime import datetime as dt

import iris
import numpy as np
from iris.coords import DimCoord
from iris.tests import IrisTest

from improver.utilities.cube_manipulation import ConcatenateCubes

from ...set_up_test_cubes import set_up_variable_cube


def check_coord_type(cube, coord):
    """
    Function to test whether coord is classified as scalar or auxiliary

    Args:
        cube (iris.cube.Cube):
            Iris cube containing coordinates to be checked
        coord (iris.coords.Coord):
            Coordinate to check
    """
    coord_scalar = True
    coord_aux = False
    cube_summary = cube.summary()
    aux_ind = cube_summary.find("Auxiliary")
    if coord in cube_summary[aux_ind:]:
        coord_scalar = False
        coord_aux = True
    return coord_scalar, coord_aux


class Test__init__(IrisTest):
    """Test the __init__ method"""

    def test_default(self):
        """Test default settings"""
        plugin = ConcatenateCubes("time")
        self.assertEqual(plugin.master_coord, "time")
        self.assertSequenceEqual(
            plugin.coords_to_associate, ["forecast_period"])
        self.assertFalse(plugin.coords_to_slice_over)

    def test_arguments(self):
        """Test custom arguments"""
        plugin = ConcatenateCubes(
            "time", coords_to_associate=["forecast_reference_time"],
            coords_to_slice_over=["realization"])
        self.assertSequenceEqual(
            plugin.coords_to_associate, ["forecast_reference_time"])
        self.assertSequenceEqual(
            plugin.coords_to_slice_over, ["realization"])

    def test_fails_unphysical_associations(self):
        """Test the plugin will not accept time, forecast period and forecast
        reference time (which describe a two-dimensional space) as all
        associated with the same dimension"""
        msg = "cannot all be associated with a single dimension"
        with self.assertRaisesRegex(ValueError, msg):
            ConcatenateCubes(
                "time", coords_to_associate=[
                    "forecast_period", "forecast_reference_time"])


class Test__associate_any_coordinate_with_master_coordinate(IrisTest):
    """Test the _associate_any_coordinate_with_master_coordinate method"""

    def setUp(self):
        """Set up default plugin and test cube"""
        self.plugin = ConcatenateCubes("time")
        data = 275.*np.ones((3, 3, 3), dtype=np.float32)
        cube = set_up_variable_cube(
            data, time=dt(2017, 1, 10, 3), frt=dt(2017, 1, 10, 0))
        # cubes can only be concatenated along an existing dimension;
        # therefore promote "time"
        self.cube = iris.util.new_axis(cube, scalar_coord="time")

    def test_forecast_period_association(self):
        """Test forecast period is correctly promoted to associate with time"""
        result = self.plugin._associate_any_coordinate_with_master_coordinate(
            self.cube)
        scalar, aux = check_coord_type(result, "forecast_period")
        self.assertFalse(scalar)
        self.assertTrue(aux)

    def test_forecast_period_not_added(self):
        """Test auxiliary coordinates aren't added if not originally present"""
        self.cube.remove_coord("forecast_period")
        result = self.plugin._associate_any_coordinate_with_master_coordinate(
            self.cube)
        self.assertNotIn("forecast_period", result.coords())

    def test_cube_with_latitude_and_height(self):
        """
        Test that the utility returns an iris.cube.Cube with a height
        coordinate, if this coordinate is added to the input cube. This checks
        that the height coordinate points are not modified.
        """
        plugin = ConcatenateCubes("latitude", coords_to_associate=["height"])

        cube = self.cube
        for latitude_slice in cube.slices_over("latitude"):
            cube = iris.util.new_axis(latitude_slice, "latitude")

        cube.add_aux_coord(
            DimCoord([10], "height", units="m"))

        result = plugin._associate_any_coordinate_with_master_coordinate(
            cube)
        self.assertArrayAlmostEqual(result.coord("height").points, [10])
        scalar, aux = check_coord_type(result, "height")
        self.assertFalse(scalar)
        self.assertTrue(aux)


class Test__slice_over_coordinate(IrisTest):
    """Test the _slice_over_coordinate method"""

    def setUp(self):
        """Set up default plugin and test cube with dimensions:
        time (1), realization (3), latitude (3), longitude (3)"""
        self.plugin = ConcatenateCubes("time")
        data = 275.*np.ones((3, 3, 3), dtype=np.float32)
        cube = set_up_variable_cube(
            data, time=dt(2017, 1, 10, 3), frt=dt(2017, 1, 10, 0))
        # cubes can only be concatenated along an existing dimension;
        # therefore promote "time"
        self.cube = iris.util.new_axis(cube, scalar_coord="time")

    def test_basic(self):
        """Test that the utility returns an iris.cube.CubeList when given an
        iris.cube.CubeList instance."""
        cubelist = iris.cube.CubeList([self.cube])
        result = self.plugin._slice_over_coordinate(cubelist, "time")
        self.assertIsInstance(result, iris.cube.CubeList)

    def test_basic_cube(self):
        """Test that the utility returns an iris.cube.CubeList when given an
        iris.cube.Cube instance."""
        result = self.plugin._slice_over_coordinate(self.cube, "time")
        self.assertIsInstance(result, iris.cube.CubeList)

    def test_reordering(self):
        """Test that the sliced coordinate is the first dimension in each
        output cube in the list, regardless of input dimension order"""
        result = self.plugin._slice_over_coordinate(self.cube, "realization")
        self.assertEqual(len(result), 3)
        for i in range(3):
            dim_coord_names = [
                coord.name() for coord in result[i].coords(dim_coords=True)]
            self.assertEqual(dim_coord_names[0], "realization")


class Test_process(IrisTest):
    """Test the process method (see also test_concatenate_cubes.py)"""

    def setUp(self):
        """Set up default plugin and test cubes."""
        self.plugin = ConcatenateCubes("time")
        data = 275.*np.ones((3, 3, 3), dtype=np.float32)
        cube = set_up_variable_cube(
            data, time=dt(2017, 1, 10, 3), frt=dt(2017, 1, 10, 0))
        # cubes can only be concatenated along an existing dimension;
        # therefore promote "time"
        self.cube = iris.util.new_axis(cube, scalar_coord="time")

        # create a cube for 3 hours later from the same forecast cycle
        self.later_cube = self.cube.copy()
        self.later_cube.coord("time").points = (
            self.later_cube.coord("time").points + 3*3600)
        self.later_cube.coord("forecast_period").points = (
            self.later_cube.coord("forecast_period").points + 3*3600)
        self.cubelist = iris.cube.CubeList([self.cube, self.later_cube])

    def test_basic(self):
        """Test that the utility returns an iris.cube.Cube."""
        result = self.plugin.process(self.cubelist)
        self.assertIsInstance(result, iris.cube.Cube)

    def test_single_cube(self):
        """Test a single cube is returned unchanged"""
        result = self.plugin.process(self.cube)
        self.assertArrayAlmostEqual(result.data, self.cube.data)
        self.assertEqual(result.metadata, self.cube.metadata)

    def test_single_item_cubelist(self):
        """Test a single item cubelist returns the cube unchanged"""
        result = self.plugin.process([self.cube])
        self.assertArrayAlmostEqual(result.data, self.cube.data)
        self.assertEqual(result.metadata, self.cube.metadata)

    def test_missing_master_coord(self):
        """Test error is raised if the master coordinate is missing"""
        self.cube.remove_coord("time")
        msg = "Master coordinate time is not present"
        with self.assertRaisesRegex(ValueError, msg):
            self.plugin.process([self.cube, self.cube])

    def test_identical_cubes(self):
        """
        Test that the utility returns the expected error message,
        if an attempt is made to concatenate identical cubes.
        """
        cubes = iris.cube.CubeList([self.cube, self.cube])
        msg = "An unexpected problem prevented concatenation."
        with self.assertRaisesRegex(iris.exceptions.ConcatenateError, msg):
            self.plugin.process(cubes)

    def test_cubelist_type_and_data(self):
        """
        Test that the utility returns an iris.cube.Cube with the expected
        resulting data, if a CubeList containing non-identical cubes
        (different values for the time coordinate) is passed in as the input.
        """
        data = self.cube.data.copy()
        expected_result = np.vstack([data, data])
        result = self.plugin.process(self.cubelist)
        self.assertIsInstance(result, iris.cube.Cube)
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

        result = ConcatenateCubes(
            "time", coords_to_slice_over=["realization"]).process(cubelist)
        self.assertIsInstance(result, iris.cube.Cube)
        self.assertArrayAlmostEqual(
            result.coord("realization").points, [0, 1, 2])

    def test_cubelist_different_number_of_realizations_time(self):
        """
        Test that the utility returns the expected error message, if a
        CubeList containing cubes with different numbers of realizations are
        passed in as the input, and the slicing done, in order to help the
        concatenation is only done over time.
        """
        plugin = ConcatenateCubes("time", coords_to_slice_over=["time"])

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
        with self.assertRaisesRegex(iris.exceptions.ConcatenateError, msg):
            plugin.process(cubelist)

    def test_cubelist_slice_over_time_only(self):
        """
        Test that the utility returns an iris.cube.Cube with the expected
        time and associated forecast period coordinates, if a CubeList
        containing cubes with different timesteps is passed in as the input.
        """
        plugin = ConcatenateCubes("time", coords_to_slice_over=["time"])

        expected_time_points = [
            self.cube.coord("time").points[0],
            self.later_cube.coord("time").points[0]]
        expected_fp_points = [
            self.cube.coord("forecast_period").points[0],
            self.later_cube.coord("forecast_period").points[0]]

        result = plugin.process(self.cubelist)
        self.assertIsInstance(result, iris.cube.Cube)
        self.assertArrayAlmostEqual(
            result.coord("time").points, expected_time_points)
        self.assertArrayAlmostEqual(
            result.coord("forecast_period").points, expected_fp_points)

    def test_cubelist_slice_over_realization_only(self):
        """
        Test that the utility returns an iris.cube.Cube with the expected
        realization coordinate, if a CubeList containing cubes with different
        realizations is passed in as the input.
        """
        plugin = ConcatenateCubes("time", coords_to_slice_over=["realization"])
        result = plugin.process(self.cubelist)
        self.assertIsInstance(result, iris.cube.Cube)
        self.assertArrayAlmostEqual(
            result.coord("realization").points, [0, 1, 2])

    def test_cubelist_different_var_names(self):
        """
        Test that the utility returns an iris.cube.Cube, if a CubeList
        containing non-identical cubes is passed in as the input.
        """
        cube1 = self.cube.copy()
        cube2 = self.later_cube.copy()
        cube1.coord("time").var_name = "time_0"
        cube2.coord("time").var_name = "time_1"

        cubelist = iris.cube.CubeList([cube1, cube2])

        result = self.plugin.process(cubelist)
        self.assertIsInstance(result, iris.cube.Cube)


if __name__ == '__main__':
    unittest.main()
