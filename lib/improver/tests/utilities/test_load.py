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
"""Unit tests for loading functionality."""

import os
import unittest
from subprocess import call as Call
from tempfile import mkdtemp

import iris
from iris.coords import DimCoord
from iris.tests import IrisTest
from iris.exceptions import ConstraintMismatchError
import numpy as np

from improver.utilities.load import load_cube, load_cubelist
from improver.utilities.save import save_netcdf

from improver.tests.set_up_test_cubes import (
    set_up_variable_cube, set_up_percentile_cube, set_up_probability_cube,
    add_coordinate)


class Test_load_cube(IrisTest):

    """Test the load function."""
    def setUp(self):
        """Set up variables for use in testing."""
        self.directory = mkdtemp()
        self.filepath = os.path.join(self.directory, "temp.nc")
        self.cube = set_up_variable_cube(np.ones((3, 3, 3), dtype=np.float32))
        save_netcdf(self.cube, self.filepath)
        self.realization_points = self.cube.coord("realization").points
        self.time_points = self.cube.coord("time").points
        self.latitude_points = self.cube.coord("latitude").points
        self.longitude_points = self.cube.coord("longitude").points

    def tearDown(self):
        """Remove temporary directories created for testing."""
        Call(['rm', '-f', self.filepath])
        Call(['rmdir', self.directory])

    def test_a_cube_is_loaded(self):
        """Test that a cube is loaded when a valid filepath is provided."""
        result = load_cube(self.filepath)
        self.assertIsInstance(result, iris.cube.Cube)

    def test_filepath_only(self):
        """Test that the realization, time, latitude and longitude coordinates
        have the expected values, when a cube is loaded from the specified
        filepath."""
        result = load_cube(self.filepath)
        self.assertArrayAlmostEqual(
            result.coord("realization").points, self.realization_points)
        self.assertArrayAlmostEqual(
            result.coord("time").points, self.time_points)
        self.assertArrayAlmostEqual(
            result.coord("latitude").points, self.latitude_points)
        self.assertArrayAlmostEqual(
            result.coord("longitude").points, self.longitude_points)

    def test_filepath_as_list(self):
        """Test that the realization, time, latitude and longitude coordinates
        have the expected values, if a cube is loaded from an input filepath
        specified in a list."""
        result = load_cube([self.filepath])
        self.assertArrayAlmostEqual(
            result.coord("realization").points, self.realization_points)
        self.assertArrayAlmostEqual(
            result.coord("time").points, self.time_points)
        self.assertArrayAlmostEqual(
            result.coord("latitude").points, self.latitude_points)
        self.assertArrayAlmostEqual(
            result.coord("longitude").points, self.longitude_points)

    def test_constraint(self):
        """Test that the realization, time, latitude and longitude coordinates
        have the expected values, if constraint are specified
        when loading a cube from a file."""
        realization_points = np.array([0])
        constr = iris.Constraint(realization=0)
        result = load_cube(self.filepath, constraints=constr)
        self.assertArrayAlmostEqual(
            result.coord("realization").points, realization_points)
        self.assertArrayAlmostEqual(
            result.coord("time").points, self.time_points)
        self.assertArrayAlmostEqual(
            result.coord("latitude").points, self.latitude_points)
        self.assertArrayAlmostEqual(
            result.coord("longitude").points, self.longitude_points)

    def test_multiple_constraints(self):
        """Test that the realization, time, latitude and longitude coordinates
        have the expected values, if multiple constraints are specified
        when loading a cube from a file."""
        realization_points = np.array([0])
        latitude_points = np.array([0])
        constr1 = iris.Constraint(realization=0)
        constr2 = iris.Constraint(latitude=lambda cell: -0.1 < cell < 0.1)
        constr = constr1 & constr2
        result = load_cube(self.filepath, constraints=constr)
        self.assertArrayAlmostEqual(
            result.coord("realization").points, realization_points)
        self.assertArrayAlmostEqual(
            result.coord("time").points, self.time_points)
        self.assertArrayAlmostEqual(
            result.coord("latitude").points, latitude_points)
        self.assertArrayAlmostEqual(
            result.coord("longitude").points, self.longitude_points)

    def test_ordering_for_realization_coordinate(self):
        """Test that the cube has been reordered, if it is originally in an
        undesirable order and the cube contains a "realization" coordinate."""
        cube = set_up_variable_cube(np.ones((3, 3, 3), dtype=np.float32))
        cube.transpose([2, 1, 0])
        save_netcdf(cube, self.filepath)
        result = load_cube(self.filepath)
        self.assertEqual(result.coord_dims("realization")[0], 0)
        self.assertArrayAlmostEqual(result.coord_dims("latitude")[0], 1)
        self.assertArrayAlmostEqual(result.coord_dims("longitude")[0], 2)

    def test_ordering_for_percentile_over_coordinate(self):
        """Test that the cube has been reordered, if it is originally in an
        undesirable order and the cube contains a "percentile_over"
        coordinate."""
        data = np.ones((3, 4, 5), dtype=np.float32)
        cube = set_up_percentile_cube(data, np.array([10, 50, 90]))
        cube.transpose([2, 1, 0])
        save_netcdf(cube, self.filepath)
        result = load_cube(self.filepath)
        self.assertEqual(
            result.coord_dims("percentile_over_realization")[0], 0)
        self.assertArrayAlmostEqual(result.coord_dims("latitude")[0], 1)
        self.assertArrayAlmostEqual(result.coord_dims("longitude")[0], 2)

    def test_ordering_for_threshold_coordinate(self):
        """Test that the cube has been reordered, if it is originally in an
        undesirable order and the cube contains a "threshold" coordinate."""
        cube = set_up_probability_cube(
            np.zeros((3, 4, 5), dtype=np.float32),
            np.array([273., 274., 275.], dtype=np.float32))
        cube.transpose([2, 1, 0])
        save_netcdf(cube, self.filepath)
        result = load_cube(self.filepath)
        self.assertEqual(result.coord_dims("threshold")[0], 0)
        self.assertArrayAlmostEqual(result.coord_dims("latitude")[0], 1)
        self.assertArrayAlmostEqual(result.coord_dims("longitude")[0], 2)

    def test_ordering_for_realization_threshold_percentile_over_coordinate(
            self):
        """Test that the cube has been reordered, if it is originally in an
        undesirable order and the cube contains a "threshold" coordinate,
        a "realization" coordinate and a "percentile_over" coordinate."""
        cube = set_up_probability_cube(
            np.zeros((3, 4, 5), dtype=np.float32),
            np.array([273., 274., 275.], dtype=np.float32))
        cube = add_coordinate(cube, [0, 1, 2], "realization")
        cube = add_coordinate(
            cube, [10, 50, 90], "percentile_over_neighbourhood")
        cube.transpose([4, 3, 2, 1, 0])
        save_netcdf(cube, self.filepath)
        result = load_cube(self.filepath)
        self.assertEqual(result.coord_dims("realization")[0], 0)
        self.assertEqual(
            result.coord_dims("percentile_over_neighbourhood")[0], 1)
        self.assertEqual(result.coord_dims("threshold")[0], 2)
        self.assertArrayAlmostEqual(result.coord_dims("latitude")[0], 3)
        self.assertArrayAlmostEqual(result.coord_dims("longitude")[0], 4)

    def test_attributes(self):
        """Test that metadata attributes are successfully stripped out."""
        result = load_cube(self.filepath)
        self.assertNotIn('bald__isPrefixedBy', result.attributes.keys())

    def test_prefix_cube_removed(self):
        """Test metadata prefix cube is discarded during load"""
        msg = "No cubes found"
        with self.assertRaisesRegexp(ValueError, msg):
            load_cube(self.filepath, 'prefixes')

    def test_no_lazy_load(self):
        """Test that the cube returned upon loading does not contain
        lazy data."""
        result = load_cube(self.filepath, no_lazy_load=True)
        self.assertFalse(result.has_lazy_data())

    def test_lazy_load(self):
        """Test that the loading works correctly with lazy loading."""
        result = load_cube(self.filepath)
        self.assertTrue(result.has_lazy_data())


class Test_load_cubelist(IrisTest):

    """Test the load function."""
    def setUp(self):
        """Set up variables for use in testing."""
        self.directory = mkdtemp()
        self.filepath = os.path.join(self.directory, "temp.nc")
        self.cube = set_up_variable_cube(np.ones((3, 3, 3), dtype=np.float32))
        save_netcdf(self.cube, self.filepath)
        self.realization_points = self.cube.coord("realization").points
        self.time_points = self.cube.coord("time").points
        self.latitude_points = self.cube.coord("latitude").points
        self.longitude_points = self.cube.coord("longitude").points
        self.low_cloud_filepath = os.path.join(self.directory, "low_cloud.nc")
        self.med_cloud_filepath = os.path.join(self.directory,
                                               "medium_cloud.nc")

    def tearDown(self):
        """Remove temporary directories created for testing."""
        Call(['rm', '-f', self.filepath])
        Call(['rm', '-f', self.low_cloud_filepath])
        Call(['rm', '-f', self.med_cloud_filepath])
        Call(['rmdir', self.directory])

    def test_a_cubelist_is_loaded(self):
        """Test that a cubelist is loaded when a valid filepath is provided."""
        result = load_cubelist(self.filepath)
        self.assertIsInstance(result, iris.cube.CubeList)

    def test_single_file(self):
        """Test that the loading works correctly, if only the filepath is
        provided."""
        result = load_cubelist(self.filepath)
        self.assertArrayAlmostEqual(
            result[0].coord("realization").points, self.realization_points)
        self.assertArrayAlmostEqual(
            result[0].coord("time").points, self.time_points)
        self.assertArrayAlmostEqual(
            result[0].coord("latitude").points, self.latitude_points)
        self.assertArrayAlmostEqual(
            result[0].coord("longitude").points, self.longitude_points)

    def test_wildcard_files(self):
        """Test that the loading works correctly, if a wildcarded filepath is
        provided."""
        filepath = os.path.join(self.directory, "*.nc")
        result = load_cubelist(filepath)
        self.assertArrayAlmostEqual(
            result[0].coord("realization").points, self.realization_points)
        self.assertArrayAlmostEqual(
            result[0].coord("time").points, self.time_points)
        self.assertArrayAlmostEqual(
            result[0].coord("latitude").points, self.latitude_points)
        self.assertArrayAlmostEqual(
            result[0].coord("longitude").points, self.longitude_points)

    def test_multiple_files(self):
        """Test that the loading works correctly, if a path to multiple files
        is provided."""
        result = load_cubelist([self.filepath, self.filepath])
        for cube in result:
            self.assertArrayAlmostEqual(
                cube.coord("realization").points, self.realization_points)
            self.assertArrayAlmostEqual(
                cube.coord("time").points, self.time_points)
            self.assertArrayAlmostEqual(
                cube.coord("latitude").points, self.latitude_points)
            self.assertArrayAlmostEqual(
                cube.coord("longitude").points, self.longitude_points)

    def test_wildcard_files_with_constraint(self):
        """Test that the loading works correctly, if a wildcarded filepath is
        provided and a constraint is provided that is only valid for a subset
        of the available files."""
        low_cloud_cube = self.cube.copy()
        low_cloud_cube.rename("low_type_cloud_area_fraction")
        save_netcdf(low_cloud_cube, self.low_cloud_filepath)
        medium_cloud_cube = self.cube.copy()
        medium_cloud_cube.rename("medium_type_cloud_area_fraction")
        save_netcdf(medium_cloud_cube, self.med_cloud_filepath)
        constr = iris.Constraint("low_type_cloud_area_fraction")
        result = load_cubelist([self.low_cloud_filepath,
                                self.med_cloud_filepath], constraints=constr)
        self.assertEqual(len(result), 1)
        self.assertArrayAlmostEqual(
            result[0].coord("realization").points, self.realization_points)
        self.assertArrayAlmostEqual(
            result[0].coord("time").points, self.time_points)
        self.assertArrayAlmostEqual(
            result[0].coord("latitude").points, self.latitude_points)
        self.assertArrayAlmostEqual(
            result[0].coord("longitude").points, self.longitude_points)

    def test_no_lazy_load(self):
        """Test that the cubelist returned upon loading does not contain
        lazy data."""
        result = load_cubelist([self.filepath, self.filepath],
                               no_lazy_load=True)
        self.assertArrayEqual([False, False],
                              [_.has_lazy_data() for _ in result])
        for cube in result:
            self.assertArrayAlmostEqual(
                cube.coord("realization").points, self.realization_points)
            self.assertArrayAlmostEqual(
                cube.coord("time").points, self.time_points)
            self.assertArrayAlmostEqual(
                cube.coord("latitude").points, self.latitude_points)
            self.assertArrayAlmostEqual(
                cube.coord("longitude").points, self.longitude_points)

    def test_lazy_load(self):
        """Test that the cubelist returned upon loading does contain
        lazy data."""
        result = load_cubelist([self.filepath, self.filepath])
        self.assertArrayEqual([True, True],
                              [_.has_lazy_data() for _ in result])


if __name__ == '__main__':
    unittest.main()
