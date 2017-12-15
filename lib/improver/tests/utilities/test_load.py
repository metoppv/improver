# -*- coding: utf-8 -*-
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
"""Unit tests for loading functionality."""

from subprocess import call as Call
from tempfile import mkdtemp
import unittest

import iris
from iris.tests import IrisTest
import numpy as np

from improver.utilities.load import load_cube, load_cubelist

from improver.tests.ensemble_calibration.ensemble_calibration.\
    helper_functions import set_up_temperature_cube

iris.FUTURE.netcdf_no_unlimited = True


class Test_load_cube(IrisTest):

    """Test the load function."""
    def setUp(self):
        """Set up variables for use in testing."""
        self.directory = mkdtemp()
        self.filepath = self.directory+"temp.nc"
        self.cube = set_up_temperature_cube()
        iris.save(self.cube, self.filepath)
        self.realization_points = np.array([0, 1, 2])
        self.time_points = np.array([402192.5])
        self.latitude_points = np.array([-45., 0., 45.])
        self.longitude_points = np.array([120., 150., 180.])

    def tearDown(self):
        """Remove temporary directories created for testing."""
        Call(['rm', '-f', self.filepath])
        Call(['rmdir', self.directory])

    def test_filepath_only(self):
        """Test that the loading works correctly, if only the filepath is
        provided."""
        result = load_cube(self.filepath)
        self.assertIsInstance(result, iris.cube.Cube)
        self.assertArrayAlmostEqual(
            result.coord("realization").points, self.realization_points)
        self.assertArrayAlmostEqual(
            result.coord("time").points, self.time_points)
        self.assertArrayAlmostEqual(
            result.coord("latitude").points, self.latitude_points)
        self.assertArrayAlmostEqual(
            result.coord("longitude").points, self.longitude_points)

    def test_filepath_as_list(self):
        """Test that the loading works correctly, if only the filepath is
        provided as a list."""
        result = load_cube([self.filepath])
        self.assertIsInstance(result, iris.cube.Cube)
        self.assertArrayAlmostEqual(
            result.coord("realization").points, self.realization_points)
        self.assertArrayAlmostEqual(
            result.coord("time").points, self.time_points)
        self.assertArrayAlmostEqual(
            result.coord("latitude").points, self.latitude_points)
        self.assertArrayAlmostEqual(
            result.coord("longitude").points, self.longitude_points)

    def test_constraint(self):
        """Test that the loading works correctly, if a constraint is
        specified."""
        realization_points = np.array([0])
        constr = iris.Constraint(realization=0)
        result = load_cube(self.filepath, constraints=constr)
        self.assertIsInstance(result, iris.cube.Cube)
        self.assertArrayAlmostEqual(
            result.coord("realization").points, realization_points)
        self.assertArrayAlmostEqual(
            result.coord("time").points, self.time_points)
        self.assertArrayAlmostEqual(
            result.coord("latitude").points, self.latitude_points)
        self.assertArrayAlmostEqual(
            result.coord("longitude").points, self.longitude_points)

    def test_multiple_constraints(self):
        """Test that the loading works correctly, if multiple constraints are
        specified."""
        realization_points = np.array([0])
        latitude_points = np.array([0])
        constr1 = iris.Constraint(realization=0)
        constr2 = iris.Constraint(latitude=lambda cell: -0.1 < cell < 0.1)
        constr = constr1 & constr2
        result = load_cube(self.filepath, constraints=constr)
        self.assertIsInstance(result, iris.cube.Cube)
        self.assertArrayAlmostEqual(
            result.coord("realization").points, realization_points)
        self.assertArrayAlmostEqual(
            result.coord("time").points, self.time_points)
        self.assertArrayAlmostEqual(
            result.coord("latitude").points, latitude_points)
        self.assertArrayAlmostEqual(
            result.coord("longitude").points, self.longitude_points)

    def test_ordering(self):
        """Test that cube has been reordered, if it is originally in an
        undesirable order."""
        filepath = self.directory+"temp.nc"
        cube = set_up_temperature_cube()
        cube.transpose([3, 2, 1, 0])
        iris.save(cube, filepath)
        result = load_cube(filepath)
        self.assertIsInstance(result, iris.cube.Cube)
        self.assertArrayAlmostEqual(
            result.coord("realization").points, self.realization_points)
        self.assertArrayAlmostEqual(
            result.coord("time").points, self.time_points)
        self.assertArrayAlmostEqual(
            result.coord("latitude").points, self.latitude_points)
        self.assertArrayAlmostEqual(
            result.coord("longitude").points, self.longitude_points)
        self.assertEqual(result.coord_dims("realization")[0], 0)
        self.assertEqual(result.coord_dims("time")[0], 1)
        self.assertArrayAlmostEqual(result.coord_dims("latitude")[0], 2)
        self.assertArrayAlmostEqual(result.coord_dims("longitude")[0], 3)


class Test_load_cubelist(IrisTest):

    """Test the load function."""
    def setUp(self):
        """Set up variables for use in testing."""
        self.directory = mkdtemp()
        self.filepath = self.directory+"temp.nc"
        self.cube = set_up_temperature_cube()
        iris.save(self.cube, self.filepath)
        self.realization_points = np.array([0, 1, 2])
        self.time_points = np.array(402192.5)
        self.latitude_points = np.array([-45., 0., 45.])
        self.longitude_points = np.array([120., 150., 180.])

    def tearDown(self):
        """Remove temporary directories created for testing."""
        Call(['rm', '-f', self.filepath])
        Call(['rmdir', self.directory])

    def test_single_file(self):
        """Test that the loading works correctly, if only the filepath is
        provided."""
        result = load_cubelist(self.filepath)
        self.assertIsInstance(result, iris.cube.CubeList)
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
        filepath = self.directory+"*.nc"
        result = load_cubelist(filepath)
        self.assertIsInstance(result, iris.cube.CubeList)
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
        self.assertIsInstance(result, iris.cube.CubeList)
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
        low_cloud_filepath = self.directory+"low_cloud.nc"
        iris.save(low_cloud_cube, low_cloud_filepath)
        medium_cloud_cube = self.cube.copy()
        medium_cloud_cube.rename("medium_type_cloud_area_fraction")
        medium_cloud_filepath = self.directory+"medium_cloud.nc"
        iris.save(medium_cloud_cube, medium_cloud_filepath)
        constr = iris.Constraint("low_type_cloud_area_fraction")
        result = load_cubelist(
            [low_cloud_filepath, medium_cloud_filepath], constraints=constr)
        self.assertIsInstance(result, iris.cube.CubeList)
        self.assertEqual(len(result), 1)
        self.assertArrayAlmostEqual(
            result[0].coord("realization").points, self.realization_points)
        self.assertArrayAlmostEqual(
            result[0].coord("time").points, self.time_points)
        self.assertArrayAlmostEqual(
            result[0].coord("latitude").points, self.latitude_points)
        self.assertArrayAlmostEqual(
            result[0].coord("longitude").points, self.longitude_points)


if __name__ == '__main__':
    unittest.main()
