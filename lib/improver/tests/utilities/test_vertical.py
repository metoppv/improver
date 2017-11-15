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
"""Unit tests for the convert_distance_into_number_of_grid_cells function from
 spatial.py."""

import unittest

import numpy as np

import iris
from iris.tests import IrisTest

from improver.utilities.vertical import VerticalIntegration
from improver.tests.ensemble_calibration.ensemble_calibration.\
    helper_functions import (
        set_up_temperature_cube)


def set_up_height_cube(height_points):
    """Create a height cube using the existing set_up_temperature_cube
    utility."""
    cube = set_up_temperature_cube()
    cubelist = iris.cube.CubeList([])
    for height_point in height_points:
        temp_cube = cube.copy()
        height_coord = iris.coords.DimCoord(height_point, "height")
        temp_cube.add_aux_coord(height_coord)
        temp_cube = iris.util.new_axis(temp_cube, "height")
        cubelist.append(temp_cube)
    return cubelist.concatenate_cube()


class Test__init__(IrisTest):

    """Test the init method."""

    def test_raise_exception(self):
        """Test that an error is raised, if the direction_of_integration
        is not valid."""
        coord_name = "height"
        direction = "sideways"
        msg = "The specified direction of integration"
        with self.assertRaisesRegexp(ValueError, msg):
            VerticalIntegration(coord_name, direction_of_integration=direction)


class Test__repr__(IrisTest):

    """Test the repr method."""

    def test_basic(self):
        """Test that the __repr__ returns the expected string."""
        coord_name = "height"
        result = str(VerticalIntegration(coord_name))
        msg = ('<VerticalIntegration: coord_name_to_integrate: height, '
               'start_point: None, end_point: None, '
               'direction_of_integration: upwards>')
        self.assertEqual(result, msg)


class Test_ensure_monotonic_in_chosen_direction(IrisTest):

    """Test the ensure_monotonic_in_chosen_direction method."""

    def setUp(self):
        """Set up the cube."""
        self.increasing_height_points = np.array([5., 10., 20.])
        self.increasing_cube = (
            set_up_height_cube(self.increasing_height_points))
        self.decreasing_height_points = np.array([20., 10., 5.])
        self.decreasing_cube = (
            set_up_height_cube(self.decreasing_height_points))
        self.decreasing_cube.coord("height").points = (
            self.decreasing_height_points)

    def test_increasing_coordinate_upwards(self):
        """Test that for a monotonically increasing coordinate, where the
        chosen direction is upward, the resulting coordinate still increases.
        """
        coord_name = "height"
        direction = "upwards"
        cube = (
            VerticalIntegration(
                coord_name, direction_of_integration=direction
                ).ensure_monotonic_in_chosen_direction(
                    self.increasing_cube))
        self.assertIsInstance(cube, iris.cube.Cube)
        self.assertArrayAlmostEqual(
            cube.coord(coord_name).points, self.increasing_height_points)

    def test_increasing_coordinate_downwards(self):
        """Test that for a monotonically increasing coordinate, where the
        chosen direction is downward, the resulting coordinate now decreases.
        """
        coord_name = "height"
        direction = "downwards"
        cube = (
            VerticalIntegration(
                coord_name, direction_of_integration=direction
                ).ensure_monotonic_in_chosen_direction(
                    self.increasing_cube))
        self.assertIsInstance(cube, iris.cube.Cube)
        self.assertArrayAlmostEqual(
            cube.coord(coord_name).points, self.decreasing_height_points)

    def test_decreasing_coordinate_upwards(self):
        """Test that for a monotonically increasing coordinate, where the
        chosen direction is upward, the resulting coordinate still
        increases."""
        coord_name = "height"
        direction = "upwards"
        cube = (
            VerticalIntegration(
                coord_name, direction_of_integration=direction
                ).ensure_monotonic_in_chosen_direction(
                    self.decreasing_cube))
        self.assertIsInstance(cube, iris.cube.Cube)
        self.assertArrayAlmostEqual(
            cube.coord(coord_name).points, self.increasing_height_points)

    def test_decreasing_coordinate_downwards(self):
        """Test that for a monotonically increasing coordinate, where the
        chosen direction is upward, the resulting coordinate now decreases."""
        coord_name = "height"
        direction = "downwards"
        cube = (
            VerticalIntegration(
                coord_name, direction_of_integration=direction
                ).ensure_monotonic_in_chosen_direction(
                    self.decreasing_cube))
        self.assertIsInstance(cube, iris.cube.Cube)
        self.assertArrayAlmostEqual(
            cube.coord(coord_name).points, self.decreasing_height_points)


class Test_process(IrisTest):

    """Test the process method."""

    def setUp(self):
        self.height_points = np.array([5., 10., 20.])
        self.cube = set_up_height_cube(self.height_points)

    def test_basic(self):
        """Test the """
        coord_name = "height"
        direction = "downwards"
        result = (
            VerticalIntegration(
                coord_name, direction_of_integration=direction
                ).process(self.cube))
        self.assertIsInstance(result, iris.cube.Cube)
        self.assertArrayAlmostEqual(
            result.coord("height").points, np.array([10., 5.]))

    def test_data(self):
        """"""
        expected = np.array([])
        coord_name = "height"
        direction = "downwards"
        result = (
            VerticalIntegration(
                coord_name, direction_of_integration=direction
                ).process(self.cube))
        self.assertIsInstance(result, iris.cube.Cube)
        self.assertArrayAlmostEqual(result.data, expected)

    def test_start_point(self):
        """"""
        expected = np.array([])
        coord_name = "height"
        start_point = 2000
        direction = "downwards"
        cube = (
            VerticalIntegration(
                coord_name, direction_of_integration=direction
                ).process(self.cube))
        self.assertIsInstance(cube, iris.cube.Cube)
        self.assertArrayAlmostEqual(result.data, expected)

    def test_end_point(self):
        """"""
        expected = np.array([])
        cube = (
            VerticalIntegration(
                coord_name, direction_of_integration=direction
                ).process(self.cube))
        self.assertIsInstance(cube, iris.cube.Cube)
        self.assertArrayAlmostEqual(result.data, expected)


if __name__ == '__main__':
    unittest.main()
