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

from improver.utilities.mathematical_operations import Integration
from improver.tests.ensemble_calibration.ensemble_calibration.\
    helper_functions import set_up_temperature_cube


def set_up_height_cube(height_points, cube=set_up_temperature_cube()):
    """Create a height cube using the existing set_up_temperature_cube
    utility."""
    ascending = False
    if np.all(np.diff(height_points) > 0):
        ascending = True

    cubelist = iris.cube.CubeList([])
    for height_point in height_points:
        temp_cube = cube.copy()
        height_coord = iris.coords.DimCoord(height_point, "height")
        if ascending:
            height_coord.attributes = {"positive": "up"}
        else:
            height_coord.attributes = {"positive": "down"}
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
            Integration(coord_name, direction_of_integration=direction)


class Test__repr__(IrisTest):

    """Test the repr method."""

    def test_basic(self):
        """Test that the __repr__ returns the expected string."""
        coord_name = "height"
        result = str(Integration(coord_name))
        msg = ('<Integration: coord_name_to_integrate: height, '
               'start_point: None, end_point: None, '
               'direction_of_integration: negative>')
        self.assertEqual(result, msg)


class Test_ensure_monotonic_increase_in_chosen_direction(IrisTest):

    """Test the ensure_monotonic_increase_in_chosen_direction method."""

    def setUp(self):
        """Set up the cube."""
        self.ascending_height_points = np.array([5., 10., 20.])
        self.ascending_cube = (
            set_up_height_cube(self.ascending_height_points))
        self.descending_height_points = np.array([20., 10., 5.])
        self.descending_cube = (
            set_up_height_cube(self.descending_height_points))
        self.descending_cube.coord("height").points = (
            self.descending_height_points)

    def test_ascending_coordinate_positive(self):
        """Test that for a monotonically ascending coordinate, where the
        chosen direction is positive, the resulting coordinate still
        increases monotonically in the positive direction."""
        coord_name = "height"
        direction = "positive"
        cube = (
            Integration(
                coord_name, direction_of_integration=direction
                ).ensure_monotonic_increase_in_chosen_direction(
                    self.ascending_cube))
        self.assertIsInstance(cube, iris.cube.Cube)
        self.assertArrayAlmostEqual(
            cube.coord(coord_name).points, self.ascending_height_points)

    def test_ascending_coordinate_negative(self):
        """Test that for a monotonically ascending coordinate, where the
        chosen direction is negative, the resulting coordinate now decreases
        monotonically in the positive direction."""
        coord_name = "height"
        direction = "negative"
        cube = (
            Integration(
                coord_name, direction_of_integration=direction
                ).ensure_monotonic_increase_in_chosen_direction(
                    self.ascending_cube))
        self.assertIsInstance(cube, iris.cube.Cube)
        self.assertArrayAlmostEqual(
            cube.coord(coord_name).points, self.descending_height_points)

    def test_descending_coordinate_positive(self):
        """Test that for a monotonically ascending coordinate, where the
        chosen direction is positive, the resulting coordinate still
        increases monotonically in the positive direction."""
        coord_name = "height"
        direction = "positive"
        cube = (
            Integration(
                coord_name, direction_of_integration=direction
                ).ensure_monotonic_increase_in_chosen_direction(
                    self.descending_cube))
        self.assertIsInstance(cube, iris.cube.Cube)
        self.assertArrayAlmostEqual(
            cube.coord(coord_name).points, self.ascending_height_points)

    def test_descending_coordinate_negative(self):
        """Test that for a monotonically ascending coordinate, where the
        chosen direction is negative, the resulting coordinate still decreases
        monotonically in the positive direction."""
        coord_name = "height"
        direction = "negative"
        cube = (
            Integration(
                coord_name, direction_of_integration=direction
                ).ensure_monotonic_increase_in_chosen_direction(
                    self.descending_cube))
        self.assertIsInstance(cube, iris.cube.Cube)
        self.assertArrayAlmostEqual(
            cube.coord(coord_name).points, self.descending_height_points)


class Test_process(IrisTest):

    """Test the process method."""

    def setUp(self):
        """Set up the cube."""
        self.height_points = np.array([5., 10., 20.])
        cube = set_up_height_cube(self.height_points)[:, 0, :, :, :]
        data = np.ones(cube.shape)
        data[1] = np.zeros(cube[1].shape)+2
        data[2] = np.zeros(cube[2].shape)+3
        data[0, :, 0, 0] = 6
        cube.data = data
        self.cube = cube

    def test_basic(self):
        """Test that a cube with the points on the chosen coordinate are
        in the expected order."""
        coord_name = "height"
        direction = "negative"
        result = (
            Integration(
                coord_name, direction_of_integration=direction
                ).process(self.cube))
        self.assertIsInstance(result, iris.cube.Cube)
        self.assertArrayAlmostEqual(
            result.coord("height").points, np.array([10., 5.]))

    def test_data(self):
        """Test that the resulting cube contains the expected data following
        vertical integration."""
        expected = np.array(
            [[[[25.00, 25.00, 25.00],
               [25.00, 25.00, 25.00],
               [25.00, 25.00, 25.00]]],
             [[[45.00, 32.50, 32.50],
               [32.50, 32.50, 32.50],
               [32.50, 32.50, 32.50]]]])
        coord_name = "height"
        direction = "negative"
        result = (
            Integration(
                coord_name, direction_of_integration=direction
                ).process(self.cube))
        self.assertIsInstance(result, iris.cube.Cube)
        self.assertArrayAlmostEqual(
            result.coord("height").points, np.array([10., 5.]))
        self.assertArrayAlmostEqual(result.data, expected)

    def test_start_point_positive(self):
        """Test that the resulting cube contains the expected data when a
        start_point is specified, so that only part of the column is
        integrated. For integration in the positive direction (equivalent to
        integrating downwards for a height coordinate), the presense of a
        start_point indicates that the integration may start above the
        lowest height within the column to be integrated."""
        expected = np.array(
            [[[[25.00, 25.00, 25.00],
               [25.00, 25.00, 25.00],
               [25.00, 25.00, 25.00]]]])
        coord_name = "height"
        start_point = 8.
        direction = "positive"
        result = (
            Integration(
                coord_name, start_point=start_point,
                direction_of_integration=direction
                ).process(self.cube))
        self.assertIsInstance(result, iris.cube.Cube)
        self.assertArrayAlmostEqual(
            result.coord("height").points, np.array([20.]))
        self.assertArrayAlmostEqual(result.data, expected)

    def test_start_point_negative(self):
        """Test that the resulting cube contains the expected data when a
        start_point is specified, so that only part of the column is
        integrated. For integration in the negative direction (equivalent to
        integrating downwards for a height coordinate), the presense of a
        start_point indicates that the integration may start below the
        highest height within the column to be integrated."""
        expected = np.array(
            [[[[20.00, 7.50, 7.50],
               [7.50, 7.50, 7.50],
               [7.50, 7.50, 7.50]]]])
        coord_name = "height"
        start_point = 18.
        direction = "negative"
        result = (
            Integration(
                coord_name, start_point=start_point,
                direction_of_integration=direction
                ).process(self.cube))
        self.assertIsInstance(result, iris.cube.Cube)
        self.assertArrayAlmostEqual(
            result.coord("height").points, np.array([5.]))
        self.assertArrayAlmostEqual(result.data, expected)

    def test_end_point_positive(self):
        """Test that the resulting cube contains the expected data when a
        end_point is specified, so that only part of the column is
        integrated. For integration in the negative direction (equivalent to
        integrating downwards for a height coordinate), the presense of an
        end_point indicates that the integration may end below the
        highest height within the column to be integrated."""
        expected = np.array(
            [[[[20.00, 7.50, 7.50],
               [7.50, 7.50, 7.50],
               [7.50, 7.50, 7.50]]]])
        coord_name = "height"
        end_point = 18.
        direction = "positive"
        result = (
            Integration(
                coord_name, end_point=end_point,
                direction_of_integration=direction
                ).process(self.cube))
        self.assertIsInstance(result, iris.cube.Cube)
        self.assertArrayAlmostEqual(
            result.coord("height").points, np.array([10.]))
        self.assertArrayAlmostEqual(result.data, expected)

    def test_end_point_negative(self):
        """Test that the resulting cube contains the expected data when a
        end_point is specified, so that only part of the column is
        integrated. For integration in the negative direction (equivalent to
        integrating downwards for a height coordinate), the presense of an
        end_point indicates that the integration may end above the
        lowest height within the column to be integrated."""
        expected = np.array(
            [[[[25.00, 25.00, 25.00],
               [25.00, 25.00, 25.00],
               [25.00, 25.00, 25.00]]]])
        coord_name = "height"
        end_point = 8.
        direction = "negative"
        result = (
            Integration(
                coord_name, end_point=end_point,
                direction_of_integration=direction
                ).process(self.cube))
        self.assertIsInstance(result, iris.cube.Cube)
        self.assertArrayAlmostEqual(
            result.coord("height").points, np.array([10.]))
        self.assertArrayAlmostEqual(result.data, expected)


if __name__ == '__main__':
    unittest.main()
