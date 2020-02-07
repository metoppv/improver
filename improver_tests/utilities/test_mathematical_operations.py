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
"""Unit tests for the plugins and functions within mathematical_operations.py
"""

import unittest

import iris
import numpy as np
from iris.tests import IrisTest

from improver.metadata.utilities import generate_mandatory_attributes
from improver.utilities.cube_manipulation import sort_coord_in_cube
from improver.utilities.mathematical_operations import Integration

from ..set_up_test_cubes import set_up_variable_cube, add_coordinate


def _set_up_height_cube(height_points, ascending=True):
    """Create cube of temperatures decreasing with height"""
    data = 280*np.ones((3, 3, 3), dtype=np.float32)
    data[1, :] = 278
    data[2, :] = 276

    cube = set_up_variable_cube(data[0].astype(np.float32))
    height_points = np.sort(height_points)
    cube = add_coordinate(cube, height_points, "height", coord_units="m")
    cube.coord("height").attributes["positive"] = "up"
    cube.data = data.astype(np.float32)

    if not ascending:
        cube = sort_coord_in_cube(cube, "height", descending=True)
        cube.coord("height").attributes["positive"] = "down"

    return cube


class Test__repr__(IrisTest):

    """Test the repr method."""

    def test_basic(self):
        """Test that the __repr__ returns the expected string."""
        coord_name = "height"
        result = str(Integration(coord_name))
        msg = ('<Integration: coord_name_to_integrate: height, '
               'start_point: None, end_point: None, '
               'positive_integration: False>')
        self.assertEqual(result, msg)


class Test_ensure_monotonic_increase_in_chosen_direction(IrisTest):

    """Test the ensure_monotonic_increase_in_chosen_direction method."""

    def setUp(self):
        """Set up the cube."""
        self.ascending_height_points = np.array([5., 10., 20.])
        self.ascending_cube = _set_up_height_cube(
            self.ascending_height_points)
        self.descending_height_points = np.array([20., 10., 5.])
        self.descending_cube = _set_up_height_cube(
            self.descending_height_points, ascending=False)
        self.plugin_positive = Integration("height", positive_integration=True)
        self.plugin_negative = Integration("height")

    def test_ascending_coordinate_positive(self):
        """Test that for a monotonically ascending coordinate, where the
        chosen direction is positive, the resulting coordinate still
        increases monotonically in the positive direction."""
        cube = (
            self.plugin_positive.ensure_monotonic_increase_in_chosen_direction(
                self.ascending_cube))
        self.assertIsInstance(cube, iris.cube.Cube)
        self.assertArrayAlmostEqual(
            cube.coord("height").points, self.ascending_height_points)

    def test_ascending_coordinate_negative(self):
        """Test that for a monotonically ascending coordinate, where the
        chosen direction is negative, the resulting coordinate now decreases
        monotonically in the positive direction."""
        cube = (
            self.plugin_negative.ensure_monotonic_increase_in_chosen_direction(
                self.ascending_cube))
        self.assertIsInstance(cube, iris.cube.Cube)
        self.assertArrayAlmostEqual(
            cube.coord("height").points, self.descending_height_points)

    def test_descending_coordinate_positive(self):
        """Test that for a monotonically ascending coordinate, where the
        chosen direction is positive, the resulting coordinate still
        increases monotonically in the positive direction."""
        cube = (
            self.plugin_positive.ensure_monotonic_increase_in_chosen_direction(
                    self.descending_cube))
        self.assertIsInstance(cube, iris.cube.Cube)
        self.assertArrayAlmostEqual(
            cube.coord("height").points, self.ascending_height_points)

    def test_descending_coordinate_negative(self):
        """Test that for a monotonically ascending coordinate, where the
        chosen direction is negative, the resulting coordinate still decreases
        monotonically in the positive direction."""
        cube = (
            self.plugin_negative.ensure_monotonic_increase_in_chosen_direction(
                    self.descending_cube))
        self.assertIsInstance(cube, iris.cube.Cube)
        self.assertArrayAlmostEqual(
            cube.coord("height").points, self.descending_height_points)


class Test_prepare_for_integration(IrisTest):

    """Test the prepare_for_integration method."""

    def setUp(self):
        """Set up the cube."""
        height_points = np.array([5., 10., 20.])
        cube = _set_up_height_cube(height_points)
        self.plugin_positive = Integration("height", positive_integration=True)
        self.plugin_positive.input_cube = cube.copy()
        self.plugin_negative = Integration("height")
        self.plugin_negative.input_cube = cube.copy()

    def test_basic(self):
        """Test that the type of the returned value is as expected and the
        expected number of items are returned."""
        result = self.plugin_negative.prepare_for_integration()
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        self.assertIsInstance(result[0], iris.cube.Cube)
        self.assertIsInstance(result[1], iris.cube.Cube)

    def test_positive_points(self):
        """Test that the expected coordinate points are returned for each
        cube when the direction of integration is positive."""
        result = self.plugin_positive.prepare_for_integration()
        self.assertArrayAlmostEqual(
            result[0].coord("height").points, np.array([10., 20.]))
        self.assertArrayAlmostEqual(
            result[1].coord("height").points, np.array([5., 10.]))

    def test_negative_points(self):
        """Test that the expected coordinate points are returned for each
        cube when the direction of integration is negative."""
        self.plugin_negative.input_cube.coord("height").points = (
            np.array([20., 10., 5.]))
        result = self.plugin_negative.prepare_for_integration()
        self.assertArrayAlmostEqual(
            result[0].coord("height").points, np.array([20., 10.]))
        self.assertArrayAlmostEqual(
            result[1].coord("height").points, np.array([10., 5.]))


class Test_perform_integration(IrisTest):

    """Test the perform_integration method."""

    def setUp(self):
        """Set up the cubes. One set of cubes for integrating in the positive
        direction and another set of cubes for integrating in the negative
        direction."""
        cube = _set_up_height_cube(np.array([5., 10., 20.]))

        data = np.zeros(cube.shape)
        data[0] = np.ones(cube[0].shape, dtype=np.int32)
        data[1] = np.full(cube[1].shape, 2, dtype=np.int32)
        data[2] = np.full(cube[2].shape, 3, dtype=np.int32)
        data[0, 0, 0] = 6
        cube.data = data

        # Cubes for integrating in the positive direction.
        self.positive_upper_bounds_cube = cube[1:, ...]
        self.positive_lower_bounds_cube = cube[:-1, ...]
        self.positive_integrated_cube = cube[1:, ...]
        self.positive_integrated_cube.data = (
            np.zeros(self.positive_integrated_cube.shape))

        # Cubes for integrating in the negative direction.
        new_cube = cube.copy()
        # Sort cube so that it is in the expected order.
        index = [[2, 1, 0], slice(None), slice(None), slice(None)]
        new_cube = new_cube[tuple(index)]
        self.negative_upper_bounds_cube = new_cube[:-1, ...]
        self.negative_lower_bounds_cube = new_cube[1:, ...]

        self.expected_data_zero_or_negative = np.array(
            [[[10.00, 25.00, 25.00],
              [25.00, 25.00, 25.00],
              [25.00, 25.00, 25.00]],
             [[30.00, 32.50, 32.50],
              [32.50, 32.50, 32.50],
              [32.50, 32.50, 32.50]]])

        self.plugin_positive = Integration("height", positive_integration=True)
        self.plugin_positive.input_cube = cube.copy()
        self.plugin_negative = Integration("height")
        self.plugin_negative.input_cube = cube.copy()

    def test_basic(self):
        """Test that a cube is returned by the perform_integration method with
        the expected coordinate points."""
        result = self.plugin_negative.perform_integration(
            self.negative_upper_bounds_cube, self.negative_lower_bounds_cube)
        self.assertIsInstance(result, iris.cube.Cube)
        self.assertArrayAlmostEqual(
            result.coord("height").points, np.array([10., 5.]))
        self.assertArrayAlmostEqual(
            result.coord("height").bounds, np.array([[10., 20.], [5., 10.]]))

    def test_positive_values_in_data(self):
        """Test that the resulting cube contains the expected data following
        vertical integration."""
        expected = np.array(
            [[[25.00, 25.00, 25.00],
              [25.00, 25.00, 25.00],
              [25.00, 25.00, 25.00]],
             [[45.00, 32.50, 32.50],
              [32.50, 32.50, 32.50],
              [32.50, 32.50, 32.50]]])
        result = self.plugin_negative.perform_integration(
            self.negative_upper_bounds_cube, self.negative_lower_bounds_cube)
        self.assertArrayAlmostEqual(
            result.coord("height").points, np.array([10., 5.]))
        self.assertArrayAlmostEqual(result.data, expected)

    def test_zero_values_in_data(self):
        """Test that the resulting cube contains the expected data following
        vertical integration where some of the values in the data are equal
        to zero. This provides a baseline as the Integration plugin is
        currently restricted so that only positive values contribute towards
        the integral."""
        self.negative_upper_bounds_cube.data[0, 0, 0] = 0
        result = self.plugin_negative.perform_integration(
            self.negative_upper_bounds_cube, self.negative_lower_bounds_cube)
        self.assertArrayAlmostEqual(
            result.coord("height").points, np.array([10., 5.]))
        self.assertArrayAlmostEqual(
            result.data, self.expected_data_zero_or_negative)

    def test_negative_and_positive_values_in_data(self):
        """Test that the resulting cube contains the expected data following
        vertical integration where some of the values in the data are negative.
        This shows that both zero values and negative values have no impact
        on the integration as the Integration plugin is currently
        restricted so that only positive values contribute towards the
        integral."""
        self.negative_upper_bounds_cube.data[0, 0, 0] = -1
        result = self.plugin_negative.perform_integration(
            self.negative_upper_bounds_cube, self.negative_lower_bounds_cube)
        self.assertArrayAlmostEqual(
            result.coord("height").points, np.array([10., 5.]))
        self.assertArrayAlmostEqual(
            result.data, self.expected_data_zero_or_negative)

    def test_start_point_positive_direction(self):
        """Test that the resulting cube contains the expected data when a
        start_point is specified, so that only part of the column is
        integrated. For integration in the positive direction (equivalent to
        integrating downwards for the height coordinate in the input cube),
        the presence of a start_point indicates that the integration may start
        above the lowest height within the column to be integrated."""
        expected = np.array(
            [[[25.00, 25.00, 25.00],
              [25.00, 25.00, 25.00],
              [25.00, 25.00, 25.00]]])
        self.plugin_positive.start_point = 8.
        result = self.plugin_positive.perform_integration(
            self.positive_upper_bounds_cube, self.positive_lower_bounds_cube)
        self.assertArrayAlmostEqual(
            result.coord("height").points, np.array([20.]))
        self.assertArrayAlmostEqual(result.data, expected)

    def test_start_point_negative_direction(self):
        """Test that the resulting cube contains the expected data when a
        start_point is specified, so that only part of the column is
        integrated. For integration in the negative direction (equivalent to
        integrating downwards for the height coordinate in the input cube),
        the presence of a start_point indicates that the integration may start
        below the highest height within the column to be integrated."""
        expected = np.array(
            [[[20.00, 7.50, 7.50],
              [7.50, 7.50, 7.50],
              [7.50, 7.50, 7.50]]])
        self.plugin_negative.start_point = 18.
        result = self.plugin_negative.perform_integration(
            self.negative_upper_bounds_cube, self.negative_lower_bounds_cube)
        self.assertArrayAlmostEqual(
            result.coord("height").points, np.array([5.]))
        self.assertArrayAlmostEqual(result.data, expected)

    def test_end_point_positive_direction(self):
        """Test that the resulting cube contains the expected data when a
        end_point is specified, so that only part of the column is
        integrated. For integration in the positive direction (equivalent to
        integrating downwards for the height coordinate in the input cube),
        the presence of an end_point indicates that the integration may end
        below the highest height within the column to be integrated."""
        expected = np.array(
            [[[20.00, 7.50, 7.50],
              [7.50, 7.50, 7.50],
              [7.50, 7.50, 7.50]]])
        self.plugin_positive.end_point = 18.
        result = self.plugin_positive.perform_integration(
            self.positive_upper_bounds_cube, self.positive_lower_bounds_cube)
        self.assertArrayAlmostEqual(
            result.coord("height").points, np.array([10.]))
        self.assertArrayAlmostEqual(result.data, expected)

    def test_end_point_negative_direction(self):
        """Test that the resulting cube contains the expected data when a
        end_point is specified, so that only part of the column is
        integrated. For integration in the negative direction (equivalent to
        integrating downwards for the height coordinate in the input cube),
        the presence of an end_point indicates that the integration may end
        above the lowest height within the column to be integrated."""
        expected = np.array(
            [[[25.00, 25.00, 25.00],
              [25.00, 25.00, 25.00],
              [25.00, 25.00, 25.00]]])
        self.plugin_negative.end_point = 8.
        result = self.plugin_negative.perform_integration(
            self.negative_upper_bounds_cube, self.negative_lower_bounds_cube)
        self.assertArrayAlmostEqual(
            result.coord("height").points, np.array([10.]))
        self.assertArrayAlmostEqual(result.data, expected)

    def test_start_point_at_bound_positive_direction(self):
        """Test that the resulting cube contains the expected data when a
        start_point is specified, so that only part of the column is
        integrated. In this instance, the start_point of 10 is equal to the
        available height levels [5., 10., 20.]. If the start_point is greater
        than a height level then integration will start from the next layer
        in the vertical. In this example, the start_point is equal to a height
        level, so the layer above the start_point is included within the
        integration.

        For integration in the positive direction (equivalent to
        integrating downwards for the height coordinate in the input cube),
        the presence of a start_point indicates that the integration may start
        above the lowest height within the column to be integrated."""
        expected = np.array(
            [[[25.00, 25.00, 25.00],
              [25.00, 25.00, 25.00],
              [25.00, 25.00, 25.00]]])
        self.plugin_positive.start_point = 10.
        result = self.plugin_positive.perform_integration(
            self.positive_upper_bounds_cube, self.positive_lower_bounds_cube)
        self.assertArrayAlmostEqual(
            result.coord("height").points, np.array([20.]))
        self.assertArrayAlmostEqual(result.data, expected)

    def test_end_point_at_bound_negative_direction(self):
        """Test that the resulting cube contains the expected data when a
        end_point is specified, so that only part of the column is
        integrated. In this instance, the end_point of 10 is equal to the
        available height levels [5., 10., 20.]. If the end_point is lower
        than a height level then integration will end. In this example,
        the end_point is equal to a height level, so the layer above the
        end_point is included within the integration.

        For integration in the negative direction (equivalent to
        integrating downwards for the height coordinate in the input cube),
        the presence of an end_point indicates that the integration may end
        above the lowest height within the column to be integrated."""
        expected = np.array(
            [[[25.00, 25.00, 25.00],
              [25.00, 25.00, 25.00],
              [25.00, 25.00, 25.00]]])
        self.plugin_negative.end_point = 10.
        result = self.plugin_negative.perform_integration(
            self.negative_upper_bounds_cube, self.negative_lower_bounds_cube)
        self.assertArrayAlmostEqual(
            result.coord("height").points, np.array([10.]))
        self.assertArrayAlmostEqual(result.data, expected)

    def test_integration_not_performed(self):
        """Test that the expected exception is raise if no integration
        can be performed, for example if the selected levels are out of
        the dataset range."""
        self.plugin_positive.start_point = 25.
        msg = "No integration could be performed for"
        with self.assertRaisesRegex(ValueError, msg):
            self.plugin_positive.perform_integration(
                self.positive_upper_bounds_cube,
                self.positive_lower_bounds_cube)


class Test_process(IrisTest):

    """Test the process method."""

    def setUp(self):
        """Set up the cube."""
        cube = _set_up_height_cube(np.array([5., 10., 20.]))
        self.coord_name = "height"
        data = np.zeros(cube.shape)
        data[0] = np.ones(cube[0].shape, dtype=np.int32)
        data[1] = np.full(cube[1].shape, 2, dtype=np.int32)
        data[2] = np.full(cube[2].shape, 3, dtype=np.int32)
        data[0, 0, 0] = 6
        cube.data = data
        self.cube = cube
        self.plugin = Integration("height")

    def test_basic(self):
        """Test that a cube with the points on the chosen coordinate are
        in the expected order."""
        result = self.plugin.process(self.cube)
        self.assertIsInstance(result, iris.cube.Cube)
        self.assertArrayAlmostEqual(
            result.coord("height").points, np.array([10., 5.]))
        self.assertArrayAlmostEqual(
            result.coord("height").bounds, np.array([[10., 20.], [5., 10.]]))

    def test_metadata(self):
        """Test that the metadata on the resulting cube is as expected"""
        expected_attributes = generate_mandatory_attributes([self.cube])
        result = self.plugin.process(self.cube)
        self.assertEqual(result.name(), self.cube.name()+'_integral')
        self.assertEqual(result.units, '{} m'.format(self.cube.units))
        self.assertDictEqual(result.attributes, expected_attributes)

    def test_data(self):
        """Test that the resulting cube contains the expected data following
        vertical integration."""
        expected = np.array(
            [[[25.00, 25.00, 25.00],
              [25.00, 25.00, 25.00],
              [25.00, 25.00, 25.00]],
             [[45.00, 32.50, 32.50],
              [32.50, 32.50, 32.50],
              [32.50, 32.50, 32.50]]])
        result = self.plugin.process(self.cube)
        self.assertArrayAlmostEqual(
            result.coord("height").points, np.array([10., 5.]))
        self.assertArrayAlmostEqual(result.data, expected)

    def test_dimension_preservation(self):
        """Test the result preserves input dimension order when the coordinate
        to integrate is not the first dimension (eg there's a leading
        realization coordinate)
        """
        cube = set_up_variable_cube(280*np.ones((3, 3, 3), dtype=np.float32))
        cube = add_coordinate(
            cube, np.array([5., 10., 20.]), "height", coord_units="m")
        cube.transpose([1, 0, 2, 3])
        expected_coord_order = [
            coord.name() for coord in cube.coords(dim_coords=True)]
        result = self.plugin.process(cube)
        self.assertEqual(result.coord_dims("height"), (1,))
        result_coord_order = [
            coord.name() for coord in result.coords(dim_coords=True)]
        self.assertListEqual(result_coord_order, expected_coord_order)


if __name__ == '__main__':
    unittest.main()
