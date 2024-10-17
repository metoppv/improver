# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
""" Tests of DifferenceBetweenAdjacentGridSquares plugin."""

import unittest

import iris
import numpy as np
import pytest
from iris.coords import CellMethod
from iris.cube import Cube
from iris.tests import IrisTest
from numpy import ma

from improver.synthetic_data.set_up_test_cubes import set_up_variable_cube
from improver.utilities.spatial import DifferenceBetweenAdjacentGridSquares


class Test_create_difference_cube(IrisTest):

    """Test the create_difference_cube method."""

    def setUp(self):
        """Set up cube."""
        data = np.array([[1, 2, 3], [2, 4, 6], [5, 10, 15]])
        self.diff_in_y_array = np.array([[1, 2, 3], [3, 6, 9]])
        self.cube = set_up_variable_cube(
            data, name="wind_speed", units="m s-1", spatial_grid="equalarea",
        )
        self.plugin = DifferenceBetweenAdjacentGridSquares()

    def test_y_dimension_equalarea(self):
        """Test differences calculated along the y dimension, equalarea grid."""
        points = self.cube.coord(axis="y").points
        expected_y_coords = (points[1:] + points[:-1]) / 2
        result = self.plugin.create_difference_cube(
            self.cube, "projection_y_coordinate", self.diff_in_y_array
        )
        self.assertIsInstance(result, Cube)
        self.assertArrayAlmostEqual(result.coord(axis="y").points, expected_y_coords)
        self.assertArrayEqual(result.data, self.diff_in_y_array)

    def test_x_dimension_equalarea(self):
        """Test differences calculated along the x dimension, equalarea grid."""
        diff_array = np.array([[1, 1], [2, 2], [5, 5]])
        points = self.cube.coord(axis="x").points
        expected_x_coords = (points[1:] + points[:-1]) / 2
        result = self.plugin.create_difference_cube(
            self.cube, "projection_x_coordinate", diff_array
        )
        self.assertIsInstance(result, Cube)
        self.assertArrayAlmostEqual(result.coord(axis="x").points, expected_x_coords)
        self.assertArrayEqual(result.data, diff_array)

    def test_x_dimension_equalarea_circular(self):
        """Test differences calculated along the x dimension when x is circular, equalarea grid."""
        diff_array = np.array([[1, 1], [2, 2], [5, 5]])
        self.cube.coord(axis="x").circular = True
        with pytest.raises(
            NotImplementedError,
            match="DifferenceBetweenAdjacentGridSquares does not support cubes with circular "
            "x-axis that do not use a geographic",
        ):
            self.plugin.create_difference_cube(
                self.cube, "projection_x_coordinate", diff_array
            )

    def test_x_dimension_for_circular_latlon_cube(self):
        """Test differences calculated along the x dimension for a cube which is circular in x."""
        test_cube_data = np.array([[1, 2, 3], [2, 4, 6], [5, 10, 15]])
        test_cube_x_grid_spacing = 120
        test_cube = set_up_variable_cube(
            test_cube_data,
            "latlon",
            x_grid_spacing=test_cube_x_grid_spacing,
            name="wind_speed",
            units="m s-1",
        )
        test_cube.coord(axis="x").circular = True
        expected_diff_array = np.array([[1, 1, -2], [2, 2, -4], [5, 5, -10]])
        expected_x_coords = np.array(
            [-60, 60, 180]
        )  # Original data are at [-120, 0, 120], therefore differences are at [-60, 60, 180].
        result = self.plugin.create_difference_cube(
            test_cube, "longitude", expected_diff_array
        )
        self.assertIsInstance(result, Cube)
        self.assertArrayAlmostEqual(result.coord(axis="x").points, expected_x_coords)
        self.assertArrayEqual(result.data, expected_diff_array)

    def test_x_dimension_for_circular_latlon_cube_360_degree_coord(self):
        """Test differences calculated along the x dimension for a cube which is circular in x."""
        test_cube_data = np.array([[1, 2, 3], [2, 4, 6], [5, 10, 15]])
        test_cube_x_grid_spacing = 120
        test_cube = set_up_variable_cube(
            test_cube_data,
            "latlon",
            x_grid_spacing=test_cube_x_grid_spacing,
            name="wind_speed",
            units="m s-1",
        )
        test_cube.coord(axis="x").bounds = [[0, 120], [120, 240], [240, 360]]
        test_cube.coord(axis="x").points = [60, 120, 300]
        test_cube.coord(axis="x").circular = True
        expected_diff_array = np.array([[1, 1, -2], [2, 2, -4], [5, 5, -10]])
        expected_x_coords = np.array(
            [90, 210, 360]
        )  # Original data are at [60, 120, 300], therefore differences are at [90, 210, 360].
        result = self.plugin.create_difference_cube(
            test_cube, "longitude", expected_diff_array
        )
        self.assertIsInstance(result, Cube)
        self.assertArrayAlmostEqual(result.coord(axis="x").points, expected_x_coords)
        self.assertArrayEqual(result.data, expected_diff_array)

    def test_othercoords(self):
        """Test that other coords are transferred properly"""
        time_coord = self.cube.coord("time")
        proj_x_coord = self.cube.coord(axis="x")
        result = self.plugin.create_difference_cube(
            self.cube, "projection_y_coordinate", self.diff_in_y_array
        )
        self.assertEqual(result.coord(axis="x"), proj_x_coord)
        self.assertEqual(result.coord("time"), time_coord)


class Test_calculate_difference(IrisTest):
    """Test the calculate_difference method."""

    def setUp(self):
        """Set up cube."""
        data = np.array([[1, 2, 3, 4], [2, 4, 6, 8], [5, 10, 15, 20]])
        self.cube = set_up_variable_cube(
            data, "equalarea", name="wind_speed", units="m s-1",
        )
        self.plugin = DifferenceBetweenAdjacentGridSquares()

    def test_x_dimension(self):
        """Test differences calculated along the x dimension."""
        expected = np.array([[1, 1, 1], [2, 2, 2], [5, 5, 5]])
        result = self.plugin.calculate_difference(
            self.cube, self.cube.coord(axis="x").name()
        )
        self.assertIsInstance(result, np.ndarray)
        self.assertArrayEqual(result, expected)

    def test_x_dimension_wraps_around_meridian(self):
        """Test differences calculated along the x dimension for a cube which is circular in x."""
        self.cube.coord(axis="x").circular = True
        expected = np.array([[1, 1, 1, -3], [2, 2, 2, -6], [5, 5, 5, -15]])
        result = self.plugin.calculate_difference(
            self.cube, self.cube.coord(axis="x").name()
        )
        self.assertIsInstance(result, np.ndarray)
        self.assertArrayEqual(result, expected)

    def test_x_dimension_wraps_around_meridian_cube_axes_flipped(self):
        """Test differences calculated along the x dimension for a cube which is circular in x."""
        self.cube.coord(axis="x").circular = True
        self.cube.transpose()
        expected = np.array([[1, 1, 1, -3], [2, 2, 2, -6], [5, 5, 5, -15]]).transpose()
        result = self.plugin.calculate_difference(
            self.cube, self.cube.coord(axis="x").name()
        )
        self.assertIsInstance(result, np.ndarray)
        self.assertArrayEqual(result, expected)

    def test_y_dimension(self):
        """Test differences calculated along the y dimension."""
        expected = np.array([[1, 2, 3, 4], [3, 6, 9, 12]])
        result = self.plugin.calculate_difference(
            self.cube, self.cube.coord(axis="y").name()
        )
        self.assertIsInstance(result, np.ndarray)
        self.assertArrayEqual(result, expected)

    def test_missing_data(self):
        """Test that the result is as expected when data is missing."""
        data = np.array(
            [[1, 2, 3, 4], [np.nan, 4, 6, 8], [5, 10, 15, 20]], dtype=np.float32
        )
        self.cube.data = data
        expected = np.array([[np.nan, 2, 3, 4], [np.nan, 6, 9, 12]])
        result = self.plugin.calculate_difference(
            self.cube, self.cube.coord(axis="y").name()
        )
        self.assertIsInstance(result, np.ndarray)
        self.assertArrayAlmostEqual(result, expected)

    def test_masked_data(self):
        """Test that the result is as expected when data is masked."""
        data = ma.array(
            [[1, 2, 3, 4], [2, 4, 6, 8], [5, 10, 15, 20]],
            mask=[[0, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0]],
        )
        self.cube.data = data
        expected = ma.array(
            [[1, 2, 3, 4], [3, 6, 9, 12]], mask=[[1, 0, 0, 0], [1, 0, 0, 0]]
        )
        result = self.plugin.calculate_difference(
            self.cube, self.cube.coord(axis="y").name()
        )
        self.assertIsInstance(result, np.ndarray)
        self.assertArrayEqual(result, expected)
        self.assertArrayEqual(result.mask, expected.mask)


class Test_process(IrisTest):
    """Test the process method."""

    def setUp(self):
        """Set up cube."""
        data = np.array([[1, 2, 3], [2, 4, 6], [5, 10, 15]])
        self.cube = set_up_variable_cube(
            data,
            name="wind_speed",
            units="m s-1",
            spatial_grid="equalarea",
            realizations=np.array([1, 2]),
        )
        self.plugin = DifferenceBetweenAdjacentGridSquares()

    def test_basic(self):
        """Test that differences are calculated along both the x and
        y dimensions and returned as separate cubes."""
        expected_x = np.array([[1, 1], [2, 2], [5, 5]])
        expected_y = np.array([[1, 2, 3], [3, 6, 9]])
        result = self.plugin.process(self.cube)
        self.assertIsInstance(result[0], Cube)
        self.assertArrayEqual(result[0].data, expected_x)
        self.assertIsInstance(result[1], Cube)
        self.assertArrayEqual(result[1].data, expected_y)

    def test_metadata(self):
        """Test the resulting metadata is correct."""
        cell_method_x = CellMethod(
            "difference", coords=["projection_x_coordinate"], intervals="1 grid length"
        )
        cell_method_y = CellMethod(
            "difference", coords=["projection_y_coordinate"], intervals="1 grid length"
        )

        result = self.plugin.process(self.cube)
        for cube, cm in zip(result, [cell_method_x, cell_method_y]):
            self.assertEqual(cube.cell_methods[0], cm)
            self.assertEqual(
                cube.attributes["form_of_difference"], "forward_difference"
            )
            self.assertEqual(cube.name(), "difference_of_wind_speed")

    def test_3d_cube(self):
        """Test the differences are calculated along both the x and
        y dimensions and returned as separate cubes when a 3d cube is input."""
        data = np.array(
            [[[1, 2, 3], [2, 4, 6], [5, 10, 15]], [[1, 2, 3], [2, 2, 6], [5, 10, 20]]]
        )
        expected_x = np.array([[[1, 1], [2, 2], [5, 5]], [[1, 1], [0, 4], [5, 10]]])
        expected_y = np.array([[[1, 2, 3], [3, 6, 9]], [[1, 0, 3], [3, 8, 14]]])
        cube = set_up_variable_cube(
            data,
            name="wind_speed",
            units="m s-1",
            spatial_grid="equalarea",
            realizations=np.array([1, 2]),
        )
        result = self.plugin.process(cube)
        self.assertIsInstance(result[0], iris.cube.Cube)
        self.assertArrayEqual(result[0].data, expected_x)
        self.assertIsInstance(result[1], iris.cube.Cube)
        self.assertArrayEqual(result[1].data, expected_y)

    def test_circular_non_geographic_cube_raises_approprate_exception(self):
        """Check for error and message with projection coord and circular x axis"""
        self.cube.coord(axis="x").circular = True
        with self.assertRaisesRegex(
            NotImplementedError,
            "DifferenceBetweenAdjacentGridSquares does not support cubes with "
            r"circular x-axis that do not use a geographic \(i.e. latlon\) coordinate system.",
        ):
            self.plugin.process(self.cube)


if __name__ == "__main__":
    unittest.main()
