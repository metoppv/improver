# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the cube_checker utility."""

import unittest
from datetime import datetime
from typing import List

import numpy as np
import pytest
from iris.cube import Cube

from improver.synthetic_data.set_up_test_cubes import (
    set_up_variable_cube,
)
from improver.utilities.cube_checker import (
    assert_spatial_coords_match,
    check_for_x_and_y_axes,
    spatial_coords_match,
)


class Test_check_for_x_and_y_axes(unittest.TestCase):
    """Test whether the cube has an x and y axis."""

    def setUp(self):
        """Set up a cube."""
        data = np.ones((1, 5, 5), dtype=np.float32)
        self.cube = set_up_variable_cube(
            data, name="precipitation_amount", units="kg m^-2", spatial_grid="equalarea"
        )

    def test_no_y_coordinate(self):
        """Test that the expected exception is raised, if there is no
        y coordinate."""
        sliced_cube = next(self.cube.slices(["projection_x_coordinate"]))
        sliced_cube.remove_coord("projection_y_coordinate")
        msg = "The cube does not contain the expected"
        with self.assertRaisesRegex(ValueError, msg):
            check_for_x_and_y_axes(sliced_cube)

    def test_no_x_coordinate(self):
        """Test that the expected exception is raised, if there is no
        x coordinate."""

        sliced_cube = next(self.cube.slices(["projection_y_coordinate"]))
        sliced_cube.remove_coord("projection_x_coordinate")
        msg = "The cube does not contain the expected"
        with self.assertRaisesRegex(ValueError, msg):
            check_for_x_and_y_axes(sliced_cube)

    def test_pass_dimension_requirement(self):
        """Pass in compatible cubes that should not raise an exception. No
        assert statement required as any other input will raise an
        exception."""
        check_for_x_and_y_axes(self.cube, require_dim_coords=True)

    def test_fail_dimension_requirement(self):
        """Test that the expected exception is raised, if there the x and y
        coordinates are not dimensional coordinates."""
        msg = "The cube does not contain the expected"
        cube = self.cube[0, :, 0]
        with self.assertRaisesRegex(ValueError, msg):
            check_for_x_and_y_axes(cube, require_dim_coords=True)


class Test_spatial_coords_match(unittest.TestCase):
    """Test for function testing cube spatial coords."""

    def setUp(self):
        """Create two unmatching cubes for spatial comparison."""
        data_a = np.ones((1, 16, 16), dtype=np.float32)
        data_b = np.ones((1, 10, 10), dtype=np.float32)
        self.cube_a = set_up_variable_cube(
            data_a,
            name="precipitation_amount",
            units="kg m^-2",
            spatial_grid="equalarea",
        )
        self.cube_b = set_up_variable_cube(
            data_b,
            name="precipitation_amount",
            units="kg m^-2",
            spatial_grid="equalarea",
        )

    def test_single_cube(self):
        """Test that True is returned if a single cube is provided as input."""
        result = spatial_coords_match([self.cube_a])
        self.assertTrue(result)

    def test_matching(self):
        """Test bool return when given one cube twice."""
        result = spatial_coords_match([self.cube_a, self.cube_a])
        self.assertTrue(result)

    def test_assert_matching(self):
        """Test for no error when test_matching is repeated with assert method."""
        assert_spatial_coords_match([self.cube_a, self.cube_a])

    def test_matching_multiple(self):
        """Test when given more than two cubes to test, these matching."""
        result = spatial_coords_match([self.cube_a, self.cube_a, self.cube_a])
        self.assertTrue(result)

    def test_copy(self):
        """Test when given one cube copied."""
        result = spatial_coords_match([self.cube_a, self.cube_a.copy()])
        self.assertTrue(result)

    def test_other_coord_diffs(self):
        """Test when given cubes that differ in non-spatial coords."""
        cube_c = self.cube_a.copy()
        r_coord = cube_c.coord("realization")
        r_coord.points = [r * 2 for r in r_coord.points]
        result = spatial_coords_match([self.cube_a, cube_c])
        self.assertTrue(result)

    def test_other_coord_bigger_diffs(self):
        """Test when given cubes that differ in shape on non-spatial coords."""
        data_c = np.ones((4, 16, 16), dtype=np.float32)
        data_c[:, 7, 7] = 0.0
        cube_c = set_up_variable_cube(
            data_c,
            name="precipitation_amount",
            units="kg m^-2",
            spatial_grid="equalarea",
        )
        r_coord = cube_c.coord("realization")
        r_coord.points = [r * 2 for r in r_coord.points]
        result = spatial_coords_match([self.cube_a, cube_c])
        self.assertTrue(result)

    def test_unmatching(self):
        """Test when given two spatially different cubes of same resolution."""
        result = spatial_coords_match([self.cube_a, self.cube_b])
        self.assertFalse(result)

    def test_assert_unmatching(self):
        """Test assert method when given two spatially different cubes of same resolution."""
        msg = "Mismatched spatial coords for "
        with self.assertRaisesRegex(ValueError, msg):
            assert_spatial_coords_match([self.cube_a, self.cube_b])

    def test_unmatching_multiple(self):
        """Test when given more than two cubes to test, these unmatching."""
        result = spatial_coords_match([self.cube_a, self.cube_b, self.cube_a])
        self.assertFalse(result)

    def test_unmatching_x(self):
        """Test when given two cubes of the same shape, but with differing
        x coordinate values."""
        cube_c = self.cube_a.copy()
        x_coord = cube_c.coord(axis="x")
        x_coord.points = [x * 2.0 for x in x_coord.points]
        result = spatial_coords_match([self.cube_a, cube_c])
        self.assertFalse(result)

    def test_unmatching_y(self):
        """Test when given two cubes of the same shape, but with differing
        y coordinate values."""
        cube_c = self.cube_a.copy()
        y_coord = cube_c.coord(axis="y")
        y_coord.points = [y * 1.01 for y in y_coord.points]
        result = spatial_coords_match([self.cube_a, cube_c])
        self.assertFalse(result)


@pytest.fixture(name="cubes")
def cubes_fixture(time_bounds) -> List[Cube]:
    """Set up matching r, y, x cubes matching Plugin requirements, with or without time
    bounds"""
    cubes = []
    data = np.ones((2, 3, 4), dtype=np.float32)
    kwargs = {}
    if time_bounds:
        kwargs["time_bounds"] = (
            datetime(2017, 11, 10, 3, 0),
            datetime(2017, 11, 10, 4, 0),
        )
    cube = set_up_variable_cube(data, **kwargs)
    for descriptor in (
        {"name": "air_temperature", "units": "K"},
        {"name": "air_pressure", "units": "Pa"},
        {"name": "relative_humidity", "units": "kg kg-1"},
    ):
        cube = cube.copy()
        cube.rename(descriptor["name"])
        cube.units = descriptor["units"]
        cubes.append(cube)
    return cubes


if __name__ == "__main__":
    unittest.main()
