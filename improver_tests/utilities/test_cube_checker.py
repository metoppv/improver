# (C) Crown copyright, Met Office. All rights reserved.
#
# This file is part of IMPROVER and is released under a BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the cube_checker utility."""

import unittest
from datetime import datetime
from typing import List

import iris
import numpy as np
import pytest
from iris.cube import Cube
from iris.exceptions import CoordinateNotFoundError
from iris.tests import IrisTest

from improver.synthetic_data.set_up_test_cubes import (
    add_coordinate,
    set_up_variable_cube,
)
from improver.utilities.cube_checker import (
    assert_spatial_coords_match,
    assert_time_coords_valid,
    check_cube_coordinates,
    check_for_x_and_y_axes,
    find_dimension_coordinate_mismatch,
    spatial_coords_match,
)


class Test_check_for_x_and_y_axes(IrisTest):

    """Test whether the cube has an x and y axis."""

    def setUp(self):
        """Set up a cube."""
        data = np.ones((1, 5, 5), dtype=np.float32)
        self.cube = set_up_variable_cube(
            data,
            name="precipitation_amount",
            units="kg m^-2",
            spatial_grid="equalarea",
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


class Test_check_cube_coordinates(IrisTest):

    """Test check_cube_coordinates successfully promotes scalar coordinates to
    dimension coordinates in a new cube if they were dimension coordinates in
    the progenitor cube."""

    def setUp(self):
        """Set up a cube."""
        data = np.ones((1, 16, 16), dtype=np.float32)
        self.cube = set_up_variable_cube(
            data,
            name="precipitation_amount",
            units="kg m^-2",
            spatial_grid="equalarea",
        )
        self.squeezed_cube = iris.util.squeeze(self.cube)

    def test_basic(self):
        """Test returns iris.cube.Cube."""
        result = check_cube_coordinates(self.cube, self.cube)
        self.assertIsInstance(result, Cube)

    def test_basic_transpose(self):
        """Test when we only want to transpose the new_cube."""
        new_cube = self.cube.copy()
        new_cube.transpose([2, 1, 0])
        result = check_cube_coordinates(self.cube, new_cube)
        self.assertEqual(result.dim_coords, self.cube.dim_coords)

    def test_coord_promotion(self):
        """Test that scalar coordinates in new_cube are promoted to dimension
        coordinates to match the parent cube."""
        result = check_cube_coordinates(self.cube, self.squeezed_cube)
        self.assertEqual(result.dim_coords, self.cube.dim_coords)
        self.assertEqual(
            result.coords(dim_coords=False), self.cube.coords(dim_coords=False)
        )

    def test_coord_promotion_and_reordering(self):
        """Test case in which a scalar coordinate are promoted but the order
        must be corrected to match the progenitor cube."""
        self.cube.transpose(new_order=[1, 0, 2])
        result = check_cube_coordinates(self.cube, self.squeezed_cube)
        self.assertEqual(result.dim_coords, self.cube.dim_coords)

    def test_permitted_exception_coordinates(self):
        """Test that if the new_cube is known to have additional coordinates
        compared with the original cube, these coordinates are listed as
        exception_coordinates and handled correctly."""
        exception_coordinates = ["realization"]
        result = check_cube_coordinates(
            self.squeezed_cube, self.cube, exception_coordinates=exception_coordinates
        )
        dim_coords = (
            tuple(self.cube.coord("realization")) + self.squeezed_cube.dim_coords
        )
        self.assertEqual(result.dim_coords, dim_coords)

    def test_no_permitted_exception_coordinates(self):
        """Test that if the new_cube has additional coordinates compared with
        the original cube, if no coordinates are listed as exception
        coordinates, then an exception will be raised."""
        msg = "The number of dimension coordinates within the new cube"
        with self.assertRaisesRegex(CoordinateNotFoundError, msg):
            check_cube_coordinates(self.squeezed_cube, self.cube)

    def test_missing_exception_coordinates(self):
        """Test that if the new_cube has additional coordinates compared with
        the original cube, if these coordinates are not listed as exception
        coordinates, then an exception will be raised."""
        exception_coordinates = ["height"]
        msg = "All permitted exception_coordinates must be on the new_cube."
        with self.assertRaisesRegex(CoordinateNotFoundError, msg):
            check_cube_coordinates(
                self.squeezed_cube,
                self.cube,
                exception_coordinates=exception_coordinates,
            )

    def test_coord_promotion_missing_scalar(self):
        """Test case in which a scalar coordinate has been lost from new_cube,
        meaning the cube undergoing checking ends up with different dimension
        coordinates to the progenitor cube. This raises an error."""
        self.squeezed_cube.remove_coord("realization")
        msg = "The number of dimension coordinates within the new cube"
        with self.assertRaisesRegex(CoordinateNotFoundError, msg):
            check_cube_coordinates(self.cube, self.squeezed_cube)


class Test_find_dimension_coordinate_mismatch(IrisTest):

    """Test if two cubes have the dimension coordinates."""

    def setUp(self):
        """Set up a cube."""
        data = np.ones((2, 16, 16), dtype=np.float32)
        self.cube = set_up_variable_cube(
            data,
            name="precipitation_amount",
            units="kg m^-2",
            spatial_grid="equalarea",
        )

    def test_no_mismatch(self):
        """Test if there is no mismatch between the dimension coordinates."""
        result = find_dimension_coordinate_mismatch(self.cube, self.cube)
        self.assertIsInstance(result, list)
        self.assertFalse(result)

    def test_mismatch_in_first_cube(self):
        """Test when finding a one-way mismatch, so that the second cube has
        a missing coordinate. This returns an empty list."""
        first_cube = self.cube.copy()
        second_cube = next(self.cube.slices_over("realization")).copy()
        second_cube.remove_coord("realization")
        result = find_dimension_coordinate_mismatch(
            first_cube, second_cube, two_way_mismatch=False
        )
        self.assertIsInstance(result, list)
        self.assertFalse(result)

    def test_mismatch_in_second_cube(self):
        """Test when finding a one-way mismatch, so that the first cube has
        a missing coordinate. This returns a list with the missing coordinate
        name."""
        first_cube = next(self.cube.slices_over("realization")).copy()
        first_cube.remove_coord("realization")
        second_cube = self.cube.copy()
        result = find_dimension_coordinate_mismatch(
            first_cube, second_cube, two_way_mismatch=False
        )
        self.assertIsInstance(result, list)
        self.assertListEqual(result, ["realization"])

    def test_two_way_mismatch(self):
        """Test when finding a two-way mismatch, when the first and second
        cube contain different coordinates."""
        first_cube = self.cube.copy()
        second_cube = next(self.cube.slices_over("realization")).copy()
        second_cube.remove_coord("realization")
        second_cube = add_coordinate(second_cube, [10, 20], "height", "m")
        result = find_dimension_coordinate_mismatch(first_cube, second_cube)
        self.assertIsInstance(result, list)
        self.assertListEqual(result, ["height", "realization"])


class Test_spatial_coords_match(IrisTest):
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
    """Set up matching r, y, x cubes matching Plugin requirements, with or without time bounds"""
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


def swap_frt_for_blend_time(cubes: List[Cube]):
    """Renames the forecast_reference_time coord on each cube to blend_time"""
    for cube in cubes:
        cube.coord("forecast_reference_time").rename("blend_time")


@pytest.mark.parametrize("blend_time", (True, False))
@pytest.mark.parametrize("time_bounds", (True, False))
@pytest.mark.parametrize("input_count", (2, 3))
def test_time_coords_valid(
    cubes: List[Cube], input_count: int, time_bounds: bool, blend_time: bool
):
    """Test that no exceptions are raised when the required conditions are met
    for either 2 or 3 cubes, with or without time bounds, with or without blend_time"""
    if blend_time:
        swap_frt_for_blend_time(cubes)
    assert_time_coords_valid(cubes[:input_count], time_bounds=time_bounds)


def inconsistent_time_bounds(cubes: List[Cube]):
    """Adds time bounds only to the first cube"""
    time_point = cubes[0].coord("time").points[0]
    cubes[0].coord("time").bounds = (time_point - 10800, time_point)


def inconsistent_time_point(cubes: List[Cube]):
    """Moves time point of first cube back by one hour"""
    cubes[0].coord("time").points = cubes[0].coord("time").points - 3600


def inconsistent_frt(cubes: List[Cube]):
    """Moves forecast_reference_time point of first cube back by one hour"""
    cubes[0].coord("forecast_reference_time").points = (
        cubes[0].coord("forecast_reference_time").points - 3600
    )


def remove_one_time_bounds(cubes: List[Cube]):
    """Removes time bounds from first cube"""
    cubes[0].coord("time").bounds = None


def remove_two_time_bounds(cubes: List[Cube]):
    """Removes time bounds from first two cubes"""
    cubes[0].coord("time").bounds = None
    cubes[1].coord("time").bounds = None


def only_one_cube(cubes: List[Cube]):
    """Removes second and third cubes"""
    cubes.pop(2)
    cubes.pop(1)


@pytest.mark.parametrize(
    "modifier, time_bounds, error_match",
    (
        (inconsistent_time_bounds, True, "^time coordinates do not match."),
        (inconsistent_time_bounds, False, "^air_temperature must not have time bounds"),
        (inconsistent_time_point, True, "^time coordinates do not match."),
        (inconsistent_time_point, False, "^time coordinates do not match."),
        (inconsistent_frt, True, "^forecast_reference_time coordinates do not match."),
        (inconsistent_frt, False, "^forecast_reference_time coordinates do not match."),
        (remove_one_time_bounds, True, "^air_temperature must have time bounds"),
        (
            remove_two_time_bounds,
            True,
            "^air_temperature and air_pressure must have time bounds",
        ),
        (only_one_cube, False, "^Need at least 2 cubes to check. Found 1",),
    ),
)
def test_time_coord_exceptions(
    cubes, modifier: callable, time_bounds: bool, error_match: str
):
    """Checks that assert_time_coords_valid raises useful errors
    when the required conditions are not met."""
    modifier(cubes)
    with pytest.raises(ValueError, match=error_match):
        assert_time_coords_valid(cubes, time_bounds=time_bounds)


if __name__ == "__main__":
    unittest.main()
