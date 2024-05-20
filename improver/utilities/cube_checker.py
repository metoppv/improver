# (C) Crown copyright, Met Office. All rights reserved.
#
# This file is part of improver and is released under a BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
""" Provides support utilities for checking cubes."""

from typing import List, Optional, Union

import iris
import numpy as np
from iris.cube import Cube, CubeList
from iris.exceptions import CoordinateNotFoundError


def check_for_x_and_y_axes(cube: Cube, require_dim_coords: bool = False) -> None:
    """
    Check whether the cube has an x and y axis, otherwise raise an error.

    Args:
        cube:
            Cube to be checked for x and y axes.
        require_dim_coords:
            If true the x and y coordinates must be dimension coordinates.

    Raises:
        ValueError : Raise an error if non-uniform increments exist between
            grid points.
    """
    for axis in ["x", "y"]:
        if require_dim_coords:
            coord = cube.coords(axis=axis, dim_coords=True)
        else:
            coord = cube.coords(axis=axis)

        if coord:
            pass
        else:
            msg = "The cube does not contain the expected {} coordinates.".format(axis)
            raise ValueError(msg)


def check_cube_coordinates(
    cube: Cube, new_cube: Cube, exception_coordinates: Optional[List[str]] = None
) -> Cube:
    """Find and promote to dimension coordinates any scalar coordinates in
    new_cube that were originally dimension coordinates in the progenitor
    cube. If coordinate is in new_cube that is not in the old cube, keep
    coordinate in its current position.

    Args:
        cube:
            The input cube that will be checked to identify the preferred
            coordinate order for the output cube.
        new_cube:
            The cube that must be checked and adjusted using the coordinate
            order from the original cube.
        exception_coordinates:
            The names of the coordinates that are permitted to be within the
            new_cube but are not available within the original cube.

    Returns:
        Modified cube with relevant scalar coordinates promoted to
        dimension coordinates with the dimension coordinates re-ordered,
        as best as can be done based on the original cube.

    Raises:
        CoordinateNotFoundError : Raised if the final dimension
            coordinates of the returned cube do not match the input cube.
        CoordinateNotFoundError : If a coordinate is within in the permitted
            exceptions but is not in the new_cube.
    """
    if exception_coordinates is None:
        exception_coordinates = []

    # Promote available and relevant scalar coordinates
    cube_dim_names = [coord.name() for coord in cube.dim_coords]
    for coord in new_cube.aux_coords[::-1]:
        if coord.name() in cube_dim_names:
            new_cube = iris.util.new_axis(new_cube, coord)
    new_cube_dim_names = [coord.name() for coord in new_cube.dim_coords]
    # If we have the wrong number of dimensions then raise an error.
    if len(cube.dim_coords) + len(exception_coordinates) != len(new_cube.dim_coords):

        msg = (
            "The number of dimension coordinates within the new cube "
            "do not match the number of dimension coordinates within the "
            "original cube plus the number of exception coordinates. "
            "\n input cube dimensions {}, new cube dimensions {}".format(
                cube_dim_names, new_cube_dim_names
            )
        )
        raise CoordinateNotFoundError(msg)

    # Ensure dimension order matches
    new_cube_dimension_order = {
        coord.name(): new_cube.coord_dims(coord.name())[0]
        for coord in new_cube.dim_coords
    }
    correct_order = []
    new_cube_only_dims = []
    for coord_name in cube_dim_names:
        correct_order.append(new_cube_dimension_order[coord_name])
    for coord_name in exception_coordinates:
        try:
            new_coord_dim = new_cube.coord_dims(coord_name)[0]
            new_cube_only_dims.append(new_coord_dim)
        except CoordinateNotFoundError:
            msg = (
                "All permitted exception_coordinates must be on the"
                " new_cube. In this case, coordinate {0} within the list "
                "of permitted exception_coordinates ({1}) is not available"
                " on the new_cube."
            ).format(coord_name, exception_coordinates)
            raise CoordinateNotFoundError(msg)

    correct_order = np.array(correct_order)
    for dim in new_cube_only_dims:
        correct_order = np.insert(correct_order, dim, dim)

    new_cube.transpose(correct_order)

    return new_cube


def find_dimension_coordinate_mismatch(
    first_cube: Cube, second_cube: Cube, two_way_mismatch: bool = True
) -> List[str]:
    """Determine if there is a mismatch between the dimension coordinates in
    two cubes.

    Args:
        first_cube:
            First cube to compare.
        second_cube:
            Second cube to compare.
        two_way_mismatch:
            If True, a two way mismatch is calculated e.g.
                second_cube - first_cube AND
                first_cube - second_cube
            If False, a one way mismatch is calculated e.g.
                second_cube - first_cube

    Returns:
        List of the dimension coordinates that are only present in
        one out of the two cubes.
    """
    first_dim_names = [coord.name() for coord in first_cube.dim_coords]
    second_dim_names = [coord.name() for coord in second_cube.dim_coords]
    if two_way_mismatch:
        mismatch = list(set(second_dim_names) - set(first_dim_names)) + list(
            set(first_dim_names) - set(second_dim_names)
        )
    else:
        mismatch = list(set(second_dim_names) - set(first_dim_names))
    return mismatch


def spatial_coords_match(cubes: Union[List, CubeList]) -> bool:
    """
    Determine if the x and y coords of all the input cubes are the same.

    Args:
        cubes:
            A list of cubes to compare.

    Returns:
        True if the x and y coords are the exactly the same to the
        precision of the floating-point values (this should be true for
        any cubes derived using cube.regrid()), otherwise False.
    """
    ref = cubes[0]
    match = True
    for cube in cubes[1:]:
        match = (
            cube.coord(axis="x") == ref.coord(axis="x")
            and cube.coord(axis="y") == ref.coord(axis="y")
            and match
        )
    return match


def assert_time_coords_valid(inputs: List[Cube], time_bounds: bool):
    """
    Raises appropriate ValueError if

    - Any input cube has or is missing time bounds (depending on time_bounds)
    - Input cube times do not match
    - Input cube forecast_reference_times do not match (unless blend_time is present)

    Note that blend_time coordinates do not have to match as it is likely that data
    from nearby blends will be used together.

    Args:
        inputs:
            List of Cubes where times should match
        time_bounds:
            When True, time bounds are checked for and compared on the input cubes.
            When False, the absence of time bounds is checked for.

    Raises:
        ValueError: If any of the stated checks fail.
    """
    if len(inputs) <= 1:
        raise ValueError(f"Need at least 2 cubes to check. Found {len(inputs)}")
    cubes_not_matching_time_bounds = [
        c.name() for c in inputs if c.coord("time").has_bounds() != time_bounds
    ]
    if cubes_not_matching_time_bounds:
        str_bool = "" if time_bounds else "not "
        msg = f"{' and '.join(cubes_not_matching_time_bounds)} must {str_bool}have time bounds"
        raise ValueError(msg)

    if inputs[0].coords("blend_time"):
        time_coords_to_check = ["time"]
    else:
        time_coords_to_check = ["time", "forecast_reference_time"]
    for time_coord_name in time_coords_to_check:
        time_coords = [c.coord(time_coord_name) for c in inputs]
        if not all([tc == time_coords[0] for tc in time_coords[1:]]):
            msg = f"{time_coord_name} coordinates do not match. \n  " + "\n  ".join(
                [f"{c.name()}: {c.coord('time')}" for c in inputs]
            )
            raise ValueError(msg)


def assert_spatial_coords_match(cubes: Union[List, CubeList]):
    """
    Raises an Exception if `spatial_coords_match` returns False.

    Args:
        cubes:
            A list of cubes to compare.

    Raises:
        ValueError if spatial coords do not match.

    """
    if not spatial_coords_match(cubes):
        raise ValueError(
            f"Mismatched spatial coords for {', '.join([c.name() for c in cubes])}"
        )
