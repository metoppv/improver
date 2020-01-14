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
""" Provides support utilities for checking cubes."""

import iris
import numpy as np
from iris.exceptions import CoordinateNotFoundError


def check_for_x_and_y_axes(cube, require_dim_coords=False):
    """
    Check whether the cube has an x and y axis, otherwise raise an error.

    Args:
        cube (iris.cube.Cube):
            Cube to be checked for x and y axes.
        require_dim_coords (bool):
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
            msg = ("The cube does not contain the expected {}"
                   "coordinates.".format(axis))
            raise ValueError(msg)


def check_cube_coordinates(cube, new_cube, exception_coordinates=None):
    """Find and promote to dimension coordinates any scalar coordinates in
    new_cube that were originally dimension coordinates in the progenitor
    cube. If coordinate is in new_cube that is not in the old cube, keep
    coordinate in its current position.

    Args:
        cube (iris.cube.Cube):
            The input cube that will be checked to identify the preferred
            coordinate order for the output cube.
        new_cube (iris.cube.Cube):
            The cube that must be checked and adjusted using the coordinate
            order from the original cube.
        exception_coordinates (list of str or None):
            The names of the coordinates that are permitted to be within the
            new_cube but are not available within the original cube.

    Returns:
        iris.cube.Cube:
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
    if (len(cube.dim_coords)+len(exception_coordinates) !=
            len(new_cube.dim_coords)):

        msg = ('The number of dimension coordinates within the new cube '
               'do not match the number of dimension coordinates within the '
               'original cube plus the number of exception coordinates. '
               '\n input cube dimensions {}, new cube dimensions {}'.format(
                   cube_dim_names, new_cube_dim_names))
        raise CoordinateNotFoundError(msg)

    # Ensure dimension order matches
    new_cube_dimension_order = {coord.name(): new_cube.coord_dims(
        coord.name())[0] for coord in new_cube.dim_coords}
    correct_order = []
    new_cube_only_dims = []
    for coord_name in cube_dim_names:
        correct_order.append(new_cube_dimension_order[coord_name])
    for coord_name in exception_coordinates:
        try:
            new_coord_dim = new_cube.coord_dims(coord_name)[0]
            new_cube_only_dims.append(new_coord_dim)
        except CoordinateNotFoundError:
            msg = ("All permitted exception_coordinates must be on the"
                   " new_cube. In this case, coordinate {0} within the list "
                   "of permitted exception_coordinates ({1}) is not available"
                   " on the new_cube.").format(
                        coord_name, exception_coordinates)
            raise CoordinateNotFoundError(msg)

    correct_order = np.array(correct_order)
    for dim in new_cube_only_dims:
        correct_order = np.insert(correct_order, dim, dim)

    new_cube.transpose(correct_order)

    return new_cube


def find_dimension_coordinate_mismatch(
        first_cube, second_cube, two_way_mismatch=True):
    """Determine if there is a mismatch between the dimension coordinates in
    two cubes.

    Args:
        first_cube (iris.cube.Cube):
            First cube to compare.
        second_cube (iris.cube.Cube):
            Second cube to compare.
        two_way_mismatch (Logical):
            If True, a two way mismatch is calculated e.g.
                second_cube - first_cube AND
                first_cube - second_cube
            If False, a one way mismatch is calculated e.g.
                second_cube - first_cube

    Returns:
        list of str:
            List of the dimension coordinates that are only present in
            one out of the two cubes.

    """
    first_dim_names = [coord.name() for coord in first_cube.dim_coords]
    second_dim_names = [coord.name() for coord in second_cube.dim_coords]
    if two_way_mismatch:
        mismatch = (list(set(second_dim_names) - set(first_dim_names)) +
                    list(set(first_dim_names) - set(second_dim_names)))
    else:
        mismatch = list(set(second_dim_names) - set(first_dim_names))
    return mismatch


def spatial_coords_match(first_cube, second_cube):
    """
    Determine if the x and y coords in the two cubes are the same.

    Args:
        first_cube (iris.cube.Cube):
            First cube to compare.
        second_cube (iris.cube.Cube):
            Second cube to compare.

    Returns:
        bool:
            True if the x and y coords are the exactly the same to the
            precision of the floating-point values (this should be true for
            any cubes derived using cube.regrid()), otherwise False.
    """
    return (first_cube.coord(axis='x') == second_cube.coord(axis='x') and
            first_cube.coord(axis='y') == second_cube.coord(axis='y'))


def time_coords_match(first_cube, second_cube, raise_exception=False):
    """
    Determine if two cubes have equivalent time, forecast_period, and
    forecast_reference_time points.

    Args:
        first_cube (iris.cube.Cube):
            First cube to compare.
        second_cube (iris.cube.Cube):
            Second cube to compare.
        raise_exception (bool):
            By default this function returns a boolean, but if this argument is
            set to True it will raise an exception if there is a mismatch in
            the coordinates.

    Returns:
        bool:
            True if the cube time coordinates are equivalent, False if they are
            not.

    Raised:
        ValueError: The two cubes are not equivalent.
        CoordinateNotFoundError: One of the expected temporal coordinates is
        not present on one or more cubes.
    """
    cubes_equivalent = True
    mismatches = []
    for coord_name in ["forecast_period", "time", "forecast_reference_time"]:
        if (first_cube.coord(coord_name) != second_cube.coord(coord_name)):
            mismatches.append(coord_name)
            cubes_equivalent = False

    if mismatches and raise_exception:
        msg = "The following coordinates of the two cubes do not match: {}"
        raise ValueError(msg.format(', '.join(mismatches)))

    return cubes_equivalent
