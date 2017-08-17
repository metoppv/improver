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
""" Provides support utilities for checking cubes."""

import iris
from iris.exceptions import CoordinateNotFoundError, InvalidCubeError
import numpy as np


def check_for_x_and_y_axes(cube):
    """
    Check whether the cube has an x and y axis, otherwise raise an error.

    Parameters
    ----------
    cube : Iris.cube.Cube
        Cube to be checked for x and y axes.

    Raises
    ------
    ValueError : Raise an error if non-uniform increments exist between
                  grid points.
    """
    for axis in ["x", "y"]:
        if cube.coords(axis=axis):
            pass
        else:
            msg = ("The cube does not contain the expected {}"
                   "coordinates.".format(axis))
            raise ValueError(msg)


def check_cube_coordinates(cube, new_cube, exception_coordinates=None):
    """
    Find and promote to dimension coordinates any scalar coordinates in
    new_cube that were originally dimension coordinates in the progenitor
    cube. If coordinate is in new_cube that is not in the old cube, keep
    coordinate in its current position.
    Parameters
    ----------
    cube : iris.cube.Cube
        The input cube provided to nbhood.
    new_cube : iris.cube.Cube
        The cube produced by the neighbourhooding process that must be
        checked for demoted dimensional coordinates.
    exception_coordinates : List of strings or None
        The names of the coordinates that are permitted to be within the
        new_cube but are not available within the original cube.
    Returns
    -------
    new_cube : iris.cube.Cube
        Modified cube with relevant scalar coordinates promoted to
        dimension coordinates with the dimension coordinates re-ordered,
        as best as can be done based on the original cube.
    Raises
    ------
    InvalidCubeError if the coordinate is not within the original cube and
    there are no permitted exceptions.
    InvalidCubeError if the coordinate is not within the original cube and
    it is not within the list of permitted exceptions.
    CoordinateNotFoundError raised if the final dimension
    coordinates of the returned cube do not match the input cube.
    """

    # Promote available and relevant scalar coordinates
    for coord in new_cube.aux_coords[::-1]:
        if coord in cube.dim_coords:
            new_cube = iris.util.new_axis(new_cube, coord)

    # Ensure dimension order matches; if lengths are unequal a coordinate
    # is missing, so raise an appropriate error.
    cube_dimension_order = {coord.name(): cube.coord_dims(coord.name())[0]
                            for coord in cube.dim_coords}

    correct_order = []
    new_cube_only_dims = []
    new_cube_dim_names = [coord.name() for coord in new_cube.dim_coords]
    cube_dim_names = [coord.name() for coord in cube.dim_coords]
    for coord_name in new_cube_dim_names:
        if coord_name in cube_dim_names:
            correct_order.append(cube_dimension_order[coord_name])
        else:
            if exception_coordinates is None:
                msg = ("The coordinate: {} is within the new_cube, "
                       "however, this is not within the original "
                       "cube. As there are no permitted "
                       "exceptions, this is not allowed. "
                       "\nnew_cube: {}"
                       "\ncube: {}").format(
                           coord_name, new_cube, cube)
                raise InvalidCubeError(msg)
            elif coord_name in exception_coordinates:
                new_coord_dim = new_cube.coord_dims(coord_name)[0]
                new_cube_only_dims.append(new_coord_dim)
            else:
                msg = ("The coordinate: {0} is within new_cube, "
                       "however, this is not within the original "
                       "cube. As {0} is not within the permitted "
                       "exceptions: {1}, this is not allowed. "
                       "\nnew_cube: {2}"
                       "\ncube: {3}").format(
                           coord_name, exception_coordinates, new_cube,
                           cube)
                raise InvalidCubeError(msg)

    correct_order = np.array(correct_order)
    for dim in new_cube_only_dims:
        correct_order[correct_order >= dim] += 1
        correct_order = np.insert(correct_order, dim, dim)

    if exception_coordinates is None:
        exception_coordinates = []

    if (len(cube_dimension_order.keys()+exception_coordinates) ==
            len(correct_order)):
        new_cube.transpose(correct_order)
    else:
        msg = ('Returned cube dimension coordinates do not match input '
               'cube dimension coordinates. \n input cube shape {} '
               ' returned cube shape {}'.format(
                   cube.shape, new_cube.shape))
        raise CoordinateNotFoundError(msg)

    return new_cube


def find_dimension_coordinate_mismatch(
        first_cube, second_cube, two_way_mismatch=True):
    """Determine if there is a mismatch between the dimension coordinates in
    two cubes.
    Parameters
    ----------
    first_cube : Iris.cube.Cube
        First cube to compare.
    second_cube : Iris.cube.Cube
        Second cube to compare.
    two_way_mismatch : Logical
        If True, a two way mismatch is calculated e.g.
            second_cube - first_cube AND
            first_cube - second_cube
        If False, a one way mismatch is calculated e.g.
            second_cube - first_cube
    Returns
    ------
    result : List
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
