# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown Copyright 2017-2018 Met Office.
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
"""Utilities for spatial padding of iris cubes."""

import numpy as np
import iris
from copy import deepcopy

from improver.utilities.cube_checker import check_for_x_and_y_axes


def pad_coord(coord, width, method):
    """
    Construct a new coordinate by extending the current coordinate by the
    padding width.

    Args:
        coord (iris.coord):
            Original coordinate which will be used as the basis of the
            new extended coordinate.
        width (int):
            The width of padding in grid cells (the extent of the
            neighbourhood radius in grid cells in a given direction).
        method (string):
            A string determining whether the coordinate is being expanded
            or contracted. Options: 'remove' to remove points from coord;
            'add' to add points to coord.

    Returns:
        iris.coord:
            Coordinate with expanded or contracted length, to be added to
            the padded or unpadded iris cube.

    Raises:
        ValueError: Raise an error if non-uniform increments exist between
                    grid points.
    """
    orig_points = coord.points
    increment = orig_points[1:] - orig_points[:-1]
    if np.isclose(np.sum(np.diff(increment)), 0):
        increment = increment[0]
    else:
        msg = ("Non-uniform increments between grid points: "
               "{}.".format(increment))
        raise ValueError(msg)

    if method == 'add':
        num_of_new_points = len(orig_points) + width + width
        new_points = (
            np.linspace(
                orig_points[0] - width*increment,
                orig_points[-1] + width*increment,
                num_of_new_points,
                dtype=np.float32)
        )
    elif method == 'remove':
        end_width = -width if width != 0 else None
        new_points = np.float32(orig_points[width:end_width])
    new_points = new_points.astype(orig_points.dtype)

    new_points_bounds = np.array(
        [new_points - 0.5*increment, new_points + 0.5*increment],
        dtype=np.float32).T
    return coord.copy(points=new_points, bounds=new_points_bounds)


def create_cube_with_new_data(cube, data, coord_x, coord_y):
    """
    Create a cube with newly created data where the metadata is copied from
    the input cube and the supplied x and y coordinates are added to the
    cube.

    Args:
        cube (Iris.cube.Cube):
            Template cube used for copying metadata and non x and y axes
            coordinates.
        data (Numpy array):
            Data to be put into the new cube.
        coord_x (Iris.coords.DimCoord):
            Coordinate to be added to the new cube to represent the x axis.
        coord_y (Iris.coords.DimCoord):
            Coordinate to be added to the new cube to represent the y axis.

    Returns:
        new_cube (Iris.cube.Cube):
            Cube built from the template cube using the requested data and
            the supplied x and y axis coordinates.
    """
    check_for_x_and_y_axes(cube)

    yname = cube.coord(axis='y').name()
    xname = cube.coord(axis='x').name()
    ycoord_dim = cube.coord_dims(yname)
    xcoord_dim = cube.coord_dims(xname)
    metadata_dict = deepcopy(cube.metadata._asdict())
    new_cube = iris.cube.Cube(data, **metadata_dict)
    for coord in cube.coords():
        if coord.name() not in [yname, xname]:
            if cube.coords(coord, dim_coords=True):
                coord_dim = cube.coord_dims(coord)
                new_cube.add_dim_coord(coord, coord_dim)
            else:
                new_cube.add_aux_coord(coord)
    if len(xcoord_dim) > 0:
        new_cube.add_dim_coord(coord_x, xcoord_dim)
    else:
        new_cube.add_aux_coord(coord_x)
    if len(ycoord_dim) > 0:
        new_cube.add_dim_coord(coord_y, ycoord_dim)
    else:
        new_cube.add_aux_coord(coord_y)
    return new_cube


def pad_cube_with_halo(cube, width_x, width_y, masked_halo=False):
    """
    Method to pad a halo around the data in an iris cube. Normally the
    masked_halo should be zero as it is considered masked data however if
    masked_halo is False then the padding calculates the mean within the
    neighbourhood radius in grid cells i.e. the neighbourhood width at
    the edge of the data and uses this mean value as the padding value.

    Args:
        cube (iris.cube.Cube):
            The original cube prior to applying padding. The cube should
            contain only x and y dimensions, so will generally be a slice
            of a cube.
        width_x, width_y (int):
            The width in x and y directions of the neighbourhood radius in
            grid cells. This will be the width of padding to be added to
            the numpy array.
        masked_halo (bool):
            masked_halo = True means that the halo will be set to 0.0,
            otherwise the halo will be filled with mean values.

    Returns:
        padded_cube (iris.cube.Cube):
            Cube containing the new padded cube, with appropriate
            changes to the cube's dimension coordinates.
    """
    check_for_x_and_y_axes(cube)

    # Pad a halo around the original data with the extent of the halo
    # given by width_y and width_x.
    if masked_halo:
        padded_data = np.pad(
            cube.data,
            ((width_y, width_y), (width_x, width_x)),
            "constant", constant_values=(0.0, 0.0))
    else:
        padded_data = np.pad(
            cube.data,
            ((width_y, width_y), (width_x, width_x)),
            "mean", stat_length=((width_y, width_y), (width_x, width_x)))
    coord_x = cube.coord(axis='x')
    padded_x_coord = pad_coord(coord_x, width_x, 'add')
    coord_y = cube.coord(axis='y')
    padded_y_coord = pad_coord(coord_y, width_y, 'add')
    padded_cube = create_cube_with_new_data(
        cube, padded_data, padded_x_coord, padded_y_coord)

    return padded_cube


def remove_halo_from_cube(cube, width_x, width_y):
    """
    Method to remove rows/columns from the edge of an iris cube.
    Used to 'unpad' cubes which have been previously padded by
    pad_cube_with_halo.

    Args:
        cube (iris.cube.Cube):
            The original cube to be trimmed of edge data. The cube should
            contain only x and y dimensions, so will generally be a slice
            of a cube.
        width_x, width_y (int):
            The width in x and y directions of the neighbourhood radius in
            grid cells. This will be the width removed from the numpy
            array.

    Returns:
        trimmed_cube (iris.cube.Cube):
            Cube containing the new trimmed cube, with appropriate
            changes to the cube's dimension coordinates.
    """
    check_for_x_and_y_axes(cube)

    end_y = -width_y if width_y != 0 else None
    end_x = -width_x if width_x != 0 else None
    trimmed_data = cube.data[width_y:end_y, width_x:end_x]
    coord_x = cube.coord(axis='x')
    trimmed_x_coord = pad_coord(coord_x, width_x, 'remove')
    coord_y = cube.coord(axis='y')
    trimmed_y_coord = pad_coord(coord_y, width_y, 'remove')
    trimmed_cube = create_cube_with_new_data(
        cube, trimmed_data, trimmed_x_coord, trimmed_y_coord)
    return trimmed_cube
