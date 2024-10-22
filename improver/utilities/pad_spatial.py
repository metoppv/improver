# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Utilities for spatial padding of iris cubes."""

from copy import deepcopy

import iris
import numpy as np
from cf_units import Unit
from iris.coords import Coord, DimCoord
from iris.cube import Cube
from numpy import ndarray

from improver.utilities.cube_checker import check_for_x_and_y_axes
from improver.utilities.cube_manipulation import (
    enforce_coordinate_ordering,
    get_dim_coord_names,
)
from improver.utilities.spatial import distance_to_number_of_grid_cells


def pad_coord(coord: Coord, width: int, method: str) -> Coord:
    """
    Construct a new coordinate by extending the current coordinate by the
    padding width.

    Args:
        coord:
            Original coordinate which will be used as the basis of the
            new extended coordinate.
        width:
            The width of padding in grid cells (the extent of the
            neighbourhood radius in grid cells in a given direction).
        method:
            A string determining whether the coordinate is being expanded
            or contracted. Options: 'remove' to remove points from coord;
            'add' to add points to coord.

    Returns:
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
        msg = "Non-uniform increments between grid points: {}.".format(increment)
        raise ValueError(msg)

    if method == "add":
        num_of_new_points = len(orig_points) + width + width
        new_points = np.linspace(
            orig_points[0] - width * increment,
            orig_points[-1] + width * increment,
            num_of_new_points,
            dtype=np.float32,
        )
    elif method == "remove":
        end_width = -width if width != 0 else None
        new_points = np.float32(orig_points[width:end_width])
    new_points = new_points.astype(orig_points.dtype)

    new_points_bounds = np.array(
        [new_points - 0.5 * increment, new_points + 0.5 * increment], dtype=np.float32
    ).T
    return coord.copy(points=new_points, bounds=new_points_bounds)


def create_cube_with_halo(cube: Cube, halo_radius: float) -> Cube:
    """
    Create a template cube defining a new grid by adding a fixed width halo
    on all sides to the input cube grid.  The cube contains no meaningful
    data.

    Args:
        cube:
            Cube on original grid
        halo_radius:
            Size of border to pad original grid, in metres

    Returns:
        New cube defining the halo-padded grid (data set to zero)
    """
    halo_size_x = distance_to_number_of_grid_cells(cube, halo_radius, axis="x")
    halo_size_y = distance_to_number_of_grid_cells(cube, halo_radius, axis="y")

    # create padded x- and y- coordinates
    x_coord = pad_coord(cube.coord(axis="x"), halo_size_x, "add")
    y_coord = pad_coord(cube.coord(axis="y"), halo_size_y, "add")

    halo_cube = iris.cube.Cube(
        np.zeros((len(y_coord.points), len(x_coord.points)), dtype=np.float32),
        long_name="grid_with_halo",
        units=Unit("no_unit"),
        dim_coords_and_dims=[(y_coord, 0), (x_coord, 1)],
    )

    return halo_cube


def _create_cube_with_padded_data(
    source_cube: Cube, data: ndarray, coord_x: DimCoord, coord_y: DimCoord
) -> Cube:
    """
    Create a cube with newly created data where the metadata is copied from
    the input cube and the supplied x and y coordinates are added to the
    cube.

    Args:
        source_cube:
            Template cube used for copying metadata and non x and y axes
            coordinates.
        data:
            Data to be put into the new cube.
        coord_x:
            Coordinate to be added to the new cube to represent the x axis.
        coord_y:
            Coordinate to be added to the new cube to represent the y axis.

    Returns:
        Cube built from the template cube using the requested data and
        the supplied x and y axis coordinates.
    """
    check_for_x_and_y_axes(source_cube)

    yname = source_cube.coord(axis="y").name()
    xname = source_cube.coord(axis="x").name()
    ycoord_dim = source_cube.coord_dims(yname)
    xcoord_dim = source_cube.coord_dims(xname)

    # inherit metadata (cube name, units, attributes etc)
    metadata_dict = deepcopy(source_cube.metadata._asdict())
    new_cube = iris.cube.Cube(data, **metadata_dict)

    # inherit non-spatial coordinates
    for coord in source_cube.coords():
        if coord.name() not in [yname, xname]:
            if source_cube.coords(coord, dim_coords=True):
                coord_dim = source_cube.coord_dims(coord)
                new_cube.add_dim_coord(coord, coord_dim)
            else:
                new_cube.add_aux_coord(coord)

    # update spatial coordinates
    if len(xcoord_dim) > 0:
        new_cube.add_dim_coord(coord_x, xcoord_dim)
    else:
        new_cube.add_aux_coord(coord_x)

    if len(ycoord_dim) > 0:
        new_cube.add_dim_coord(coord_y, ycoord_dim)
    else:
        new_cube.add_aux_coord(coord_y)

    return new_cube


def pad_cube_with_halo(
    cube: Cube, width_x: int, width_y: int, pad_method: str = "constant"
) -> Cube:
    """
    Method to pad a halo around the data in an iris cube.  If halo_with_data
    is False, the halo is filled with zeros.  Otherwise the padding calculates
    a mean within half the padding width with which to fill the halo region.

    Args:
        cube:
            The original cube prior to applying padding. The cube should
            contain only x and y dimensions, so will generally be a slice
            of a cube.
        width_x:
            The width in x directions of the neighbourhood radius in
            grid cells. This will be the width of padding to be added to
            the numpy array.
        width_y:
            The width in y directions of the neighbourhood radius in
            grid cells. This will be the width of padding to be added to
            the numpy array.
        pad_method:
            The numpy.pad method with which to populate the halo. The default
            is 'constant' which will populate the region with zeros. All other
            np.pad methods are accepted, though they are not fully configurable.

    Returns:
        Cube containing the new padded cube, with appropriate
        changes to the cube's dimension coordinates.
    """
    check_for_x_and_y_axes(cube)

    # Pad a halo around the original data with the extent of the halo
    # given by width_y and width_x.
    kwargs = {
        "stat_length": ((width_y // 2, width_y // 2), (width_x // 2, width_x // 2))
    }
    if pad_method == "constant":
        kwargs = {"constant_values": (0.0, 0.0)}
    if pad_method == "symmetric":
        kwargs = {}

    padded_data = np.pad(
        cube.data, ((width_y, width_y), (width_x, width_x)), mode=pad_method, **kwargs
    )

    coord_x = cube.coord(axis="x")
    padded_x_coord = pad_coord(coord_x, width_x, "add")
    coord_y = cube.coord(axis="y")
    padded_y_coord = pad_coord(coord_y, width_y, "add")
    padded_cube = _create_cube_with_padded_data(
        cube, padded_data, padded_x_coord, padded_y_coord
    )

    return padded_cube


def remove_cube_halo(cube: Cube, halo_radius: float) -> Cube:
    """
    Remove halo of halo_radius from a cube.

    This function converts the halo radius into
    the number of grid points in the x and y coordinate
    that need to be removed. It then calls remove_halo_from_cube
    which only acts on a cube with x and y coordinates so we
    need to slice the cube and them merge the cube back together
    ensuring the resulting cube has the same dimension coordinates.

    Args:
        cube:
            Cube on extended grid
        halo_radius:
            Size of border to remove, in metres

    Returns:
        New cube with the halo removed.
    """
    halo_size_x = distance_to_number_of_grid_cells(cube, halo_radius, axis="x")
    halo_size_y = distance_to_number_of_grid_cells(cube, halo_radius, axis="y")

    result_slices = iris.cube.CubeList()
    for cube_slice in cube.slices([cube.coord(axis="y"), cube.coord(axis="x")]):
        cube_halo = remove_halo_from_cube(cube_slice, halo_size_x, halo_size_y)
        result_slices.append(cube_halo)
    result = result_slices.merge_cube()

    # re-promote any scalar dimensions lost in slice / merge
    req_dims = get_dim_coord_names(cube)
    present_dims = get_dim_coord_names(result)
    for coord in req_dims:
        if coord not in present_dims:
            result = iris.util.new_axis(result, coord)

    # re-order (needed if scalar dimensions have been re-added)
    enforce_coordinate_ordering(result, req_dims)

    return result


def remove_halo_from_cube(cube: Cube, width_x: float, width_y: float) -> Cube:
    """
    Method to remove rows/columns from the edge of an iris cube.
    Used to 'unpad' cubes which have been previously padded by
    pad_cube_with_halo.

    Args:
        cube:
            The original cube to be trimmed of edge data. The cube should
            contain only x and y dimensions, so will generally be a slice
            of a cube.
        width_x:
            The width in x directions of the neighbourhood radius in
            grid cells. This will be the width of padding to be added to
            the numpy array.
        width_y:
            The width in y directions of the neighbourhood radius in
            grid cells. This will be the width of padding to be added to
            the numpy array.

    Returns:
        Cube containing the new trimmed cube, with appropriate
        changes to the cube's dimension coordinates.
    """
    check_for_x_and_y_axes(cube)

    end_y = -width_y if width_y != 0 else None
    end_x = -width_x if width_x != 0 else None
    trimmed_data = cube.data[width_y:end_y, width_x:end_x]
    coord_x = cube.coord(axis="x")
    trimmed_x_coord = pad_coord(coord_x, width_x, "remove")
    coord_y = cube.coord(axis="y")
    trimmed_y_coord = pad_coord(coord_y, width_y, "remove")
    trimmed_cube = _create_cube_with_padded_data(
        cube, trimmed_data, trimmed_x_coord, trimmed_y_coord
    )
    return trimmed_cube
