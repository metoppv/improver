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
""" Provides support utilities."""

import copy
from iris.coords import CellMethod, DimCoord
from iris.cube import Cube
from iris.exceptions import InvalidCubeError
import numpy as np


class DiscreteDifferenceBetweenAdjacentGridSquares(object):

    """
    Calculate the discrete difference between adjacent grid squares within
    a cube. The discrete difference is calculated along the x and y axis
    individually.
    """

    def __init__(self):
        """
        Initialise class.
        """
        pass

    @staticmethod
    def create_discrete_difference_cube(
            cube, coord_name, diffs_along_axis):
        """
        Put the discrete difference array into a cube with the appropriate
        metadata.

        Parameters
        ----------
        cube : Iris.cube.Cube
            Cube from which the discrete differences will be calculated.
        coord_name : String
            The name of the coordinate over which the discrete difference
            will be calculated.
        diffs_along_axis : numpy array
            Array containing the discrete differences.

        Returns
        -------
        diff_cube : Iris.cube.Cube
            Cube after the discrete differences have been calculated along the
            specified axis.
        """
        points = cube.coord(coord_name).points
        mean_points = (points[1:] + points[:-1]) / 2

        # Copy cube metadata and coordinates into a new cube.
        # Create a new coordinate for the coordinate along which the discrete
        # difference has been calculated.
        metadata_dict = copy.deepcopy(cube.metadata._asdict())
        diff_cube = Cube(diffs_along_axis, **metadata_dict)

        for coord in cube.dim_coords:
            dims = cube.coord_dims(coord)
            if coord.name() in [coord_name]:
                coord = DimCoord(
                    mean_points, standard_name=coord.standard_name,
                    units=coord.units)
            diff_cube.add_dim_coord(coord.copy(), dims)
        for coord in cube.aux_coords:
            dims = cube.coord_dims(coord)
            diff_cube.add_aux_coord(coord.copy(), dims)
        for coord in cube.derived_coords:
            dims = cube.coord_dims(coord)
            diff_cube.add_aux_coord(coord.copy(), dims)

        # Add metadata to indicate that a discrete difference has been
        # calculated.
        cell_method = CellMethod("discrete_difference", coords=[coord_name])
        diff_cube.add_cell_method(cell_method)
        diff_cube.attributes["direction_of_discrete_difference"] = (
            "forward_difference")
        return diff_cube

    def calculate_discrete_difference(self, cube, coord_axis):
        """
        Calculate the discrete difference along the axis specified by the
        coordinate.

        Parameters
        ----------
        cube : Iris.cube.Cube
            Cube from which the discrete differences will be calculated.
        coord_axis : String
            Short-hand reference for the x or y coordinate, as allowed by
            iris.util.guess_coord_axis.

        Returns
        -------
        diff_along_cube : Iris.cube.Cube
            Cube after the discrete differences have been calculated along the
            specified axis.
        """
        coord_name = cube.coord(axis=coord_axis).name()
        diff_axis = cube.coord_dims(coord_name)[0]
        diffs_along_axis = np.diff(cube.data, axis=diff_axis)
        diff_along_cube = self.create_discrete_difference_cube(
            cube, coord_name, diffs_along_axis)
        return diff_along_cube

    def process(self, cube):
        """
        Calculate the discrete difference along the x and y axes and return
        the result in separate cubes.

        Parameters
        ----------
        cube : Iris.cube.Cube
            Cube from which the discrete differences will be calculated.

        Returns
        -------
        diff_along_y_cube : Iris.cube.Cube
            Cube after the discrete differences have been calculated along the
            y axis.
        diff_along_x_cube : Iris.cube.Cube
            Cube after the discrete differences have been calculated along the
            x axis.
        """
        if cube.ndim > 2:
            msg = ("The input cube must have two dimensions: y and x."
                   "Instead the input cube was {}".format(cube))
            raise InvalidCubeError(msg)
        diff_along_y_cube = self.calculate_discrete_difference(cube, "y")
        diff_along_x_cube = self.calculate_discrete_difference(cube, "x")
        return diff_along_x_cube, diff_along_y_cube
