# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown Copyright 2017-2020 Met Office.
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
"""Module containing plugins for combining cubes"""

import iris
import numpy as np
from iris.cube import CubeList
from iris.exceptions import CoordinateNotFoundError

from improver import BasePlugin
from improver.metadata.check_datatypes import enforce_dtype
from improver.metadata.probabilistic import (
    extract_diagnostic_name,
    find_threshold_coordinate,
)
from improver.utilities.cube_manipulation import (
    enforce_coordinate_ordering,
    expand_bounds,
)


class CubeCombiner(BasePlugin):
    """Plugin for combining cubes using linear operators"""

    COMBINE_OPERATORS = {
        "+": np.add,
        "add": np.add,
        "-": np.subtract,
        "subtract": np.subtract,
        "max": np.maximum,
        "min": np.minimum,
        "mean": np.add,
    }  # mean is calculated in two steps: sum and normalise

    def __init__(self, operation):
        """Create a CubeCombiner plugin

        Args:
            operation (str):
                Operation (+, - etc) to apply to the incoming cubes.

        Raises:
            ValueError: if operation is not recognised in dictionary
        """
        try:
            self.operator = self.COMBINE_OPERATORS[operation]
        except KeyError:
            msg = "Unknown operation {}".format(operation)
            raise ValueError(msg)
        self.operation = operation

    @staticmethod
    def _coords_are_broadcastable(coord1, coord2):
        """
        Broadcastable coords will differ only in length, so create a copy of one with
        the points and bounds of the other and compare. Also ensure length of at least
        one of the coords is 1.
        """
        coord_copy = coord1.copy(coord2.points, bounds=coord2.bounds)

        return (coord_copy == coord2) and (
            (len(coord1.points) == 1) or (len(coord2.points) == 1)
        )

    def _check_dimensions_match(self, cube_list):
        """
        Check all coordinate dimensions on the input cubes are equal or broadcastable

        Args:
            cube_list (iris.cube.CubeList or list):
                List of cubes to compare

        Raises:
            ValueError: If dimension coordinates do not match
        """
        ref_coords = cube_list[0].coords(dim_coords=True)
        for cube in cube_list[1:]:
            coords = cube.coords(dim_coords=True)
            compare = [
                (a == b) or self._coords_are_broadcastable(a, b)
                for a, b in zip(coords, ref_coords)
            ]
            if not np.all(compare):
                msg = (
                    "Cannot combine cubes with different dimensions:\n"
                    "{} and {}".format(repr(cube_list[0]), repr(cube))
                )
                raise ValueError(msg)

    @staticmethod
    def _get_expanded_coord_names(cube_list):
        """
        Get names of coordinates whose bounds need expanding and points
        recalculating after combining cubes. These are the scalar coordinates
        that are present on all input cubes, but have different values.

        Args:
            cube_list (iris.cube.CubeList or list):
                List of cubes to that will be combined

        Returns:
            list of str:
                List of coordinate names to expand
        """
        shared_scalar_coords = {
            coord.name() for coord in cube_list[0].coords(dim_coords=False)
        }
        for cube in cube_list[1:]:
            cube_scalar_coords = {
                coord.name() for coord in cube.coords(dim_coords=False)
            }
            shared_scalar_coords = shared_scalar_coords & cube_scalar_coords

        expanded_coords = []
        for cube in cube_list[1:]:
            for coord in shared_scalar_coords:
                if (
                    cube.coord(coord) != cube_list[0].coord(coord)
                    and coord not in expanded_coords
                ):
                    expanded_coords.append(coord)
        return expanded_coords

    def _combine_cube_data(self, cube_list):
        """
        Perform cumulative operation to combine cube data

        Args:
            cube_list (iris.cube.CubeList or list)

        Returns:
            iris.cube.Cube

        Raises:
            ValueError: if the operation results in an escalated datatype
        """
        result = cube_list[0].copy()
        for cube in cube_list[1:]:
            result.data = self.operator(result.data, cube.data)

        if self.operation == "mean":
            result.data = result.data / len(cube_list)

        enforce_dtype(self.operation, cube_list, result)

        return result

    def process(
        self, cube_list, new_diagnostic_name, use_midpoint=False,
    ):
        """
        Combine data and metadata from a list of input cubes into a single
        cube, using the specified operation to combine the cube data.  The
        first cube in the input list provides the template for the combined
        cube metadata.

        Args:
            cube_list (iris.cube.CubeList or list):
                List of cubes to combine.
            new_diagnostic_name (str):
                New name for the combined diagnostic.
            use_midpoint (bool):
                Determines the nature of the points and bounds for expanded
                coordinates.  If False, the upper bound of the coordinate is
                used as the point values.  If True, the midpoint is used.

        Returns:
            iris.cube.Cube:
                Cube containing the combined data.

        Raises:
            ValueError: If the cube_list contains only one cube.
            TypeError: If combining data results in float64 data.
        """
        if len(cube_list) < 2:
            msg = "Expecting 2 or more cubes in cube_list"
            raise ValueError(msg)

        self._check_dimensions_match(cube_list)
        result = self._combine_cube_data(cube_list)
        expanded_coord_names = self._get_expanded_coord_names(cube_list)
        if expanded_coord_names:
            result = expand_bounds(
                result, cube_list, expanded_coord_names, use_midpoint=use_midpoint
            )
        result.rename(new_diagnostic_name)
        return result


class CubeMultiplier(CubeCombiner):
    """Class to multiply input cubes

    The behaviour for the "multiply" operation is different from
    other types of cube combination.  The only valid use case for
    "multiply" is to apply a factor that conditions an input probability
    field - that is, to apply Bayes Theorem.  The input probability is
    therefore used as the source of ALL input metadata, and should always
    be the first cube in the input list.  The factor(s) by which this is
    multiplied are not compared for any mis-match in scalar coordinates.

    """

    def __init__(self):
        """Create a CubeMultiplier plugin"""
        self.operator = np.multiply
        self.operation = "multiply"
        self.broadcast_coords = None

    def _setup_coords_for_broadcast(self, cube_list):
        """
        Adds a scalar DimCoord to any subsequent cube in cube_list so that they all include all of
        the coords specified in self.broadcast_coords in the right order.

        Args:
            cube_list: (iris.cube.CubeList)

        Returns:
            iris.cube.CubeList
                Updated version of cube_list

        """
        for coord in self.broadcast_coords:
            target_cube = cube_list[0]
            try:
                if coord == "threshold":
                    target_coord = find_threshold_coordinate(target_cube)
                else:
                    target_coord = target_cube.coord(coord)
            except CoordinateNotFoundError:
                raise CoordinateNotFoundError(
                    f"Cannot find coord {coord} in {repr(target_cube)} to broadcast to."
                )
            new_list = CubeList([])
            for cube in cube_list:
                try:
                    found_coord = cube.coord(target_coord)
                except CoordinateNotFoundError:
                    new_coord = target_coord.copy([0], bounds=None)
                    cube = cube.copy()
                    cube.add_aux_coord(new_coord, None)
                    cube = iris.util.new_axis(cube, new_coord)
                    enforce_coordinate_ordering(
                        cube, [d.name() for d in target_cube.coords(dim_coords=True)]
                    )
                else:
                    if found_coord not in cube.dim_coords:
                        # We don't expect the coord to already exist in a scalar form as
                        # this would indicate that the broadcast-from cube is only valid
                        # for part of the new dimension and therefore should be rejected.
                        raise TypeError(
                            f"Cannot broadcast to coord {coord} as it already exists as an AuxCoord"
                        )
                new_list.append(cube)
            cube_list = new_list
        return cube_list

    def process(
        self, cube_list, new_diagnostic_name, broadcast_to_coords=None,
    ):
        """
        Combine data and metadata from a list of input cubes into a single
        cube, using the specified operation to combine the cube data.  The
        first cube in the input list provides the template for the combined
        cube metadata.

        NOTE the behaviour for the "multiply" operation is different from
        other types of cube combination.  The only valid use case for
        "multiply" is to apply a factor that conditions an input probability
        field - that is, to apply Bayes Theorem.  The input probability is
        therefore used as the source of ALL input metadata, and should always
        be the first cube in the input list.  The factor(s) by which this is
        multiplied are not compared for any mis-match in scalar coordinates,
        neither do they to contribute to expanded bounds.

        TODO the "multiply" case should be factored out into a separate plugin
        given its substantial differences from other combine use cases.

        Args:
            cube_list (iris.cube.CubeList or list):
                List of cubes to combine.
            new_diagnostic_name (str):
                New name for the combined diagnostic.
            broadcast_to_coords (list):
                Specifies a list of coord names that exist only on the first cube that
                the other cube(s) need(s) broadcasting to prior to the combine.

        Returns:
            iris.cube.Cube:
                Cube containing the combined data.

        Raises:
            ValueError: If the cube_list contains only one cube.
            TypeError: If combining data results in float64 data.
        """
        if len(cube_list) < 2:
            msg = "Expecting 2 or more cubes in cube_list"
            raise ValueError(msg)

        self.broadcast_coords = broadcast_to_coords
        if self.broadcast_coords:
            cube_list = self._setup_coords_for_broadcast(cube_list)
        self._check_dimensions_match(cube_list)

        result = self._combine_cube_data(cube_list)

        if self.broadcast_coords and "threshold" in self.broadcast_coords:
            probabilistic_name = cube_list[0].name()
            diagnostic_name = extract_diagnostic_name(probabilistic_name)

            # Rename the threshold coordinate to match the name of the diagnostic
            # that results from the combine operation.
            result.coord(var_name="threshold").rename(new_diagnostic_name)
            result.coord(new_diagnostic_name).var_name = "threshold"

            new_diagnostic_name = probabilistic_name.replace(
                diagnostic_name, new_diagnostic_name
            )

        result.rename(new_diagnostic_name)

        return result
