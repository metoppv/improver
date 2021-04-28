# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown Copyright 2017-2021 Met Office.
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

from operator import eq
from typing import Callable, List, Union

import iris
import numpy as np
from iris.coords import DimCoord
from iris.cube import Cube, CubeList
from iris.exceptions import CoordinateNotFoundError

from improver import BasePlugin
from improver.metadata.check_datatypes import enforce_dtype
from improver.metadata.probabilistic import (
    find_threshold_coordinate,
    get_diagnostic_cube_name_from_probability_name,
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

    def __init__(self, operation: str) -> None:
        """Create a CubeCombiner plugin

        Args:
            operation:
                Operation (+, - etc) to apply to the incoming cubes.

        Raises:
            ValueError: if operation is not recognised in dictionary
        """
        try:
            self.operator = self.COMBINE_OPERATORS[operation]
        except KeyError:
            msg = "Unknown operation {}".format(operation)
            raise ValueError(msg)

        self.normalise = operation == "mean"

    @staticmethod
    def _check_dimensions_match(
        cube_list: Union[List[Cube], CubeList], comparators: List[Callable] = [eq],
    ) -> None:
        """
        Check all coordinate dimensions on the input cubes match according to
        the comparators specified.

        Args:
            cube_list:
                List of cubes to compare
            comparators:
                Comparison operators, at least one of which must return "True"
                for each coordinate in order for the match to be valid

        Raises:
            ValueError: If dimension coordinates do not match
        """
        ref_coords = cube_list[0].coords(dim_coords=True)
        for cube in cube_list[1:]:
            coords = cube.coords(dim_coords=True)
            compare = [
                np.any([comp(a, b) for comp in comparators])
                for a, b in zip(coords, ref_coords)
            ]
            if not np.all(compare):
                msg = (
                    "Cannot combine cubes with different dimensions:\n"
                    "{} and {}".format(repr(cube_list[0]), repr(cube))
                )
                raise ValueError(msg)

    @staticmethod
    def _get_expanded_coord_names(cube_list: Union[List[Cube], CubeList]) -> List[str]:
        """
        Get names of coordinates whose bounds need expanding and points
        recalculating after combining cubes. These are the scalar coordinates
        that are present on all input cubes, but have different values.

        Args:
            cube_list:
                List of cubes to that will be combined

        Returns:
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

    def _combine_cube_data(self, cube_list: Union[List[Cube], CubeList]) -> Cube:
        """
        Perform cumulative operation to combine cube data

        Args:
            cube_list

        Returns:
            Combined cube

        Raises:
            TypeError: if the operation results in an escalated datatype
        """
        result = cube_list[0].copy()
        for cube in cube_list[1:]:
            result.data = self.operator(result.data, cube.data)

        if self.normalise:
            result.data = result.data / len(cube_list)

        enforce_dtype(str(self.operator), cube_list, result)

        return result

    def process(
        self, cube_list: Union[List[Cube], CubeList], new_diagnostic_name: str,
    ) -> Cube:
        """
        Combine data and metadata from a list of input cubes into a single
        cube, using the specified operation to combine the cube data.  The
        first cube in the input list provides the template for the combined
        cube metadata.
        If coordinates are expanded as a result of this combine operation
        (e.g. expanding time for accumulations / max in period) the upper bound
        of the new coordinate will also be used as the point for the new coordinate.

        Args:
            cube_list:
                List of cubes to combine.
            new_diagnostic_name:
                New name for the combined diagnostic.

        Returns:
            Cube containing the combined data.

        Raises:
            ValueError: If the cube_list contains only one cube.
        """
        if len(cube_list) < 2:
            msg = "Expecting 2 or more cubes in cube_list"
            raise ValueError(msg)

        self._check_dimensions_match(cube_list)
        result = self._combine_cube_data(cube_list)
        expanded_coord_names = self._get_expanded_coord_names(cube_list)
        if expanded_coord_names:
            result = expand_bounds(result, cube_list, expanded_coord_names)
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

    def __init__(self) -> None:
        """Create a CubeMultiplier plugin"""
        self.operator = np.multiply
        self.normalise = False

    def _setup_coords_for_broadcast(self, cube_list: CubeList) -> CubeList:
        """
        Adds a scalar threshold to any subsequent cube in cube_list so that they all
        match the dimensions, in order, of the first cube in the list

        Args:
            cube_list

        Returns:
            Updated version of cube_list

        Raises:
            CoordinateNotFoundError: if there is no threshold coordinate on the
                first cube in the list
            TypeError: if there is a scalar threshold coordinate on any of the
                later cubes, which would indicate that the cube is only valid for
                a single threshold and should not be broadcast to all thresholds.
        """
        target_cube = cube_list[0]
        try:
            target_coord = find_threshold_coordinate(target_cube)
        except CoordinateNotFoundError:
            raise CoordinateNotFoundError(
                f"Cannot find coord threshold in {repr(target_cube)} to broadcast to"
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
                    msg = "Cannot broadcast to coord threshold as it already exists as an AuxCoord"
                    raise TypeError(msg)
            new_list.append(cube)

        return new_list

    @staticmethod
    def _coords_are_broadcastable(coord1: DimCoord, coord2: DimCoord) -> bool:
        """
        Broadcastable coords will differ only in length, so create a copy of one with
        the points and bounds of the other and compare. Also ensure length of at least
        one of the coords is 1.
        """
        coord_copy = coord1.copy(coord2.points, bounds=coord2.bounds)

        return (coord_copy == coord2) and (
            (len(coord1.points) == 1) or (len(coord2.points) == 1)
        )

    def process(
        self,
        cube_list: Union[List[Cube], CubeList],
        new_diagnostic_name: str,
        broadcast_to_threshold: bool = False,
    ) -> Cube:
        """
        Multiply data from a list of input cubes into a single cube.  The first
        cube in the input list provides the combined cube metadata.

        Args:
            cube_list:
                List of cubes to combine.
            new_diagnostic_name:
                New name for the combined diagnostic.  This should be the diagnostic
                name, eg rainfall_rate or rainfall_rate_in_vicinity, rather than the
                name of the probabilistic output cube.
            broadcast_to_threshold:
                True if the first cube has a threshold coordinate to which the
                following cube(s) need(s) to be broadcast prior to combining data.

        Returns:
            Cube containing the combined data.

        Raises:
            ValueError: If the cube_list contains only one cube.
            TypeError: If combining data results in float64 data.
        """
        if len(cube_list) < 2:
            msg = "Expecting 2 or more cubes in cube_list"
            raise ValueError(msg)

        if broadcast_to_threshold:
            cube_list = self._setup_coords_for_broadcast(cube_list)

        self._check_dimensions_match(
            cube_list, comparators=[eq, self._coords_are_broadcastable]
        )

        result = self._combine_cube_data(cube_list)

        if broadcast_to_threshold:
            probabilistic_name = cube_list[0].name()
            diagnostic_name = get_diagnostic_cube_name_from_probability_name(
                probabilistic_name
            )

            # Rename the threshold coordinate to match the name of the diagnostic
            # that results from the combine operation.
            new_threshold_name = new_diagnostic_name.replace("_in_vicinity", "")
            result.coord(var_name="threshold").rename(new_threshold_name)
            result.coord(new_threshold_name).var_name = "threshold"

            new_diagnostic_name = probabilistic_name.replace(
                diagnostic_name, new_diagnostic_name
            )

        result.rename(new_diagnostic_name)

        return result
