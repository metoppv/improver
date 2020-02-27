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
"""Module containing plugin for CubeCombiner."""

import numpy as np

from improver import BasePlugin
from improver.utilities.cube_manipulation import expand_bounds


class CubeCombiner(BasePlugin):

    """Plugin for combining cubes.

    """

    COMBINE_OPERATORS = {
        "+": np.add,
        "add": np.add,
        "-": np.subtract,
        "subtract": np.subtract,
        "*": np.multiply,
        "multiply": np.multiply,
        "max": np.maximum,
        "min": np.minimum,
        "mean": np.add}  # mean is calculated in two steps: sum and normalise

    def __init__(self, operation, warnings_on=False):
        """
        Create a CubeCombiner plugin

        Args:
            operation (str):
                Operation (+, - etc) to apply to the incoming cubes.
            warnings_on (bool):
                If True output warnings for mismatching metadata.

        Raises:
            ValueError: Unknown operation.

        """
        try:
            self.operator = self.COMBINE_OPERATORS[operation]
        except KeyError:
            msg = 'Unknown operation {}'.format(operation)
            raise ValueError(msg)
        self.operation = operation
        self.warnings_on = warnings_on

    def __repr__(self):
        """Represent the configured plugin instance as a string."""
        desc = ('<CubeCombiner: operation=' +
                '{}, warnings_on = {}>'.format(self.operation,
                                               self.warnings_on))
        return desc

    @staticmethod
    def _check_dimensions_match(cube_list):
        """
        Check all coordinate dimensions on the input cubes are equal

        Args:
            cube_list (iris.cube.CubeList or list):
                List of cubes to compare

        Raises:
            ValueError: If dimension coordinates do not match
        """
        ref_coords = cube_list[0].coords(dim_coords=True)
        for cube in cube_list[1:]:
            coords = cube.coords(dim_coords=True)
            compare = [a == b for a, b in zip(coords, ref_coords)]
            if not np.all(compare):
                msg = ("Cannot combine cubes with different dimensions:\n"
                       "{} and {}".format(repr(cube_list[0]), repr(cube)))
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
            coord.name() for coord in cube_list[0].coords(dim_coords=False)}
        for cube in cube_list[1:]:
            cube_scalar_coords = {
                coord.name() for coord in cube.coords(dim_coords=False)}
            shared_scalar_coords = shared_scalar_coords & cube_scalar_coords

        expanded_coords = []
        for cube in cube_list[1:]:
            for coord in shared_scalar_coords:
                if (cube.coord(coord) != cube_list[0].coord(coord) and
                        coord not in expanded_coords):
                    expanded_coords.append(coord)
        return expanded_coords

    def process(self, cube_list, new_diagnostic_name, use_midpoint=False):
        """
        Combine data and metadata from a list of input cubes into a single
        cube, using the specified operation to combine the cube data.

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
            ValueError: If the cubelist contains only one cube.
        """
        if len(cube_list) < 2:
            msg = 'Expecting 2 or more cubes in cube_list'
            raise ValueError(msg)

        self._check_dimensions_match(cube_list)

        # perform operation (add, subtract, min, max, multiply) cumulatively
        result = cube_list[0].copy()
        for cube in cube_list[1:]:
            result.data = self.operator(result.data, cube.data)

        # normalise mean (for which self.operator is np.add)
        if self.operation == 'mean':
            result.data = result.data / len(cube_list)

        # update any coordinates that have been expanded, and rename output
        expanded_coord_names = self._get_expanded_coord_names(cube_list)
        if expanded_coord_names:
            result = expand_bounds(result, cube_list, expanded_coord_names,
                                   use_midpoint=use_midpoint)
        result.rename(new_diagnostic_name)

        return result
