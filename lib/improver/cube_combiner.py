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
"""Module containing plugin for CubeCombiner."""

import numpy as np

import iris

from improver.utilities.cube_metadata import (
    resolve_metadata_diff, amend_metadata)


class CubeCombiner(object):

    """Plugin for combining cubes.

    """

    def __init__(self, operation, warnings_on=False):
        """
        Create a CubeCombiner plugin

        Args:
            operation (str):
                Operation (+, - etc) to apply to the incoming cubes.
        Keyword Args:
            warnings_on (bool):
                If True output warnings for mismatching metadata.

        Raises:
            ValueError: Unknown operation.

        """
        possible_operations = ['+', 'add',
                               '-', 'subtract',
                               '*', 'multiply',
                               'max', 'min', 'mean']

        if operation in possible_operations:
            self.operation = operation
        else:
            msg = 'Unknown operation {}'.format(operation)
            raise ValueError(msg)
        self.warnings_on = warnings_on

    def __repr__(self):
        """Represent the configured plugin instance as a string."""
        desc = ('<CubeCombiner: operation=' +
                '{}, warnings_on = {}>'.format(self.operation,
                                               self.warnings_on))
        return desc

    @staticmethod
    def expand_bounds(result_cube, cubelist, coord, point):
        """Alter a coord such that bounds are expanded to cover
        the entire range of the input cubes.

        For example, in the case of time cubes if the input cubes have
        bounds of [0000Z, 0100Z] & [0100Z, 0200Z] then the output cube will
        have bounds of [0000Z,0200Z]

        Args:
            result_cube (iris.cube.Cube):
                A cube with metadata for the results.
            cubelist (iris.cube.CubeList):
                The list of cubes with coordinates to be combined
            coord (str):
                The coordinate to be combined.
            point (str):
                The method of calculating the new point for the coordinate.
                Currently accepts:

                    | 'mid' - halfway between the bounds
                    | 'upper' - equal to the upper bound
        Returns:
            result (iris.cube.Cube):
                Cube with coord expanded.

                n.b. If argument point == 'mid' then python will convert
                result.coord('coord').points[0] to a float.

                This is to ensure that midpoints are not accidentally
                rounded down by Python's default integer divide behavour:
                For example if you combined cubes for accumulation for three
                hours the midpoint should be 1.5 hours into the period.
                Integer behaviour would give 3 / 2 = 1
        """
        if len(result_cube.coord(coord).points) != 1:
            emsg = ('the expand bounds function should only be used on a'
                    'coordinate with a single point. The coordinate \"{}\" '
                    'has {} points.')
            raise ValueError(emsg.format(
                coord,
                len(result_cube.coord(coord).points)))

        bounds = ([cube.coord(coord).bounds for cube in cubelist])
        if any(b is None for b in bounds):
            points = ([cube.coord(coord).points for cube in cubelist])
            new_low_bound = np.min(points)
            new_top_bound = np.max(points)
        else:
            new_low_bound = np.min(bounds)
            new_top_bound = np.max(bounds)
        result_cube.coord(coord).bounds = [[new_low_bound, new_top_bound]]

        if point == 'mid':
            result_cube.coord(coord).points = [((new_top_bound -
                                                 new_low_bound) / 2.) +
                                               new_low_bound]
        elif point == 'upper':
            result_cube.coord(coord).points = [new_top_bound]
        return result_cube

    @staticmethod
    def combine(cube1, cube2, operation):
        """
        Combine cube data

        Args:
            cube1 (iris.cube.Cube):
                Cube containing data to be combined.
            cube2 (iris.cube.Cube):
                Cube containing data to be combined.
            operation (str):
                Operation (+, - etc) to apply to the incoming cubes)

        Returns:
            result (iris.cube.Cube):
                Cube containing the combined data.
        Raises:
            ValueError: Unknown operation.

        """
        result = cube1
        if operation == '+' or operation == 'add' or operation == 'mean':
            result.data = cube1.data + cube2.data
        elif operation == '-' or operation == 'subtract':
            result.data = cube1.data - cube2.data
        elif operation == '*' or operation == 'multiply':
            result.data = cube1.data * cube2.data
        elif operation == 'min':
            result.data = np.minimum(cube1.data, cube2.data)
        elif operation == 'max':
            result.data = np.maximum(cube1.data, cube2.data)
        else:
            msg = 'Unknown operation {}'.format(operation)
            raise ValueError(msg)

        return result

    def process(self, cube_list, new_diagnostic_name,
                revised_coords=None,
                revised_attributes=None,
                expanded_coord=None):
        """
        Create a combined cube.

        Args:
            cube_list (iris.cube.CubeList):
                Cube List contain the cubes to combine.
            new_diagnostic_name (str):
                New name for the combined diagnostic.
        Keyword Args:
            revised_coords (dict or None):
                Revised coordinates for combined cube.
            revised_attributes (dict or None):
                Revised attributes for combined cube.

        Returns:
            result (iris.cube.Cube):
                Cube containing the combined data.

        """
        if not isinstance(cube_list, iris.cube.CubeList):
            msg = ('Expecting data to be an instance of '
                   'iris.cube.CubeList but is'
                   ' {0:s}.'.format(type(cube_list)))
            raise TypeError(msg)
        if len(cube_list) < 2:
            msg = 'Expecting 2 or more cubes in cube_list'
            raise ValueError(msg)

        # resulting cube will be based on the first cube.
        data_type = cube_list[0].dtype
        result = cube_list[0].copy()

        for ind in range(1, len(cube_list)):
            cube1, cube2 = (
                resolve_metadata_diff(result.copy(),
                                      cube_list[ind].copy(),
                                      warnings_on=self.warnings_on))
            result = self.combine(cube1,
                                  cube2,
                                  self.operation)

        if self.operation == 'mean':
            result.data = result.data / len(cube_list)

        # If cube has coord bounds that we want to expand
        if expanded_coord:
            for coord, treatment in expanded_coord.items():
                result = self.expand_bounds(result,
                                            cube_list,
                                            coord=coord,
                                            point=treatment)

        result = amend_metadata(result,
                                new_diagnostic_name,
                                data_type,
                                revised_coords,
                                revised_attributes,
                                warnings_on=self.warnings_on)

        return result
