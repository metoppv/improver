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
"""Module containing plugin for CubeCombiner."""

import numpy as np
import iris
import warnings
from iris import FUTURE

from improver.utilities.cube_manipulation import (compare_attributes,
                                                  compare_coords)

FUTURE.netcdf_promote = True


class CubeCombiner(object):

    """Plugin for combining cubes.

    """

    def __init__(self, operation):
        """
        Create a CubeCombiner plugin

        Args:
            operation (str):
                Operation (+, - etc) to apply to the incoming cubes.

        Raises:
            ValueError: Unknown operation.
        """
        possible_operations = ['+', 'add',
                               '-', 'subtract',
                               '*', 'multiple',
                               'max', 'min', 'mean']
        if operation in possible_operations:
            self.operation = operation
        else:
            msg = 'Unknown operation {}'.format(operation)
            raise ValueError(msg)

    def __repr__(self):
        """Represent the configured plugin instance as a string."""
        desc = '<CubeCombiner: operation={}>'.format(self.operation)
        return desc

    @staticmethod
    def resolve_metadata_diff(cube1, cube2, revised_coords,
                              revised_attributes,
                              warnings_on=False):
        """Resolve any differences in  metadata between cubes.

        Args:
            cube1 (iris.cube.Cube):
                Cube containing data to be combined.
            cube2 (iris.cube.Cube):
                Cube containing data to be combined.
            revised_coords (dict or None):
                Revised coordinates for combined cube.
            revised_attributes (dict or None):
                Revised attributes for combined cube.
            warnings_on (bool):
                If True output warnings for mismatching metadata.
        Returns:
            (tuple): tuple containing
                **result1** (iris.cube.Cube):
                    Cube with corrected Metadata.
                **result2** (iris.cube.Cube):
                    Cube with corrected Metadata.
        """
        cubes = iris.cube.CubeList([cube1, cube2])
        unmatching_attributes = compare_attributes(cubes)
        unmatching_coords = compare_coords(cubes)

        result1 = cube1
        result2 = cube2

        return result1, result2

    @staticmethod
    def amend_metadata(cube,
                       new_diagnostic_name,
                       data_type,
                       revised_coords,
                       revised_attributes,
                       warnings_on=False):
        """Amend the metadata in the combined cube.

        Args:
            cube (iris.cube.Cube):
                Cube containing combined data.
            new_diagnostic_name (str):
                New name for the combined diagnostic.
            data_type (numpy.dtype):
                data type of cube data.
            revised_coords (dict or None):
                Revised coordinates for combined cube.
            revised_attributes (dict or None):
                Revised attributes for combined cube.
        Keyword Args:
            warnings_on (bool):
                If True output warnings for mismatching metadata.
        Returns:
            result (iris.cube.Cube):
                Cube with corrected Metadata.
        """
        result = cube
        result.data = result.data.astype(data_type)
        result.rename(new_diagnostic_name)

        if revised_coords:
            for key in revised_coords:
                # If and exising coordinate.
                if key in [coord.name() for coord in cube.coords()]:
                    new_coord = cube.coord(key)
                    changes = revised_coords[key]
                    if changes == 'delete':
                        if len(new_coord.points) != 1:
                            msg = ("Can only remove a coordinate of len 1"
                                   " coord  = {}".format(key))
                            raise ValueError(msg)
                        result.remove_coord(key)
                        result = iris.util.squeeze(result)
                    else:
                        if 'points' in changes:
                            new_points = np.array(changes['points'])
                            if (len(new_points) ==
                                    len(new_coord.points)):
                                new_coord.points = new_points
                            else:
                                msg = ("Mismatch in points in existing"
                                       " coord and updated metadata for "
                                       " coord {}".format(key))
                                raise ValueError(msg)
                        if 'bounds' in changes:
                            new_bounds = np.array(changes['bounds'])
                            if new_coord.bounds:
                                if (len(new_bounds) ==
                                        len(new_coord.bounds)):
                                    new_coord.bounds = new_bounds
                                else:
                                    msg = ("Mismatch in bounds in existing"
                                           " coord and updated metadata for "
                                           " coord {}".format(key))
                                    raise ValueError(msg)
                            else:
                                new_coord.bounds = new_bounds
                        if 'units' in changes:
                            new_coord.units = changes['units']
                        if warnings_on:
                            msg = ("Updated coordinate "
                                   "{}".format(key) +
                                   "with {}".format(changes))
                            warnings.warn(msg)
                # Adding coord
                else:
                    changes = revised_coords[key]
                    print 'Adding coords here'
                    points = None
                    bounds = None

        if revised_attributes:
            for key in revised_attributes:
                if revised_attributes[key] == 'delete':
                    result.attributes.pop(key)
                    if warnings_on:
                        msg = ("Deleted attribute "
                               "{}".format(key))
                        warnings.warn(msg)
                else:
                    result.attributes[key] = revised_attributes[key]

        return result

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

        if operation == '+' or operation == 'add' or operation == 'mean':
            result = cube1 + cube2
        elif operation == '-' or operation == 'subtract':
            result = cube1 - cube2
        elif operation == '*' or operation == 'multiple':
            result = cube1 * cube2
        elif operation == 'min':
            result = cube1
            result.data = np.minimum(cube1.data, cube2.data)
        elif operation == 'max':
            result = cube1
            result.data = np.maximum(cube1.data, cube2.data)
        else:
            msg = 'Unknown operation {}'.format(operation)
            raise ValueError(msg)

        return result

    def process(self, cube_list, new_diagnostic_name,
                revised_coords=None,
                revised_attributes=None,
                warnings_on=False):
        """
        Create a combined cube.

        Args:
            cube_list (iris.cube.CubeList):
                Cube List contain the cubes to combine.
            new_diagnostic_name (str):
                New name for the combined diagnostic.
            revised_coords (dict or None):
                Revised coordinates for combined cube.
            revised_attributes (dict or None):
                Revised attributes for combined cube.
            warnings_on (bool):
                True output warnings for mismatching metadata.

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
                self.resolve_metadata_diff(result.copy(),
                                           cube_list[ind].copy(),
                                           revised_coords,
                                           revised_attributes,
                                           warnings_on))
            result = self.combine(cube1,
                                  cube2,
                                  self.operation)

        if self.operation == 'mean':
            result = result / len(cube_list)

        result = self.amend_metadata(result,
                                     new_diagnostic_name,
                                     data_type,
                                     revised_coords,
                                     revised_attributes,
                                     warnings_on)

        return result
