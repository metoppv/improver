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

import iris
import numpy as np

from improver.utilities.cube_manipulation import expand_bounds
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

    def combine(self, cube1, cube2):
        """
        Combine cube data

        Args:
            cube1 (iris.cube.Cube):
                Cube containing data to be combined.
            cube2 (iris.cube.Cube):
                Cube containing data to be combined.
        Returns:
            result (iris.cube.Cube):
                Cube containing the combined data.
        """
        result = cube1
        if (self.operation == '+' or self.operation == 'add' or
                self.operation == 'mean'):
            result.data = cube1.data + cube2.data
        elif self.operation == '-' or self.operation == 'subtract':
            result.data = cube1.data - cube2.data
        elif self.operation == '*' or self.operation == 'multiply':
            result.data = cube1.data * cube2.data
        elif self.operation == 'min':
            result.data = np.minimum(cube1.data, cube2.data)
        elif self.operation == 'max':
            result.data = np.maximum(cube1.data, cube2.data)

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
            revised_coords (dict or None):
                Revised coordinates for combined cube.
            revised_attributes (dict or None):
                Revised attributes for combined cube.
            expanded_coord (dict or None):
                Coordinates to be expanded as a key, with the value
                indicating whether the upper or mid point of the coordinate
                should be used as the point value, e.g.
                {'time': 'upper'}.
        Returns:
            result (iris.cube.Cube):
                Cube containing the combined data.
        Raises:
            TypeError: If cube_list is not an iris.cube.CubeList.
            ValueError: If the cubelist contains only one cube.
        """
        if not isinstance(cube_list, iris.cube.CubeList):
            msg = ('Expecting data to be an instance of iris.cube.CubeList '
                   'but is {}.'.format(type(cube_list)))
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
            result = self.combine(cube1, cube2)

        if self.operation == 'mean':
            result.data = result.data / len(cube_list)

        # If cube has coord bounds that we want to expand
        if expanded_coord:
            result = expand_bounds(result, cube_list, expanded_coord)

        result = amend_metadata(result,
                                new_diagnostic_name,
                                data_type,
                                revised_coords,
                                revised_attributes,
                                warnings_on=self.warnings_on)

        return result
