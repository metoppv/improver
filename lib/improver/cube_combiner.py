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

import warnings

import iris
from iris import FUTURE

FUTURE.netcdf_promote = True


class CubeCombiner(object):

    """Plugin for combining cubes.

    """

    def __init__(self, operation):
        """
        Create a CubeCombiner plugin
        
        Args:
            operation (str):
                Operation (+, - etc) to apply to the incoming cubes)
        """
        possible_operations = [ '+', '-', '*', 'max', 'min', 'mean']
        if operation in possible_operations:
            self.operation = operation

    def __repr__(self):
        """Represent the configured plugin instance as a string."""
        desc = '<CubeCombiner: operation={}>'.format(self.operation)
        return desc

    def add_metadata(self, cube):
        """Add metadata to cube.

        Args:
            cube (iris.cube.Cube):
                Cube containing the wind-gust diagnostic data.
        Returns:
            result (iris.cube.Cube):
                Cube containing the wind-gust diagnostic data with
                corrected Metadata.

        """
        result = cube

        return result

    def process(self, cube_list):
        """
        Create a cube.

        Args:
            cube_list (iris.cube.CubeList):
                Cube List contain the cubes to combine.

        Returns:
            result (iris.cube.Cube):
                Cube containing the combined data.

        """
        data_type = cube_list[0].dtype
        result = cube_list[0]
        result.data = result.data.astype(data_type)

        return result
