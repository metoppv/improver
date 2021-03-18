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
"""Plugin to standardise metadata"""

import iris
import numpy as np
from iris.exceptions import CoordinateNotFoundError

from improver import BasePlugin
from improver.metadata.amend import amend_attributes
from improver.metadata.check_datatypes import (
    check_units,
    get_required_dtype,
    get_required_units,
)
from improver.metadata.constants.time_types import TIME_COORDS
from improver.utilities.round import round_close


class StandardiseMetadata(BasePlugin):
    """Plugin to standardise cube metadata"""

    @staticmethod
    def _collapse_scalar_dimensions(cube):
        """
        Demote any scalar dimensions (excluding "realization") on the input
        cube to auxiliary coordinates.

        Returns:
            iris.cube.Cube
        """
        coords_to_collapse = []
        for coord in cube.coords(dim_coords=True):
            if len(coord.points) == 1 and "realization" not in coord.name():
                coords_to_collapse.append(coord)
        for coord in coords_to_collapse:
            cube = next(cube.slices_over(coord))
        return cube

    @staticmethod
    def _remove_scalar_coords(cube, coords_to_remove):
        """Removes named coordinates from the input cube."""
        for coord in coords_to_remove:
            try:
                cube.remove_coord(coord)
            except CoordinateNotFoundError:
                continue

    @staticmethod
    def _standardise_dtypes_and_units(cube):
        """
        Modify input cube in place to conform to mandatory dtype and unit
        standards.

        Args:
            cube (iris.cube.Cube:
                Cube to be updated in place

        """

        def as_correct_dtype(obj, required_dtype):
            """
            Returns an object updated if necessary to the required dtype

            Args:
                obj (np.ndarray):
                    The object to be updated
                required_dtype (np.dtype):
                    The dtype required

            Returns:
                np.ndarray
            """
            if obj.dtype != required_dtype:
                return obj.astype(required_dtype)
            return obj

        cube.data = as_correct_dtype(cube.data, get_required_dtype(cube))
        for coord in cube.coords():
            if coord.name() in TIME_COORDS and not check_units(coord):
                coord.convert_units(get_required_units(coord))
            req_dtype = get_required_dtype(coord)
            # ensure points and bounds have the same dtype
            if np.issubdtype(req_dtype, np.integer):
                coord.points = round_close(coord.points)
            coord.points = as_correct_dtype(coord.points, req_dtype)
            if coord.has_bounds():
                if np.issubdtype(req_dtype, np.integer):
                    coord.bounds = round_close(coord.bounds)
                coord.bounds = as_correct_dtype(coord.bounds, req_dtype)

    def process(
        self,
        cube,
        new_name=None,
        new_units=None,
        coords_to_remove=None,
        attributes_dict=None,
    ):
        """
        Perform compulsory and user-configurable metadata adjustments.  The
        compulsory adjustments are to collapse any scalar dimensions apart from
        realization (which is expected always to be a dimension); to cast the cube
        data and coordinates into suitable datatypes; and to convert time-related
        metadata into the required units.

        Args:
            cube (iris.cube.Cube):
                Input cube to be standardised
            new_name (str or None):
                Optional rename for output cube
            new_units (str or None):
                Optional unit conversion for output cube
            coords_to_remove (list of str or None):
                Optional list of scalar coordinates to remove from output cube
            attributes_dict (dict or None):
                Optional dictionary of required attribute updates. Keys are
                attribute names, and values are the required value or "remove".

        Returns:
            iris.cube.Cube
        """
        cube = self._collapse_scalar_dimensions(cube)

        if new_name:
            cube.rename(new_name)
        if new_units:
            cube.convert_units(new_units)
        if coords_to_remove:
            self._remove_scalar_coords(cube, coords_to_remove)
        if attributes_dict:
            amend_attributes(cube, attributes_dict)

        # this must be done after unit conversion as if the input is an integer
        # field, unit conversion outputs the new data as float64
        self._standardise_dtypes_and_units(cube)

        return cube
