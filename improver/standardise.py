# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown copyright. The Met Office.
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

from typing import Any, Dict, List, Optional

import numpy as np
from iris.coords import CellMethod
from iris.cube import Cube, CubeList
from iris.exceptions import CoordinateNotFoundError
from numpy import dtype, ndarray

from improver import BasePlugin
from improver.metadata.amend import amend_attributes
from improver.metadata.check_datatypes import (
    _is_time_coord,
    check_units,
    get_required_dtype,
    get_required_units,
)
from improver.metadata.constants.time_types import TIME_COORDS
from improver.utilities.round import round_close


class StandardiseMetadata(BasePlugin):
    """Plugin to standardise cube metadata"""

    @staticmethod
    def _rm_air_temperature_status_flag(cube: Cube) -> Cube:
        """
        Remove air_temperature status_flag coord by applying as NaN to cube data.

        See https://github.com/metoppv/improver/pull/1839 for further details.
        """
        coord_name = "air_temperature status_flag"
        try:
            coord = cube.coord(coord_name)
        except CoordinateNotFoundError:
            coord = None

        if coord:
            if coord.attributes != {
                "flag_meanings": "above_surface_pressure below_surface_pressure",
                "flag_values": np.array([0, 1], dtype="int8"),
            }:
                raise ValueError(
                    f"'{coord_name}' coordinate is not of the expected form."
                )
            ncube = CubeList()

            try:
                cube_iterator = cube.slices_over("realization")
            except CoordinateNotFoundError:
                cube_iterator = [cube]

            for cc in cube_iterator:
                coord = cc.coord(coord_name)
                if np.ma.is_masked(coord.points):
                    raise ValueError(
                        f"'{coord_name}' coordinate has unexpected mask values."
                    )
                mask = np.asarray(coord.points)
                cc.data[mask.astype(bool)] = np.nan
                cc.remove_coord(coord_name)
                ncube.append(cc)
            cube = ncube.merge_cube()
        return cube

    @staticmethod
    def _collapse_scalar_dimensions(cube: Cube) -> Cube:
        """
        Demote any scalar dimensions (excluding "realization") on the input
        cube to auxiliary coordinates.

        Args:
            cube: The cube

        Returns:
            The collapsed cube
        """
        coords_to_collapse = []
        for coord in cube.coords(dim_coords=True):
            if len(coord.points) == 1 and "realization" not in coord.name():
                coords_to_collapse.append(coord)
        for coord in coords_to_collapse:
            cube = next(cube.slices_over(coord))
        return cube

    @staticmethod
    def _remove_scalar_coords(cube: Cube, coords_to_remove: List[str]) -> None:
        """Removes named coordinates from the input cube."""
        for coord in coords_to_remove:
            try:
                cube.remove_coord(coord)
            except CoordinateNotFoundError:
                continue

    @staticmethod
    def _modify_scalar_coord_value(cube: Cube, coord_modification: Dict[str, float]) -> None:
        """Modifies the value of each specified scalar coord (dictionary key)
        to the provided value (dictionary value). Note that data types are not
        enforced here as the subsequent enforcment step will fulfil this
        requirement. Units are assumed to be the same as the original
        coordinate value. Modifying multi-valued coordinates or time
        coordinates is specifically prevented as there is greater scope to
        harm data integrity (i.e. the description of the data and the data
        becoming unaligned).

        Args:
            cube:
                Cube to be updated in place
            coord_modification:
                Dictionary defining the coordinates (keys) to be modified
                and the values (values) to which they should be set.
        """
        for coord, value in coord_modification.items():
            if cube.coords(coord):
                if cube.coords(coord, dim_coords=True):
                    raise ValueError(
                        "Modifying dimension coordinate values it not allowed "
                        "due to the risk of introducing errors."
                    )
                if _is_time_coord(cube.coord(coord)):
                    raise ValueError(
                        "Modifying time coordinates it not allowed."
                    )
                cube.coord(coord).points = np.array([value])

    @staticmethod
    def _standardise_dtypes_and_units(cube: Cube) -> None:
        """
        Modify input cube in place to conform to mandatory dtype and unit
        standards.

        Args:
            cube:
                Cube to be updated in place
        """

        def as_correct_dtype(obj: ndarray, required_dtype: dtype) -> ndarray:
            """
            Returns an object updated if necessary to the required dtype

            Args:
                obj:
                    The object to be updated
                required_dtype:
                    The dtype required

            Returns:
                The updated object
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

    @staticmethod
    def _discard_redundant_cell_methods(cube: Cube) -> None:
        """
        Removes cell method "point": "time" from cube if present.
        """
        if not cube.cell_methods:
            return
        removable_cms = [
            CellMethod(method="point", coords="time"),
        ]
        updated_cms = []
        for cm in cube.cell_methods:
            if cm in removable_cms:
                continue
            updated_cms.append(cm)

        cube.cell_methods = updated_cms

    def process(
        self,
        cube: Cube,
        new_name: Optional[str] = None,
        new_units: Optional[str] = None,
        coords_to_remove: Optional[List[str]] = None,
        coord_modification: Optional[Dict[str, float]] = None,
        attributes_dict: Optional[Dict[str, Any]] = None,
    ) -> Cube:
        """
        Perform compulsory and user-configurable metadata adjustments.  The
        compulsory adjustments are:

        - to collapse any scalar dimensions apart from realization (which is expected
          always to be a dimension);
        - to cast the cube data and coordinates into suitable datatypes;
        - to convert time-related metadata into the required units
        - to remove cell method ("point": "time").

        Args:
            cube:
                Input cube to be standardised
            new_name:
                Optional rename for output cube
            new_units:
                Optional unit conversion for output cube
            coords_to_remove:
                Optional list of scalar coordinates to remove from output cube
            coord_modification:
                Optional dictionary used to directly modify the values of
                scalar coordinates. To be used with extreme caution.
                For example this dictionary might take the form:
                {"height": 1.5} to set the height coordinate to have a value
                of 1.5m (assuming original units of m).
                This can be used to align e.g. temperatures defined at slightly
                different heights where this difference is considered small
                enough to ignore. Type is inferred, so provide a value of 2
                will result in an integer type, whilst a value of 2.0 will
                result in a float type.
            attributes_dict:
                Optional dictionary of required attribute updates. Keys are
                attribute names, and values are the required changes.
                See improver.metadata.amend.amend_attributes for details.

        Returns:
            The processed cube
        """
        cube = self._rm_air_temperature_status_flag(cube)
        cube = self._collapse_scalar_dimensions(cube)

        if new_name:
            cube.rename(new_name)
        if new_units:
            cube.convert_units(new_units)
        if coords_to_remove:
            self._remove_scalar_coords(cube, coords_to_remove)
        if coord_modification:
            self._modify_scalar_coord_value(cube, coord_modification)
        if attributes_dict:
            amend_attributes(cube, attributes_dict)
        self._discard_redundant_cell_methods(cube)

        # this must be done after unit conversion as if the input is an integer
        # field, unit conversion outputs the new data as float64
        self._standardise_dtypes_and_units(cube)

        return cube
