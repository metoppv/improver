# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Plugin to standardise metadata"""

import warnings
from typing import Any, Dict, List, Optional

import numpy as np
from iris.coords import CellMethod
from iris.cube import Cube, CubeList
from iris.exceptions import AncillaryVariableNotFoundError, CoordinateNotFoundError
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
from improver.utilities.common_input_handle import as_cube
from improver.utilities.round import round_close


class StandardiseMetadata(BasePlugin):
    """Plugin to standardise cube metadata"""

    def __init__(
        self,
        new_name: Optional[str] = None,
        new_units: Optional[str] = None,
        coords_to_remove: Optional[List[str]] = None,
        coord_modification: Optional[Dict[str, float]] = None,
        attributes_dict: Optional[Dict[str, Any]] = None,
        ancillary_variables_to_remove: Optional[List[str]] = None,
    ):
        """
        Instantiate our class for standardising cube metadata.

        Args:
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
                enough to ignore. Type is inferred, so providing a value of 2
                will result in an integer type, whilst a value of 2.0 will
                result in a float type.
            attributes_dict:
                Optional dictionary of required attribute updates. Keys are
                attribute names, and values are the required changes.
                See improver.metadata.amend.amend_attributes for details.
            ancillary_variables_to_remove:
                Optional list of ancillary variable names to remove from the
                output cube.
        """
        self._new_name = new_name
        self._new_units = new_units
        self._coords_to_remove = coords_to_remove
        self._coord_modification = coord_modification
        self._attributes_dict = attributes_dict
        self._ancillary_variables_to_remove = ancillary_variables_to_remove

    @staticmethod
    def _remove_air_temperature_status_flag(cube: Cube) -> Cube:
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
    def _remove_ancillary_variables(
        cube: Cube, ancillary_variables_to_remove: List[str]
    ) -> None:
        """Removes named ancillary variables from the input cube."""
        for var in ancillary_variables_to_remove:
            try:
                cube.remove_ancillary_variable(var)
            except AncillaryVariableNotFoundError:
                warnings.warn(
                    f"Ancillary variable '{var}' not found in cube '{cube.name()}'.",
                    UserWarning
                )
                continue

    @staticmethod
    def _modify_scalar_coord_value(
        cube: Cube, coord_modification: Dict[str, float]
    ) -> None:
        """Modifies the value of each specified scalar coord (dictionary key)
        to the provided value (dictionary value). Note that data types are not
        enforced here as the subsequent enforcement step will fulfil this
        requirement. Units are assumed to be the same as the original
        coordinate value. Modifying multi-valued coordinates or time
        coordinates is specifically prevented as there is greater scope to
        harm data integrity (i.e. the description of the data and the data
        becoming misaligned).

        If the coordinate does not exist the modification request is silently
        skipped.

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
                        "Modifying dimension coordinate values is not allowed "
                        "due to the risk of introducing errors."
                    )
                if hasattr(value, "__len__") and len(value) > 1:
                    raise ValueError(
                        "Modifying multi-valued coordinates is not allowed. "
                        "This functionality should be used only for very "
                        "modest changes to scalar coordinates."
                    )
                if _is_time_coord(cube.coord(coord)):
                    raise ValueError("Modifying time coordinates is not allowed.")
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
        removable_cms = [CellMethod(method="point", coords="time")]
        updated_cms = []
        for cm in cube.cell_methods:
            if cm in removable_cms:
                continue
            updated_cms.append(cm)

        cube.cell_methods = updated_cms

    @staticmethod
    def _remove_long_name_if_standard_name(cube: Cube) -> None:
        """
        Remove the long_name attribute from cubes if the cube also has a standard_name defined
        """

        if cube.standard_name and cube.long_name:
            cube.long_name = None

    def process(self, cube: Cube) -> Cube:
        """
        Perform compulsory and user-configurable metadata adjustments.
        The compulsory adjustments are:

        - to collapse any scalar dimensions apart from realization (which is expected
          always to be a dimension);
        - to cast the cube data and coordinates into suitable datatypes;
        - to convert time-related metadata into the required units
        - to remove cell method ("point": "time").

        If the air_temperature data is required, this can be retained by
        removing the `air_temperature status_flag` as part of the standardise step
        so that the process of masking this data with NaNs is bypassed.
        See https://github.com/metoppv/improver/pull/1839 for further information.

        Args:
            cube:
                Input cube to be standardised

        Returns:
            The processed cube
        """
        cube = as_cube(cube)
        # It is necessary to have the `_coords_to_remove step` first
        # so that it allows keeping the air temperature data for
        # a future calculation. Removing the `air_temperature status_flag`
        # means the air temperature data will then not be masked by NaNs,
        # as happens in the `_remove_air_temperature_status_flag` step if
        # the flag is not removed.
        if self._coords_to_remove:
            self._remove_scalar_coords(cube, self._coords_to_remove)
        if self._ancillary_variables_to_remove:
            self._remove_ancillary_variables(cube, self._ancillary_variables_to_remove)
        cube = self._remove_air_temperature_status_flag(cube)
        cube = self._collapse_scalar_dimensions(cube)
        if self._new_name:
            cube.rename(self._new_name)
        if self._new_units:
            cube.convert_units(self._new_units)
        if self._coord_modification:
            self._modify_scalar_coord_value(cube, self._coord_modification)
        if self._attributes_dict:
            amend_attributes(cube, self._attributes_dict)
        self._discard_redundant_cell_methods(cube)
        self._remove_long_name_if_standard_name(cube)
        # this must be done after unit conversion as if the input is an integer
        # field, unit conversion outputs the new data as float64
        self._standardise_dtypes_and_units(cube)
        return cube
