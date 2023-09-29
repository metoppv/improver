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
""" Utilities to parse a list of constraints and extract matching subcube """

import operator
from ast import literal_eval
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from iris import Constraint
from iris.cube import Cube
from iris.exceptions import CoordinateNotFoundError

from improver import BasePlugin
from improver.metadata.constants import FLOAT_DTYPE
from improver.utilities.cube_constraints import create_sorted_lambda_constraint
from improver.utilities.cube_manipulation import get_dim_coord_names


def parse_range_string_to_dict(value: str) -> Dict[str, str]:
    """
    Splits up a string in the form "[min:max:step]" into a list of
    [min, max, step].

    Args:
        value:
            A string containing the range information.
            It is assumed that the input value is of the form: "[2:10]".

    Returns:
        A list containing the min and max (and step).
    """
    value = value.replace("[", "").replace("]", "").split(":")
    return dict(zip(["min", "max", "step"], value))


def create_constraint(value: Union[float, List[float]]) -> Union[Callable, List[int]]:
    """
    Constructs an appropriate constraint for matching numerical values if they
    are floating point. If not, the original values are returned as a list
    (even if they were single valued on entry).

    Args:
        value:
            Constraint values that are being used to match against values in a
            cube for the purposes of extracting elements of the cube.

    Returns:
        If the input value(s) are floating point this function returns a
        lambda function that will enable for approximate matching to
        ensure they can be matched to cube values. If the inputs are int
        or non-numeric, they will be returned unchanged, except for single
        values that will have become lists.
    """
    if not isinstance(value, list):
        value = [value]

    if np.issubdtype(np.array(value).dtype, np.number):
        return lambda cell: any(np.isclose(cell.point, value))
    return value


def parse_constraint_list(
    constraints: List[str], units: Optional[List[str]] = None
) -> Tuple[Constraint, Optional[Dict], Optional[float], Optional[Dict]]:
    """
    For simple constraints of a key=value format, these are passed in as a
    list of strings and converted to key-value pairs prior to creating the
    constraints.
    For more complex constraints, the list of strings given as input
    are evaluated by parsing for specific identifiers and then the constraints
    are created as required.
    The simple key-value pairs and other constraints are merged into a single
    constraint.

    Args:
        constraints:
            List of string constraints with keys and values split by "=":
            e.g: ["kw1=val1", "kw2 = val2", "kw3=val3"], where the vals
            could include ranges e.g. [0:20] or ranges with a step value e.g.
            [0:20:3].
        units:
            List of units (as strings) corresponding to each coordinate in the
            list of constraints.  One or more "units" may be None, and units
            may only be associated with coordinate constraints.

    Returns:
        - A combination of all the constraints that were supplied.
        - A dictionary of unit keys and values
        - A list containing the min and max values for a longitude constraint
        - A dictionary of coordinate and the step value, i.e. a step of 2 will
          skip every other point
    """

    if units is None:
        list_units = len(constraints) * [None]
        units_dict = None
    else:
        if len(units) != len(constraints):
            msg = "units list must match constraints"
            raise ValueError(msg)
        list_units = units
        units_dict = {}

    simple_constraints_dict = {}
    complex_constraints = []
    longitude_constraint = None
    thinning_values = {}
    for constraint_pair, unit_val in zip(constraints, list_units):
        key, value = constraint_pair.split("=", 1)
        key = key.strip(" ")
        value = value.strip(" ")

        if ":" in value:
            range_dict = parse_range_string_to_dict(value)

            # longitude is a circular coordinate, so needs to be treated in a
            # different way to a normal constraint
            if key == "longitude":
                longitude_constraint = [
                    FLOAT_DTYPE(range_dict[k]) for k in ["min", "max"]
                ]
            else:
                complex_constraints.append(
                    create_sorted_lambda_constraint(
                        key, [range_dict["min"], range_dict["max"]]
                    )
                )
            if range_dict.get("step", None):
                thinning_values[key] = int(range_dict["step"])
        else:
            try:
                typed_value = literal_eval(value)
            except ValueError:
                simple_constraints_dict[key] = value
            else:
                simple_constraints_dict[key] = create_constraint(typed_value)

        if unit_val is not None and unit_val.capitalize() != "None":
            units_dict[key] = unit_val.strip(" ")

    if simple_constraints_dict:
        simple_constraints = Constraint(**simple_constraints_dict)
    else:
        simple_constraints = None

    constraints = simple_constraints
    for constr in complex_constraints:
        constraints = constraints & constr

    return constraints, units_dict, longitude_constraint, thinning_values


def apply_extraction(
    cube: Cube,
    constraint: Constraint,
    units: Optional[Dict] = None,
    use_original_units: bool = True,
    longitude_constraint: Optional[List] = None,
) -> Cube:
    """
    Using a set of constraints, extract a subcube from the provided cube if it
    is available.

    Args:
        cube:
            The cube from which a subcube is to be extracted.
        constraint:
            The constraint or ConstraintCombination that will be used to
            extract a subcube from the input cube.
        units:
            A dictionary of units for the constraints. Supplied if any
            coordinate constraints are provided in different units from those
            of the input cube (eg precip in mm/h for cube threshold in m/s).
        use_original_units:
            Boolean to state whether the coordinates used in the extraction
            should be converted back to their original units. The default is
            True, indicating that the units should be converted back to the
            original units.
        longitude_constraint:
            List containing the min and max values for the longitude.
            This has to be treated separately to the normal constraints due
            to the circular nature of longitude.

    Returns:
        A single cube matching the input constraints, or None if no subcube
        is found within cube that matches the constraints.
    """
    if units is None:
        output_cube = cube.extract(constraint)
    else:
        original_units = {}
        for coord in units.keys():
            original_units[coord] = cube.coord(coord).units
            cube.coord(coord).convert_units(units[coord])
        output_cube = cube.extract(constraint)
        if use_original_units:
            for coord in original_units:
                cube.coord(coord).convert_units(original_units[coord])
                try:
                    output_cube.coord(coord).convert_units(original_units[coord])
                except AttributeError:
                    # an empty output cube (None) is handled by the CLI
                    pass

    if longitude_constraint:
        output_cube = output_cube.intersection(
            longitude=longitude_constraint, ignore_bounds=True
        )
        # TODO: Below can be removed when https://github.com/SciTools/iris/issues/4119
        # is fixed
        output_cube.coord("longitude").points = output_cube.coord(
            "longitude"
        ).points.astype(FLOAT_DTYPE)
        if output_cube.coord("longitude").bounds is not None:
            output_cube.coord("longitude").bounds = output_cube.coord(
                "longitude"
            ).bounds.astype(FLOAT_DTYPE)

    return output_cube


def extract_subcube(
    cube: Cube,
    constraints: List[str],
    units: Optional[List[str]] = None,
    use_original_units: bool = True,
) -> Optional[Cube]:
    """
    Using a set of constraints, extract a subcube from the provided cube if it
    is available.

    Args:
        cube:
            The cube from which a subcube is to be extracted.
        constraints:
            List of string constraints with keys and values split by "=":
            e.g: ["kw1=val1", "kw2 = val2", "kw3=val3"].
        units:
            List of units (as strings) corresponding to each coordinate in the
            list of constraints.  One or more "units" may be None, and units
            may only be associated with coordinate constraints.
        use_original_units:
            Boolean to state whether the coordinates used in the extraction
            should be converted back to their original units. The default is
            True, indicating that the units should be converted back to the
            original units.

    Returns:
        A single cube matching the input constraints, or None if no subcube
        is found within cube that matches the constraints.
    """
    constraints, units, longitude_constraint, thinning_dict = parse_constraint_list(
        constraints, units=units
    )
    output_cube = apply_extraction(
        cube, constraints, units, use_original_units, longitude_constraint
    )

    if thinning_dict:
        output_cube = thin_cube(output_cube, thinning_dict)

    return output_cube


def thin_cube(cube: Cube, thinning_dict: Dict[str, int]) -> Cube:
    """
    Thin the coordinate by taking every X points, defined in the thinning dict
    as {coordinate: X}

    Args:
        cube:
            The cube containing the coordinates to be thinned.
        thinning_dict:
            A dictionary of coordinate and the step value, i.e. a step of 2
            will skip every other point

    Returns:
        A cube with thinned coordinates.
    """
    coord_names = get_dim_coord_names(cube)
    slices = [slice(None, None, None)] * len(coord_names)
    for key, val in thinning_dict.items():
        slices[coord_names.index(key)] = slice(None, None, val)
    return cube[tuple(slices)]


class ExtractLevel(BasePlugin):
    """
    Class for extracting a pressure or height surface from data-on-levels.
    """

    def __init__(
        self, positive_correlation: bool, value_of_level: float,
    ):
        """Sets up Class
            Args:
                positive_correlation:
                    Set to True when the variable generally increases as pressure increase or when the variable
                    generally increases as height decreases.
                value_of_level:
                    The value of the input cube for which the pressure or height level is required
        """

        self.positive_correlation = positive_correlation
        self.value_of_level = value_of_level
        self.coordinate=None

    def coordinate_grid(self, variable_on_levels: Cube) -> np.ndarray:
        """Creates a pressure or height grid of the same shape as variable_on_levels cube.
        It is populated at every grid square and for every realization with
        a column of all pressure or height levels taken from variable_on_levels's pressure or height
        coordinate

        Args:
            variable_on_levels:
                Cube of some variable with pressure or height levels
        Returns:
            An n dimensional array with the same dimensions as variable_on_levels containing,
            at every grid square and for every realization, a column of all pressure or height levels
            taken from variable_on_levels's pressure or height coordinate
        """

        required_shape = variable_on_levels.shape
        coordinate_points = variable_on_levels.coord(self.coordinate).points
        (coordinate_axis,) = variable_on_levels.coord_dims(self.coordinate)
        coordinate_shape = np.ones_like(required_shape)
        coordinate_shape[coordinate_axis] = required_shape[coordinate_axis]
        coordinate_array = np.broadcast_to(
            coordinate_points.reshape(coordinate_shape), required_shape
        )
        return coordinate_array

    def fill_invalid(self, cube: Cube):
        """Populate any invalid values in the source data with the neighbouring value in that
        column plus a small difference (determined by the least_significant_digit attribute,
        or 10^-2). This results in columns that do not have repeated values as repeated values
        confuse the stratify.interpolate method.

        Args:
            cube:
                Cube of variable on levels (3D) (modified in-place).
"""
        if np.isfinite(cube.data).all() and not np.ma.is_masked(cube.data):
            return
        data = np.ma.masked_invalid(cube.data)
        (coordinate_axis,) = cube.coord_dims(self.coordinate)
        coordinate_points = cube.coord(self.coordinate).points
        # Find the least significant increment to use as an offset for filling missing values.
        # We don't really care so long as it is non-zero and has the same sign.
        increasing_order = np.all(np.diff(cube.coord(self.coordinate).points) > 0)
        sign = 1 if self.positive_correlation == increasing_order else -1
        v_increment = sign * 10 ** (-cube.attributes.get("least_significant_digit", 2))
        self._one_way_fill(
            data, coordinate_axis, coordinate_points, v_increment, reverse=True
        )
        cube.data = data
        if np.isfinite(cube.data).all() and not np.ma.is_masked(cube.data):
            # We've filled everything in. No need to try again.
            return
        self._one_way_fill(
            data, coordinate_axis, coordinate_points, v_increment, reverse=False
        )
        cube.data = data

    @staticmethod
    def _one_way_fill(
        data: np.ma.MaskedArray,
        coordinate_axis: int,
        coordinate_points: np.ndarray,
        v_increment: np.ndarray,
        reverse: bool = False,
    ):
        """
        Scans through the pressure or height axis forwards or backwards, filling any missing data with the
        previous value plus (or minus in reverse) the specified increment. Running this in both
        directions will therefore populate all columns so long as there is at least one valid
        data point to start with.
        """
        last_slice = [slice(None)] * data.ndim
        if reverse:
            local_increment = -v_increment
            first = len(coordinate_points) - 2
            last = -1
            step = -1
            last_slice[coordinate_axis] = slice(
                len(coordinate_points) - 1, len(coordinate_points)
            )
        else:
            local_increment = v_increment
            first = 1
            last = len(coordinate_points)
            step = 1
            last_slice[coordinate_axis] = slice(0, 1)

        for c in range(first, last, step):
            c_slice = [slice(None)] * data.ndim
            c_slice[coordinate_axis] = slice(c, c + 1)
            data[c_slice] = np.ma.where(
                data.mask[c_slice], data[last_slice] + local_increment, data[c_slice],
            )
            last_slice = c_slice

    def fill_in_bounds(
        self, value_of_variable: np.ma.MaskedArray, source_cube: Cube
    ) -> np.ndarray:
        """Update any undefined value_of_variable values with the maximum or minimum
        pressure or height for that column. This occurs when the requested variable value falls above
        or below the entire column.
        Maximum pressure or height is chosen if the maximum data value in the column is lower than
        the value of self.value_of_level.

        Args:
            value_of_variable:
                2D array of the pressure or height at the required variable value, masked True
                where this method needs to fill it in. (modified in-place)
            source_cube:
                Cube of variable on levels (3D) which shows where the lowest
                and highest valid values are.
        Returns:
            Updated value_of_variable array
        """
        if not np.ma.is_masked(value_of_variable):
            return value_of_variable.data
        coord = source_cube.coord(self.coordinate)
        max_coordinate = coord.points.max()
        min_coordinate = coord.points.min()
        (coordinate_axis,) = source_cube.coord_dims(self.coordinate)
        max_index = np.argmax(coord.points)
        # The values at the maximum pressure or height will be compared with the requested variable value
        # using an appropriate operator based on whether value and pressure (or height) are increasing.
        comparator = operator.lt if self.positive_correlation else operator.gt
        values_at_max = source_cube.data.take(
            axis=coordinate_axis, indices=max_index
        )
        # First, fill in missing values at the maximum end of the pressure or height coordinate:
        value_of_variable = np.ma.where(
            np.logical_and(
                comparator(values_at_max, self.value_of_level),
                value_of_variable.mask,
            ),
            max_coordinate,
            value_of_variable,
        )
        # Now fill in remaining missing values which should be at the minimum end:
        value_of_variable = np.where(
            value_of_variable.mask, min_coordinate, value_of_variable,
        )
        return value_of_variable

    def _make_template_cube(
        self, result_data: np.array, variable_on_levels: Cube
    ) -> Cube:
        """Creates a cube of the variable on a pressure or height level based on the input cube"""
        template_cube = next(
            variable_on_levels.slices_over([self.coordinate])
        ).copy(result_data)
        if self.coordinate=="pressure":
            template_cube.rename(
                "pressure_of_atmosphere_at_"
                f"{self.value_of_level}{variable_on_levels.units}"
            )
        else:
            template_cube.rename(
                "height_at_"
                f"{self.value_of_level}{variable_on_levels.units}"
            )

        template_cube.units = variable_on_levels.coord(self.coordinate).units
        template_cube.remove_coord(self.coordinate)
        return template_cube

    def process(self, variable_on_levels: Cube) -> Cube:
        """Extracts the pressure or height level (depending on which is present on the cube) 
        where the environment variable first intersects self.value_of_level starting at a pressure or height
        value near the surface and ascending in altitude from there.
        Where the surface falls outside the available data, the maximum or minimum
        of the surface will be returned, even if the source data has no value at that point.

        Args:
            variable_on_levels:
                A cube of data on pressure or height levels
        Returns:
            A cube of the environment pressure or height at self.value_of_level
        """
        from stratify import interpolate

        # if both a pressure and height coordinate exists then pressure takes priority
        # over the height coordinate
        try:
            self.coordinate=variable_on_levels.coord("pressure").name()
        except CoordinateNotFoundError:
            self.coordinate=variable_on_levels.coord("height").name()

        self.fill_invalid(variable_on_levels)

        if "realization" in [c.name() for c in variable_on_levels.dim_coords]:
            slicer = variable_on_levels.slices_over((["realization"]))
            one_slice = variable_on_levels.slices_over(
                (["realization"])
            ).next()
            has_r_coord = True
        else:
            slicer = [variable_on_levels]
            one_slice = variable_on_levels
            has_r_coord = False
        grid = self.coordinate_grid(one_slice).astype(np.float32)
        (coordinate_axis,) = one_slice.coord_dims(self.coordinate)
        result_data = np.empty_like(
            variable_on_levels.slices_over(([self.coordinate])).next().data
        )
        for i, zyx_slice in enumerate(slicer):
            interp_data = interpolate(
                np.array([self.value_of_level], dtype=np.float32),
                zyx_slice.data.data,
                grid,
                axis=coordinate_axis,
            ).squeeze(axis=coordinate_axis)
            if has_r_coord:
                result_data[i] = interp_data
            else:
                result_data = interp_data
        result_data = np.ma.masked_invalid(result_data)
        result_data = self.fill_in_bounds(result_data, variable_on_levels)

        output_cube = self._make_template_cube(
            result_data, variable_on_levels
        )
        return output_cube
