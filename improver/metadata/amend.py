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
"""Module containing utilities for modifying cube metadata"""
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Tuple, Union

from iris.coords import CellMethod
from iris.cube import Cube, CubeList

from improver.metadata.constants.mo_attributes import (
    GRID_ID_LOOKUP,
    MOSG_GRID_DEFINITION,
)
from improver.metadata.probabilistic import (
    get_diagnostic_cube_name_from_probability_name,
    get_threshold_coord_name_from_probability_name,
    is_probability,
)


def update_stage_v110_metadata(cube: Cube) -> None:
    """Translates attributes relating to the grid_id attribute from StaGE
    version 1.1.0 to later StaGE versions.
    Cubes that have no "grid_id" attribute are not recognised as v1.1.0 and
    are ignored.

    Args:
        cube:
            Cube to modify attributes in (modified in place)
    """
    try:
        grid_id = cube.attributes.pop("grid_id")
    except KeyError:
        # Not a version 1.1.0 grid, do nothing
        return
    cube.attributes.update(MOSG_GRID_DEFINITION[GRID_ID_LOOKUP[grid_id]])
    cube.attributes["mosg__grid_version"] = "1.1.0"


def amend_attributes(cube: Cube, attributes_dict: Dict[str, Any]) -> None:
    """
    Add, update or remove attributes from a cube.  Modifies cube in place.

    Args:
        cube:
            Input cube
        attributes_dict:
            Dictionary containing items of the form {attribute_name: value}.
            The "value" item is either the string "remove" or the new value
            of the attribute required.
    """
    for attribute_name, value in attributes_dict.items():
        if value == "remove":
            cube.attributes.pop(attribute_name, None)
        else:
            cube.attributes[attribute_name] = value


def set_history_attribute(cube: Cube, value: str, append: bool = False) -> None:
    """Add a history attribute to a cube. This uses the current datetime to
    generate the timestamp for the history attribute. The new history attribute
    will overwrite any existing history attribute unless the "append" option is
    set to True. The history attribute is of the form "Timestamp: Description".

    Args:
        cube:
            The cube to which the history attribute will be added.
        value:
            String defining details to be included in the history attribute.
        append:
            If True, add to the existing history rather than replacing the
            existing attribute.  Default is False.
    """
    timestamp = datetime.strftime(
        datetime.now(timezone(timedelta(0), name="Z")), "%Y-%m-%dT%H:%M:%S%Z"
    )
    new_history = "{}: {}".format(timestamp, value)
    if append and "history" in cube.attributes.keys():
        cube.attributes["history"] += "; {}".format(new_history)
    else:
        cube.attributes["history"] = new_history


def update_model_id_attr_attribute(
    cubes: Union[List[Cube], CubeList], model_id_attr: str
) -> Dict:
    """Update the dictionary with the unique values of the model_id_attr
    attribute from within the input cubes. The model_id_attr attribute is
    expected on all cubes.

    Args:
        cubes:
            List of input cubes that might have a model_id_attr attribute.
        model_id_attr:
            Name of attribute expected on the input cubes. This attribute is
            expected on the cubes as a space-separated string.

    Returns:
        Dictionary containing a model_id_attr key, if available.

    Raises:
        AttributeError: Expected to find the model_id_attr attribute on all
            cubes.
    """
    attr_in_cubes = [model_id_attr in c.attributes for c in cubes]
    if not all(attr_in_cubes):
        msg = f"Expected to find {model_id_attr} attribute on all cubes"
        raise AttributeError(msg)

    attr_list = [a for c in cubes for a in c.attributes[model_id_attr].split(" ")]
    return {model_id_attr: " ".join(sorted(set(attr_list)))}


def update_diagnostic_name(source_cube: Cube, new_diagnostic_name: str, result: Cube):
    """
    Used for renaming the threshold coordinate and modifying cell methods
    where necessary; excludes the in_vicinity component.

    Args:
        source_cube: An original cube before any processing took place. Can be the same cube as
            result.
        new_diagnostic_name: The new diagnostic name to apply to result.
        result: The cube that needs to be modified in place.

    """
    new_base_name = new_diagnostic_name.replace("_in_variable_vicinity", "")
    new_base_name = new_base_name.replace("_in_vicinity", "")
    original_name = source_cube.name()

    if is_probability(source_cube):
        diagnostic_name = get_diagnostic_cube_name_from_probability_name(original_name)
        # Rename the threshold coordinate to match the name of the diagnostic
        # that results from the combine operation.
        result.coord(var_name="threshold").rename(new_base_name)
        result.coord(new_base_name).var_name = "threshold"

        new_diagnostic_name = original_name.replace(
            diagnostic_name, new_diagnostic_name
        )
    # Modify cell methods that include the variable name to match the new
    # name.
    cell_methods = source_cube.cell_methods
    if cell_methods:
        result.cell_methods = _update_cell_methods(
            cell_methods, original_name, new_base_name
        )
    result.rename(new_diagnostic_name)


def _update_cell_methods(
    cell_methods: Tuple[CellMethod], original_name: str, new_diagnostic_name: str,
) -> List[CellMethod]:
    """
    Update any cell methods that include a comment that refers to the
    diagnostic name to refer instead to the new diagnostic name. Those cell
    methods that do not include the diagnostic name are passed through
    unmodified.

    Args:
        cell_methods:
            The cell methods found on the cube that is being used as the
            metadata template.
        original_name:
            The full name of the metadata template cube.
        new_diagnostic_name:
            The new diagnostic name to use in the modified cell methods.

    Returns:
        A list of modified cell methods to replace the originals.
    """
    try:
        # strip probability and vicinity components to provide the diagnostic name
        diagnostic_name = get_threshold_coord_name_from_probability_name(original_name)
    except ValueError:
        diagnostic_name = original_name

    new_cell_methods = []
    for cell_method in cell_methods:
        try:
            (cell_comment,) = cell_method.comments
        except ValueError:
            new_cell_methods.append(cell_method)
        else:
            if diagnostic_name in cell_comment:
                new_cell_methods.append(
                    CellMethod(
                        cell_method.method,
                        coords=cell_method.coord_names,
                        intervals=cell_method.intervals,
                        comments=f"of {new_diagnostic_name}",
                    )
                )
            else:
                new_cell_methods.append(cell_method)
    return new_cell_methods
