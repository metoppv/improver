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
"""Utilities to support weighted blending"""

from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
from iris.cube import Cube, CubeList
from numpy import int64

from improver.blending import MODEL_BLEND_COORD, MODEL_NAME_COORD
from improver.metadata.amend import amend_attributes
from improver.metadata.constants.attributes import (
    MANDATORY_ATTRIBUTE_DEFAULTS,
    MANDATORY_ATTRIBUTES,
)
from improver.metadata.constants.time_types import TIME_COORDS
from improver.metadata.forecast_times import add_blend_time, forecast_period_coord
from improver.utilities.round import round_close
from improver.utilities.temporal import cycletime_to_number


def find_blend_dim_coord(cube: Cube, blend_coord: str) -> str:
    """
    Find the name of the dimension coordinate across which to perform the blend,
    since the input "blend_coord" may be an auxiliary coordinate.

    Args:
        cube:
            Cube to be blended
        blend_coord:
            Name of coordinate to blend over

    Returns:
        Name of dimension coordinate associated with blend dimension

    Raises:
        ValueError:
            If blend coordinate is associated with more or less than one dimension
    """
    blend_dim = cube.coord_dims(blend_coord)
    if len(blend_dim) != 1:
        if len(blend_dim) < 1:
            msg = f"Blend coordinate {blend_coord} has no associated dimension"
        else:
            msg = (
                "Blend coordinate must only be across one dimension. Coordinate "
                f"{blend_coord} is associated with dimensions {blend_dim}."
            )
        raise ValueError(msg)

    return cube.coord(dimensions=blend_dim[0], dim_coords=True).name()


def get_coords_to_remove(cube: Cube, blend_coord: str) -> Optional[List[str]]:
    """
    Generate a list of coordinate names associated with the blend
    dimension.  Unless these are time-related coordinates, they should be
    removed after blending.

    Args:
        cube:
            Cube to be blended
        blend_coord:
            Name of coordinate over which the blend will be performed

    Returns:
        List of names of coordinates to remove
    """
    try:
        (blend_dim,) = cube.coord_dims(blend_coord)
    except ValueError:
        # occurs if the blend coordinate is scalar
        if blend_coord == MODEL_BLEND_COORD:
            return [MODEL_BLEND_COORD, MODEL_NAME_COORD]
        return None

    crds_to_remove = []
    for coord in cube.coords():
        if coord.name() in TIME_COORDS:
            continue
        if blend_dim in cube.coord_dims(coord):
            crds_to_remove.append(coord.name())
    return crds_to_remove


def update_blended_metadata(
    cube: Cube,
    blend_coord: str,
    coords_to_remove: Optional[List[str]] = None,
    cycletime: Optional[str] = None,
    attributes_dict: Optional[Dict[str, str]] = None,
    model_id_attr: Optional[str] = None,
) -> None:
    """
    Update metadata as required after blending
    - For cycle and model blending, set a single forecast reference time
    and period using current cycletime
    - For model blending, add attribute detailing the contributing models
    - Remove scalar coordinates that were previously associated with the
    blend dimension
    - Update attributes as specified via process arguments
    - Set any missing mandatory arguments to their default values

    Modifies cube in place.

    Args:
        cube:
            Blended cube
        blend_coord:
            Name of coordinate over which blending has been performed
        coords_to_remove:
            Name of scalar coordinates to be removed from the blended cube
        cycletime:
            Current cycletime in YYYYMMDDTHHmmZ format
        model_id_attr:
            Name of attribute for use in model blending, to record the names of
            contributing models on the blended output
        attributes_dict:
            Optional user-defined attributes to add to the cube
    """
    if blend_coord in ["forecast_reference_time", MODEL_BLEND_COORD]:
        _set_blended_time_coords(cube, cycletime)

    if blend_coord == MODEL_BLEND_COORD:
        (contributing_models,) = cube.coord(MODEL_NAME_COORD).points
        # iris concatenates string coordinates as a "|"-separated string
        cube.attributes[model_id_attr] = " ".join(
            sorted(contributing_models.split("|"))
        )

    if coords_to_remove is not None:
        for coord in coords_to_remove:
            cube.remove_coord(coord)

    if attributes_dict is not None:
        amend_attributes(cube, attributes_dict)

    for attr in MANDATORY_ATTRIBUTES:
        if attr not in cube.attributes:
            cube.attributes[attr] = MANDATORY_ATTRIBUTE_DEFAULTS[attr]


def _set_blended_time_coords(blended_cube: Cube, cycletime: Optional[str]) -> None:
    """
    For cycle and model blending:
    - Add a "blend_time" coordinate equal to the current cycletime
    - Update the forecast reference time and forecast period coordinate points
    to reflect the current cycle time (behaviour is DEPRECATED)
    - Remove any bounds from the forecast reference time (behaviour is DEPRECATED)
    - Mark the forecast reference time and forecast period as DEPRECATED

    Modifies cube in place.

    Args:
        blended_cube
        cycletime:
            Current cycletime in YYYYMMDDTHHmmZ format
    """
    try:
        cycletime_point = _get_cycletime_point(blended_cube, cycletime)
    except TypeError:
        raise ValueError("Current cycle time is required for cycle and model blending")

    add_blend_time(blended_cube, cycletime)
    blended_cube.coord("forecast_reference_time").points = [cycletime_point]
    blended_cube.coord("forecast_reference_time").bounds = None
    if blended_cube.coords("forecast_period"):
        blended_cube.remove_coord("forecast_period")
    new_forecast_period = forecast_period_coord(blended_cube)
    time_dim = blended_cube.coord_dims("time")
    blended_cube.add_aux_coord(new_forecast_period, data_dims=time_dim)
    for coord in ["forecast_period", "forecast_reference_time"]:
        msg = f"{coord} will be removed in future and should not be used"
        blended_cube.coord(coord).attributes.update({"deprecation_message": msg})


def _get_cycletime_point(cube: Cube, cycletime: str) -> int64:
    """
    For cycle and model blending, establish the current cycletime to set on
    the cube after blending.

    Args:
        blended_cube
        cycletime:
            Current cycletime in YYYYMMDDTHHmmZ format

    Returns:
        Cycle time point in units matching the input cube forecast reference
        time coordinate
    """
    frt_coord = cube.coord("forecast_reference_time")
    frt_units = frt_coord.units.origin
    frt_calendar = frt_coord.units.calendar
    # raises TypeError if cycletime is None
    cycletime_point = cycletime_to_number(
        cycletime, time_unit=frt_units, calendar=frt_calendar
    )
    return round_close(cycletime_point, dtype=np.int64)


def set_record_run_attr(
    cubelist: CubeList, record_run_attr: str, model_id_attr: Optional[str]
) -> None:
    """Set a record_run attribute that records the model identifier and
    forecast reference time of each cube in the cubelist. From the list of cubes,
    pre-existing record_run attributes, model IDs and forecast reference
    times are extracted as required to build a new record_run attribute.

    The new attribute is applied to each cube in the cubelist in preparation
    for blending / combining the cubes. The resulting composite product will
    have a record of the contributing models and their associated forecast
    reference times.

    There are three ways this method may work:

      - None of the input cubes have an existing record_run attribute.
        The model_id_attr argument must be provided to enable the model
        identifiers to be extracted and used in conjunction with the forecast
        reference time to build the record_run attribute.
      - All of the input cubes have an existing record_run attribute. The
        model_id_attr argument is not required as a new record_run attribute
        will be constructed by combining the existing record_run attributes on
        each input cube.
      - Some of the input cubes have an existing record_run attribute, and some
        have not. The model_id_attr argument must be provided so that those cubes
        without an existing record_run attribute can be interrogated for their
        model identifier.

    The cubes are modified in place.

    Args:
        cubelist:
            Cubes from which to obtain model and cycle information, and to which
            the resulting run record attribute is added.
        record_run_attr:
            The name of the record run attribute that is to be created.
        model_id_attr:
            The name of the attribute that contains the source model information.

    Raises:
        ValueError: If model_id_attr is not set and is required to construct a
                    new record_run_attr.
        Exception: The model_id_attr name provided is not present on one or more
                   of the input cubes.
    """
    if not model_id_attr and not all(
        [record_run_attr in cube.attributes for cube in cubelist]
    ):
        raise ValueError(
            f"Not all input cubes contain an existing {record_run_attr} attribute. "
            "A model_id_attr argument must be provided to enable the construction "
            f"of a new {record_run_attr} attribute."
        )

    cycle_strings = []
    for cube in cubelist:
        if record_run_attr in cube.attributes:
            model_attrs = cube.attributes[record_run_attr].splitlines()
            for model_attr in model_attrs:
                if model_attr not in cycle_strings:
                    cycle_strings.append(model_attr)
            continue

        if model_id_attr not in cube.attributes:
            raise Exception(
                f"Failure to record run information in '{record_run_attr}' "
                "during blend: no model id attribute found in cube. "
                f"Cube attributes: {cube.attributes}"
            )

        cycle = datetime.utcfromtimestamp(
            cube.coord("forecast_reference_time").points[0]
        )
        cycle_str = cycle.strftime("%Y%m%dT%H%MZ")

        blending_weight = ""  # TODO: include actual blending weight here.
        run_attr = f"{cube.attributes[model_id_attr]}:{cycle_str}:{blending_weight}"
        if run_attr not in cycle_strings:
            cycle_strings.append(run_attr)

    cycle_strings.sort()
    for cube in cubelist:
        cube.attributes[record_run_attr] = "\n".join(cycle_strings)
