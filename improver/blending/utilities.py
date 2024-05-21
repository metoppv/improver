# (C) Crown copyright, Met Office. All rights reserved.
#
# This file is part of IMPROVER and is released under a BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Utilities to support weighted blending"""

from datetime import datetime
from typing import Dict, List, Optional, Union

import numpy as np
from iris import Constraint
from iris.coords import AuxCoord
from iris.cube import Cube, CubeList
from iris.exceptions import CoordinateNotFoundError
from numpy import int64

from improver.blending import (
    MODEL_BLEND_COORD,
    MODEL_NAME_COORD,
    RECORD_COORD,
    WEIGHT_FORMAT,
)
from improver.metadata.amend import amend_attributes
from improver.metadata.constants.attributes import (
    MANDATORY_ATTRIBUTE_DEFAULTS,
    MANDATORY_ATTRIBUTES,
)
from improver.metadata.constants.time_types import DT_FORMAT, TIME_COORDS
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


def get_coords_to_remove(cube: Cube, blend_coord: str) -> List[str]:
    """
    Generate a list of coordinate names associated with the blend
    dimension.  Unless these are time-related coordinates, they should be
    removed after blending. An empty list is returned if there are no
    coordinates to remove.

    Args:
        cube:
            Cube to be blended
        blend_coord:
            Name of coordinate over which the blend will be performed

    Returns:
        List of names of coordinates to remove
    """
    crds_to_remove = []
    try:
        (blend_dim,) = cube.coord_dims(blend_coord)
    except ValueError:
        # occurs if the blend coordinate is scalar, in which case the
        # RECORD_COORD must be added manually if present as scalar coords
        # are not associated with one another.
        if cube.coords(RECORD_COORD):
            crds_to_remove.append(RECORD_COORD)

        if blend_coord == MODEL_BLEND_COORD:
            crds_to_remove.extend([MODEL_BLEND_COORD, MODEL_NAME_COORD])

        return crds_to_remove

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
    - Update the forecast reference time to reflect the current cycle time
      (behaviour is DEPRECATED)
    - If a forecast period coordinate is present this will be updated relative
      to the current cycle time (behaviour is DEPRECATED)
    - Remove any bounds from the forecast reference time (behaviour is DEPRECATED)
    - Mark any forecast reference time and forecast period coordinates as DEPRECATED

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
        try:
            blended_cube.coord(coord).attributes.update({"deprecation_message": msg})
        except CoordinateNotFoundError:
            pass


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


def store_record_run_as_coord(
    cubelist: CubeList, record_run_attr: str, model_id_attr: Optional[str]
) -> None:
    """Stores model identifiers and forecast_reference_times on the input
    cubes as auxiliary coordinates. These are used to construct record_run
    attributes that can be applied to a cube produced by merging or combining
    these cubes. By storing this information as a coordinate it will persist
    in merged cubes in which forecast_reference_times are otherwise updated.
    It also allows the data to be discarded when cubes for blending are
    filtered to remove zero-weight contributors, ensuring the resulting
    attribute is consistent with what has actually contributed to a blended
    forecast.

    From the list of cubes, pre-existing record_run attributes, model IDs and
    forecast reference times are extracted as required. These are combined
    and stored as a RECORD_COORD auxiliary coordinate. The constructed
    attribute has the form:

        model : cycle : weight

    e.g.:

        uk_ens : 20201105T1200Z : 1.0

    If a new record is being constructed the weight is always set to 1. This
    will be updated using the weights applied in blending the cubes. When
    combining (rather than blending) cubes (e.g. weather symbols) it is
    expected that each input will already possess a record_run attribute
    that contains the real weights used.

    Existing records may contain multiple of these attribute components joined
    with newline characters.

    There are two ways this method may work:

      - All of the input cubes have an existing record_run attribute. The
        model_id_attr argument is not required as the existing record_run
        attribute is used to create the RECORD_COORD.
      - Some or none of the input cubes have an existing record_run attribute.
        The model_id_attr argument must be provided so that those cubes
        without an existing record_run attribute can be interrogated for their
        model identifier. This is used to create a model:cycle:weight
        attribute that is stored in the RECORD_COORD.

    Args:
        cubelist:
            Cubes from which to obtain model and cycle information, or an existing
            record_run attribute.
        record_run_attr:
            The name of the record_run attribute that may exist on the input cubes.
        model_id_attr:
            The name of the model_id attribute that may exist on the input cubes.

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

    for cube in cubelist:
        if record_run_attr in cube.attributes:
            run_attr = cube.attributes[record_run_attr]
            record_coord = AuxCoord([run_attr], long_name=RECORD_COORD)
            cube.add_aux_coord(record_coord)
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
        cycle_str = cycle.strftime(DT_FORMAT)

        blending_weight = 1
        run_attr = (
            f"{cube.attributes[model_id_attr]}:{cycle_str}:"
            f"{blending_weight:{WEIGHT_FORMAT}}"
        )

        record_coord = AuxCoord([run_attr], long_name=RECORD_COORD)
        cube.add_aux_coord(record_coord)


def record_run_coord_to_attr(
    target: Cube,
    source: Union[Cube, CubeList, List[Cube]],
    record_run_attr: str,
    discard_weights: bool = False,
) -> None:
    """
    Extracts record_run entries from the RECORD_COORD auxiliary
    coordinate on the source cube. These are joined together and added
    as the record_run_attr attribute to the target cube.

    Args:
        target:
            Cube to which the record_run_attr should be added. This is the
            product of blending the data in the source cube.
        source:
            The cubes merged together in preparation for blending. This cube
            contains the RECORD_COORD that includes the record_run entries
            to use in constructing the attribute.
        record_run_attr:
            The name of the attribute used to store model and cycle sources.
        discard_weights:
            If cubes have been combined rather than blended the weights may
            well not provide much information, i.e. in weather symbols. They
            may also not be normalised and could give a total contributing
            weight much larger than 1. In these cases, setting discard_weights
            to true will produce an attribute that records only the models
            and cycles without the weights.
    """
    if isinstance(source, Cube):
        source_data = source.coord(RECORD_COORD).points
    else:
        source_data = []
        for cube in source:
            source_data.extend(cube.coord(RECORD_COORD).points)

    if discard_weights:
        all_points = []
        for point in source_data:
            all_points.extend(point.split("\n"))
        all_points = [point.rsplit(":", 1)[0] + ":" for point in all_points]
        source_data = list(set(all_points))
    target.attributes[record_run_attr] = "\n".join(sorted(source_data))


def update_record_run_weights(cube: Cube, weights: Cube, blend_coord: str) -> Cube:
    """
    Update each weight component of record_run to reflect the blending
    weights being applied.

    When cycle blending, the weights recorded in the record_run entries will
    all initially be set to 1 following their creation. These are modified
    to reflect the weight each cycle contributes to the blend.

    When model blending the record_run entries will likely be multiple,
    reflecting the several cycles that have already been blended. These are
    split up so that the weights can be modified to reflect their final
    contribution to the model blend.

    As an example, consider 4 deterministic cycles that have been cycle blended
    using equal weights. Each cycle will have a weight of 0.25. These might
    then be model blended with another model, each contributing 0.5 of the
    total. Each of the 4 deterministic cycles will end up with a modified
    weight of 0.125 contributing to the final blend.

    Args:
        cube:
            The cubes merged together in preparation for blending. These
            contain the RECORD_COORD in which the weights need to be
            updated.
        weights:
            The weights cube that is being applied in the blending. Note
            that these should be simple weights, rather than spatial
            weights. The attribute cannot hope to capture the spatially
            varying weights, so the high-level contribution is recorded
            ignoring local variation.
        blend_coord:
            The coordinate over which blending is being performed.

    Returns:
        A copy of the input cube with the updated weights.
    """
    cubes = CubeList()
    for cslice in cube.slices_over(blend_coord):
        blend_point = cslice.coord(blend_coord).cell(0).point
        weight = weights.extract(
            Constraint(coord_values={blend_coord: lambda cell: cell == blend_point})
        )
        run_records = cslice.coord(RECORD_COORD).points[0].split("\n")
        updated_records = []
        for run_record in run_records:
            components = run_record.rsplit(":", 1)
            model_cycle = components[0]
            try:
                value = float(components[-1]) * weight.data
            except ValueError as err:
                msg = (
                    "record_run attributes are expected to be of the form "
                    "'model_id:model_cycle:blending weight'. The weight is "
                    f"absent in this case: {run_record}. This suggests one "
                    "of the sources of data is using an older version of "
                    "IMPROVER that did not include the weights in this "
                    "attribute."
                )
                raise ValueError(msg) from err

            updated_records.append(f"{model_cycle}:{value:{WEIGHT_FORMAT}}")
        cslice.coord(RECORD_COORD).points = "\n".join(sorted(updated_records))
        cubes.append(cslice)

    return cubes.merge_cube()
