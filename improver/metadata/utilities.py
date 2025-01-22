# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""General IMPROVER metadata utilities"""

import hashlib
import pprint
from typing import Any, Dict, List, Optional, Type, Union

import dask.array as da
import iris
import numpy as np
from cf_units import Unit
from iris.cube import Cube, CubeList
from numpy import ndarray
from numpy.ma.core import MaskedArray

from improver.metadata.constants.attributes import (
    MANDATORY_ATTRIBUTE_DEFAULTS,
    MANDATORY_ATTRIBUTES,
)
from improver.metadata.constants.time_types import TIME_COORDS


def create_new_diagnostic_cube(
    name: str,
    units: Union[Unit, str],
    template_cube: Cube,
    mandatory_attributes: Union[Dict[str, str], Dict],
    optional_attributes: Optional[Union[Dict[str, str], Dict]] = None,
    data: Optional[Union[MaskedArray, ndarray]] = None,
    dtype: Type = np.float32,
) -> Cube:
    """
    Creates a new diagnostic cube with suitable metadata.

    Args:
        name:
            Standard or long name for output cube
        units:
            Units for output cube
        template_cube:
            Cube from which to copy dimensional and auxiliary coordinates
        mandatory_attributes:
            Dictionary containing values for the mandatory attributes
            "title", "source" and "institution".  These are overridden by
            values in the optional_attributes dictionary, if specified.
        optional_attributes:
            Dictionary of optional attribute names and values.  If values for
            mandatory attributes are included in this dictionary they override
            the values of mandatory_attributes.
        data:
            Data array.  If not set, cube is filled with zeros using a lazy
            data object, as this will be overwritten later by the caller
            routine.
        dtype:
            Datatype for dummy cube data if "data" argument is None.

    Returns:
        Cube with correct metadata to accommodate new diagnostic field
    """
    attributes = mandatory_attributes
    if optional_attributes is not None:
        attributes.update(optional_attributes)

    error_msg = ""
    for attr in MANDATORY_ATTRIBUTES:
        if attr not in attributes:
            error_msg += "{} attribute is required\n".format(attr)
    if error_msg:
        raise ValueError(error_msg)

    if data is None:
        data = da.zeros_like(template_cube.core_data(), dtype=dtype)

    aux_coords_and_dims, dim_coords_and_dims = [
        [
            (coord.copy(), template_cube.coord_dims(coord))
            for coord in getattr(template_cube, coord_type)
        ]
        for coord_type in ("aux_coords", "dim_coords")
    ]

    cube = iris.cube.Cube(
        data,
        units=units,
        attributes=attributes,
        dim_coords_and_dims=dim_coords_and_dims,
        aux_coords_and_dims=aux_coords_and_dims,
    )
    cube.rename(name)

    return cube


def generate_mandatory_attributes(
    diagnostic_cubes: List[Cube], model_id_attr: Optional[str] = None
) -> Dict[str, str]:
    """
    Function to generate mandatory attributes for new diagnostics that are
    generated using several different model diagnostics as input to the
    calculation.  If all input diagnostics have the same attribute use this,
    otherwise set a default value.

    Args:
        diagnostic_cubes:
            List of diagnostic cubes used in calculating the new diagnostic
        model_id_attr:
            Name of attribute used to identify source model for blending,
            if required

    Returns:
        Dictionary of mandatory attribute "key": "value" pairs.
    """
    missing_value = object()
    attr_dicts = [cube.attributes for cube in diagnostic_cubes]
    required_attributes = [model_id_attr] if model_id_attr else []
    attributes = MANDATORY_ATTRIBUTE_DEFAULTS.copy()
    for attr in MANDATORY_ATTRIBUTES + required_attributes:
        unique_values = {d.get(attr, missing_value) for d in attr_dicts}
        if len(unique_values) == 1 and missing_value not in unique_values:
            (attributes[attr],) = unique_values
        elif attr in required_attributes:
            msg = (
                'Required attribute "{}" is missing or '
                "not the same on all input cubes"
            )
            raise ValueError(msg.format(attr))
    return attributes


def generate_hash(data_in: Any) -> str:
    """
    Generate a hash from the data_in that can be used to uniquely identify
    equivalent data_in.

    Args:
        data_in:
            The data from which a hash is to be generated. This can be of any
            type that can be pretty printed.

    Returns:
        A hexadecimal string which is a hash hexdigest of the data as a
        string.
    """
    bytestring = pprint.pformat(data_in).encode("utf-8")
    return hashlib.sha256(bytestring).hexdigest()


def create_coordinate_hash(cube: Cube) -> str:
    """
    Generate a hash based on the input cube's x and y coordinates. This
    acts as a unique identifier for the grid which can be used to allow two
    grids to be compared.

    Args:
        cube:
            The cube from which x and y coordinates will be used to
            generate a hash.

    Returns:
        A hash created using the x and y coordinates of the input cube.
    """
    hashable_data = []
    for axis in ("x", "y"):
        coord = cube.coord(axis=axis)
        hashable_data.extend(
            [
                list(coord.points),
                list(coord.bounds) if isinstance(coord.bounds, list) else None,
                coord.standard_name,
                coord.long_name,
                coord.coord_system,
                coord.units,
            ]
        )
    return generate_hash(hashable_data)


def check_grid_match(cubes: Union[List[Cube], CubeList]) -> None:
    """
    Checks that cubes are on, or originate from, compatible coordinate grids.
    Each cube is first checked for an existing 'model_grid_hash' which can be
    used to encode coordinate information on cubes that do not themselves
    contain a coordinate grid (e.g. spotdata cubes). If this is not found a new
    hash is generated to enable comparison. If the cubes are not compatible, an
    exception is raised to prevent the use of unmatched cubes.

    Args:
        cubes:
            A list of cubes to check for grid compatibility.

    Raises:
        ValueError: Raised if the cubes are not on matching grids as
                    identified by the model_grid_hash.
    """

    def _get_grid_hash(cube):
        try:
            cube_hash = cube.attributes["model_grid_hash"]
        except KeyError:
            cube_hash = create_coordinate_hash(cube)
        return cube_hash

    cubes = iter(cubes)
    reference_hash = _get_grid_hash(next(cubes))

    for cube in cubes:
        cube_hash = _get_grid_hash(cube)
        if cube_hash != reference_hash:
            raise ValueError(
                "Cubes do not share or originate from the same "
                "grid, so cannot be used together."
            )


def get_model_id_attr(cubes: List[Cube], model_id_attr: str) -> str:
    """
    Gets the specified model ID attribute from a list of input cubes, checking
    that the value is the same on all those cubes in the process.

    Args:
        cubes:
            List of cubes to get the attribute from
        model_id_attr:
            Attribute name

    Returns:
        The unique attribute value

    """
    try:
        model_id_value = {cube.attributes[model_id_attr] for cube in cubes}
    except KeyError as error:
        failing_cubes = [
            cube.name()
            for cube in cubes
            if not cube.attributes.get(model_id_attr, False)
        ]
        raise ValueError(
            f"Model ID attribute {model_id_attr} not present for {', '.join(failing_cubes)}."
        ) from error
    if len(model_id_value) != 1:
        raise ValueError(
            f"Attribute {model_id_attr} must be the same for all input cubes. "
            f"{' != '.join(model_id_value)}"
        )
    (model_id_value,) = model_id_value
    return model_id_value


def enforce_time_point_standard(cube: Cube):
    """
    Enforce the IMPROVER standard of a coordinate point that aligns with the
    upper bound of the period for time, forecast_period, and forecast
    reference time coordinates.

    The cube is modified in place.

    Args:
        cube:
            Cube to enforce the IMPROVER standard on.
    """
    for crd in TIME_COORDS:
        try:
            cube.coord(crd).points = [bound[-1] for bound in cube.coord(crd).bounds]
        except (iris.exceptions.CoordinateNotFoundError, TypeError):
            pass
