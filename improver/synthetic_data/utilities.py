# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""
Functions required for additional processing in generate_metadata_cube CLI.
"""

from typing import Any, Dict, List, Tuple


def _error_more_than_one_leading_dimension():
    """Raises an error to inform the user that only one leading dimension can be
    provided in the input data."""
    raise ValueError(
        'Only one of "realization", "percentile" or "probability" dimensions should be provided.'
    )


def get_leading_dimension(coord_data: Dict[str, Any]) -> Tuple[List[float], str]:
    """Gets leading dimension values from coords nested dictionary and sets cube
    type based on what dimension key is used.

    Args:
        coord_data:
            Dictionary containing values to use for either realizations, percentiles or
            thresholds.

    Returns:
        A tuple containing the list of values to use for the leading dimension and
        a string specifying what cube type to create.
    """
    leading_dimension = None
    cube_type = "variable"

    if "realizations" in coord_data:
        leading_dimension = coord_data["realizations"]

    if "percentiles" in coord_data:
        if leading_dimension is not None:
            _error_more_than_one_leading_dimension()

        leading_dimension = coord_data["percentiles"]
        cube_type = "percentile"

    if "thresholds" in coord_data:
        if leading_dimension is not None:
            _error_more_than_one_leading_dimension()

        leading_dimension = coord_data["thresholds"]
        cube_type = "probability"

    return leading_dimension, cube_type


def get_vertical_levels(coord_data: Dict[str, Any]) -> Tuple[List[float], bool, bool]:
    """Gets vertical level values from coords nested dictionary and sets pressure
    value based on whether heights or pressures key is used.

    Args:
        coord_data:
            Dictionary containing values to use for either height or pressure levels.

    Returns:
        A tuple containing a list of values to use for the height/pressure dimension
        and a bool specifying whether the coordinate should be created as height
        levels or pressure levels.
    """
    vertical_levels = None
    pressure = False
    height = False

    if "heights" in coord_data:
        vertical_levels = coord_data["heights"]
        height = True
    elif "pressures" in coord_data:
        vertical_levels = coord_data["pressures"]
        pressure = True

    return vertical_levels, pressure, height
