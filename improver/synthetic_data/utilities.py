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


def get_height_levels(coord_data: Dict[str, Any]) -> Tuple[List[float], bool]:
    """Gets height level values from coords nested dictionary and sets pressure
    value based on whether heights or pressures key is used.

    Args:
        coord_data:
            Dictionary containing values to use for either height or pressure levels.

    Returns:
        A tuple containing a list of values to use for the height/pressure dimension
        and a bool specifying whether the coordinate should be created as height
        levels or pressure levels.
    """
    height_levels = None
    pressure = False

    if "heights" in coord_data:
        height_levels = coord_data["heights"]
    elif "pressures" in coord_data:
        height_levels = coord_data["pressures"]
        pressure = True

    return height_levels, pressure
