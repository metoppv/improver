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
"""Module to generate a metadata cube."""

from datetime import datetime, timedelta
from typing import Any, List, Optional, Tuple

import numpy as np
from iris.cube import Cube
from iris.std_names import STD_NAMES
from iris.util import squeeze
from numpy import ndarray

from improver.synthetic_data.set_up_test_cubes import (
    set_up_percentile_cube,
    set_up_probability_cube,
    set_up_variable_cube,
)

DEFAULT_GRID_SPACING = {"latlon": 0.02, "equalarea": 2000}
DEFAULT_SPATIAL_GRID = "latlon"
DEFAULT_TIME = datetime(2017, 11, 10, 4, 0)
CUBE_TYPES = ("variable", "percentile", "probability")


def _get_units(name: str) -> str:
    """ Get output variable units from iris.std_names.STD_NAMES """
    try:
        units = STD_NAMES[name]["canonical_units"]
    except KeyError:
        raise ValueError("Units of {} are not known.".format(name))

    return units


def _create_time_bounds(time: datetime, time_period: int) -> Tuple[datetime, datetime]:
    """ Create time bounds using time - time_period as the lower bound and time as the
    upper bound"""
    lower_bound = time - timedelta(minutes=time_period)
    upper_bound = time

    return (lower_bound, upper_bound)


def _create_data_array(
    ensemble_members: int,
    leading_dimension: Optional[List[float]],
    npoints: int,
    height_levels: Optional[List[float]],
) -> ndarray:
    """ Create data array of specified shape filled with zeros """
    if leading_dimension is not None:
        nleading_dimension = len(leading_dimension)
    else:
        nleading_dimension = ensemble_members

    if height_levels is not None:
        nheight_levels = len(height_levels)
    else:
        nheight_levels = None

    data_shape = []

    if nleading_dimension > 1:
        data_shape.append(nleading_dimension)
    if nheight_levels is not None:
        data_shape.append(nheight_levels)

    data_shape.append(npoints)
    data_shape.append(npoints)

    return np.zeros(data_shape, dtype=np.float32)


def generate_metadata(
    name: str = "air_pressure_at_sea_level",
    units: Optional[str] = None,
    time_period: Optional[int] = None,
    ensemble_members: int = 8,
    leading_dimension: Optional[List[float]] = None,
    cube_type: str = "variable",
    spp__relative_to_threshold: str = "greater_than",
    npoints: int = 71,
    **kwargs: Any,
) -> Cube:
    """ Generate a cube with metadata only.

    Args:
        name:
            Output variable name, or if creating a probability cube the name of the
            underlying variable to which the probability field applies.
        units:
            Output variable units, or if creating a probability cube the units of the
            underlying variable / threshold.
        time_period:
            The period in minutes between the time bounds. This is used to calculate
            the lower time bound. If unset the diagnostic will be instantaneous, i.e.
            without time bounds.
        ensemble_members:
            Number of ensemble members. Default 8, unless percentile or probability set
            to True.
        leading_dimension:
            List of realizations, percentiles or thresholds.
        cube_type:
            The type of cube to be generated. Permitted values are "variable",
            "percentile" or "probability".
        spp__relative_to_threshold:
            Value of the attribute "spp__relative_to_threshold" which is required for
            IMPROVER probability cubes.
        npoints:
            Number of points along each of the y and x spatial axes.
        **kwargs:
            Additional keyword arguments to pass to the required cube setup function.

    Returns:
        Output of set_up_variable_cube(), set_up_percentile_cube() or
        set_up_probability_cube()
    """
    if cube_type not in CUBE_TYPES:
        raise ValueError(
            'Cube type {} not supported. Specify one of "variable", "percentile" or "probability".'.format(
                cube_type
            )
        )

    if "spatial_grid" in kwargs and kwargs["spatial_grid"] not in (
        "latlon",
        "equalarea",
    ):
        raise ValueError(
            "Spatial grid {} not supported. Specify either latlon or equalarea.".format(
                kwargs["spatial_grid"]
            )
        )

    if (
        "domain_corner" in kwargs
        and kwargs["domain_corner"] is not None
        and len(kwargs["domain_corner"]) != 2
    ):
        raise ValueError("Domain corner must be a list or tuple of length 2.")

    if units is None:
        units = _get_units(name)

    # If time_period specified, create time bounds using time as upper bound
    if time_period is not None:
        if "time" not in kwargs:
            kwargs["time"] = DEFAULT_TIME

        time_bounds = _create_time_bounds(kwargs["time"], time_period)
        kwargs["time_bounds"] = time_bounds

    # If grid_spacing not specified, use default for requested spatial grid
    if "grid_spacing" not in kwargs or kwargs["grid_spacing"] is None:
        if "spatial_grid" not in kwargs:
            kwargs["spatial_grid"] = DEFAULT_SPATIAL_GRID

        kwargs["grid_spacing"] = DEFAULT_GRID_SPACING[kwargs["spatial_grid"]]

    # Create ndimensional array of zeros
    if "height_levels" not in kwargs:
        kwargs["height_levels"] = None

    data = _create_data_array(
        ensemble_members, leading_dimension, npoints, kwargs["height_levels"]
    )

    # Set up requested cube
    if cube_type == "percentile":
        metadata_cube = set_up_percentile_cube(
            data, percentiles=leading_dimension, name=name, units=units, **kwargs,
        )
    elif cube_type == "probability":
        metadata_cube = set_up_probability_cube(
            data,
            leading_dimension,
            variable_name=name,
            threshold_units=units,
            spp__relative_to_threshold=spp__relative_to_threshold,
            **kwargs,
        )
    else:
        metadata_cube = set_up_variable_cube(
            data, name=name, units=units, realizations=leading_dimension, **kwargs,
        )

    metadata_cube = squeeze(metadata_cube)

    return metadata_cube
