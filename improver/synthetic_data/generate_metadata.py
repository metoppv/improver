# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown Copyright 2017-2020 Met Office.
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

from datetime import datetime

import numpy as np
from iris.std_names import STD_NAMES

from improver.synthetic_data.set_up_test_cubes import set_up_variable_cube


def _get_units(name):
    """ Get output variable units from iris.std_names.STD_NAMES """
    try:
        units = STD_NAMES[name]["canonical_units"]
    except KeyError:
        raise ValueError("Units of {} are not known.".format(name))

    return units


def _create_data_array(ensemble_members, npoints, height_levels):
    """ Create data array of specified shape filled with zeros """
    if height_levels is not None:
        nheights = len(height_levels)

        if ensemble_members > 1:
            data_shape = (ensemble_members, nheights, npoints, npoints)
        else:
            data_shape = (nheights, npoints, npoints)
    elif ensemble_members > 1:
        data_shape = (ensemble_members, npoints, npoints)
    else:
        data_shape = (npoints, npoints)

    return np.zeros(data_shape, dtype=int)


def generate_metadata(
    name="air_temperature",
    units=None,
    spatial_grid="latlon",
    time=datetime(2017, 11, 10, 4, 0),
    frt=datetime(2017, 11, 10, 4, 0),
    ensemble_members=8,
    attributes=None,
    resolution=None,
    domain_corner=None,
    npoints=71,
    height_levels=None,
):
    """ Generate a cube with metadata only.

    Args:
        name (str):
            Output variable name.
        units (Optional[str]):
            Output variable units.
        spatial_grid (Optional[str]):
            What type of x/y coordinate values to use.  Permitted values are
            "latlon" or "equalarea".
        time (Optional[datetime.datetime]):
            Single cube validity time.
        frt (Optional[datetime.datetime]):
            Single cube forecast reference time.
        ensemble_members (Optional[int]):
            Number of ensemble members.
        attributes (Optional[Dict]):
            Dictionary of additional metadata attributes.
        resolution (Optional[float]):
            Resolution of grid (metres or degrees).
        domain_corner (Optional[Tuple[float, float]]):
            Bottom left corner of grid domain (y,x) (degrees for latlon or metres for equalarea).
        npoints (Optional[int]):
            Number of points along a single axis.
        height_levels (Optional[List[float]]):
            List of altitude/pressure levels.

    Returns:
        iris.cube.Cube:
            Output of set_up_variable_cube()
    """
    if units is None:
        units = _get_units(name)

    if spatial_grid not in ("latlon", "equalarea"):
        raise ValueError(
            "Spatial grid {} not supported. Choose either latlon or equalarea.".format(
                spatial_grid
            )
        )

    if resolution is None:
        if spatial_grid == "latlon":
            resolution = 0.02
        elif spatial_grid == "equalarea":
            resolution = 2000

    data = _create_data_array(ensemble_members, npoints, height_levels)

    metadata_cube = set_up_variable_cube(
        data,
        name=name,
        units=units,
        spatial_grid=spatial_grid,
        time=time,
        frt=frt,
        attributes=attributes,
        grid_spacing=resolution,
        domain_corner=domain_corner,
        height_levels=height_levels,
    )

    return metadata_cube
