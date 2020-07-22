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

import numpy as np
from iris.std_names import STD_NAMES

from improver.metadata.constants.mo_attributes import MOSG_GRID_DEFINITION
from improver.synthetic_data.set_up_test_cubes import set_up_variable_cube
from improver.utilities.temporal import cycletime_to_datetime


def _get_unit(name):
    try:
        units = STD_NAMES[name]["canonical_units"]
        return units
    except KeyError:
        raise KeyError("Units of {} are not known.".format(name))


def _create_data_array(ensemble_members, npoints, height_levels):
    data = None
    if ensemble_members > 1:
        data = np.zeros((ensemble_members, npoints, npoints), dtype=int)
    else:
        data = np.zeros((npoints, npoints), dtype=int)

    return data


def generate_metadata(
    name,
    spatial_grid="latlon",
    time="20171110T0400Z",
    frt="20171110T0400Z",
    ensemble_members=8,
    realizations=None,
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
        spatial_grid (Optional[str]):
            What type of x/y coordinate values to use.  Permitted values are
            "latlon" or "equalarea".
        time (Optional[str]):
            Single cube validity time. Datetime string of format YYYYMMDDTHHMMZ.
        frt (Optional[str]):
            Single cube forecast reference time. Datetime string of format YYYYMMDDTHHMMZ.
        ensemble_members (Optional[int]):
            Number of ensemble members.
        realizations (Optional[csv]):
            CSV list of realization input data.
        attributes (Optional[str]):
            JSON file of additional metadata attributes.
        resolution (Optional[float]):
            Resolution of grid (metres or degrees).
        domain_corner (Optional[Tuple[float, float]]):
            Bottom left corner of grid domain (y,x) (degrees for latlon or metres for equalarea).
        npoints (Optional[int]):
            Number of points along a single axis.
        height_levels (Optional[int]):
            Number of altitude/pressure levels.

    Returns:
        iris.cube.Cube:
            Output of set_up_variable_cube()
    """
    units = _get_unit(name)

    if spatial_grid != "latlon" and spatial_grid != "equalarea":
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

    data = None
    if realizations is None:
        data = _create_data_array(ensemble_members, npoints, height_levels)
    else:
        data = realizations

    standard_grid_metadata = None
    if ensemble_members > 1:
        standard_grid_metadata = "uk_ens"
    else:
        standard_grid_metadata = "uk_det"

    # Convert str time and frt to datetime
    time = cycletime_to_datetime(time)
    frt = cycletime_to_datetime(frt)

    metadata_cube = set_up_variable_cube(
        data,
        name=name,
        units=units,
        spatial_grid=spatial_grid,
        time=time,
        # time_bounds=None,
        frt=frt,
        realizations=realizations,
        # include_scalar_coords=None,
        attributes=attributes,
        standard_grid_metadata=standard_grid_metadata,
        grid_spacing=resolution,
        domain_corner=domain_corner,
    )

    return metadata_cube
