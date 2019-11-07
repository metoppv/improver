# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown Copyright 2017-2019 Met Office.
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
"""Module updating blended metadata"""

import numpy as np

from improver.metadata.amend import amend_attributes
from improver.metadata.constants import TIME_COORDINATES
from improver.utilities.temporal import (
    cycletime_to_number, forecast_period_coord)


TIME_COORDINATES = ["time", "forecast_period", "forecast_reference_time"]


def _update_blended_forecast_reference_time(
        cube, reference_frt_coord, cycletime):
    """
    For model or cycle blended cubes, a single forecast reference time is set.
    This may be user-specified, otherwise will correspond to the most recent
    contributing forecast.  Modifies cube in place.

    Args:
        cube (iris.cube.Cube):
            Cube containing the metadata to be adjusted
        frt_coord (iris.coords.Coord):
            Reference forecast_reference_time coordinate
        cycletime (str or None):
            The cycletime in a YYYYMMDDTHHMMZ format e.g. 20171122T0100Z
    """
    frt_coord = cube.coord("forecast_reference_time")
    frt_units = frt_coord.units.origin

    if cycletime is None:
        reference_frt_coord.convert_units(frt_units)
        new_cycletime = np.max(reference_frt_coord.points)
    else:
        frt_calendar = frt_coord.units.calendar
        new_cycletime = cycletime_to_number(
                cycletime, time_unit=frt_units, calendar=frt_calendar)
        frt_type = frt_coord.dtype
        new_cycletime = np.round(new_cycletime).astype(frt_type)
    cube.coord("forecast_reference_time").points = new_cycletime
    cube.coord("forecast_reference_time").bounds = None


def _recalculate_forecast_period(cube):
    """
    Recalculates the forecast period in place and associates it with the
    time dimension, where the forecast reference time has been updated.
    """
    if cube.coords("forecast_period"):
        cube.remove_coord("forecast_period")
    new_forecast_period = forecast_period_coord(cube)
    time_dim = cube.coord_dims("time")
    cube.add_aux_coord(new_forecast_period, data_dims=time_dim)


def update_blended_metadata(
        cube, blend_coord, frt_coord, cycletime=None, attributes_dict=None):
    """
    Update metadata after blending.  Modifies cube in place.

    Args:
        cube (iris.cube.Cube):
            Cube containing the metadata to be adjusted
        blend_coord (str):
            Name of the coordinate that has been blended. This allows updates
            to forecast reference time and attributes to be restricted to cycle
            and model blends.
        frt_coord (iris.coords.Coord or None):
            Reference forecast reference time coordinate, or None.  Required
            for cycle and model blending.
        cycletime (str or None):
            The cycletime in a YYYYMMDDTHHMMZ format e.g. 20171122T0100Z
        attributes_dict (dict or None):
            Changes to be made to attributes on the blended cube
    """
    if blend_coord in ["forecast_reference_time", "model_id"]:
        _update_blended_forecast_reference_time(cube, frt_coord, cycletime)
        _recalculate_forecast_period(cube)

    # update attributes
    if attributes_dict is not None:
        amend_attributes(cube, attributes_dict)

    # remove appropriate scalar coordinates
    crds_to_remove = []
    if blend_coord == "model_id":
        crds_to_remove = ["model_id", "model_configuration"]
    elif blend_coord not in TIME_COORDINATES:
        crds_to_remove = [blend_coord]

    for crd in crds_to_remove:
        if cube.coords(crd) and cube.coord(crd).shape == (1,):
            cube.remove_coord(crd)
