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
"""Utilities to manipulate forecast time coordinates"""

import warnings

import iris
import numpy as np
from cf_units import Unit
from iris.exceptions import CoordinateNotFoundError

from improver.metadata.constants import FLOAT_TYPES
from improver.metadata.constants.time_types import TIME_COORDS
from improver.metadata.check_datatypes import check_mandatory_standards
from improver.utilities.cube_manipulation import build_coordinate
from improver.utilities.temporal import cycletime_to_datetime


def forecast_period_coord(cube, force_lead_time_calculation=False):
    """
    Return the lead time coordinate (forecast_period) from a cube, either by
    reading an existing forecast_period coordinate, or by calculating the
    difference between time and forecast_reference_time.

    Args:
        cube (iris.cube.Cube):
            Cube from which the lead times will be determined.
        force_lead_time_calculation (bool):
            Force the lead time to be calculated from the
            forecast_reference_time and the time coordinate, even if
            the forecast_period coordinate exists. Default is False.

    Returns:
        iris.coords.Coord:
            New forecast_period coord. A DimCoord is returned if the
            forecast_period coord is already present in the cube as a
            DimCoord and this coord does not need changing, otherwise
            it will be an AuxCoord.
    """
    create_dim_coord = False
    if cube.coords("forecast_period"):
        if isinstance(cube.coord("forecast_period"), iris.coords.DimCoord):
            create_dim_coord = True

    if cube.coords("forecast_period") and not force_lead_time_calculation:
        result_coord = cube.coord("forecast_period").copy()

    elif cube.coords("time") and cube.coords("forecast_reference_time"):
        # Cube must adhere to mandatory standards for safe time calculations
        check_mandatory_standards(cube)
        # Try to calculate forecast period from forecast reference time and
        # time coordinates
        result_coord = _calculate_forecast_period(
            cube.coord("time"), cube.coord("forecast_reference_time"),
            dim_coord=create_dim_coord)

    else:
        msg = ("The forecast period coordinate is not available within {}."
               "The time coordinate and forecast_reference_time "
               "coordinate were also not available for calculating "
               "the forecast_period.".format(cube))
        raise CoordinateNotFoundError(msg)

    return result_coord


def _calculate_forecast_period(time_coord, frt_coord, dim_coord=False,
                               coord_spec=TIME_COORDS['forecast_period']):
    """
    Calculate a forecast period from existing time and forecast reference
    time coordinates.

    Args:
        time_coord (iris.coords.Coord):
            Time coordinate
        frt_coord (iris.coords.Coord):
            Forecast reference coordinate
        dim_coord (bool):
            If true, create an iris.coords.DimCoord instance.  Default is to
            create an iris.coords.AuxCoord.
        coord_spec (collections.namedtuple):
            Specification of units and dtype for the forecast_period
            coordinate.

    Returns:
        iris.coords.Coord:
            Forecast period coordinate corresponding to the input times and
            forecast reference times specified

    Warns:
        UserWarning: If any calculated forecast periods are negative
    """
    # use cell() access method to get datetime.datetime instances
    time_points = np.array([c.point for c in time_coord.cells()])
    forecast_reference_time_points = np.array(
        [c.point for c in frt_coord.cells()])
    required_lead_times = (
        time_points - forecast_reference_time_points)
    required_lead_times = np.array(
        [x.total_seconds() for x in required_lead_times])

    if time_coord.bounds is not None:
        time_bounds = np.array([c.bound for c in time_coord.cells()])
        required_lead_time_bounds = (
            time_bounds - forecast_reference_time_points)
        required_lead_time_bounds = np.array(
            [[b.total_seconds() for b in x]
             for x in required_lead_time_bounds])
    else:
        required_lead_time_bounds = None

    coord_type = iris.coords.DimCoord if dim_coord else iris.coords.AuxCoord
    result_coord = coord_type(
        required_lead_times,
        standard_name='forecast_period',
        bounds=required_lead_time_bounds,
        units="seconds")

    result_coord.convert_units(coord_spec.units)

    if coord_spec.dtype not in FLOAT_TYPES:
        result_coord.points = np.around(result_coord.points)
        if result_coord.bounds is not None:
            result_coord.bounds = np.around(result_coord.bounds)

    result_coord.points = result_coord.points.astype(coord_spec.dtype)
    if result_coord.bounds is not None:
        result_coord.bounds = result_coord.bounds.astype(coord_spec.dtype)

    if np.any(result_coord.points < 0):
        msg = ("The values for the time {} and "
               "forecast_reference_time {} coordinates from the "
               "input cube have produced negative values for the "
               "forecast_period. A forecast does not generate "
               "values in the past.").format(time_coord.points,
                                             frt_coord.points)
        warnings.warn(msg)

    return result_coord


def rebadge_forecasts_as_latest_cycle(cubes, cycletime=None):
    """
    Function to update the forecast_reference_time and forecast_period
    on a list of input forecasts to match either a given cycletime, or
    the most recent forecast in the list (proxy for the current cycle).

    Args:
        cubes (iris.cube.CubeList or list of iris.cube.Cube):
            Cubes that will have their forecast_reference_time and
            forecast_period updated.
        cycletime (str or None):
            Required forecast reference time in a YYYYMMDDTHHMMZ format
            e.g. 20171122T0100Z. If None, the latest forecast reference
            time is used.

    Returns:
        iris.cube.CubeList:
            Updated cubes
    """
    if cycletime is None and len(cubes) == 1:
        return cubes
    cycle_datetime = (find_latest_cycletime(cubes) if cycletime is None
                      else cycletime_to_datetime(cycletime))
    return unify_cycletime(cubes, cycle_datetime)


def unify_cycletime(cubes, cycletime):
    """
    Function to unify the forecast_reference_time and update forecast_period.
    The cycletime specified is used as the forecast_reference_time, and the
    forecast_period is recalculated using the time coordinate and updated
    forecast_reference_time.

    Args:
        cubes (iris.cube.CubeList or list of iris.cube.Cube):
            Cubes that will have their forecast_reference_time and
            forecast_period updated. Any bounds on the forecast_reference_time
            coordinate will be discarded.
        cycletime (datetime.datetime):
            Datetime for the cycletime that will be used to replace the
            forecast_reference_time on the individual cubes.

    Returns:
        iris.cube.CubeList:
            Updated cubes

    Raises:
        ValueError: if forecast_reference_time is a dimension coordinate
    """
    result_cubes = iris.cube.CubeList([])
    for cube in cubes:
        cube = cube.copy()
        frt_coord_name = 'forecast_reference_time'
        coord_type_spec = TIME_COORDS[frt_coord_name]
        coord_units = Unit(coord_type_spec.units)
        frt_points = np.around(
            [coord_units.date2num(cycletime)]).astype(coord_type_spec.dtype)
        frt_coord = build_coordinate(
            frt_points, standard_name=frt_coord_name, bounds=None,
            template_coord=cube.coord(frt_coord_name),
            units=coord_units)

        cube.remove_coord(frt_coord_name)
        cube.add_aux_coord(frt_coord, data_dims=None)

        # Update the forecast period for consistency within each cube
        if cube.coords("forecast_period"):
            cube.remove_coord("forecast_period")
        fp_coord = forecast_period_coord(
            cube, force_lead_time_calculation=True)
        cube.add_aux_coord(fp_coord, data_dims=cube.coord_dims("time"))
        result_cubes.append(cube)
    return result_cubes


def find_latest_cycletime(cubelist):
    """
    Find the latest cycletime from the cubes in a cubelist and convert it into
    a datetime object.

    Args:
        cubelist (iris.cube.CubeList or list of iris.cube.Cube):
            A list of cubes each containing single time step from different
            forecast cycles.

    Returns:
        datetime.datetime:
            A datetime object corresponding to the latest forecast reference
            time in the input cubelist.
    """
    # Get cycle time as latest forecast reference time
    if any([cube.coord_dims("forecast_reference_time")
            for cube in cubelist]):
        raise ValueError(
            "Expecting scalar forecast_reference_time for each input "
            "cube - cannot replace a dimension coordinate")

    frt_coord = cubelist[0].coord("forecast_reference_time").copy()
    for cube in cubelist:
        next_coord = cube.coord("forecast_reference_time").copy()
        next_coord.convert_units(frt_coord.units)
        if next_coord.points[0] > frt_coord.points[0]:
            frt_coord = next_coord
    cycletime, = frt_coord.units.num2date(frt_coord.points)
    return cycletime
