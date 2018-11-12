# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown Copyright 2017-2018 Met Office.
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
Functions to set up cubes for unit tests
"""

from cf_units import Unit, date2num
from datetime import datetime

import numpy as np
import iris
from iris.coords import DimCoord

TIME_UNIT = "seconds since 1970-01-01 00:00:00"
CALENDAR = "gregorian"


def construct_xy_coords(ypoints, xpoints, spatial_grid):
    """
    Construct x/y spatial dimension coordinates

    Args:
        ypoints (int):
            Number of grid points required along the y-axis
        xpoints (int):
            Number of grid points required along the x-axis
        spatial_grid (str):
            Specifier to produce either a lat-lon or equal area grid

    Returns:
        y_coord, x_coord (tuple):
            Tuple of iris.coord.DimCoord instances
    """
    if spatial_grid == 'latlon':
        # make a lat-lon grid including the UK area
        y_coord = DimCoord(
            np.linspace(-20.0, 20.0, ypoints, dtype=np.float32),
            "latitude", units="degrees")
        x_coord = DimCoord(
            np.linspace(40, 80, xpoints, dtype=np.float32),
            "longitude", units="degrees")
    else:
        # use UK eastings and northings
        y_coord = DimCoord(
            np.linspace(-100000, 900000, ypoints, dtype=np.float32),
            "projection_y_coordinate", units="metres")
        x_coord = DimCoord(
            np.linspace(-400000, 600000, xpoints, dtype=np.float32),
            "projection_x_coordinate", units="metres")
    return y_coord, x_coord


def construct_scalar_time_coords(time, frt):
    """
    Construct scalar time coordinates as aux_coord list

    Args:
        time (datetime.datetime or None):
            Single time point
        frt (datetime.datetime or None):
            Single forecast reference time point

    Returns:
        coord_dims (list):
            List of iris.coord.DimCoord instances with the associated "None"
            dimension (format required by iris.cube.Cube initialisation).
    """
    time_point_seconds = (
        date2num(time, TIME_UNIT, CALENDAR) if time is not None else
        date2num(datetime(2017, 11, 10, 4, 0), TIME_UNIT, CALENDAR))
    frt_point_seconds = (
        date2num(frt, TIME_UNIT, CALENDAR) if frt is not None else
        date2num(datetime(2017, 11, 10, 0, 0), TIME_UNIT, CALENDAR))
    fp_point_seconds = time_point_seconds - frt_point_seconds

    time_coord = DimCoord(time_point_seconds, "time", units=TIME_UNIT)
    frt_coord = DimCoord(
        frt_point_seconds, "forecast_reference_time", units=TIME_UNIT)
    fp_coord = DimCoord(fp_point_seconds, "forecast_period", units="seconds")

    coord_dims = [(time_coord, None), (frt_coord, None), (fp_coord, None)]
    return coord_dims


def set_up_variable_cube(data, name='air_temperature', units='K',
                         spatial_grid='latlon', time=None, frt=None,
                         realizations=None, include_scalar_coords=None,
                         attributes=None):
    """
    Set up a cube containing a single variable field with:
    - x/y spatial dimensions (equal area or lat / lon)
    - optional leading "realization" dimension
    - "time", "forecast_reference_time" and "forecast_period" scalar coords
    - option to specify additional scalar coordinates
    - configurable attributes

    Args:
        data (np.ndarray):
            2D or 3D array of data to put into the cube

    Kwargs:
        name (str):
            Variable name (standard / long)
        units (str):
            Variable units
        spatial_grid (str):
            What type of x/y coordinate values to use.  Default is "latlon",
            otherwise uses "projection_[x|y]_coordinate".
        time (datetime.datetime):
            Single cube validity time
        frt (datetime.datetime):
            Single cube forecast reference time
        realizations (list):
            List of forecast realizations.  If not present, taken from the
            leading dimension of the input data array (if 3D).
        include_scalar_coords (list):
            List of iris.coords.DimCoord or AuxCoord instances of length 1.
        attributes (dict):
            Optional cube attributes.
    """
    # construct spatial dimension coordimates
    ypoints = data.shape[-2]
    xpoints = data.shape[-1]
    y_coord, x_coord = construct_xy_coords(ypoints, xpoints, spatial_grid)

    # construct realization dimension for 3D data, and dim_coords list
    ndims = len(data.shape)
    if ndims == 3:
        if realizations is not None:
            if len(realizations) != data.shape[0]:
                raise ValueError(
                    'Cannot generate {} realizations from data of shape '
                    '{}'.format(len(realizations), data.shape))
        else:
            realizations = np.arange(data.shape[0]).astype(np.int32)
        realization_coord = DimCoord(realizations, "realization", units="1")
        dim_coords = [(realization_coord, 0), (y_coord, 1), (x_coord, 2)]
    elif ndims == 2:
        dim_coords = [(y_coord, 0), (x_coord, 1)]
    else:
        raise ValueError(
            'Expected 2 or 3 dimensions on input data: got {}'.format(ndims))

    # construct list of aux_coords_and_dims
    scalar_coords = construct_scalar_time_coords(time, frt, fp)
    if include_scalar_coords is not None:
        for coord in include_scalar_coords:
            scalar_coords.append((coord, None))

    # create data cube
    cube = iris.cube.Cube(data, long_name=name, units=units,
                          dim_coords_and_dims=dim_coords,
                          aux_coords_and_dims=scalar_coords,
                          attributes=attributes)
    return cube


def set_up_percentile_cube(data, percentiles, name='air_temperature',
                           units='K',
                           percentile_dim_name='percentile_over_realization',
                           spatial_grid='latlon', time=None, frt=None,
                           percentiles=None, include_scalar_coords=None,
                           attributes=None):
    """
    Set up a cube containing percentiles of a variable with:
    - x/y spatial dimensions (equal area or lat / lon)
    - leading "percentile" dimension
    - "time", "forecast_reference_time" and "forecast_period" scalar coords
    - option to specify additional scalar coordinates
    - configurable attributes

    Args:
        data (np.ndarray):
            3D array of data to put into the cube
        percentiles (list):
            List of int / float percentile values whose length must match the
            first dimension on the input data cube

    Kwargs:
        standard_name (str):
            Variable standard name
        units (str):
            Variable units
        spatial_grid (str):
            What type of x/y coordinate values to use.  Default is "latlon",
            otherwise uses "projection_[x|y]_coordinate".
        percentile_dim_name (str):
            Name for the percentile dimension.
        time (datetime.datetime):
            Single cube validity time
        frt (datetime.datetime):
            Single cube forecast reference time
        include_scalar_coords (list):
            List of iris.coords.DimCoord or AuxCoord instances of length 1.
        attributes (dict):
            Optional cube attributes.
    """
    cube = set_up_variable_cube(
        data, name=name, units=units, spatial_grid=spatial_grid,
        time=time, frt=frt, realizations=percentiles, attributes=attributes,
        include_scalar_coords=include_scalar_coords)
    cube.coord("realization").rename(percentile_dim_name)
    cube.coord(percentile_dim_name).units = Unit("%")
    return cube


def set_up_probability_cube(data, thresholds, variable_name='air_temperature',
                            threshold_units='K', relative_to_threshold='above',
                            spatial_grid='latlon', time=None, frt=None,
                            percentiles=None, include_scalar_coords=None,
                            attributes=None):
    """
    Set up a cube containing probabilities at thresholds with:
    - x/y spatial dimensions (equal area or lat / lon)
    - leading "threshold" dimension
    - "time", "forecast_reference_time" and "forecast_period" scalar coords
    - option to specify additional scalar coordinates
    - "relative_to_threshold" attribute (default "above")
    - default or configurable attributes
    - configurable cube data, name conforms to "probability_of_X" convention

    Args:
        data (np.ndarray):
            3D array of data to put into the cube
        thresholds (list):
            List of int / float threshold values whose length must match the
            first dimension on the input data cube

    Kwargs:
        variable_name (str):
            Name of the underlying variable to which the probability field
            applies
        threshold_units (str):
            Units of the underlying variable / threshold.
        spatial_grid (str):
            What type of x/y coordinate values to use.  Default is "latlon",
            otherwise uses "projection_[x|y]_coordinate".
        time (datetime.datetime):
            Single cube validity time
        frt (datetime.datetime):
            Single cube forecast reference time
        include_scalar_coords (list):
            List of iris.coords.DimCoord or AuxCoord instances of length 1.
        attributes (dict):
            Optional cube attributes.
    """
    # create a "relative to threshold" attribute
    if attributes is None:
        attributes = {'relative_to_threshold': relative_to_threshold}
    else:
        attributes['relative_to_threshold'] = relative_to_threshold

    name = 'probability_of_{}'.format(variable_name)
    cube = set_up_variable_cube(
        data, name=name, units='1', spatial_grid=spatial_grid,
        time=time, frt=frt, realizations=thresholds, attributes=attributes,
        include_scalar_coords=include_scalar_coords)
    cube.coord("realization").rename("threshold")
    cube.coord("threshold").units = Unit(threshold_units)
    return cube


def set_up_cube_list(data, name='air_temperature', units='K',
                     spatial_grid='latlon', time_points=None, frt_points=None,
                     realizations=None, percentiles=None, thresholds=None,
                     include_scalar_coords=None, attributes=None):
    """
    Set up list of variable, percentile or probability cubes with a differing
    scalar coordinate value(s).  Option to merge into a single concatenated
    cube.  If adding forecast_reference_time coordinate, should calculate and
    add a forecast_period AuxCoord to match.
    """






