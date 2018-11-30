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
Functions to set up variable, multi-realization, percentile and probability
cubes for unit tests.  Standardises time units and spatial coordinates,
including coordinate order expected by IMPROVER plugins.
"""

from cf_units import Unit, date2num
from datetime import datetime

import numpy as np
import iris
from iris.coords import DimCoord
from iris.exceptions import CoordinateNotFoundError

from improver.grids import GLOBAL_GRID_CCRS, STANDARD_GRID_CCRS
from improver.utilities.cube_metadata import MOSG_GRID_DEFINITION
from improver.utilities.cube_checker import check_cube_not_float64

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
            Specifier to produce either a "latlon" or "equalarea" grid

    Returns:
        y_coord, x_coord (tuple):
            Tuple of iris.coord.DimCoord instances
    """
    if spatial_grid == 'latlon':
        # make a lat-lon grid including the UK area
        y_coord = DimCoord(
            np.linspace(-20.0, 20.0, ypoints, dtype=np.float32),
            "latitude", units="degrees", coord_system=GLOBAL_GRID_CCRS)
        x_coord = DimCoord(
            np.linspace(40, 80, xpoints, dtype=np.float32),
            "longitude", units="degrees", coord_system=GLOBAL_GRID_CCRS)
    elif 'equalarea':
        # use UK eastings and northings on standard grid
        y_coord = DimCoord(
            np.linspace(-100000, 900000, ypoints, dtype=np.float32),
            "projection_y_coordinate", units="metres",
            coord_system=STANDARD_GRID_CCRS)
        x_coord = DimCoord(
            np.linspace(-400000, 600000, xpoints, dtype=np.float32),
            "projection_x_coordinate", units="metres",
            coord_system=STANDARD_GRID_CCRS)
    else:
        raise ValueError('Grid type {} not recognised'.format(spatial_grid))

    return y_coord, x_coord


def construct_scalar_time_coords(time, time_bounds, frt):
    """
    Construct scalar time coordinates as aux_coord list

    Args:
        time (datetime.datetime):
            Single time point
        time_bounds (tuple or list of datetime.datetime instances or None):
            Lower and upper bound on time point, if required
        frt (datetime.datetime):
            Single forecast reference time point

    Returns:
        coord_dims (list):
            List of iris.coord.DimCoord instances with the associated "None"
            dimension (format required by iris.cube.Cube initialisation).
    """
    # generate time coordinate points
    time_point_seconds = np.round(
        date2num(time, TIME_UNIT, CALENDAR)).astype(np.int64)
    frt_point_seconds = np.round(
        date2num(frt, TIME_UNIT, CALENDAR)).astype(np.int64)

    if time_point_seconds < frt_point_seconds:
        raise ValueError('Cannot set up cube with negative forecast period')
    fp_point_seconds = time_point_seconds - frt_point_seconds

    # parse bounds if required
    if time_bounds is not None:
        lower_bound = np.round(
            date2num(time_bounds[0], TIME_UNIT, CALENDAR)).astype(np.int64)
        upper_bound = np.round(
            date2num(time_bounds[1], TIME_UNIT, CALENDAR)).astype(np.int64)
        bounds = (min(lower_bound, upper_bound),
                  max(lower_bound, upper_bound))
        if time_point_seconds < bounds[0] or time_point_seconds > bounds[1]:
            raise ValueError(
                'Time point {} not within bounds {}-{}'.format(
                    time, time_bounds[0], time_bounds[1]))
    else:
        bounds = None

    # create coordinates
    time_coord = DimCoord(
        time_point_seconds, "time", units=TIME_UNIT, bounds=bounds)
    frt_coord = DimCoord(
        frt_point_seconds, "forecast_reference_time", units=TIME_UNIT)
    fp_coord = DimCoord(fp_point_seconds, "forecast_period", units="seconds")

    coord_dims = [(time_coord, None), (frt_coord, None), (fp_coord, None)]
    return coord_dims


def set_up_variable_cube(data, name='air_temperature', units='K',
                         spatial_grid='latlon',
                         time=datetime(2017, 11, 10, 4, 0), time_bounds=None,
                         frt=datetime(2017, 11, 10, 0, 0), realizations=None,
                         include_scalar_coords=None, attributes=None,
                         standard_grid_metadata=None):
    """
    Set up a cube containing a single variable field with:
    - x/y spatial dimensions (equal area or lat / lon)
    - optional leading "realization" dimension
    - "time", "forecast_reference_time" and "forecast_period" scalar coords
    - option to specify additional scalar coordinates
    - configurable attributes

    Args:
        data (np.ndarray):
            2D (y-x ordered) or 3D (realization-y-x ordered) array of data
            to put into the cube.

    Kwargs:
        name (str):
            Variable name (standard / long)
        units (str):
            Variable units
        spatial_grid (str):
            What type of x/y coordinate values to use.  Permitted values are
            "latlon" or "equalarea".
        time (datetime.datetime):
            Single cube validity time
        time_bounds (tuple or list of datetime.datetime instances):
            Lower and upper bound on time point, if required
        frt (datetime.datetime):
            Single cube forecast reference time
        realizations (list):
            List of forecast realizations.  If not present, taken from the
            leading dimension of the input data array (if 3D).
        include_scalar_coords (list):
            List of iris.coords.DimCoord or AuxCoord instances of length 1.
        attributes (dict):
            Optional cube attributes.
        standard_grid_metadata (str):
            Recognised mosg__model_configuration for which to set up Met
            Office standard grid attributes.  Should be 'uk_det', 'uk_ens',
            'gl_det' or 'gl_ens'.
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
    scalar_coords = construct_scalar_time_coords(time, time_bounds, frt)
    if include_scalar_coords is not None:
        for coord in include_scalar_coords:
            scalar_coords.append((coord, None))

    # set up attributes
    cube_attrs = {}
    if standard_grid_metadata is not None:
        cube_attrs.update(MOSG_GRID_DEFINITION[standard_grid_metadata])
    if attributes is not None:
        cube_attrs.update(attributes)

    # create data cube
    cube = iris.cube.Cube(data, long_name=name, units=units,
                          dim_coords_and_dims=dim_coords,
                          aux_coords_and_dims=scalar_coords,
                          attributes=cube_attrs)

    # don't allow unit tests to set up invalid cubes
    check_cube_not_float64(cube)

    return cube


def set_up_percentile_cube(data, percentiles, name='air_temperature',
                           units='K', spatial_grid='latlon',
                           percentile_dim_name='percentile_over_realization',
                           time=datetime(2017, 11, 10, 4, 0), time_bounds=None,
                           frt=datetime(2017, 11, 10, 0, 0),
                           include_scalar_coords=None, attributes=None,
                           standard_grid_metadata=None):
    """
    Set up a cube containing percentiles of a variable with:
    - x/y spatial dimensions (equal area or lat / lon)
    - leading "percentile" dimension
    - "time", "forecast_reference_time" and "forecast_period" scalar coords
    - option to specify additional scalar coordinates
    - configurable attributes

    Args:
        data (np.ndarray):
            3D (percentile-y-x ordered) array of data to put into the cube
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
        time_bounds (tuple or list of datetime.datetime instances):
            Lower and upper bound on time point, if required
        frt (datetime.datetime):
            Single cube forecast reference time
        include_scalar_coords (list):
            List of iris.coords.DimCoord or AuxCoord instances of length 1.
        attributes (dict):
            Optional cube attributes.
        standard_grid_metadata (str):
            Recognised mosg__model_configuration for which to set up Met
            Office standard grid attributes.  Should be 'uk_det', 'uk_ens',
            'gl_det' or 'gl_ens'.
    """
    cube = set_up_variable_cube(
        data, name=name, units=units, spatial_grid=spatial_grid,
        time=time, frt=frt, realizations=percentiles, attributes=attributes,
        include_scalar_coords=include_scalar_coords,
        standard_grid_metadata=standard_grid_metadata)
    cube.coord("realization").rename(percentile_dim_name)
    cube.coord(percentile_dim_name).units = Unit("%")
    return cube


def set_up_probability_cube(data, thresholds, variable_name='air_temperature',
                            threshold_units='K', relative_to_threshold='above',
                            spatial_grid='latlon',
                            time=datetime(2017, 11, 10, 4, 0),
                            time_bounds=None,
                            frt=datetime(2017, 11, 10, 0, 0),
                            include_scalar_coords=None, attributes=None,
                            standard_grid_metadata=None):
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
            3D (threshold-y-x ordered) array of data to put into the cube
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
        relative_to_threshold (str):
            Value of the attribute "relative_to_threshold" which is required
            for IMPROVER probability cubes.
        time (datetime.datetime):
            Single cube validity time
        time_bounds (tuple or list of datetime.datetime instances):
            Lower and upper bound on time point, if required
        frt (datetime.datetime):
            Single cube forecast reference time
        include_scalar_coords (list):
            List of iris.coords.DimCoord or AuxCoord instances of length 1.
        attributes (dict):
            Optional cube attributes.
        standard_grid_metadata (str):
            Recognised mosg__model_configuration for which to set up Met
            Office standard grid attributes.  Should be 'uk_det', 'uk_ens',
            'gl_det' or 'gl_ens'.
    """
    # create a "relative to threshold" attribute
    if attributes is None:
        attributes = {'relative_to_threshold': relative_to_threshold}
    else:
        attributes['relative_to_threshold'] = relative_to_threshold

    if variable_name.startswith('probability_of_'):
        name = variable_name
    else:
        name = 'probability_of_{}'.format(variable_name)
    cube = set_up_variable_cube(
        data, name=name, units='1', spatial_grid=spatial_grid,
        time=time, frt=frt, realizations=thresholds, attributes=attributes,
        include_scalar_coords=include_scalar_coords,
        standard_grid_metadata=standard_grid_metadata)
    cube.coord("realization").rename("threshold")
    cube.coord("threshold").units = Unit(threshold_units)
    return cube


def add_coordinate(incube, coord_points, coord_name, coord_units=None,
                   dtype=np.float32, order=None):
    """
    Function to duplicate a sample cube with an additional coordinate to create
    a cubelist. The cubelist is merged to create a single cube, which can be
    reordered to place the new coordinate in the required position.

    Args:
        incube (iris.cube.Cube):
            Cube to be duplicated.
        coord_points (list):
            Values for the coordinate.
        coord_name (str):
            Long name of the coordinate to be added.

    Kwargs:
        coord_units (str):
            Coordinate unit required.
        dtype (type):
            Datatype for coordinate points.
        order (list):
            Optional list of integers to reorder the dimensions on the new
            merged cube.  For example, if the new coordinate is required to
            be in position 1 on a 4D cube, use order=[1, 0, 2, 3] to swap the
            new coordinate position with that of the original leading
            coordinate.

    Returns:
        iris.cube.Cube:
            Cube containing an additional dimension coordinate.
    """
    # if the coordinate already exists as a scalar coordinate, remove it
    cube = incube.copy()
    try:
        cube.remove_coord(coord_name)
    except CoordinateNotFoundError:
        pass

    cubes = iris.cube.CubeList([])
    for val in coord_points:
        temp_cube = cube.copy()
        temp_cube.add_aux_coord(
            DimCoord(np.array([val], dtype=dtype), long_name=coord_name,
                     units=coord_units))
        cubes.append(temp_cube)

    new_cube = cubes.merge_cube()
    if order is not None:
        new_cube.transpose(order)

    return new_cube
