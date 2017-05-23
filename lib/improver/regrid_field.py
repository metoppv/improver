# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown Copyright 2017 Met Office.
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
"""Module containing temporary regridding utility for improver
   ancillary generation module"""

from biggus import ConstantArray
import iris
import numpy as np
from iris.coord_systems import GeogCS
from iris.fileformats.pp import EARTH_RADIUS
try:
    from iris.coord_systems import LambertAzimuthalEqualArea
except ImportError:
    from iris import __version__
    raise ImportError(
        'iris version {} does not support the '
        'Lambert azimuthal equal-area projection.'.format(__version__))


# The standardised grid.
# Set up the new coordinate reference system for the new projection.
SEMI_MAJOR_AXIS = 6378137.0
INVERSE_FLATTENING = 298.257222101


def _make_grid_cube(x_coord, y_coord, bounds=True):
    """
    Creates a two-dimensional Cube with the given one-dimensional coordinates.

    Returns
    -------
    Cube
        A grid with the defined coordinates.

    """
    assert x_coord.ndim == 1
    assert y_coord.ndim == 1
    if bounds:
        x_coord.guess_bounds()
        y_coord.guess_bounds()
    cube = iris.cube.Cube(ConstantArray((len(x_coord.points),
                                         len(y_coord.points))))
    cube.add_dim_coord(x_coord, 0)
    cube.add_dim_coord(y_coord, 1)
    return cube


def _make_standard_grid(n_x, min_x, max_x, n_y, min_y, max_y,
                        coord_system, standard_x_coord_name,
                        standard_y_coord_name, units, bounds=True):
    """
    Creates a two-dimensional Cube that represents the standard grid.

    Returns
    -------
    Cube
        A global grid with the requested resolution.

    """
    x_coord = iris.coords.DimCoord(np.linspace(min_x, max_x, n_x,
                                               endpoint=True),
                                   standard_x_coord_name, units=units,
                                   coord_system=coord_system)
    y_coord = iris.coords.DimCoord(np.linspace(min_y, max_y, n_y,
                                               endpoint=True),
                                   standard_y_coord_name, units=units,
                                   coord_system=coord_system)
    return _make_grid_cube(x_coord, y_coord, bounds=bounds)


def _make_global_grid(n_lat, min_lat, max_lat, n_lon, min_lon, max_lon,
                      bounds=True):
    """
    Creates a two-dimensional Cube that represents the standard global grid.

    Returns
    -------
    Cube
        A global grid with the requested resolution.

    """
    cs = GeogCS(EARTH_RADIUS)
    lat_coord = iris.coords.DimCoord(np.linspace(min_lat, max_lat, n_lat),
                                     'latitude', units='degrees',
                                     coord_system=cs)
    lon_coord = iris.coords.DimCoord(np.linspace(min_lon, max_lon, n_lon),
                                     'longitude', units='degrees',
                                     coord_system=cs)
    return _make_grid_cube(lat_coord, lon_coord, bounds=bounds)


ELLIPSOID = GeogCS(semi_major_axis=SEMI_MAJOR_AXIS,
                   inverse_flattening=INVERSE_FLATTENING)
STANDARD_CRS = LambertAzimuthalEqualArea(
    latitude_of_projection_origin=54.9,
    longitude_of_projection_origin=-2.5,
    ellipsoid=ELLIPSOID)
STANDARD_GRIDS = {}
STANDARD_GRIDS['ukvx'] = _make_standard_grid(1042, -1158000, 924000,
                                             970, -1036000, 902000,
                                             STANDARD_CRS,
                                             'projection_x_coordinate',
                                             'projection_y_coordinate', 'm',
                                             bounds=True)

# The standardised grid.
STANDARD_GRIDS['glm'] = _make_global_grid(1920, -89.953125, 89.953125,
                                          2560, -180.0, 179.859375)


def regrid_field(field, grid):
    '''
    Regrids fields onto the standard grid

    Inputs
    -------
    field : cube
        cube to be regridded onto Standard_Grid
    grid : string
        the grid we wish to interpolate to

    Exceptions
    -----------
    - Raises a ValueError if NaNs are found in the field following regridding
        (this would indicate the input field domain was smaller than the
         standard grid) UNLESS: grid is global field.l
    '''
    if grid is 'glm':
        field = field.regrid(STANDARD_GRIDS[grid],
                             iris.analysis.Linear())
    else:
        field = field.regrid(STANDARD_GRIDS[grid],
                             iris.analysis.Linear(extrapolation_mode='nan'))
    if np.any(np.isnan(field.data)):
        msg = 'Model domain must be larger than Standard grid domain'
        raise ValueError(msg)
    return field
