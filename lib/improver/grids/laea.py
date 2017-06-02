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
"""Module providing the LAMBERT AZIMUTHAL EQUAL AREA grid."""

import iris
import numpy as np
from iris.coord_systems import GeogCS
try:
    from iris.coord_systems import LambertAzimuthalEqualArea
except ImportError:
    from iris import __version__
    raise ImportError(
        'iris version {} does not support the '
        'Lambert azimuthal equal-area projection.'.format(__version__))

from improver.grids import _make_grid_cube


# The standardised grid.
# Set up the new coordinate reference system for the new projection.
SEMI_MAJOR_AXIS = 6378137.0
INVERSE_FLATTENING = 298.257222101
ELLIPSOID = GeogCS(semi_major_axis=SEMI_MAJOR_AXIS,
                   inverse_flattening=INVERSE_FLATTENING)
STANDARD_CRS = LambertAzimuthalEqualArea(
    latitude_of_projection_origin=54.9,
    longitude_of_projection_origin=-2.5,
    ellipsoid=ELLIPSOID)


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


# Lambert Azimuthal Equal Areas grid
UK_LAEA_GRID = _make_standard_grid(
    1042, -1158000, 924000, 970, -1036000, 902000,
    STANDARD_CRS, 'projection_x_coordinate', 'projection_y_coordinate',
    'm', bounds=True)
