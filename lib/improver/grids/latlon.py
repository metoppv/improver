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
"""Module providing the latitude-longitude grid."""

import iris
import numpy as np
from iris.coord_systems import GeogCS
from iris.fileformats.pp import EARTH_RADIUS

from improver.grids import _make_grid_cube


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


GLOBAL_LATLON_GRID = _make_global_grid(
    1920, -89.953125, 89.953125, 2560, -180.0, 179.859375)
