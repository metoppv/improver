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
"""Functions to create spotdata cubes."""

import iris
from iris.coords import DimCoord, AuxCoord


def build_spotdata_cube(data, name, units,
                        altitude, latitude, longitude, wmo_id,
                        scalar_coords=None,
                        neighbour_methods=None, neighbour_methods_dim=1,
                        grid_attributes=None, grid_attributes_dim=2)
    """
    Function to build a spotdata cube

    Args:
        data (float or np.ndarray):
            Float spot data or array of data points from several sites.
            The spot index should be the first dimension if the array is
            multi-dimensional.
        name (str):
            Cube name (eg 'air_temperature')
        units (str):
            Cube units (eg 'K')
        altitude (float or np.ndarray):
            Float or 1d array of site altitudes in metres
        latitude (float or np.ndarray):
            Float or 1d array of site latitudes in degrees
        longitude (float or np.ndarray):
            Float or 1d array of site longitudes in degrees
        wmo_id (float or np.ndarray):
            Float or 1d array of site 5-digit WMO identifiers

    Kwargs:
        scalar_coords (list):
            Optional list of iris.coord.AuxCoord instances
        neighbour_methods (list):
            Optional list of neighbour method names
        neighbour_methods_dim (int):
            Data dimension to match the neighbour method list
        grid_attributes (list):
            Optional list of grid attribute names
        grid_attributes_dim (int):
            Data dimension to match the grid attributes list
    """

    # construct auxiliary coordinates
    alt_coord = DimCoord(altitude, 'altitude', units='m')
    lat_coord = DimCoord(latitude, 'latitude', units='degrees')
    lon_coord = DimCoord(longitude, 'longitude', units='degrees')
    id_coord = AuxCoord(wmo_id, 'wmo_id')

    # construct dimension coordinates
    n_sites = len(data) if isinstance(data, np.ndarray) else 1
    spot_index = DimCoord(n_sites, 'spot_index', units='1')
    dim_coords_and_dims = [(spot_index, 0)]

    if neighbour_methods is not None:
        neighbour_methods_coord = DimCoord(
            len(neighbour_methods), 'neighbour_selection_method', units='1')
        ne




