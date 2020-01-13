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
"""Functions to create spotdata cubes."""

import iris
import numpy as np
from iris.coords import AuxCoord, DimCoord


def build_spotdata_cube(data, name, units,
                        altitude, latitude, longitude, wmo_id,
                        scalar_coords=None,
                        neighbour_methods=None, neighbour_methods_dim=1,
                        grid_attributes=None, grid_attributes_dim=2):
    """
    Function to build a spotdata cube with expected dimension and auxiliary
    coordinate structure.

    It can be used to create spot data cubes. In this case the data is the
    spot data values at each site, and the coordinates that describe each site.

    It can also be used to create cubes which describe the grid points that are
    used to extract each site from a gridded field, for different selection
    method. The selection methods are specified by the neighbour_methods
    coordinate. The grid_attribute coordinate encapsulates information required
    to extract data, for example the x/y indices that identify the grid point
    neighbour.

    .. See the documentation for examples of these cubes.
    .. include:: extended_documentation/spotdata/build_spotdata_cube/
       build_spotdata_cube_examples.rst

    Args:
        data (float or numpy.ndarray):
            Float spot data or array of data points from several sites.
            The spot index should be the first dimension if the array is
            multi-dimensional (see optional additional dimensions below).
        name (str):
            Cube name (eg 'air_temperature')
        units (str):
            Cube units (eg 'K')
        altitude (float or numpy.ndarray):
            Float or 1d array of site altitudes in metres
        latitude (float or numpy.ndarray):
            Float or 1d array of site latitudes in degrees
        longitude (float or numpy.ndarray):
            Float or 1d array of site longitudes in degrees
        wmo_id (str or list):
            String or list of site 5-digit WMO identifiers
        scalar_coords (list):
            Optional list of iris.coords.AuxCoord instances
        neighbour_methods (list):
            Optional list of neighbour method names, e.g. 'nearest'
        neighbour_methods_dim (int):
            Data dimension to match the neighbour method list
        grid_attributes (list):
            Optional list of grid attribute names, e.g. x-index, y-index
        grid_attributes_dim (int):
            Data dimension to match the grid attributes list
    """

    # construct auxiliary coordinates
    alt_coord = AuxCoord(altitude, 'altitude', units='m')
    lat_coord = AuxCoord(latitude, 'latitude', units='degrees')
    lon_coord = AuxCoord(longitude, 'longitude', units='degrees')
    id_coord = AuxCoord(wmo_id, long_name='wmo_id', units='no_unit')

    aux_coords_and_dims = []
    for coord in [alt_coord, lat_coord, lon_coord, id_coord]:
        aux_coords_and_dims.append((coord, 0))

    # append scalar coordinates
    if scalar_coords is not None:
        for coord in scalar_coords:
            aux_coords_and_dims.append((coord, None))

    # construct dimension coordinates
    if np.isscalar(data):
        data = np.array([data])
    spot_index = DimCoord(
        np.arange(len(data), dtype=np.int32), long_name='spot_index',
        units='1')
    dim_coords_and_dims = [(spot_index, 0)]

    if neighbour_methods is not None:
        neighbour_methods_coord = DimCoord(
            np.arange(len(neighbour_methods), dtype=np.int32),
            long_name='neighbour_selection_method', units='1')
        neighbour_methods_key = AuxCoord(
            neighbour_methods, long_name='neighbour_selection_method_name',
            units='no_unit')

        dim_coords_and_dims.append((neighbour_methods_coord,
                                    neighbour_methods_dim))
        aux_coords_and_dims.append((neighbour_methods_key,
                                    neighbour_methods_dim))

    if grid_attributes is not None:
        grid_attributes_coord = DimCoord(
            np.arange(len(grid_attributes), dtype=np.int32),
            long_name='grid_attributes', units='1')
        grid_attributes_key = AuxCoord(
            grid_attributes, long_name='grid_attributes_key', units='no_unit')

        dim_coords_and_dims.append((grid_attributes_coord,
                                    grid_attributes_dim))
        aux_coords_and_dims.append((grid_attributes_key,
                                    grid_attributes_dim))

    # create output cube
    spot_cube = iris.cube.Cube(
        data, long_name=name, units=units,
        dim_coords_and_dims=dim_coords_and_dims,
        aux_coords_and_dims=aux_coords_and_dims)
    # rename to force a standard name to be set if name is valid
    spot_cube.rename(name)

    return spot_cube
