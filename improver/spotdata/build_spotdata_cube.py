# (C) Crown copyright, Met Office. All rights reserved.
#
# This file is part of IMPROVER and is released under a BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Functions to create spotdata cubes."""

from typing import List, Optional, Union

import iris
import numpy as np
from iris.coords import AuxCoord, Coord, DimCoord
from iris.cube import Cube
from numpy import ndarray

from . import UNIQUE_ID_ATTRIBUTE


def build_spotdata_cube(
    data: ndarray,
    name: str,
    units: str,
    altitude: ndarray,
    latitude: ndarray,
    longitude: ndarray,
    wmo_id: Union[str, List[str]],
    unique_site_id: Optional[Union[List[str], ndarray]] = None,
    unique_site_id_key: Optional[str] = None,
    scalar_coords: Optional[List[Coord]] = None,
    auxiliary_coords: Optional[List[Coord]] = None,
    neighbour_methods: Optional[List[str]] = None,
    grid_attributes: Optional[List[str]] = None,
    additional_dims: Optional[List[Coord]] = None,
    additional_dims_aux: Optional[List[List[AuxCoord]]] = None,
) -> Cube:
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
        data:
            Float spot data or array of data points from several sites.
            The spot index should be the last dimension if the array is
            multi-dimensional (see optional additional dimensions below).
        name:
            Cube name (eg 'air_temperature')
        units:
            Cube units (eg 'K')
        altitude:
            Float or 1d array of site altitudes in metres
        latitude:
            Float or 1d array of site latitudes in degrees
        longitude:
            Float or 1d array of site longitudes in degrees
        wmo_id:
            String or list of 5-digit WMO site identifiers.
        unique_site_id:
            Optional list of 8-digit unique site identifiers. If provided, this
            is expected to be a complete list with a unique identifier for every site.
        unique_site_id_key:
            String to name the unique_site_id coordinate. Required if
            unique_site_id is in use.
        scalar_coords:
            Optional list of iris.coords.Coord instances
        auxiliary_coords:
            Optional list of iris.coords.Coord instances which are non-scalar.
        neighbour_methods:
            Optional list of neighbour method names, e.g. 'nearest'
        grid_attributes:
            Optional list of grid attribute names, e.g. x-index, y-index
        additional_dims:
            Optional list of additional dimensions to preceed the spot data dimension.
        additional_dims_aux:
            Optional list of auxiliary coordinates associated with each dimension in
            additional_dims

    Returns:
        A cube containing the extracted spot data with spot data being the final dimension.
    """

    # construct auxiliary coordinates
    alt_coord = AuxCoord(altitude, "altitude", units="m")
    lat_coord = AuxCoord(latitude, "latitude", units="degrees")
    lon_coord = AuxCoord(longitude, "longitude", units="degrees")
    wmo_id_coord = AuxCoord(wmo_id, long_name="wmo_id", units="no_unit")
    if unique_site_id is not None:
        if not unique_site_id_key:
            raise ValueError(
                "A unique_site_id_key must be provided if a unique_site_id is"
                " provided."
            )
        unique_id_coord = AuxCoord(
            unique_site_id,
            long_name=unique_site_id_key,
            units="no_unit",
            attributes={UNIQUE_ID_ATTRIBUTE: "true"},
        )

    aux_coords_and_dims = []

    # append scalar coordinates
    if scalar_coords is not None:
        for coord in scalar_coords:
            aux_coords_and_dims.append((coord, None))

    # construct dimension coordinates
    if np.isscalar(data):
        data = np.array([data])
    spot_index = DimCoord(
        np.arange(data.shape[-1], dtype=np.int32), long_name="spot_index", units="1"
    )

    dim_coords_and_dims = []
    current_dim = 0

    if neighbour_methods is not None:
        neighbour_methods_coord = DimCoord(
            np.arange(len(neighbour_methods), dtype=np.int32),
            long_name="neighbour_selection_method",
            units="1",
        )
        neighbour_methods_key = AuxCoord(
            neighbour_methods,
            long_name="neighbour_selection_method_name",
            units="no_unit",
        )

        dim_coords_and_dims.append((neighbour_methods_coord, current_dim))
        aux_coords_and_dims.append((neighbour_methods_key, current_dim))
        current_dim += 1

    if grid_attributes is not None:
        grid_attributes_coord = DimCoord(
            np.arange(len(grid_attributes), dtype=np.int32),
            long_name="grid_attributes",
            units="1",
        )
        grid_attributes_key = AuxCoord(
            grid_attributes, long_name="grid_attributes_key", units="no_unit"
        )

        dim_coords_and_dims.append((grid_attributes_coord, current_dim))
        aux_coords_and_dims.append((grid_attributes_key, current_dim))
        current_dim += 1

    if additional_dims is not None:
        for coord, aux_coords in zip(
            additional_dims, additional_dims_aux or [[] for _ in additional_dims]
        ):
            dim_coords_and_dims.append((coord, current_dim))
            for aux_coord in aux_coords:
                aux_coords_and_dims.append((aux_coord, current_dim))
            current_dim += 1

    dim_coords_and_dims.append((spot_index, current_dim))
    coords = [alt_coord, lat_coord, lon_coord, wmo_id_coord]
    if unique_site_id is not None:
        coords.append(unique_id_coord)
    for coord in coords:
        aux_coords_and_dims.append((coord, current_dim))

    # append non-scalar auxiliary coordinates
    if auxiliary_coords:
        for coord in auxiliary_coords:
            aux_coords_and_dims.append((coord, current_dim))

    # create output cube
    spot_cube = iris.cube.Cube(
        data,
        long_name=name,
        units=units,
        dim_coords_and_dims=dim_coords_and_dims,
        aux_coords_and_dims=aux_coords_and_dims,
    )
    # rename to force a standard name to be set if name is valid
    spot_cube.rename(name)

    return spot_cube
