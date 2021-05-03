# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown Copyright 2017-2021 Met Office.
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
Grid handling for regridding
"""

import iris
import numpy as np
from iris.cube import Cube
from scipy.interpolate import RegularGridInterpolator

COMMON_LAT_NAMES = ["latitude", "lat", "lats"]
COMMON_LON_NAMES = ["longitude", "lon", "lons"]


def get_cube_coord_names(cube):
    """
    Get all coordinate names from a cube

    Args:
         cube (iris.cube.Cube):
            input cube

    Return:
        List[str]:
            List of coordinate names
    """
    return [coord.standard_name for coord in cube.dim_coords]


def variable_name(cube, names):
    """
    Identify the name of a variable from a list of possible candidates.

    Args:
         cube (iris.cube.Cube):
            input cube
         names (List[str]):
            possible name list
    Return:
         str:
            matching name of the variable
    """
    coord_names = set(get_cube_coord_names(cube))
    matched_names = coord_names.intersection(names)
    if not matched_names:
        raise ValueError(f"Unable to find a variable matching {names}")
    elif len(matched_names) > 1:
        raise ValueError(f"find more than a variable matching {names}")
    else:
        return list(matched_names)[0]


def latlon_names(cube):
    """
    Identify the names of the latitude and longitude dimensions of cube

    Args:
        cube (iris.cube.Cube):
            input cube

    Return:
        str: names of latitude and longitude
    """
    lats_name = variable_name(cube, COMMON_LAT_NAMES)
    lons_name = variable_name(cube, COMMON_LON_NAMES)
    return lats_name, lons_name


def latlon_from_cube(cube):
    """
    Produce an array of latitude-longitude coordinates used by an Iris cube

    Args:
       cube(iris.cube.Cube):
           cube information

    Return:
       numpy.ndarray:
           latitude-longitude pairs (N x 2)
    """
    lats_name, lons_name = latlon_names(cube)
    lats_data = cube.coord(lats_name).points
    lons_data = cube.coord(lons_name).points
    lats_mesh, lons_mesh = np.meshgrid(lats_data, lons_data, indexing="ij")
    latlon = np.dstack((lats_mesh, lons_mesh)).reshape((-1, 2))
    return latlon


def get_grid_spacing(cube):
    """
    get cube grid size (cube in even lats/lons system)

    Args:
        cube (iris.cube.Cube):
            input cube
    Return:
        lat_d,lon_d (float):
            latitude/logitude grid size
    """
    lats_name, lons_name = latlon_names(cube)
    lat_d = cube.coord(lats_name).points[1] - cube.coord(lats_name).points[0]
    lon_d = cube.coord(lons_name).points[1] - cube.coord(lons_name).points[0]
    return lat_d, lon_d


def unflatten_spatial_dimensions(
    regrid_result, cube_out_mask, in_values, lats_index, lons_index
):
    """
    Reshape numpy array regrid_result from (lat*lon,...) to (....,lat,lon)
    or from (projy*projx,...) to (...,projy,projx)

    Args:
        regrid_result (numpy.ndarray):
            array of regridded result in (lat*lon,....) or (projy*projx,...)
        cube_out_mask (iris.cube.Cube):
            target grid cube (for getting grid dimension here)
        in_values (numpy.ndarray):
            reshaped source data (in _reshape_data_cube)
        lats_index(int):
            index of lats or projy coord in reshaped array
        lons_index(int):
            index of lons or projx coord in reshaped array

    Returns:
        Union[numpy.ndarray, numpy.ma.core.MaskedArray]:
            Reshaped data array
    """
    cube_out_dim0 = cube_out_mask.coord(axis="y").shape[0]
    cube_out_dim1 = cube_out_mask.coord(axis="x").shape[0]
    latlon_shape = [cube_out_dim0, cube_out_dim1] + list(in_values.shape[1:])

    regrid_result = np.reshape(regrid_result, latlon_shape)
    regrid_result = np.swapaxes(regrid_result, 1, lons_index)
    regrid_result = np.swapaxes(regrid_result, 0, lats_index)
    return regrid_result


def flatten_spatial_dimensions(cube):
    """
    Reshape data cube from (....,lat,lon) into data (lat*lon,...)

    Args:
        cube (iris.cube.Cube):
            original data cube

    Returns:
        in_values(numpy.ndarray or numpy.ma.core.MaskedArray)
            Reshaped data array
        lats_index,lons_index (int):
            lattitude/logitude indexes in cube coord.
    """
    in_values = cube.data
    coord_names = get_cube_coord_names(cube)
    lats_name, lons_name = latlon_names(cube)
    lats_index = coord_names.index(lats_name)
    lons_index = coord_names.index(lons_name)

    in_values = np.swapaxes(in_values, 0, lats_index)
    in_values = np.swapaxes(in_values, 1, lons_index)

    lats_len = int(in_values.shape[0])
    lons_len = int(in_values.shape[1])
    latlon_shape = [lats_len * lons_len] + list(in_values.shape[2:])
    in_values = np.reshape(in_values, latlon_shape)
    return in_values, lats_index, lons_index


def convert_from_projection_to_latlons(cube_out, cube_in):
    """
    convert cube_out's LambertAzimuthalEqualArea's coord to GeogCS's lats/lons
    output grid (cube_out) could be in LambertAzimuthalEqualArea system
    cube_in is in GeogCS's lats/lons system.
    Args:
        cube_out (iris.cube.Cube):
            target cube with LambertAzimuthalEqualArea's coord system
        cube_in (iris.cube.Cube):
            source cube with GeorCS (as reference coord system for conversion)

    Returns:
        numpy.ndarray:
            latitude-longitude pairs for target grid points
    """

    # get coordinate points in native projection & transfer into xx,yy(1D)
    proj_x = cube_out.coord("projection_x_coordinate").points
    proj_y = cube_out.coord("projection_y_coordinate").points
    yy, xx = np.meshgrid(proj_y, proj_x, indexing="ij")
    yy = yy.flatten()
    xx = xx.flatten()

    # extract the native projection and convert it to a cartopy projection:
    cs_nat = cube_out.coord_system()
    cs_nat_cart = cs_nat.as_cartopy_projection()

    # find target projection,convert it to a cartopy projection
    cs_tgt = cube_in.coord("latitude").coord_system
    cs_tgt_cart = cs_tgt.as_cartopy_projection()

    # use cartopy's transform to convert coord.in native proj to coord in target proj
    lons, lats, _ = cs_tgt_cart.transform_points(cs_nat_cart, xx, yy).T

    out_latlons = np.dstack((lats, lons)).squeeze()

    return out_latlons


def classify_output_surface_type(cube_out_mask):
    """
    Classify surface types of target grid points based on a binary True/False land mask

    Args:
        cube_out_mask (iris.cube.Cube):
            land_sea mask information cube for target grid (land=>1)

    Return:
        numpy.ndarray:
        1D land-sea mask information for 1D-ordered target grid points
    """
    # cube y-axis => latitude or projection-y
    cube_out_dim0 = cube_out_mask.coord(axis="y").shape[0]
    cube_out_dim1 = cube_out_mask.coord(axis="x").shape[0]
    out_classified = cube_out_mask.data.reshape(cube_out_dim0 * cube_out_dim1)
    return out_classified


def classify_input_surface_type(cube_in_mask, classify_latlons):
    """
    Classify surface types of source grid points based on a binary True/False land mask
    cube_in_mask's grid could be different from input source grid of NWP results

    Args:
        cube_in_mask (iris.cube.Cube):
            land_sea mask information cube for input source grid(land=>1)
            should in GeogCS's lats/lons coordinate system
        classify_latlons(numpy.ndarray):
            latitude and longitude source grid points to classify (N x 2)

    Returns:
        numpy.ndarray: classifications (N) for 1D-ordered source grid points
    """
    in_land_mask = cube_in_mask.data
    lats_name, lons_name = latlon_names(cube_in_mask)
    in_land_mask_lats = cube_in_mask.coord(lats_name).points
    in_land_mask_lons = cube_in_mask.coord(lons_name).points

    mask_rg_interp = RegularGridInterpolator(
        (in_land_mask_lats, in_land_mask_lons),
        in_land_mask,
        method="nearest",
        bounds_error=False,
        fill_value=0.0,
    )
    is_land = np.bool_(mask_rg_interp(classify_latlons))
    return is_land


def similar_surface_classify(in_is_land, out_is_land, nearest_in_indexes):
    """
    Classify surface types as matched (True) or unmatched(False) between target points
    and their source point

    Args:
        in_is_land (numpy.ndarray):
            source point classifications (N)
        out_is_land (numpy.ndarray):
            target point classifications (M)
        nearest_in_indexes (numpy.ndarray)
            indexes of input points nearby output points (M x K)

    Return:
        numpy.ndarray:
            Boolean true if input surface type matches output or no matches (M x K)
    """
    k = nearest_in_indexes.shape[1]
    out_is_land_bcast = np.broadcast_to(
        out_is_land, (k, out_is_land.shape[0])
    ).transpose()  # dimensions M x K

    # classify the input points surrounding each output point
    nearest_is_land = in_is_land[nearest_in_indexes]  # dimensions M x K

    # these input points surrounding output points have the same surface type
    nearest_same_type = np.logical_not(
        np.logical_xor(nearest_is_land, out_is_land_bcast)
    )  # dimensions M x K

    return nearest_same_type


def slice_cube_by_domain(cube_in, output_domain):
    """
    extract cube domain to be consistent as cube_reference's domain

    Args:
        cube_in (iris.cube.Cube):
            input data cube to be sliced
        output_domain(tuple):
            lat_max,lon_max,lat_min,lon_min

    Returns:
        cube_in (iris.cube.Cube):
            data cube after slicing
    """
    lat_max, lon_max, lat_min, lon_min = output_domain
    lat_d, lon_d = get_grid_spacing(cube_in)

    domain = iris.Constraint(
        latitude=lambda val: lat_min - 2.0 * lat_d < val < lat_max + 2.0 * lat_d
    ) & iris.Constraint(
        longitude=lambda val: lon_min - 2.0 * lon_d < val < lon_max + 2.0 * lon_d
    )

    cube_in = cube_in.extract(domain)

    return cube_in


def slice_mask_cube_by_domain(cube_in, cube_in_mask, output_domain):
    """
    extract cube domain to be consistent as cube_reference's domain

    Args:
        cube_in (iris.cube.Cube):
            input data cube to be sliced
        cube_in_mask (iris.cube.Cube):
            input maskcube to be sliced
        output_domain (Tuple[float, float, float, float]):
            lat_max, lon_max, lat_min, lon_min
    Returns:
        Tuple[iris.cube.cube, iris.cube.Cube]:
            data cube after slicing, mask cube after slicing
    """
    lat_max, lon_max, lat_min, lon_min = output_domain
    lat_d_1, lon_d_1 = get_grid_spacing(cube_in)
    lat_d_2, lon_d_2 = get_grid_spacing(cube_in_mask)
    lat_d = lat_d_1 if lat_d_1 > lat_d_2 else lat_d_2
    lon_d = lon_d_1 if lon_d_1 > lon_d_2 else lon_d_2

    domain = iris.Constraint(
        latitude=lambda val: lat_min - 2.0 * lat_d < val < lat_max + 2.0 * lat_d
    ) & iris.Constraint(
        longitude=lambda val: lon_min - 2.0 * lon_d < val < lon_max + 2.0 * lon_d
    )

    cube_in = cube_in.extract(domain)
    cube_in_mask = cube_in_mask.extract(domain)

    return cube_in, cube_in_mask


def create_regrid_cube(cube_array, cube_in, cube_out):
    """
    create a regridded cube from regridded value(numpy array)
    source cube cube_in must be in  GeogCS's lats/lons system
    tergat cube_out either lats/lons system or LambertAzimuthalEqualArea system

    Args:
        cube_array (numpy ndarray):
            regridded value (multidimensional)
        cube_in (iris.cube.Cube):
            source cube (for value's non-grid dimensions and attributes)
        cube_out (iris.cube.Cube):
            target cube (for target grid information)

    Returns:
         iris.cube.Cube: regridded result cube
    """
    cube_coord_names = get_cube_coord_names(cube_in)
    lats_name, lons_name = latlon_names(cube_in)
    cube_coord_names.remove(lats_name)
    cube_coord_names.remove(lons_name)

    cube_v = Cube(cube_array)
    cube_v.attributes = cube_in.attributes

    cube_v.var_name = cube_in.var_name
    cube_v.standard_name = cube_in.standard_name
    cube_v.units = cube_in.units

    ndim = len(cube_coord_names)
    for i, val in enumerate(cube_coord_names):
        cube_v.add_dim_coord(cube_in.coord(val), i)

    cube_coord_names = get_cube_coord_names(cube_out)
    if "projection_y_coordinate" in cube_coord_names:
        cord_1 = "projection_y_coordinate"
        cord_2 = "projection_x_coordinate"
    else:
        cord_1, cord_2 = latlon_names(cube_out)

    cube_v.add_dim_coord(cube_out.coord(cord_1), ndim)
    cube_v.add_dim_coord(cube_out.coord(cord_2), ndim + 1)

    for coord in cube_in.aux_coords:
        dims = np.array(cube_in.coord_dims(coord)) + 1
        cube_v.add_aux_coord(coord.copy(), dims)

    return cube_v
