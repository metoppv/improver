# (C) Crown copyright, Met Office. All rights reserved.
#
# This file is part of improver and is released under a BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""
Grid handling for regridding
"""
from typing import List, Tuple, Union

import iris
import numpy as np
from iris.cube import Cube
from numpy import ndarray
from numpy.ma.core import MaskedArray
from scipy.interpolate import RegularGridInterpolator

from improver.utilities.cube_manipulation import sort_coord_in_cube
from improver.utilities.spatial import calculate_grid_spacing, lat_lon_determine


def ensure_ascending_coord(cube: Cube) -> Cube:
    """
    Check if cube coordinates ascending. if not, make it ascending

    Args:
        cube:
            Input source cube.

    Returns:
        Cube with ascending coordinates
    """
    for ax in ("x", "y"):
        if cube.coord(axis=ax).points[0] > cube.coord(axis=ax).points[-1]:
            cube = sort_coord_in_cube(cube, cube.coord(axis=ax).standard_name)
    return cube


def calculate_input_grid_spacing(cube_in: Cube) -> Tuple[float, float]:
    """
    Calculate grid spacing in latitude and logitude.
    Check if input source grid is on even-spacing and ascending lat/lon system.

    Args:
        cube_in:
            Input source cube.

    Returns:
        - Grid spacing in latitude, in degree.
        - Grid spacing in logitude, in degree.

    Raises:
        ValueError:
            If input grid is not on a latitude/longitude system or
            input grid coordinates are not ascending.
    """
    # check if in lat/lon system
    if lat_lon_determine(cube_in) is not None:
        raise ValueError("Input grid is not on a latitude/longitude system")

    # calculate grid spacing
    lon_spacing = calculate_grid_spacing(cube_in, "degree", axis="x", rtol=4.0e-5)
    lat_spacing = calculate_grid_spacing(cube_in, "degree", axis="y", rtol=4.0e-5)

    y_coord = cube_in.coord(axis="y").points
    x_coord = cube_in.coord(axis="x").points
    if x_coord[-1] < x_coord[0] or y_coord[-1] < y_coord[0]:
        raise ValueError("Input grid coordinates are not ascending.")
    return lat_spacing, lon_spacing


def get_cube_coord_names(cube: Cube) -> List[str]:
    """
    Get all coordinate names from a cube.

    Args:
        cube:
            Input cube.

    Returns:
        List of coordinate names.
    """
    return [coord.standard_name for coord in cube.dim_coords]


def latlon_names(cube: Cube) -> Tuple[str, str]:
    """
    Identify the names of the latitude and longitude dimensions of cube.

    Args:
        cube:
            Input cube.

    Returns:
        - Name of latitude dimension of cube.
        - Name of longitude dimension of cube.
    """
    lats_name = cube.coord(axis="y").standard_name
    lons_name = cube.coord(axis="x").standard_name
    return lats_name, lons_name


def latlon_from_cube(cube: Cube) -> ndarray:
    """
    Produce an array of latitude-longitude coordinates used by an Iris cube.

    Args:
        cube:
            Cube with spatial coords.

    Returns:
        Latitude-longitude pairs (N x 2).
    """
    lats_name, lons_name = latlon_names(cube)
    lats_data = cube.coord(lats_name).points
    lons_data = cube.coord(lons_name).points
    lats_mesh, lons_mesh = np.meshgrid(lats_data, lons_data, indexing="ij")
    latlon = np.dstack((lats_mesh, lons_mesh)).reshape((-1, 2))
    return latlon


def unflatten_spatial_dimensions(
    regrid_result: ndarray,
    cube_out_mask: Cube,
    in_values: ndarray,
    lats_index: int,
    lons_index: int,
) -> Union[ndarray, MaskedArray]:
    """
    Reshape numpy array regrid_result from (lat*lon,...) to (....,lat,lon)
    or from (projy*projx,...) to (...,projy,projx).

    Args:
        regrid_result:
            Array of regridded result in (lat*lon,....) or (projy*projx,...).
        cube_out_mask:
            Target grid cube (for getting grid dimension here).
        in_values:
            Reshaped source data (in _reshape_data_cube).
        lats_index:
            Index of lats or projy coord in reshaped array.
        lons_index:
            Index of lons or projx coord in reshaped array.

    Returns:
        Reshaped data array.
    """
    cube_out_dim0 = cube_out_mask.coord(axis="y").shape[0]
    cube_out_dim1 = cube_out_mask.coord(axis="x").shape[0]
    latlon_shape = [cube_out_dim0, cube_out_dim1] + list(in_values.shape[1:])

    regrid_result = np.reshape(regrid_result, latlon_shape)
    regrid_result = np.swapaxes(regrid_result, 1, lons_index)
    regrid_result = np.swapaxes(regrid_result, 0, lats_index)
    return regrid_result


def flatten_spatial_dimensions(
    cube: Cube,
) -> Tuple[Union[ndarray, MaskedArray], int, int]:
    """
    Reshape data cube from (....,lat,lon) into data (lat*lon,...).

    Args:
        cube:
            Original data cube.

    Returns:
        - Reshaped data array.
        - Index of latitude cube coords.
        - Index of longitude cube coords.
    """
    in_values = cube.data
    lats_name, lons_name = latlon_names(cube)
    lats_index = cube.coord_dims(lats_name)[0]
    lons_index = cube.coord_dims(lons_name)[0]

    in_values = np.swapaxes(in_values, 0, lats_index)
    in_values = np.swapaxes(in_values, 1, lons_index)

    lats_len = int(in_values.shape[0])
    lons_len = int(in_values.shape[1])
    latlon_shape = [lats_len * lons_len] + list(in_values.shape[2:])
    in_values = np.reshape(in_values, latlon_shape)
    return in_values, lats_index, lons_index


def classify_output_surface_type(cube_out_mask: Cube) -> ndarray:
    """
    Classify surface types of target grid points based on a binary True/False land mask.

    Args:
        cube_out_mask:
            land_sea mask information cube for target grid (land=1)

    Returns:
        1D land-sea mask information for 1D-ordered target grid points
    """
    # cube y-axis => latitude or projection-y
    cube_out_dim0 = cube_out_mask.coord(axis="y").shape[0]
    cube_out_dim1 = cube_out_mask.coord(axis="x").shape[0]
    out_classified = cube_out_mask.data.reshape(cube_out_dim0 * cube_out_dim1)
    return out_classified


def classify_input_surface_type(
    cube_in_mask: Cube, classify_latlons: ndarray
) -> ndarray:
    """
    Classify surface types of source grid points based on a binary True/False land mask
    cube_in_mask's grid could be different from input source grid of NWP results.

    Args:
        cube_in_mask:
            Land_sea mask information cube for input source grid (land=1)
            which should be in GeogCS's lats/lons coordinate system.
        classify_latlons:
            Latitude and longitude source grid points to classify (N x 2).

    Returns:
        Classifications (N) for 1D-ordered source grid points.
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


def similar_surface_classify(
    in_is_land: ndarray, out_is_land: ndarray, nearest_in_indexes: ndarray
) -> ndarray:
    """
    Classify surface types as matched (True) or unmatched(False) between target points
    and their source point.

    Args:
        in_is_land:
            Source point classifications (N).
        out_is_land:
            Target point classifications (M).
        nearest_in_indexes:
            Indexes of input points nearby output points (M x K).

    Return:
        Boolean true if input surface type matches output or no matches (M x K).
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


def slice_cube_by_domain(
    cube_in: Cube, output_domain: Tuple[float, float, float, float]
) -> Cube:
    """
    Extract cube domain to be consistent as cube_reference's domain.

    Args:
        cube_in:
            Input data cube to be sliced.
        output_domain:
            Lat_max, lon_max, lat_min, lon_min.

    Returns:
        Data cube after slicing.
    """
    lat_max, lon_max, lat_min, lon_min = output_domain
    lat_d, lon_d = calculate_input_grid_spacing(cube_in)

    domain = iris.Constraint(
        latitude=lambda val: lat_min - 2.0 * lat_d < val < lat_max + 2.0 * lat_d
    ) & iris.Constraint(
        longitude=lambda val: lon_min - 2.0 * lon_d < val < lon_max + 2.0 * lon_d
    )

    cube_in = cube_in.extract(domain)

    return cube_in


def slice_mask_cube_by_domain(
    cube_in: Cube, cube_in_mask: Cube, output_domain: Tuple[float, float, float, float]
) -> Tuple[Cube, Cube]:
    """
    Extract cube domain to be consistent as cube_reference's domain.

    Args:
        cube_in:
            Input data cube to be sliced.
        cube_in_mask:
            Input mask cube to be sliced.
        output_domain:
            Lat_max, lon_max, lat_min, lon_min.

    Returns:
        - Data cube after slicing.
        - Mask cube after slicing.
    """
    lat_max, lon_max, lat_min, lon_min = output_domain
    lat_d_1, lon_d_1 = calculate_input_grid_spacing(cube_in)
    lat_d_2, lon_d_2 = calculate_input_grid_spacing(cube_in_mask)
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


def create_regrid_cube(cube_array: ndarray, cube_in: Cube, cube_out: Cube) -> Cube:
    """
    Create a regridded cube from regridded value(numpy array).
    Source cube_in must be in regular latitude/longitude coordinates.
    Target cube_out can be either regular latitude/longitude grid or equal area.

    Args:
        cube_array:
            regridded value (multidimensional)
        cube_in:
            source cube (for value's non-grid dimensions and attributes)
        cube_out:
            target cube (for target grid information)

    Returns:
        Regridded result cube
    """
    # generate a cube based on new data and cube_in
    cube_v = Cube(cube_array, units=cube_in.units, attributes=cube_in.attributes)
    cube_v.rename(cube_in.standard_name or cube_in.long_name)
    cube_v.var_name = cube_in.var_name

    # use dim_coord from cube_in except lat/lon
    cube_coord_names = get_cube_coord_names(cube_in)
    lats_name, lons_name = latlon_names(cube_in)
    cube_coord_names.remove(lats_name)
    cube_coord_names.remove(lons_name)

    ndim = len(cube_coord_names)
    for i, val in enumerate(cube_coord_names):
        cube_v.add_dim_coord(cube_in.coord(val), i)

    # Put in suitable spatial coord from cube_out into cube_in
    cord_1, cord_2 = latlon_names(cube_out)
    cube_v.add_dim_coord(cube_out.coord(cord_1), ndim)
    cube_v.add_dim_coord(cube_out.coord(cord_2), ndim + 1)

    # add all aus_coords from cube_in

    for coord in cube_in.aux_coords:
        cube_v.add_aux_coord(coord.copy(), cube_in.coord_dims(coord))

    return cube_v


def group_target_points_with_source_domain(
    cube_in: Cube, out_latlons: ndarray
) -> Tuple[ndarray, ndarray]:
    """
    Group cube_out's grid points into outside or inside cube_in's domain.

    Args:
        cube_in:
            Source cube.
        out_latlons:
            Target points's latitude-longitudes.

    Returns:
        - Index array of target points outside input domain.
        - Index array of target points inside input domain.
    """

    # get latitude and longitude coordinates of cube_in
    lat_coord = cube_in.coord(axis="y").points
    lon_coord = cube_in.coord(axis="x").points

    in_lat_max, in_lat_min = np.max(lat_coord), np.min(lat_coord)
    in_lon_max, in_lon_min = np.max(lon_coord), np.min(lon_coord)

    lat = out_latlons[:, 0]
    lon = out_latlons[:, 1]

    # check target point coordinates inside/outside input source domain
    in_domain_lat = np.logical_and(lat >= in_lat_min, lat <= in_lat_max)
    in_domain_lon = np.logical_and(lon >= in_lon_min, lon <= in_lon_max)
    in_domain = np.logical_and(in_domain_lat, in_domain_lon)

    outside_input_domain_index = np.where(np.logical_not(in_domain))[0]
    inside_input_domain_index = np.where(in_domain)[0]

    return outside_input_domain_index, inside_input_domain_index


def mask_target_points_outside_source_domain(
    total_out_point_num: int,
    outside_input_domain_index: ndarray,
    inside_input_domain_index: ndarray,
    regrid_result: Union[ndarray, MaskedArray],
) -> Union[ndarray, MaskedArray]:
    """
    Mask target points outside cube_in's domain.

    Args:
        total_out_point_num:
            Total number of target points
        outside_input_domain_index:
            Index array of target points outside input domain.
        inside_input_domain_index:
            Index array of target points inside input domain.
        regrid_result:
            Array of regridded result in (lat*lon,....) or (projy*projx,...).

    Returns:
        Array of regridded result in (lat*lon,....) or (projy*projx,...).
    """
    # masked cube_out grid points which are out of cube_in range

    output_shape = [total_out_point_num] + list(regrid_result.shape[1:])
    if isinstance(regrid_result, np.ma.MaskedArray):
        output = np.ma.zeros(output_shape, dtype=np.float32)
        output.mask = np.full(output_shape, True, dtype=bool)
        output.mask[inside_input_domain_index] = regrid_result.mask
        output.data[inside_input_domain_index] = regrid_result.data
    else:
        output = np.zeros(output_shape, dtype=np.float32)
        output[inside_input_domain_index] = regrid_result
        output[outside_input_domain_index] = np.nan

    return output
