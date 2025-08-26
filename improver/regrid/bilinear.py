# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""
.. Further information is available in:
.. include:: extended_documentation/regrid/
   bilinear_land_sea.rst

"""

from typing import Tuple, Union

import numpy as np
from numpy import ndarray
from numpy.ma.core import MaskedArray

from improver.regrid.grid import similar_surface_classify
from improver.regrid.idw import (
    OPTIMUM_IDW_POWER,
    inverse_distance_weighting,
    nearest_input_pts,
)

NUM_NEIGHBOURS = 4


def apply_weights(
    indexes: ndarray, in_values: Union[ndarray, MaskedArray], weights: ndarray
) -> Union[ndarray, MaskedArray]:
    """
    Apply bilinear weight of source points for target value.

    Args:
        indexes:
            Array of source grid point number for target grid points.
        weights:
            Array of source grid point weighting for target grid points.
        in_values:
            Input values (maybe multidimensional).

    Returns:
        Regridded values for target points.
    """
    input_array_masked = False
    if isinstance(in_values, MaskedArray):
        input_array_masked = True
        in_values = np.ma.filled(in_values, np.nan)
    in_values_expanded = in_values[indexes]

    weighted = np.transpose(
        np.multiply(np.transpose(weights), np.transpose(in_values_expanded))
    )
    out_values = np.sum(weighted, axis=1)
    if input_array_masked:
        out_values = np.ma.masked_invalid(out_values)

    return out_values


def basic_indexes(
    out_latlons: ndarray,
    in_latlons: ndarray,
    in_lons_size: int,
    lat_spacing: float,
    lon_spacing: float,
) -> ndarray:
    """
    Locating source points for each target point.

    Args:
        in_latlons:
            Source points's latitude-longitudes.
        out_latlons:
            Target points's latitude-longitudes.
        in_lons_size:
            Source grid's longitude dimension.
        lat_spacing:
            Input grid latitude spacing, in degree.
        lon_spacing:
            Input grid longitude spacing, in degree.

    Returns:
        Updated array of source grid point number for all target grid points.
        Array shape is (total number of target points, 4).
    """
    # Calculate input/output offset, expressed in terms of the spacing
    n_lat = (out_latlons[:, 0] - in_latlons[0, 0]) // lat_spacing
    m_lon = (out_latlons[:, 1] - in_latlons[0, 1]) // lon_spacing
    n_lat = n_lat.astype(int)
    m_lon = m_lon.astype(int)

    # Four surrounding input points for each output point, in a rectangle shape
    index1 = n_lat * in_lons_size + m_lon
    index2 = index1 + in_lons_size
    index3 = index2 + 1
    index4 = index1 + 1

    # Rearrange order to match expected output style
    # note: lat (X) but ordering  (lat0, lon0)(lat0,lon1)....(lat0,lon_last),(lat1,lon0),
    indexes = np.transpose([index1, index2, index3, index4])

    # if identical max latitude and/or longitude between source/target grids,index algorithm
    # needs change at relevant boundary
    lat_max_in, lon_max_in = in_latlons.max(axis=0)
    lat_max_out, lon_max_out = out_latlons.max(axis=0)

    lat_max_equal = np.isclose(lat_max_in, lat_max_out)
    lon_max_equal = np.isclose(lon_max_in, lon_max_out)

    if lat_max_equal or lon_max_equal:
        indexes = adjust_boundary_indexes(
            in_lons_size,
            lat_max_equal,
            lon_max_equal,
            lat_max_in,
            lon_max_in,
            out_latlons,
            indexes,
        )

    return indexes


def adjust_boundary_indexes(
    in_lons_size: int,
    lat_max_equal: bool,
    lon_max_equal: bool,
    lat_max_in: float,
    lon_max_in: float,
    out_latlons: ndarray,
    indexes: ndarray,
) -> ndarray:
    """
    Adjust surrounding source point indexes for boundary target points.
    it is required when maximum latitude and logitude are identical between
    source and target grids.

    Args:
        in_lons_size:
            Source grid's longitude dimension.
        lat_max_equal:
            Whether maximum latitude is identical between source/targin grids.
        lon_max_equal:
            Whether maximum longitude is identical between source/targin grids.
        lat_max_in:
            Input grid's maximum latitude.
        lon_max_in:
            Input grid's maximum longtitude.
        out_latlons:
            Target points's latitude-longitudes.
        indexes:
            Updated array of source grid point number for all target grid points.

    Returns:
        Updated array of source grid point number for all target grid points.
        Array shape is (total number of target points, 4).
    """
    # find a list of target points with its latitude
    if lat_max_equal:
        point_lat_max = np.where(np.isclose(out_latlons[:, 0], lat_max_in))[0]
    if lon_max_equal:
        point_lon_max = np.where(np.isclose(out_latlons[:, 1], lon_max_in))[0]

    if lon_max_equal and lat_max_equal:
        point_lat_lon_max_index = np.where(
            np.isclose(out_latlons[point_lat_max, 1], lon_max_in)
        )[0]

        # if point_lat_lon_max_index exists, handle it.
        if point_lat_lon_max_index.size > 0:
            point_lat_lon_max = point_lat_max[point_lat_lon_max_index[0]]
            point_lat_max = np.delete(
                point_lat_max, np.where(point_lat_max == point_lat_lon_max)[0]
            )
            point_lon_max = np.delete(
                point_lon_max, np.where(point_lon_max == point_lat_lon_max)[0]
            )
            indexes[point_lat_lon_max, 2] = indexes[point_lat_lon_max, 0]
            indexes[point_lat_lon_max, 1] = indexes[point_lat_lon_max, 2] - 1
            indexes[point_lat_lon_max, 0] = indexes[point_lat_lon_max, 1] - in_lons_size
            indexes[point_lat_lon_max, 3] = indexes[point_lat_lon_max, 0] + 1

    if lat_max_equal:
        indexes[point_lat_max, 1] = indexes[point_lat_max, 0]
        indexes[point_lat_max, 2] = indexes[point_lat_max, 1] + 1
        indexes[point_lat_max, 0] = indexes[point_lat_max, 1] - in_lons_size
        indexes[point_lat_max, 3] = indexes[point_lat_max, 0] + 1

    if lon_max_equal:
        indexes[point_lon_max, 0] = indexes[point_lon_max, 0] - 1
        indexes[point_lon_max, 1] = indexes[point_lon_max, 1] - 1
        indexes[point_lon_max, 2] = indexes[point_lon_max, 1] + 1
        indexes[point_lon_max, 3] = indexes[point_lon_max, 0] + 1

    return indexes


def basic_weights(
    index_range: ndarray,
    indexes: ndarray,
    out_latlons: ndarray,
    in_latlons: ndarray,
    lat_spacing: float,
    lon_spacing: float,
) -> ndarray:
    """
    Calculate weighting for selecting target points using standard bilinear function.

    Args:
        index_range:
            A list of target points.
        indexes:
            Array of source grid point number for all target grid points.
        in_latlons:
            Source points's latitude-longitudes.
        out_latlons:
            Target points's latitude-longitudes.
        lat_spacing:
            Input grid latitude spacing, in degree.
        lon_spacing:
            Input grid longitude spacing, in degree.

    Returns:
        Weighting array of source grid point number for target grid points.
    """
    # Set up input points spacing values
    latlon_area = lat_spacing * lon_spacing

    out_lats = out_latlons[index_range, 0]
    out_lons = out_latlons[index_range, 1]
    # Calculate weights for four surrounding input points
    # Input point 1 - left bottom input point
    lat_1 = in_latlons[indexes[index_range, 0], 0]
    lon_1 = in_latlons[indexes[index_range, 0], 1]
    # Input point 2 - left-top input point
    lat2_lat = lat_1 + lat_spacing - out_lats
    # Input point 3 - right top input point
    lon3_lon = lon_1 + lon_spacing - out_lons
    lon_lon1 = out_lons - lon_1
    lat_lat1 = out_lats - lat_1
    # Input point 4 (right-bottom) is implied by the others

    weight1 = lat2_lat * lon3_lon / latlon_area
    weight2 = lat_lat1 * lon3_lon / latlon_area
    weight3 = lat_lat1 * lon_lon1 / latlon_area
    weight4 = lat2_lat * lon_lon1 / latlon_area
    weights = np.transpose([weight1, weight2, weight3, weight4])
    return weights


def adjust_for_surface_mismatch(
    in_latlons: ndarray,
    out_latlons: ndarray,
    in_classified: ndarray,
    out_classified: ndarray,
    weights: ndarray,
    indexes: ndarray,
    surface_type_mask: ndarray,
    in_lons_size: int,
    vicinity: float,
    lat_spacing: float,
    lon_spacing: float,
) -> Tuple[ndarray, ndarray]:
    """
    Updating source points and weighting for mismatched-source-point cases.

    1. Triangle interpolation function is used for only one mismatched source point and
       target point is within the triangle formed with three matched sourced point.
    2. In one of 3 cases (a)one false source points, three true source points but the target
       point is outside triangle (b)Two false source points, two true source points (c) three
       false source points, one true source pointfor, find four surrounding source points
       using KDtree, and regridding with inverse distance weighting(IDW) if matched source
       point is available.
    3. In case of four mismatched source points(zero matched source point), Look up 8 points
       with specified distance limit (input) using KD tree, and then check if there are any
       same-type source points. If yes, pick up the points of the same type, and do IDW
       interpolation. If no, ignore surface type and just do normal bilinear interpolation.

    Args:
        in_latlons:
            Source points's latitude-longitudes.
        out_latlons:
            Target points's latitude-longitudes.
        in_classified:
            Land_sea type for source grid points (land ->True).
        out_classified:
            Land_sea type for terget grid points (land ->True).
        weights:
            Array of source grid point weighting for all target grid points.
        indexes:
            Array of source grid point number for all target grid points.
        surface_type_mask:
            Numpy ndarray of bool, true if source point type matches target point type.
        in_lons_size:
            Longitude dimension in cube_in.
        vicinity:
            Radius of specified searching domain, in meter.
        lat_spacing:
            Input grid latitude spacing, in degree.
        lon_spacing:
            Input grid longitude spacing, in degree.

    Returns:
        - Updated array of source grid point weights for all target grid points.
        - Updated array of source grid point index for all target grid points.
    """
    count_same_surface_type = np.count_nonzero(surface_type_mask, axis=1)

    # Initialise weights to zero at locations with mismatched surface types
    mismatched_surface_type = np.where(np.logical_not(surface_type_mask))
    weights[mismatched_surface_type] = 0.0

    # Cases with one mismatched input point by adjusting bilinear weights
    # leftover_bilinear is special cases of using inverse distance weighting
    one_mismatch = np.where((count_same_surface_type == 3))[0]
    weights, leftover_bilinear = one_mismatched_input_point(
        one_mismatch,
        surface_type_mask,
        indexes,
        weights,
        out_latlons,
        in_latlons,
        lat_spacing,
        lon_spacing,
    )

    # Cases with two and three mismatched input points
    three_mismatch = np.where(count_same_surface_type == 1)[0]
    two_mismatch = np.where(count_same_surface_type == 2)[0]

    # Use inverse distance weighting to handle the cases with 2/3 mismatched input points
    # and the leftover one mismatched cases that were found to involve extrapolation
    apply_idw_indexes = np.concatenate(
        (leftover_bilinear, two_mismatch, three_mismatch)
    )
    indexes, weights, leftover_idw = inverse_distance_weighting(
        apply_idw_indexes,
        in_latlons,
        out_latlons,
        indexes,
        weights,
        in_classified,
        out_classified,
    )

    # Cases with all four input points having mismatched surface types compared
    # to the output point. These are lakes (water surrounded by land) and islands
    # (land surrounded by water). Leftovers from IDW are cases where IDW was
    # unable to find a matching surface type.
    four_mismatch = np.where((count_same_surface_type == 0))[0]
    if four_mismatch.shape[0] > 0 or leftover_idw.shape[0] > 0:
        four_mismatch = np.concatenate((four_mismatch, leftover_idw))
        weights, indexes, surface_type_mask = lakes_islands(
            four_mismatch,
            weights,
            indexes,
            surface_type_mask,
            in_latlons,
            out_latlons,
            in_classified,
            out_classified,
            in_lons_size,
            vicinity,
            lat_spacing,
            lon_spacing,
        )

    return weights, indexes


def one_mismatched_input_point(
    one_mismatch_indexes: ndarray,
    surface_type_mask: ndarray,
    indexes: ndarray,
    weights: ndarray,
    out_latlons: ndarray,
    in_latlons: ndarray,
    lat_spacing: float,
    lon_spacing: float,
) -> Tuple[ndarray, ndarray]:
    """
    Updating source points and weighting for one mismatched source-point cases.
    If target is not within the triangle formed by 3 matched-sourced-points,
    updating of weights is deferred to inverse-distance-weight method.

    Args:
        one_mismatch_indexes:
            Selected target points which have 1 false source point.
        surface_type_mask:
            Numpy ndarray of bool, true if source point type matches target point type.
        indexes:
            Array of source grid point number for all target grid points.
        weights:
            Array of source grid point weighting for all target grid points.
        in_latlons:
            Source points's latitude-longitudes.
        out_latlons:
            Target points's latitude-longitudes.
        in_lons_size:
            Source grid's longitude dimension.
        lat_spacing:
            Input grid latitude spacing, in degree.
        lon_spacing:
            Input grid longitude spacing, in degree.

    Returns:
        - Updated weights - array of source grid point weighting for target grid points.
        - Excluded indexes - target points which are not handled in this function.
    """
    lat_lon_area = lat_spacing * lon_spacing
    excluded_indexes = np.array([], dtype=int)

    # Process groups of output points which have each one of the four surrounding
    # input points as mismatched surface type
    for i in range(NUM_NEIGHBOURS):
        # Determine group of output points to process in this iteration
        inverse_mask_i = np.where(
            np.logical_not(surface_type_mask[one_mismatch_indexes, i])
        )[0]
        indexes_with_i_mismatched = one_mismatch_indexes[inverse_mask_i]

        # Extract subset of output lat/lon
        out_lats = out_latlons[indexes_with_i_mismatched, 0]
        out_lons = out_latlons[indexes_with_i_mismatched, 1]

        # Calculate lat/lon of left bottom input point
        lat_1 = in_latlons[indexes[indexes_with_i_mismatched, 0], 0]
        lon_1 = in_latlons[indexes[indexes_with_i_mismatched, 0], 1]

        # Calculate updated weights
        if i == 0:
            lat2_lat = lat_1 + lat_spacing - out_lats
            lon3_lon = lon_1 + lon_spacing - out_lons
            weight2 = lat_spacing * lon3_lon / lat_lon_area
            weight4 = lat2_lat * lon_spacing / lat_lon_area
            weight1 = np.zeros(weight2.shape[0], dtype=np.float32)
            weight3 = np.ones(weight2.shape[0], dtype=np.float32)
            weight3 -= weight2 + weight4
        elif i == 1:
            lon3_lon = lon_1 + lon_spacing - out_lons
            lat_lat1 = out_lats - lat_1
            weight1 = lat_spacing * lon3_lon / lat_lon_area
            weight3 = lat_lat1 * lon_spacing / lat_lon_area
            weight2 = np.zeros(weight1.shape[0], dtype=np.float32)
            weight4 = np.ones(weight1.shape[0], dtype=np.float32)
            weight4 -= weight1 + weight3
        elif i == 2:
            lon_lon1 = out_lons - lon_1
            lat_lat1 = out_lats - lat_1
            weight2 = lat_lat1 * lon_spacing / lat_lon_area
            weight4 = lat_spacing * lon_lon1 / lat_lon_area
            weight3 = np.zeros(weight2.shape[0], dtype=np.float32)
            weight1 = np.ones(weight2.shape[0], dtype=np.float32)
            weight1 -= weight2 + weight4
        else:  # i == 3
            lat2_lat = lat_1 + lat_spacing - out_lats
            lon_lon1 = out_lons - lon_1
            weight3 = lat_spacing * lon_lon1 / lat_lon_area
            weight1 = lat2_lat * lon_spacing / lat_lon_area
            weight4 = np.zeros(weight1.shape[0], dtype=np.float32)
            weight2 = np.ones(weight1.shape[0], dtype=np.float32)
            weight2 -= weight1 + weight3

        # Gather weights into array so they can be inserted
        weights_i = np.transpose([weight1, weight2, weight3, weight4])
        # Exclude updating of weights where target outside triangle of 3 true sources
        weights_i_positive = weights_i > -1.0e-8
        all_weights_positive = np.all(weights_i_positive, axis=1)

        not_all_weights_positive = np.where(np.logical_not(all_weights_positive))[0]

        if not_all_weights_positive.shape[0] > 0:
            good_update_indexes = np.where(all_weights_positive)[0]
            weights[indexes_with_i_mismatched[good_update_indexes]] = weights_i[
                good_update_indexes
            ]
            # Keep track of excluded points
            excluded_index = indexes_with_i_mismatched[not_all_weights_positive]
            excluded_indexes = np.concatenate((excluded_indexes, excluded_index))
        else:
            # Apply all updated weights as-is
            weights[indexes_with_i_mismatched] = weights_i

    return weights, excluded_indexes


def lakes_islands(
    lake_island_indexes: ndarray,
    weights: ndarray,
    indexes: ndarray,
    surface_type_mask: ndarray,
    in_latlons: ndarray,
    out_latlons: ndarray,
    in_classified: ndarray,
    out_classified: ndarray,
    in_lons_size: int,
    vicinity: float,
    lat_spacing: float,
    lon_spacing: float,
) -> Tuple[ndarray, ndarray, ndarray]:
    """
    Updating source points and weighting for 4-false-source-point cases.
    These are lakes (water surrounded by land) and islands (land surrounded by water).
    Note that a similar function can be found in nearest.py for nearest
    neighbour regridding rather than bilinear regridding.

    Args:
        lake_island_indexes:
            Selected target points which have 4 false source points.
        in_latlons:
            Source points's latitude-longitudes.
        out_latlons:
            Target points's latitude-longitudes.
        surface_type_mask:
            Numpy ndarray of bool, true if source point type matches target point type.
        indexes:
            Array of source grid point number for all target grid points.
        weights:
            Array of source grid point weighting for all target grid points.
        in_classified:
            Land_sea type for source grid points (land ->True).
        out_classified:
            Land_sea type for terget grid points (land ->True).
        in_lons_size:
            Source grid's longitude dimension.
        vicinity:
            Radius of specified searching domain, in meter.
        lat_spacing:
            Input grid latitude spacing, in degree.
        lon_spacing:
            Input grid longitude spacing, in degree.

    Returns:
        - Updated weights - source point weighting for all target grid points.
        - Updated indexes - source grid point number for all target grid points.
        - Updated surface_type_mask - matching info between source/target point types.
    """

    # increase 4 points to 8 points
    out_latlons_updates = out_latlons[lake_island_indexes]

    # Consider a larger area of 8 nearest points to look for more distant same
    # surface type input points
    # more than 8 points are within searching limits not considered here
    k_nearest = 8
    distances_updates, indexes_updates = nearest_input_pts(
        in_latlons, out_latlons_updates, k_nearest
    )

    # Update output surface classification and surface type mask
    out_classified_updates = out_classified[lake_island_indexes]
    surface_type_mask_updates = similar_surface_classify(
        in_classified, out_classified_updates, indexes_updates
    )

    # Where distance is outside specified vicinity, set surface type to be mismatched
    # so that it will not be used, update surface type mask again
    distance_not_in_vicinity = distances_updates > vicinity
    surface_type_mask_updates = np.where(
        distance_not_in_vicinity, False, surface_type_mask_updates
    )

    count_matching_surface = np.count_nonzero(surface_type_mask_updates, axis=1)
    points_with_no_match = np.where(count_matching_surface == 0)[0]

    # If the expanded search area hasn't found any same surface type matches anywhere
    # just ignore surface type, use normal bilinear approach
    if points_with_no_match.shape[0] > 0:
        # revert back to the basic bilinear weights, indexes unchanged
        no_match_indexes = lake_island_indexes[points_with_no_match]
        weights[no_match_indexes] = basic_weights(
            no_match_indexes, indexes, out_latlons, in_latlons, lat_spacing, lon_spacing
        )

    points_with_match = np.where(count_matching_surface > 0)[0]
    count_of_points_with_match = points_with_match.shape[0]

    # if no further processing can be done, return early
    if count_of_points_with_match == 0:
        return weights, indexes, surface_type_mask

    # Where a same surface type match has been found among the 8 nearest inputs, apply
    # inverse distance weighting with those matched points
    new_distances = np.zeros([count_of_points_with_match, NUM_NEIGHBOURS])
    for point_idx in range(points_with_match.shape[0]):
        match_indexes = lake_island_indexes[points_with_match[point_idx]]
        # Reset all input weight and surface type to mismatched
        weights[match_indexes, :] = 0.0
        surface_type_mask[match_indexes, :] = False
        # Loop through 8 nearest points found
        good_count = 0
        for i in range(k_nearest):
            # Look for an input point with same surface type as output
            if surface_type_mask_updates[points_with_match[point_idx], i]:
                # When found, update the indexes and surface mask to use that improved input point
                indexes[match_indexes, good_count] = indexes_updates[
                    points_with_match[point_idx], i
                ]
                surface_type_mask[match_indexes, good_count] = True  # other mask =false
                new_distances[point_idx, good_count] = distances_updates[
                    points_with_match[point_idx], i
                ]
                good_count += 1
                # Use a maximum of four same surface type input points
                # This is kind of like how bilinear uses four nearby input points
                if good_count == 4:
                    break

    lake_island_with_match = lake_island_indexes[points_with_match]
    # Similar to inverse_distance_weighting in idw.py
    not_mask = np.logical_not(surface_type_mask[lake_island_with_match])
    masked_distances = np.where(not_mask, np.float32(np.inf), new_distances)
    masked_distances += np.finfo(np.float32).eps
    inv_distances = 1.0 / masked_distances
    # add power 1.80 for inverse diatance weight
    inv_distances_power = np.power(inv_distances, OPTIMUM_IDW_POWER)
    inv_distances_sum = np.sum(inv_distances_power, axis=1)
    inv_distances_sum = 1.0 / inv_distances_sum
    weights_idw = inv_distances_power * inv_distances_sum.reshape(-1, 1)
    weights[lake_island_with_match] = weights_idw

    return weights, indexes, surface_type_mask
