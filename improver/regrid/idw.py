# (C) Crown copyright, Met Office. All rights reserved.
#
# This file is part of improver and is released under a BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""
Inverse distance weighting interpolation functions
"""

from typing import Tuple

import numpy as np
from cartopy.crs import Geocentric, Geodetic
from numpy import ndarray
from scipy.spatial.ckdtree import cKDTree as KDTree

from improver.regrid.grid import similar_surface_classify

# An optimal distance scaling power was found through minimising regridding RMSE
# using inverse distance weighting on a collection of sample surface temperature grids
OPTIMUM_IDW_POWER = 1.80


def inverse_distance_weighting(
    idw_out_indexes: ndarray,
    in_latlons: ndarray,
    out_latlons: ndarray,
    indexes: ndarray,
    weights: ndarray,
    in_classified: ndarray,
    out_classified: ndarray,
) -> Tuple[ndarray, ndarray, ndarray]:
    """
    Locating source points and calculating inverse distance weights for selective target points.

    Args:
        idw_out_indexes:
            Selected target points which will use Inverse Distance Weighting(idw) approach.
        in_latlons:
            Source points's latitude-longitudes.
        out_latlons:
            Target points's latitude-longitudes.
        indexes:
            Array of source grid point number for all target grid points.
        weights:
            Array of source grid point weighting for all target grid points.
        in_classified:
            Land_sea type for source grid points (land ->True).
        out_classified:
            Land_sea type for target grid points (land ->True).

    Returns:
        - Updated Indexes - source grid point number for all target grid points.
        - Updated weights - array from each target grid point to its source grid points.
        - Output_points_no_match - special target points without matching source points.
    """

    out_latlons_updates = out_latlons[idw_out_indexes]
    k_nearest = 4
    distances_updates, indexes_updates = nearest_input_pts(
        in_latlons, out_latlons_updates, k_nearest
    )

    out_classified_updates = out_classified[idw_out_indexes]
    surface_type_mask_updates = similar_surface_classify(
        in_classified, out_classified_updates, indexes_updates
    )

    # There may be some output points with no matching nearby surface type (lakes/islands)
    # Output an array of these so that the calling function can apply further processing to those
    count_matching_surface = np.count_nonzero(surface_type_mask_updates, axis=1)
    points_with_no_match = (np.where(count_matching_surface == 0))[0]
    output_points_no_match = idw_out_indexes[points_with_no_match]

    # Apply inverse distance weighting to points that do have matching surface type input
    points_with_match = (np.where(count_matching_surface > 0))[0]
    output_points_match = idw_out_indexes[points_with_match]

    # Convert mask to be true where input points should not be considered
    not_mask = np.logical_not(surface_type_mask_updates[points_with_match])

    # Replace distances with infinity where they should not be used
    masked_distances = np.where(
        not_mask, np.float32(np.inf), distances_updates[points_with_match]
    )

    # Add a small amount to all distances to avoid division by zero when taking the inverse
    masked_distances += np.finfo(np.float32).eps
    # Invert the distances, sum the k surrounding points, scale to produce weights
    inv_distances = 1.0 / masked_distances
    # add power 1.80 for inverse diatance weight
    inv_distances_power = np.power(inv_distances, OPTIMUM_IDW_POWER)
    inv_distances_sum = np.sum(inv_distances_power, axis=1)
    inv_distances_sum = 1.0 / inv_distances_sum
    weights_idw = inv_distances_power * inv_distances_sum.reshape(-1, 1)

    # Update indexes and weights with new values
    indexes[output_points_match] = indexes_updates[points_with_match]
    weights[output_points_match] = weights_idw
    return indexes, weights, output_points_no_match


def nearest_input_pts(
    in_latlons: ndarray, out_latlons: ndarray, k: int
) -> Tuple[ndarray, ndarray]:
    """
    Find k nearest source (input) points to each target (output)
    point, using a KDtree.

    Args:
        in_latlons:
            Source grid points' latitude-longitudes (N x 2).
        out_latlons:
            Target grid points' latitude-longitudes (M x 2).
        k:
            Number of points surrounding each output point.

    Return:
        - Distances from target grid point to source grid points (M x K).
        - Indexes of those source points (M x K).
    """
    # Convert input latitude and longitude to XYZ coordinates, then create KDtree
    in_x, in_y, in_z = ecef_coords(in_latlons[:, 0].flat, in_latlons[:, 1].flat)
    in_coords = np.c_[in_x, in_y, in_z]
    in_kdtree = KDTree(in_coords)
    # Convert output to XYZ and query the KDtree for nearby input points
    out_x, out_y, out_z = ecef_coords(out_latlons[:, 0].flat, out_latlons[:, 1].flat)
    out_coords = np.c_[out_x, out_y, out_z]
    distances, indexes = in_kdtree.query(out_coords, k)
    # Avoid single dimension output for k=1 case
    if distances.ndim == 1:
        distances = np.expand_dims(distances, axis=1)
    if indexes.ndim == 1:
        indexes = np.expand_dims(indexes, axis=1)
    return distances, indexes


def ecef_coords(lats: ndarray, lons: ndarray) -> Tuple[ndarray, ndarray, ndarray]:
    """
    Transform latitude-longitude coordinates to earth centred, earth fixed
    cartesian XYZ coordinates.

    Args:
        lats:
            Latitude coordinates.
        lons:
            Longitude coordinates.

    Returns:
        - X transformed coordinates.
        - Y transformed coordinates.
        - Z transformed coordinates.
    """
    # Cartopy Geodetic and Geocentric both default to the WGS84 datum
    spherical_latlon_crs = Geodetic()
    ecef_crs = Geocentric()
    xyz = ecef_crs.transform_points(
        spherical_latlon_crs, np.array(lons), np.array(lats)
    )
    return xyz[..., 0], xyz[..., 1], xyz[..., 2]
