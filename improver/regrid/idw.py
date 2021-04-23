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
Inverse distance weighting interpolation functions
"""

import numpy as np
from scipy.spatial.ckdtree import cKDTree as KDTree

# WGS84: The World Geodetic System 1984
WGS84_A = 6378137.0
WGS84_IF = 298.257223563
WGS84_F = 1.0 / WGS84_IF
WGS84_E = np.sqrt((2.0 * WGS84_F) - (WGS84_F * WGS84_F))


def inverse_distance_weighting(
    idw_out_indexes,
    in_latlons,
    out_latlons,
    indexes,
    weights,
    in_classified,
    out_classified,
):
    """
    Locating source points and calculating inverse distance weights for selective target points

    Args:
        idw_out_indexes (numpy.ndarray):
            selected target points which will use Inverse Distance Weighting(idw) approach
        in_latlons (numpy.ndarray):
            Source points's latitude-longitudes
        out_latlons (numpy.ndarray):
            Target points's latitude-longitudes
        indexes (numpy.ndarray):
            array of source grid point number for all target grid points
        weights (numpy.ndarray):
            array of source grid point weighting for all target grid points
        in_classified (numpy.ndarray):
            land_sea type for source grid points (land =>True)
        out_classified (numpy.ndarray):
            land_sea type for target grid points (land =>True)

    Returns:
        Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]:
            updated Indexes - source grid point number for all target grid points.
            updated weights - array from each target grid point to its source grid points
            output_points_no_match - special target points without matching source points
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
    optimum_power = 1.80
    inv_distances_power = np.power(inv_distances, optimum_power)
    inv_distances_sum = np.sum(inv_distances_power, axis=1)
    inv_distances_sum = 1.0 / inv_distances_sum
    weights_idw = inv_distances_power * inv_distances_sum.reshape(-1, 1)

    # Update indexes and weights with new values
    indexes[output_points_match] = indexes_updates[points_with_match]
    weights[output_points_match] = weights_idw
    return indexes, weights, output_points_no_match


def nearest_input_pts(in_latlons, out_latlons, k):
    """
    Find k nearest source (input) points to each target (output)
    point, using a KDtree

    Args:
        in_latlons (numpy.ndarray):
            Source grid points' latitude-longitudes (N x 2)
        out_latlons (numpy.ndarray):
            Target grid points' latitude-longitudes (M x 2)
        k (int):
            Number of points surrounding each output point

    Return:
        Tuple[numpy.ndarray, numpy.ndarray]:
            Distances from target grid point to source grid points and indexes
            of those points (M x K)
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


def ecef_coords(lats, lons, alts=np.array(0.0)):
    """
    Transforms the coordinates to Earth Centred Earth Fixed coordinates
    with WGS84 parameters. used in function _nearest_input_pts
    Args:
        lats(numpy.ndarray):
            latitude coordinates
        lons(numpy.ndarray):
            longitude coordinates
        alts(numpy.ndarray):
            altitudes coordinates
    Return:
        Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]:
            Transformed coordinates
    """
    rlats = np.deg2rad(lats)
    rlons = np.deg2rad(lons)
    clats = np.cos(rlats)
    clons = np.cos(rlons)
    slats = np.sin(rlats)
    slons = np.sin(rlons)
    n = WGS84_A / np.sqrt(1.0 - (WGS84_E * WGS84_E * slats * slats))
    x = (n + alts) * clats * clons
    y = (n + alts) * clats * slons
    z = (n * (1.0 - (WGS84_E * WGS84_E)) + alts) * slats
    return x, y, z


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
