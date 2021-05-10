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
Plugin to regrid using custom nearest and bilinear methods, both with
land-sea awareness
"""

import numpy as np

from improver import BasePlugin
from improver.regrid.bilinear import (
    adjust_for_surface_mismatch,
    apply_weights,
    basic_indexes,
    basic_weights,
)
from improver.regrid.grid import (
    calculate_input_grid_spacing,
    classify_input_surface_type,
    classify_output_surface_type,
    create_regrid_cube,
    flatten_spatial_dimensions,
    latlon_from_cube,
    similar_surface_classify,
    slice_cube_by_domain,
    slice_mask_cube_by_domain,
    unflatten_spatial_dimensions,
)
from improver.regrid.nearest import nearest_regrid, nearest_with_mask_regrid
from improver.utilities.spatial import transform_grid_to_lat_lon

NEAREST = "nearest"
BILINEAR = "bilinear"
WITH_MASK = "-with-mask"
BILINEAR2 = f"{BILINEAR}-2"
NEAREST2 = f"{NEAREST}-2"
NEAREST_MASK2 = f"{NEAREST}{WITH_MASK}-2"
BILINEAR_MASK2 = f"{BILINEAR}{WITH_MASK}-2"
NUM_NEIGHBOURS = 4


class RegridWithLandSeaMask(BasePlugin):
    """
    Nearest-neighbour and bilinear regridding with or without land-sea mask
    awareness. When land-sea mask considered, surface-type-mismatched source
    points are excluded from field regridding calculation for target points.
    Note: regrid_mode options are "nearest-2", "nearest-with-mask-2","bilinear-2",
    and "bilinear-with-mask-2" in this class.
    """

    def __init__(self, regrid_mode="bilinear-2", vicinity_radius=25000.0):
        """
        Initialise class

        Args:
            regrid_mode (str):
                Mode of interpolation in regridding.  Valid options are "bilinear-2",
                "nearest-2","nearest-with-mask-2" and "bilinear-with-mask-2". 
                The last two options trigger adjustment of regridded points to match
                source points in terms of land / sea type.
            vicinity_radius (float):
                Radius of vicinity to search for a coastline, in metres
        """
        self.regrid_mode = regrid_mode
        self.vicinity = vicinity_radius

    def process(self, cube_in, cube_in_mask, cube_out_mask):
        """
        Regridding considering land_sea mask. please note cube_in must use
        lats/lons rectlinear system(GeogCS). cube_in_mask and cube_in could be
        different  resolution. cube_our could be either in lats/lons rectlinear
        system or LambertAzimuthalEqualArea system.

        Args:
            cube_in (iris.cube.Cube):
                Cube of data to be regridded
            cube_in_mask (iris.cube.Cube):
                Cube of land_binary_mask data ((land:1, sea:0). used to determine
                where the input model data is representing land and sea points.
            cube_out_mask (iris.cube.Cube):
                Cube of land_binary_mask data on target grid (land:1, sea:0).

        Returns:
            iris.cube.Cube:
                Regridded result cube
        """
        # check if input source grid is on even-spacing, ascending lat/lon system
        # return grid spacing for latitude and logitude
        lat_spacing, lon_spacing = calculate_input_grid_spacing(cube_in)

        # Gather output latitude/longitudes from output template cube
        if (
            cube_out_mask.coord(axis="x").standard_name == "projection_x_coordinate"
            and cube_out_mask.coord(axis="y").standard_name == "projection_y_coordinate"
        ):
            out_latlons = np.dstack(transform_grid_to_lat_lon(cube_out_mask)).reshape(
                (-1, 2)
            )
        else:
            out_latlons = latlon_from_cube(cube_out_mask)

        # Subset the input cube so that extra spatial area beyond the output is removed
        # This is a performance optimisation to reduce the size of the dataset being processed
        lat_max, lon_max = out_latlons.max(axis=0)
        lat_min, lon_min = out_latlons.min(axis=0)
        if WITH_MASK in self.regrid_mode:
            cube_in, cube_in_mask = slice_mask_cube_by_domain(
                cube_in, cube_in_mask, (lat_max, lon_max, lat_min, lon_min)
            )
        else:  # not WITH_MASK
            cube_in = slice_cube_by_domain(
                cube_in, (lat_max, lon_max, lat_min, lon_min)
            )

        # Gather input latitude/longitudes from input cube
        in_latlons = latlon_from_cube(cube_in)
        # Number of grid points in X dimension is used to work out length of flattened array
        # stripes for finding surrounding points for bilinear interpolation
        in_lons_size = cube_in.coord(axis="x").shape[0]  # longitude

        # Reshape input data so that spatial dimensions can be handled as one
        in_values, lats_index, lons_index = flatten_spatial_dimensions(cube_in)

        # Locate nearby input points for output points
        indexes = basic_indexes(
            out_latlons, in_latlons, in_lons_size, lat_spacing, lon_spacing
        )

        if WITH_MASK in self.regrid_mode:
            in_classified = classify_input_surface_type(cube_in_mask, in_latlons)
            out_classified = classify_output_surface_type(cube_out_mask)
            # Identify mismatched surface types from input and output classifications
            surface_type_mask = similar_surface_classify(
                in_classified, out_classified, indexes
            )

        # Initialise distances and weights to zero. Weights are only used for the bilinear case
        distances = np.zeros((out_latlons.shape[0], NUM_NEIGHBOURS), dtype=np.float32)
        weights = np.zeros((out_latlons.shape[0], NUM_NEIGHBOURS), dtype=np.float32)

        # handle nearest option
        if NEAREST in self.regrid_mode:
            for i in range(NUM_NEIGHBOURS):
                distances[:, i] = np.square(
                    in_latlons[indexes[:, i], 0] - out_latlons[:, 0]
                ) + np.square(in_latlons[indexes[:, i], 1] - out_latlons[:, 1])

            # for nearest-with-mask-2,adjust indexes and distance for mismatched
            # surface type location
            if WITH_MASK in self.regrid_mode:
                distances, indexes = nearest_with_mask_regrid(
                    distances,
                    indexes,
                    surface_type_mask,
                    in_latlons,
                    out_latlons,
                    in_classified,
                    out_classified,
                    self.vicinity,
                )

            # apply nearest distance rule
            output_flat = nearest_regrid(distances, indexes, in_values)

        elif BILINEAR in self.regrid_mode:
            # Assume all four nearby points are same surface type and calculate default weights
            # These will be updated for mask/mismatched surface type further below
            # pylint: disable=unsubscriptable-object
            index_range = np.arange(weights.shape[0])
            weights[index_range] = basic_weights(
                index_range,
                indexes,
                out_latlons,
                in_latlons,
                in_lons_size,
                lat_spacing,
                lon_spacing,
            )

            if WITH_MASK in self.regrid_mode:
                # For bilinear-with-mask-2, adjust weights and indexes for mismatched
                # surface type locations
                weights, indexes = adjust_for_surface_mismatch(
                    in_latlons,
                    out_latlons,
                    in_classified,
                    out_classified,
                    weights,
                    indexes,
                    surface_type_mask,
                    in_lons_size,
                    self.vicinity,
                    lat_spacing,
                    lon_spacing,
                )

            # apply bilinear rule
            output_flat = apply_weights(indexes, in_values, weights)

        # Un-flatten spatial dimensions and put into output cube
        output_array = unflatten_spatial_dimensions(
            output_flat, cube_out_mask, in_values, lats_index, lons_index
        )
        output_cube = create_regrid_cube(output_array, cube_in, cube_out_mask)

        return output_cube
