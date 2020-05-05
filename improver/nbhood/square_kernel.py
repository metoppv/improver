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
"""This module contains methods for square neighbourhood processing."""

import iris
import numpy as np

from improver.nbhood.circular_kernel import check_radius_against_distance
from improver.utilities.cube_checker import (
    check_cube_coordinates,
    check_for_x_and_y_axes,
)
from improver.utilities.cube_manipulation import clip_cube_data
from improver.utilities.mathematical_operations import fast_cumsum_2d
from improver.utilities.neighbourhood_tools import boxsum
from improver.utilities.pad_spatial import pad_cube_with_halo, remove_halo_from_cube
from improver.utilities.spatial import distance_to_number_of_grid_cells


class SquareNeighbourhood:

    """
    Methods for use in application of a square neighbourhood.
    """

    def __init__(self, weighted_mode=True, sum_or_fraction="fraction", re_mask=True):
        """
        Initialise class.

        Args:
            weighted_mode (bool):
                This is included to allow a standard interface for both the
                square and circular neighbourhood plugins.
            sum_or_fraction (str):
                Identifier for whether sum or fraction should be returned from
                neighbourhooding. The sum represents the sum of the
                neighbourhood. The fraction represents the sum of the
                neighbourhood divided by the neighbourhood area.
                Valid options are "sum" or "fraction".
            re_mask (bool):
                If re_mask is True, the original un-neighbourhood processed
                mask is applied to mask out the neighbourhood processed cube.
                If re_mask is False, the original un-neighbourhood processed
                mask is not applied. Therefore, the neighbourhood processing
                may result in values being present in areas that were
                originally masked.
        """
        self.weighted_mode = weighted_mode
        if sum_or_fraction not in ["sum", "fraction"]:
            msg = (
                "The neighbourhood output can either be in the form of a "
                "sum of all the points in the neighbourhood or a fraction "
                "of the sum of the neighbourhood divided by the "
                "neighbourhood area. The {} option is invalid. "
                "Valid options are 'sum' or 'fraction'."
            )
            raise ValueError(msg)
        self.sum_or_fraction = sum_or_fraction
        self.re_mask = re_mask

    def __repr__(self):
        """Represent the configured plugin instance as a string."""
        result = (
            "<SquareNeighbourhood: weighted_mode: {}, "
            "sum_or_fraction: {}, re_mask: {}>"
        )
        return result.format(self.weighted_mode, self.sum_or_fraction, self.re_mask)

    @staticmethod
    def _calculate_neighbourhood(data, mask, nb_size, sum_only, re_mask, name):
        if not sum_only:
            min_val = np.nanmin(data)
            max_val = np.nanmax(data)

        cumsum = fast_cumsum_2d if data.ndim == 2 else True

        # Use 64-bit types for enough precision in accumulations.
        area_mask_dtype = np.int64
        if mask is None:
            area_mask = np.ones(data.shape, dtype=area_mask_dtype)
        else:
            area_mask = np.array(mask, dtype=area_mask_dtype, copy=False)
        # Data mask to be eventually used for re-masking.
        # (This is OK even if mask is None, it gives a scalar False mask then.)
        data_mask = mask == 0
        if isinstance(data, np.ma.MaskedArray):
            # Include data mask if masked array
            data_mask = data_mask | data.mask
            data = data.data
        if issubclass(data.dtype.type, np.complexfloating):
            data_dtype = np.complex128
        elif name.startswith("probability_of"):
            data_dtype = np.float32
            cumsum = True
        else:
            data_dtype = np.float64
        data = np.array(data, dtype=data_dtype)
        nan_mask = np.isnan(data)
        zero_mask = nan_mask | data_mask
        np.copyto(area_mask, 0, where=zero_mask)
        np.copyto(data, 0, where=zero_mask)

        data = boxsum(data, nb_size, cumsum=cumsum, mode="constant")
        if not sum_only:
            area_sum = boxsum(area_mask, nb_size, cumsum=cumsum, mode="constant")
            with np.errstate(divide="ignore", invalid="ignore"):
                data = data / area_sum
            mask_invalid = (area_sum == 0) | nan_mask
            np.copyto(data, np.nan, where=mask_invalid)
            data = data.clip(min_val, max_val)

        if issubclass(data.dtype.type, np.complexfloating):
            data_dtype = np.complex64
        else:
            data_dtype = np.float32
        data = data.astype(data_dtype)

        if re_mask:
            data = np.ma.masked_array(data, data_mask, copy=False)

        return data

    def run(self, cube, radius, mask_cube=None):
        """
        Call the methods required to apply a square neighbourhood
        method to a cube.

        The steps undertaken are:

        1. Set up cubes by determining, if the arrays are masked.
        2. Pad the input array with a halo and then calculate the neighbourhood
           of the haloed array.
        3. Remove the halo from the neighbourhooded array and deal with a mask,
           if required.

        Args:
            cube (iris.cube.Cube):
                Cube containing the array to which the square neighbourhood
                will be applied.
            radius (float):
                Radius in metres for use in specifying the number of
                grid cells used to create a square neighbourhood.
            mask_cube (iris.cube.Cube):
                Cube containing the array to be used as a mask.

        Returns:
            iris.cube.Cube:
                Cube containing the smoothed field after the square
                neighbourhood method has been applied.
        """
        # If the data is masked, the mask will be processed as well as the
        # original_data * mask array.
        check_radius_against_distance(cube, radius)
        original_attributes = cube.attributes
        original_methods = cube.cell_methods
        grid_cells = distance_to_number_of_grid_cells(cube, radius)
        nb_size = 2 * grid_cells + 1
        try:
            mask_cube_data = mask_cube.data
        except AttributeError:
            mask_cube_data = None

        result_slices = iris.cube.CubeList()
        for cube_slice in cube.slices([cube.coord(axis="y"), cube.coord(axis="x")]):
            cube_slice.data = self._calculate_neighbourhood(
                cube_slice.data,
                mask_cube_data,
                nb_size,
                self.sum_or_fraction == "sum",
                self.re_mask,
                cube.name(),
            )
            result_slices.append(cube_slice)

        neighbourhood_averaged_cube = result_slices.merge_cube()

        neighbourhood_averaged_cube.cell_methods = original_methods
        neighbourhood_averaged_cube.attributes = original_attributes

        neighbourhood_averaged_cube = check_cube_coordinates(
            cube, neighbourhood_averaged_cube
        )
        return neighbourhood_averaged_cube
