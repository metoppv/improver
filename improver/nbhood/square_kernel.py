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
"""This module contains methods for neighbourhood processing."""

from typing import List, Optional, Union

import iris
import numpy as np
from iris.cube import Cube
from numpy import ndarray
from scipy.ndimage.filters import correlate

from improver import PostProcessingPlugin
from improver.nbhood.circular_kernel import (
    check_radius_against_distance,
    circular_kernel,
)
from improver.nbhood.nbhood import BaseNeighbourhoodProcessing
from improver.utilities.cube_checker import check_cube_coordinates
from improver.utilities.neighbourhood_tools import boxsum
from improver.utilities.spatial import (
    check_if_grid_is_equal_area,
    distance_to_number_of_grid_cells,
)


class NeighbourhoodProcessing(PostProcessingPlugin, BaseNeighbourhoodProcessing):
    """Class for applying neighbourhood processing to produce a smoothed field
    within the chosen neighbourhood."""

    def __init__(
        self,
        neighbourhood_method: str,
        radii: Union[float, List[float]],
        lead_times: Optional[List] = None,
        weighted_mode: bool = True,
        sum_only: bool = False,
        re_mask: bool = True,
    ) -> None:
        """
        Initialise class.

        Args:
            neighbourhood_method:
                Name of the neighbourhood method to use. Options: 'circular',
                'square'.
            radii:
                The radii in metres of the neighbourhood to apply.
                Rounded up to convert into integer number of grid
                points east and north, based on the characteristic spacing
                at the zero indices of the cube projection-x and y coords.
            lead_times:
                List of lead times or forecast periods, at which the radii
                within 'radii' are defined. The lead times are expected
                in hours.
            weighted_mode:
                If True, use a circle for neighbourhood kernel with
                weighting decreasing with radius.
                If False, use a circle with constant weighting.
            sum_only:
                If true, return neighbourhood sum instead of mean.
            re_mask:
                If re_mask is True, the original un-neighbourhood processed
                mask is applied to mask out the neighbourhood processed cube.
                If re_mask is False, the original un-neighbourhood processed
                mask is not applied. Therefore, the neighbourhood processing
                may result in values being present in areas that were
                originally masked.
        """
        super(NeighbourhoodProcessing, self).__init__(radii, lead_times=lead_times)
        if neighbourhood_method in ["square", "circular"]:
            self.neighbourhood_method = neighbourhood_method
        else:
            msg = "{} is not a valid neighbourhood_method.".format(neighbourhood_method)
            raise ValueError(msg)
        # if weighted_mode and neighbourhood_method != "circular":
        #     msg = "weighted_mode can only be used if neighbourhood_method is circular"
        #     raise ValueError(msg)
        self.weighted_mode = weighted_mode
        self.sum_only = sum_only
        self.re_mask = re_mask

    def _calculate_neighbourhood(self, data: ndarray, mask: ndarray) -> ndarray:
        """
        Apply neighbourhood processing.

        Args:
            data:
                Input data array.
            mask:
                Mask of valid input data elements.

        Returns:
            Array containing the smoothed field after the square
            neighbourhood method has been applied.
        """
        if not self.sum_only:
            min_val = np.nanmin(data)
            max_val = np.nanmax(data)

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
            # Include data mask if masked array.
            data_mask = data_mask | data.mask
            data = data.data

        # Working type.
        if issubclass(data.dtype.type, np.complexfloating):
            data_dtype = np.complex128
        else:
            data_dtype = np.float64
        data = np.array(data, dtype=data_dtype)

        # Replace invalid elements with zeros.
        nan_mask = np.isnan(data)
        zero_mask = nan_mask | data_mask
        np.copyto(area_mask, 0, where=zero_mask)
        np.copyto(data, 0, where=zero_mask)
        # Calculate neighbourhood totals for input data.
        if self.neighbourhood_method == "square":
            data = boxsum(data, self.nb_size, mode="constant")
        elif self.neighbourhood_method == "circular":
            data = correlate(data, self.kernel, mode="nearest")
        if not self.sum_only:
            # Calculate neighbourhood totals for mask.
            if self.neighbourhood_method == "square":
                area_sum = boxsum(area_mask, self.nb_size, mode="constant")
            elif self.neighbourhood_method == "circular":
                area_sum = correlate(
                    area_mask.astype(np.float32), self.kernel, mode="nearest"
                )

            with np.errstate(divide="ignore", invalid="ignore"):
                # Calculate neighbourhood mean.
                data = data / area_sum
            mask_invalid = (area_sum == 0) | nan_mask
            np.copyto(data, np.nan, where=mask_invalid)
            data = data.clip(min_val, max_val)

        # Output type.
        if issubclass(data.dtype.type, np.complexfloating):
            data_dtype = np.complex64
        else:
            data_dtype = np.float32
        data = data.astype(data_dtype)

        if self.re_mask:
            data = np.ma.masked_array(data, data_mask, copy=False)

        return data

    def process(self, cube: Cube, mask_cube: Optional[Cube] = None) -> Cube:
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
            cube:
                Cube containing the array to which the square neighbourhood
                will be applied.
            mask_cube:
                Cube containing the array to be used as a mask.

        Returns:
            Cube containing the smoothed field after the square
            neighbourhood method has been applied.
        """
        super(NeighbourhoodProcessing, self).process(cube)
        check_if_grid_is_equal_area(cube)

        # If the data is masked, the mask will be processed as well as the
        # original_data * mask array.
        check_radius_against_distance(cube, self.radius)
        original_attributes = cube.attributes
        original_methods = cube.cell_methods
        grid_cells = distance_to_number_of_grid_cells(cube, self.radius)
        if self.neighbourhood_method == "circular":
            self.kernel = circular_kernel(grid_cells, self.weighted_mode)
        elif self.neighbourhood_method == "square":
            self.nb_size = 2 * grid_cells + 1

        try:
            mask_cube_data = mask_cube.data
        except AttributeError:
            mask_cube_data = None

        result_slices = iris.cube.CubeList()
        for cube_slice in cube.slices([cube.coord(axis="y"), cube.coord(axis="x")]):
            cube_slice.data = self._calculate_neighbourhood(
                cube_slice.data, mask_cube_data
            )
            result_slices.append(cube_slice)
        neighbourhood_averaged_cube = result_slices.merge_cube()

        neighbourhood_averaged_cube.cell_methods = original_methods
        neighbourhood_averaged_cube.attributes = original_attributes

        neighbourhood_averaged_cube = check_cube_coordinates(
            cube, neighbourhood_averaged_cube
        )
        return neighbourhood_averaged_cube
