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
"""Utilities for using neighbourhood processing."""

from typing import List, Optional, Union

import iris
import numpy as np
import numpy.ma as ma
from iris.cube import Cube

from improver import PostProcessingPlugin
from improver.nbhood.nbhood import NeighbourhoodProcessing
from improver.utilities.cube_checker import (
    check_cube_coordinates,
    find_dimension_coordinate_mismatch,
)
from improver.utilities.cube_manipulation import collapsed


class ApplyNeighbourhoodProcessingWithAMask(PostProcessingPlugin):

    r"""Class for applying neighbourhood processing when passing in a mask
    cube that is iterated over.

    Example:

        This plugin is designed to work with a set of masks which help you
        select points which are similar and only use these in your
        neighbourhood. The most obvious example of this is to divide the
        points in your cube into bands of similar orographic height.
        ::

            ..............................
            Band 2        ---
            ............./...\.../\.......
            Band 1      /     ---  \
            .........../............\.....
            Band 0    /              --
            ........--.................\..

        In this case the mask cube that comes in has a "topographic_zone"
        coordinate and each slice along this dimension has a 2D mask,
        masking out any points which are outside the topographic band that is
        described by the "topographic_zone" coordinate.

        The result from this plugin is a cube which has applied
        neighbourhooding to the plugin *n* times for the *n* bands in the mask
        cube. Each topography mask has been applied to the input cube in turn,
        resulting in a cube with a "topographic_zone" coordinate which is
        returned from this plugin.

        The re_mask option can be used to return the resulting cube with a
        "topographic_zone" coordinate, with each slice over the
        "topographic_zone" masked using the input mask cube.

        Otherwise if weights are provided then you can weight between
        adjacent bands when you collapse the new "topographic_zone" coordinate.
        This takes into account the result from the neighbourhood processing to
        adjust the weights for points in a "topographic_zone" that don't have
        a valid result.

        For example below we have two points A and B. Say point A was halfway
        between the midpoint and top of the lower band. We would want to
        generate a final result by weighting 0.75 times to neighbourhooded
        value from the bottom band and 0.25 times the neighbourhooded value in
        the upper band. For point B we would take equal weightings between the
        bands. There is a plugin to generate these weights:
        :class:`~improver.generate_ancillaries.\
        generate_topographic_zone_weights.GenerateTopographicZoneWeights`
        ::

                        A             B
                    ..........................
            band 2

                    ..................x.......
                        x
            band 1
                    ..........................

        We may need to adjust the weights if there is missing data in the
        adjacent band. If we look at the diagram with labelled bands, points
        that are near the top of band 2 could be weighted with band 3, except
        there are no nearby points in band 3. In this case the neighbourhood
        code puts NaNs in band 3 and we want to take 100% of band 2. This can
        be easily done by renormalization of the weights, which happens
        automatically within the numpy functions called within the
        iris collapse method.

        After collapsing the "topographic_zone" coordinate we end up with a
        cube with the same dimensions as the original cube, but the neighbourhood
        processing has been applied using masks so that only similar points
        are used in the neighbourhood.

        See also :class:`~improver.generate_ancillaries.generate_ancillary.\
        GenerateOrographyBandAncils`
        for a plugin for generating topographic band masks.

    """

    def __init__(
        self,
        coord_for_masking: str,
        radii: Union[float, List[float]],
        lead_times: Optional[List[float]] = None,
        collapse_weights: Optional[Cube] = None,
        weighted_mode: bool = True,
        sum_or_fraction: str = "fraction",
        re_mask: bool = False,
    ) -> None:
        """
        Initialise the class.

        Args:
            coord_for_masking:
                String matching the name of the coordinate that will be used
                for masking.
            radii:
                The radii in metres of the neighbourhood to apply.
                Rounded up to convert into integer number of grid
                points east and north, based on the characteristic spacing
                at the zero indices of the cube projection-x and y coords.
            collapse_weights:
                A cube from an ancillary file containing the weights for each
                point in the 'coord_for_masking' at each grid point. If given,
                the coord_for_masking coordinate will be collapsed using these
                weights resulting in an output cube with the same dimensions
                as the input cube. Otherwise the output cube will have the
                coord_for_masking from the supplied mask_cube as an additional
                dimension. The data in this cube may be an instance of
                numpy.ma.masked_array, for example if sea points have been
                set to np.nan and masked in order for them to be discounted
                when calculating the result. In this case the result returned
                from the process method of this plugin will have the same
                points masked.
            lead_times:
                List of lead times or forecast periods, at which the radii
                within 'radii' are defined. The lead times are expected
                in hours.
            weighted_mode:
                If True, use a circle for neighbourhood kernel with
                weighting decreasing with radius.
                If False, use a circle with constant weighting.
            sum_or_fraction:
                Identifier for whether sum or fraction should be returned from
                neighbourhooding. The sum represents the sum of the
                neighbourhood.
                The fraction represents the sum of the neighbourhood divided by
                the neighbourhood area. "fraction" is the default.
                Valid options are "sum" or "fraction".
            re_mask:
                If re_mask is True, the original un-neighbourhood processed
                mask is applied to mask out the neighbourhood processed cube.
                If re_mask is False, the original un-neighbourhood processed
                mask is not applied. Therefore, the neighbourhood processing
                may result in values being present in areas that were
                originally masked.
                This should be set to False if using collapse_weights.
        """
        self.coord_for_masking = coord_for_masking
        self.neighbourhood_method = "square"
        self.radii = radii
        self.lead_times = lead_times
        self.collapse_weights = collapse_weights
        self.weighted_mode = weighted_mode
        self.sum_or_fraction = sum_or_fraction
        self.re_mask = re_mask
        # Check that if collapse_weights are provided then re_mask is set to False
        if self.collapse_weights is not None and re_mask is True:
            message = "re_mask should be set to False when using collapse_weights"
            raise ValueError(message)

    def collapse_mask_coord(self, cube: Cube) -> Cube:
        """
        Collapse the chosen coordinate with the available weights. The result
        of the neighbourhood processing is taken into account to renormalize
        any weights corresponding to a NaN in the result from neighbourhooding.
        In this case the weights are re-normalized so that we do not lose
        probability.

        Args:
            cube:
                Cube containing the array to which the square neighbourhood
                with a mask has been applied.
                Dimensions self.coord_for_masking, y and x.

        Returns:
            Cube containing the weighted mean from neighbourhood after
            collapsing the chosen coordinate.
        """
        # Mask out any NaNs in the neighbourhood data so that Iris ignores
        # them when calculating the weighted mean.
        cube.data = ma.masked_invalid(cube.data, copy=False)
        # Collapse the coord_for_masking. Renormalization of the weights happen
        # within the underlying call to a numpy function within the Iris method.
        result = collapsed(
            cube,
            self.coord_for_masking,
            iris.analysis.MEAN,
            weights=self.collapse_weights.data,
        )
        # Set masked invalid data points back to np.nans
        if np.ma.is_masked(result.data):
            result.data.data[result.data.mask] = np.nan
        # Remove references to self.coord_masked in the result cube.
        result.remove_coord(self.coord_for_masking)
        return result

    def process(self, cube: Cube, mask_cube: Cube) -> Cube:
        """
        Apply neighbourhood processing with a mask to the input cube,
        collapsing the coord_for_masking if collapse_weights have been provided.

        Args:
            cube:
                Cube containing the array to which the square neighbourhood
                will be applied.
            mask_cube:
                Cube containing the array to be used as a mask. The data in
                this array is not an instance of numpy.ma.MaskedArray. Any sea
                points that should be ignored are set to zeros in every layer
                of the mask_cube.

        Returns:
            Cube containing the smoothed field after the square
            neighbourhood method has been applied when applying masking
            for each point along the coord_for_masking coordinate.
            The resulting cube is concatenated so that the dimension
            coordinates match the input cube.
        """
        plugin = NeighbourhoodProcessing(
            self.neighbourhood_method,
            self.radii,
            lead_times=self.lead_times,
            weighted_mode=self.weighted_mode,
            sum_or_fraction=self.sum_or_fraction,
            re_mask=self.re_mask,
        )
        yname = cube.coord(axis="y").name()
        xname = cube.coord(axis="x").name()
        result_slices = iris.cube.CubeList([])
        # Take 2D slices of the input cube for memory issues.
        prev_x_y_slice = None
        for x_y_slice in cube.slices([yname, xname]):
            if prev_x_y_slice is not None and np.array_equal(
                prev_x_y_slice.data, x_y_slice.data
            ):
                # Use same result as last time!
                prev_result = result_slices[-1].copy()
                for coord in x_y_slice.coords(dim_coords=False):
                    prev_result.coord(coord).points = coord.points.copy()
                result_slices.append(prev_result)
                continue
            prev_x_y_slice = x_y_slice

            cube_slices = iris.cube.CubeList([])
            # Apply each mask in in mask_cube to the 2D input slice.
            for mask_slice in mask_cube.slices_over(self.coord_for_masking):
                output_cube = plugin(x_y_slice, mask_cube=mask_slice)
                coord_object = mask_slice.coord(self.coord_for_masking).copy()
                output_cube.add_aux_coord(coord_object)
                output_cube = iris.util.new_axis(output_cube, self.coord_for_masking)
                cube_slices.append(output_cube)
            concatenated_cube = cube_slices.concatenate_cube()
            if self.collapse_weights is not None:
                concatenated_cube = self.collapse_mask_coord(concatenated_cube)
            result_slices.append(concatenated_cube)
        result = result_slices.merge_cube()
        # Promote any single value dimension coordinates if they were
        # dimension on the input cube.
        exception_coordinates = find_dimension_coordinate_mismatch(
            cube, result, two_way_mismatch=False
        )
        result = check_cube_coordinates(
            cube, result, exception_coordinates=exception_coordinates
        )
        return result
