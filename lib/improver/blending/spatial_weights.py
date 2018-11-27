# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown Copyright 2017-2018 Met Office.
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
"""Module to adjust weights spatially based on missing data in input cubes."""

import iris
import numpy as np

from scipy.ndimage.morphology import distance_transform_edt

from improver.utilities.rescale import rescale
from improver.utilities.cube_checker import check_cube_coordinates


class SpatialWeightsForMissingData(object):
    """
    Plugin for adjusting weights spatially based on missing data in the input
    cube that will be collapsed using these weights.

    The plugin does the following steps:

        1. Creates an initial spatial weights basked on the mask in the
           input cube giving zero weight to where there is masked data and
           a weight of 1 where there is valid data.
        2. Make these weights fuzzy by smoothing the boundary between where
           there is valid data and no valid data. This keeps areas of zero
           weight, but reduces nearby grid points with a weight of 1 depending
           on how far they are from a grid point with a weight of zero. The
           range of the effect is controlled by the supplied fuzzy_length
        3. Multiplies the fuzzy spatial weights by the one_dimensional weights
           supplied to the plugin.
        4. Normalises the weights along the coordinatethat will be collapsed
           when blending is carried out using these weights.
    """

    def __init__(self, fuzzy_length=10):
        """
        Initialise class.
        Keyword Args:
            fuzzy_length (int or float):
                The length in terms of number of grid squares, over which the
                weights from the input data mask are smoothed.
        """
        self.fuzzy_length = fuzzy_length

    def __repr__(self):
        """Represent the configured plugin instance as a string."""
        result = ('<SpatialWeightsForMissingData: fuzzy_length: {}>'.format(
                      self.fuzzy_length))
        return result

    @staticmethod
    def create_initial_weights_from_mask(cube):
        """
        Generate a cube with weights generated from the mask of the input cube.
        Where the data is masked we set the weight to zero, otherwise the
        weight is one.

        Args:
            cube (iris.cube.Cube):
                A cube containing the data we want to collapse by doing a
                weighted blend along a given cordinate.
        Return:
            masked_weights_cube (iris.cube.Cube):
                A cube containing an initial set of weights based on the mask
                on the input cube.
        Rasies:
            ValueError : If the input cube does not have a mask.
        """
        if np.ma.is_masked(cube.data):
            weights_data = np.where(cube.data.mask, 0, 1).astype(np.float32)
        else:
            message = ("Input cube to SpatialWeightsForMissingData "
                       "must be masked")
            raise ValueError(message)
        masked_weights_cube = cube.copy(data=weights_data)
        masked_weights_cube.rename("weights")
        return masked_weights_cube

    def create_fuzzy_spatial_weights(self, masked_weights_cube):
        """
        Create fuzzy weights around points in the masked_weights_cube with
        zero weight.

        Args:
            masked_weights_cube (iris.cube.Cube):
                A cube containing an initial set of weights based on the mask
                on the input cube.
        Returns:
            result (iris.cube.Cube):
                A cube contiaing the fuzzy weights calculated based on the
                masked_weights_cube. The dimension order may have changed from
                the input cube as it has been sliced over x and y coordinates.
        """
        result = iris.cube.CubeList()
        x_coord = masked_weights_cube.coord(axis='x').name()
        y_coord = masked_weights_cube.coord(axis='y').name()
        # The distance_transform_edt works on N-D cubes, so we want to make
        # sure we only apply it to x-y slices.
        for weights in masked_weights_cube.slices([x_coord, y_coord]):
            if np.all(weights.data == 1.0):
                # distance_transform_edt doesn't produce what we want if there
                # are no zeros present.
                result.append(weights.copy())
            else:
                fuzzy_data = distance_transform_edt(weights.data == 1., 1)
                fuzzy_data = fuzzy_data.astype(np.float32)
                rescaled_fuzzy_data = rescale(
                    fuzzy_data, data_range=[0., self.fuzzy_length],
                    clip=True)
                result.append(weights.copy(data=rescaled_fuzzy_data))
        result = result.merge_cube()
        return result

    @staticmethod
    def multiply_weights(masked_weights_cube, one_dimensional_weights_cube,
                         collapsing_coord):
        """
        Multiply two cubes together by taking slices along the coordinate
        matching the collapsing_coord string.

        Args:
            masked_weights_cube (iris.cube.Cube):
                A cube with spatial weights and any other leading dimensions.
                This cube must have a coordinate matching the name given by
                collapsing_coord and this coordinate must be the same length
                as the corresponding coordinate in the
                one_dimensional_weights_cube.
            one_dimensional_weights_cube (iris.cube.Cube):
                A cube with one_dimensional weights. The only dimension
                coordinate in this cube matches the string given by
                collapsing_coord and the length of this coord must match the
                length of the same coordinate in masked_weights_cube.
            collapsing_coord (string):
                The string that will match to a coordinate in both input cube.
                This is the coordinate that the input cubes will be sliced
                along and then multipled. The corresponds to the coordinate
                used to collapse a cube using the weights generated by this
                plugin.

        Returns:
            result (iris.cube.Cube):
                A cube with the same dimensions as the input cube, but with
                the weights multiplied by the weights from the
                one_dimensional_weights_cube. The collapsing_coord will be the
                leading dimension on the output cube.
        """
        result = iris.cube.CubeList()
        if (masked_weights_cube.coord(collapsing_coord) !=
                one_dimensional_weights_cube.coord(collapsing_coord)):
            message = ("The collapsing_coord {} does not match on "
                       "masked_weights_cube and "
                       "one_dimensional_weights_cube".format(collapsing_coord))
            raise ValueError(message)
        for masked_weight_slice, one_dimensional_weight in zip(
                masked_weights_cube.slices_over(collapsing_coord),
                one_dimensional_weights_cube.slices_over(collapsing_coord)):
            masked_weight_slice.data = (
                masked_weight_slice.data * one_dimensional_weight.data)
            result.append(masked_weight_slice)
        result = result.merge_cube()
        return result

    @staticmethod
    def normalised_masked_weights(weights_cube, collapsing_coord):
        """
        Normalise spatial weights along dimension associated with the
        collapsing_coord. If for a given point the sum of the weights along
        the collapsing_coord is zero then the returned normalised weight for
        that point will also be zero. This correspsonds to the case where
        there is missing data for that point for all slices along the
        collapsing_coord.

        Args:
            weights_cube (iris.cube.Cube):
                A cube with spatial weights and any other leading dimension.
                This cube must have a coordinate matching the name given by
                collapsing_coord which corresponds to the dimension along
                which the normalisation is needed.
            collapsing_coord (string):
                The string that will match to a coordinate in both input cube.
                This coordinate corresponds to the dimension along which the
                normalisation is needed.

        Returns:
            result (iris.cube.Cube):
                A cube with the same dimensions as the input cube, but with
                the weights normalised along the collapsing_coord dimension.
                The collapsing_coord will be the leading dimension on the
                output cube.
        """
        normalisation_axis = weights_cube.coord_dims(collapsing_coord)
        summed_weights = weights_cube.collapsed(
            collapsing_coord, iris.analysis.SUM)
        result = iris.cube.CubeList()
        # Slice over collapsing_coord so the dimensions match.
        for weight_slice in (
                weights_cube.slices_over(collapsing_coord)):
            # Only divide where the sum of weights are positive. Setting
            # the out keyword args sets the default value for where
            # the sum of the weights are less than zero.
            normalised_data = np.divide(
                weight_slice.data, summed_weights.data,
                out=np.zeros_like(weight_slice.data),
                where=(summed_weights.data > 0))
            result.append(weight_slice.copy(data=normalised_data))
        return result.merge_cube()

    def process(self, cube_to_collapse, one_dimensional_weights_cube,
                collapsing_coord):
        """
        Create fuzzy spatial weights based on missing data in the cube we
        are going to collapse and combine these with 1D weights along the
        collapsing_coord.

        Args:
            cube_to_collapse (iris.cube.Cube):
                The cube that will be collapsed along the collapsing_coord
                using the spatial weights generated using this plugin. Must
                be masked where there is invalid data.
            one_dimensional_weights_cube (iris.cube.Cube):
                A cube containing a single dimension coordinate with the same
                name given collapsing_coord. This cube contains 1D weights
                that will be applied along the collapsing_coord but need
                adjusting spatially based on missing data.
            collapsing_coord (string):
                A string containing the name of the coordinate that the
                cube_to_collapse will be collapsed along. Also matches the
                coordinate in one_dimensional_weights_cube.

        Returns:
            result (iris.cube.Cube):
                A cube containing normalised spatial weights based on the
                cube_to_collapsemask and the one_dimensional weights supplied.
                Contains the same dimensions in the same order as
                cube_to_collapse.
        """
        masked_weights_cube = self.create_initial_weights_from_mask(
            cube_to_collapse)
        masked_weights_cube = self.create_fuzzy_spatial_weights(
            masked_weights_cube)
        final_weights = self.multiply_weights(
            masked_weights_cube, one_dimensional_weights_cube,
            collapsing_coord)
        final_weights = self.normalised_masked_weights(
            final_weights, collapsing_coord)
        # Check dimensions
        final_weights = check_cube_coordinates(cube_to_collapse, final_weights)
        return final_weights
