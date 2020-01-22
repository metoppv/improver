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
"""Module to adjust weights spatially based on missing data in input cubes."""

import warnings

import iris
import numpy as np
from scipy.ndimage.morphology import distance_transform_edt

from improver import BasePlugin
from improver.utilities.cube_manipulation import collapsed
from improver.utilities.rescale import rescale


class SpatiallyVaryingWeightsFromMask(BasePlugin):
    """
    Plugin for adjusting weights spatially based on missing data in the input
    cube. It takes in an initial one dimensional cube of weights which would
    be used to collapse a dimension on the input cube and outputs weights
    which have been adjusted based on the presence of missing data in x-y
    slices of input data. The resulting weights cube has a x and y dimensions
    in addition to the one dimension in the initial cube of weights.

    The plugin does the following steps:

        1. Creates an initial spatial weights based on the mask in the
           input cube giving zero weight to where there is masked data and
           a weight of 1 where there is valid data.
        2. Make these weights fuzzy by smoothing the boundary between where
           there is valid data and no valid data. This keeps areas of zero
           weight, but reduces nearby grid points with a weight of 1 depending
           on how far they are from a grid point with a weight of zero. The
           range of the effect is controlled by the supplied fuzzy_length
        3. Multiplies the fuzzy spatial weights by the one dimensional weights
           supplied to the plugin.
        4. Normalises the weights along the coordinate that will be collapsed
           when blending is carried out using these weights.
    """

    def __init__(self, fuzzy_length=10):
        """
        Initialise class.

        Args:
            fuzzy_length (int or float):
                The length in terms of number of grid squares, over which the
                weights from the input data mask are smoothed. This is used
                when calculating a fuzzy weighting based on how far away each
                grid point is from a masked point. The distance is taken as
                the euclidean distance from the masked point, so the fuzzy
                length can be a non-integer value. Any points that are at least
                this distance away from a masked point keep a weight of one,
                and any points closer than this distance to a masked point have
                a weight of less than one based on how close to the masked
                point they are.
        """
        self.fuzzy_length = fuzzy_length

    def __repr__(self):
        """Represent the configured plugin instance as a string."""
        result = ('<SpatiallyVaryingWeightsFromMask: fuzzy_length: {}>'.format(
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
                weighted blend along a given coordinate.
        Returns:
            iris.cube.Cube:
                A cube containing an initial set of weights based on the mask
                on the input cube.
        Rasies:
            ValueError : If the input cube does not have a mask.
        """
        if np.ma.is_masked(cube.data):
            weights_data = np.where(cube.data.mask, 0, 1).astype(np.float32)
        else:
            weights_data = np.ones(cube.data.shape, dtype=np.float32)
            message = ("Input cube to SpatiallyVaryingWeightsFromMask "
                       "must be masked")
            warnings.warn(message)
        weights_from_mask = cube.copy(data=weights_data)
        weights_from_mask.rename("weights")
        return weights_from_mask

    def smooth_initial_weights(self, weights_from_mask):
        """
        Create fuzzy weights around points in the weights_from_mask with
        zero weight.

        This works by doing an euclidean distance transform based on how far
        each grid point is from a masked point. This returns an array
        containing the distance each point is from the nearest masked point.
        The result is then rescaled so that any point that are at least as far
        as the fuzzy_length away from a masked point are set
        back to a weight of one and any points that are closer than the
        fuzzy_length to a masked point are scaled to be between 0 and 1.

        Args:
            weights_from_mask (iris.cube.Cube):
                A cube containing an initial set of weights based on the mask
                on the input cube.
        Returns:
            iris.cube.Cube:
                A cube containing the fuzzy weights calculated based on the
                weights_from_mask. The dimension order may have changed from
                the input cube as it has been sliced over x and y coordinates.
        """
        result = iris.cube.CubeList()
        x_coord = weights_from_mask.coord(axis='x').name()
        y_coord = weights_from_mask.coord(axis='y').name()
        # The distance_transform_edt works on N-D cubes, so we want to make
        # sure we only apply it to x-y slices.
        for weights in weights_from_mask.slices([y_coord, x_coord]):
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
    def multiply_weights(weights_from_mask, one_dimensional_weights_cube,
                         blend_coord):
        """
        Multiply two cubes together by taking slices along the coordinate
        matching the blend_coord string.

        Args:
            weights_from_mask (iris.cube.Cube):
                A cube with spatial weights and any other leading dimensions.
                This cube must have a coordinate matching the name given by
                blend_coord and this coordinate must be the same length
                as the corresponding coordinate in the
                one_dimensional_weights_cube.
            one_dimensional_weights_cube (iris.cube.Cube):
                A cube with one_dimensional weights. The only dimension
                coordinate in this cube matches the string given by
                blend_coord and the length of this coord must match the
                length of the same coordinate in weights_from_mask.
            blend_coord (str):
                The string that will match to a coordinate in both input cube.
                This is the coordinate that the input cubes will be sliced
                along and then multiplied. The corresponds to the coordinate
                used to collapse a cube using the weights generated by this
                plugin.

        Returns:
            iris.cube.Cube:
                A cube with the same dimensions as the input cube, but with
                the weights multiplied by the weights from the
                one_dimensional_weights_cube. The blend_coord will be the
                leading dimension on the output cube.
        """
        result = iris.cube.CubeList()
        if (weights_from_mask.coord(blend_coord) !=
                one_dimensional_weights_cube.coord(blend_coord)):
            message = ("The blend_coord {} does not match on "
                       "weights_from_mask and "
                       "one_dimensional_weights_cube".format(blend_coord))
            raise ValueError(message)
        for masked_weight_slice, one_dimensional_weight in zip(
                weights_from_mask.slices_over(blend_coord),
                one_dimensional_weights_cube.slices_over(blend_coord)):
            masked_weight_slice.data = (
                masked_weight_slice.data * one_dimensional_weight.data)
            result.append(masked_weight_slice)
        result = result.merge_cube()
        return result

    @staticmethod
    def normalised_masked_weights(weights_cube, blend_coord):
        """
        Normalise spatial weights along dimension associated with the
        blend_coord. If for a given point the sum of the weights along
        the blend_coord is zero then the returned normalised weight for
        that point will also be zero. This correspsonds to the case where
        there is missing data for that point for all slices along the
        blend_coord.

        Args:
            weights_cube (iris.cube.Cube):
                A cube with spatial weights and any other leading dimension.
                This cube must have a coordinate matching the name given by
                blend_coord which corresponds to the dimension along
                which the normalisation is needed.
            blend_coord (str):
                The string that will match to a coordinate in both input cube.
                This coordinate corresponds to the dimension along which the
                normalisation is needed.

        Returns:
            iris.cube.Cube:
                A cube with the same dimensions as the input cube, but with
                the weights normalised along the blend_coord dimension.
                The blend_coord will be the leading dimension on the
                output cube.
        """
        summed_weights = collapsed(weights_cube,
                                   blend_coord,
                                   iris.analysis.SUM)

        result = iris.cube.CubeList()
        # Slice over blend_coord so the dimensions match.
        for weight_slice in (
                weights_cube.slices_over(blend_coord)):
            # Only divide where the sum of weights are positive. Setting
            # the out keyword args sets the default value for where
            # the sum of the weights are zero.
            normalised_data = np.divide(
                weight_slice.data, summed_weights.data,
                out=np.zeros_like(weight_slice.data),
                where=(summed_weights.data > 0))
            result.append(weight_slice.copy(data=normalised_data))
        return result.merge_cube()

    @staticmethod
    def create_template_slice(cube_to_collapse, blend_coord):
        """
        Create a template cube from a slice of the cube we are collapsing.
        The slice will be over blend_coord, y and x and will remove any other
        dimensions. This means that the resulting spatial weights won't vary in
        any other dimension other than the blend_coord. If the mask does
        vary in another dimension an error is raised.

        Args:
            cube_to_collapse (iris.cube.Cube):
                The cube that will be collapsed along the blend_coord
                using the spatial weights generated using this plugin. Must
                be masked where there is invalid data.
            blend_coord (str):
                A string containing the name of the coordinate that the
                cube_to_collapse will be collapsed along. Also matches the
                coordinate in one_dimensional_weights_cube.

        Returns:
            iris.cube.Cube:
                A cube containing dimension coordinates blend_coord, y, x,
                with all other dimensions stripped out.

        Raises:
            ValueError: if the blend coordinate is associated with more than
                        one dimension on the cube to collapse, or no dimension
            ValueError: if the mask on cube_to_collapse varies along a
                        dimension other than the dimension associated with
                        blend_coord.
        """
        # Takes slices over x, y coord_to_collapse
        # Find blend dim coord associated with blend coord, sometimes we name a
        # blend_coord that is a aux_coord associated with a dim coord rather
        # than using the name of the dim_coord itself.
        # Here we reset blend_coord to the name of the dim_coord to catch the
        # case where blend_coord is an aux_coord.
        blend_dim = cube_to_collapse.coord_dims(blend_coord)
        if len(blend_dim) == 1:
            blend_dim = blend_dim[0]
        else:
            message = (
                "Blend coordinate must only be across one dimension. "
                "Coordinate {} is associated with dimensions {}")
            message = message.format(blend_coord, blend_dim)
            raise ValueError(message)
        blend_coord = cube_to_collapse.coord(
            dimensions=blend_dim, dim_coords=True).name()
        # Find original dim coords in input cube
        original_dim_coords = [
            coord.name() for coord in cube_to_collapse.dim_coords]
        # Slice over relevant coords.
        x_coord = cube_to_collapse.coord(axis='x').name()
        y_coord = cube_to_collapse.coord(axis='y').name()
        coords_to_slice_over = [blend_coord, y_coord, x_coord]
        slices = cube_to_collapse.slices(coords_to_slice_over)
        # Check they all have the same mask
        first_slice = next(slices)
        if np.ma.is_masked(first_slice.data):
            first_mask = first_slice.data.mask
            for cube_slice in slices:
                if not np.all(cube_slice.data.mask == first_mask):
                    message = (
                        "The mask on the input cube can only vary along the "
                        "blend_coord, differences in the mask were found "
                        "along another dimension")
                    raise ValueError(message)
        # Remove old dim coords
        for coord in original_dim_coords:
            if coord not in coords_to_slice_over:
                first_slice.remove_coord(coord)
        # Return slice template
        return first_slice

    def process(self, cube_to_collapse, one_dimensional_weights_cube,
                blend_coord):
        """
        Create fuzzy spatial weights based on missing data in the cube we
        are going to collapse and combine these with 1D weights along the
        blend_coord.

        Args:
            cube_to_collapse (iris.cube.Cube):
                The cube that will be collapsed along the blend_coord
                using the spatial weights generated using this plugin. Must
                be masked where there is invalid data. The mask may only
                vary along the blend_coord, and not along any other dimensions
                on the cube.
            one_dimensional_weights_cube (iris.cube.Cube):
                A cube containing a single dimension coordinate with the same
                name given blend_coord. This cube contains 1D weights
                that will be applied along the blend_coord but need
                adjusting spatially based on missing data.
            blend_coord (str):
                A string containing the name of the coordinate that the
                cube_to_collapse will be collapsed along. Also matches the
                coordinate in one_dimensional_weights_cube.

        Returns:
            iris.cube.Cube:
                A cube containing normalised spatial weights based on the
                cube_to_collapsemask and the one_dimensional weights supplied.
                Contains the dimensions, blend_coord, y, x.
        """
        template_cube = self.create_template_slice(
            cube_to_collapse, blend_coord)
        weights_from_mask = self.create_initial_weights_from_mask(
            template_cube)
        weights_from_mask = self.smooth_initial_weights(
            weights_from_mask)
        final_weights = self.multiply_weights(
            weights_from_mask, one_dimensional_weights_cube,
            blend_coord)
        final_weights = self.normalised_masked_weights(
            final_weights, blend_coord)
        return final_weights
