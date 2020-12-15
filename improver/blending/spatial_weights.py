# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown Copyright 2017-2020 Met Office.
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
from improver.metadata.constants import FLOAT_DTYPE
from improver.utilities.rescale import rescale


class SpatiallyVaryingWeightsFromMask(BasePlugin):
    """
    Plugin for adjusting weights spatially based on masked data in the input
    cube. It takes in an initial one dimensional cube of weights which would
    be used to collapse a dimension on the input cube and outputs weights
    which have been adjusted based on the presence of missing data in x-y
    slices of input data. The resulting weights cube has a x and y dimensions
    in addition to the one dimension in the initial cube of weights.
    """

    def __init__(self, blend_coord, fuzzy_length=10):
        """
        Initialise class.

        Args:
            blend_coord (str):
                Coordinate over which the input 1D weights will vary
            fuzzy_length (int or float):
                Distance, in grid squares, over which the weights from the input
                data mask are smoothed. This is used to calculate a fuzzy
                scaling factor for the input weights based on how far away each
                grid point is from a masked point. The distance is taken as
                the euclidean distance from the masked point, so the fuzzy
                length can be a non-integer value. Any points that are at least
                this distance away from a masked point keep a weight of one,
                and any points closer than this distance to a masked point have
                a weight of less than one based on how close to the masked
                point they are.
        """
        self.fuzzy_length = fuzzy_length
        self.blend_coord = blend_coord
        self.blend_axis = None

    def __repr__(self):
        """Represent the configured plugin instance as a string."""
        result = "<SpatiallyVaryingWeightsFromMask: fuzzy_length: {}>".format(
            self.fuzzy_length
        )
        return result

    def _create_template_slice(self, cube_to_collapse):
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
        blend_dim = cube_to_collapse.coord_dims(self.blend_coord)
        if len(blend_dim) == 1:
            blend_dim = blend_dim[0]
        else:
            message = (
                "Blend coordinate must only be across one dimension. "
                "Coordinate {} is associated with dimensions {}"
            )
            message = message.format(self.blend_coord, blend_dim)
            raise ValueError(message)
        self.blend_coord = cube_to_collapse.coord(
            dimensions=blend_dim, dim_coords=True
        ).name()
        # Find original dim coords in input cube
        original_dim_coords = [coord.name() for coord in cube_to_collapse.dim_coords]
        # Slice over relevant coords.
        x_coord = cube_to_collapse.coord(axis="x").name()
        y_coord = cube_to_collapse.coord(axis="y").name()
        coords_to_slice_over = [self.blend_coord, y_coord, x_coord]
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
                        "along another dimension"
                    )
                    raise ValueError(message)
        # Remove old dim coords
        for coord in original_dim_coords:
            if coord not in coords_to_slice_over:
                first_slice.remove_coord(coord)
        # Return slice template
        return first_slice

    def _normalise_initial_weights(self, weights):
        """Normalise weights so that they add up to 1 along the blend dimension
        at each spatial point.  This is different from the normalisation that
        happens after the application of fuzzy smoothing near mask boundaries.
        Modifies weights cube in place.

        Args:
            weights (iris.cube.Cube):
                3D weights containing zeros for masked points, but before
                fuzzy smoothing
        """
        weights_sum = np.sum(weights.data, axis=self.blend_axis)
        weights_sum = np.broadcast_to(weights_sum, weights.shape)
        weights.data = np.where(
            weights_sum > 0, np.divide(weights.data, weights_sum), 0
        ).astype(FLOAT_DTYPE)

    def _rescale_masked_weights(self, weights):
        """Apply fuzzy smoothing to weights at the edge of masked areas

        Args:
            weights (iris.cube.Cube):
                Pre-normalised weights where the weights of masked data points
                have been set to 0

        Returns:
            (tuple): tuple containing:
                **weights** (iris.cube.Cube):
                    Weights where MASKED slices have been rescaled, but UNMASKED
                    slices have not
                **rescaled** (iris.cube.Cube):
                    Binary (0/1) map showing which weights have been rescaled
        """
        is_rescaled = iris.cube.CubeList()
        rescaled_weights = iris.cube.CubeList()
        for weights_slice in weights.slices_over(self.blend_coord):
            weights_nonzero = np.where(weights_slice.data > 0, True, False)
            if np.all(weights_nonzero):
                # if there are no masked points in this slice, keep current weights
                # and mark as unchanged (not rescaled)
                rescaled_weights.append(weights_slice)
                is_rescaled.append(weights_slice.copy(data=~weights_nonzero))
            else:
                weights_orig = weights_slice.data.copy()

                # calculate the distance to the nearest invalid point, in grid squares,
                # for each point on the grid
                distance = distance_transform_edt(weights_nonzero)

                # calculate a 0-1 scaling factor based on the distance from the
                # nearest invalid data point, which scales between 1 at the fuzzy length
                # towards 0 for points closest to the edge of the mask
                fuzzy_factor = rescale(
                    distance, data_range=[0.0, self.fuzzy_length], clip=True
                )

                # multiply existing weights by fuzzy scaling factor
                rescaled_weights_data = np.multiply(
                    weights_slice.data, fuzzy_factor
                ).astype(FLOAT_DTYPE)
                rescaled_weights.append(weights_slice.copy(data=rescaled_weights_data))

                # identify spatial points where weights have been rescaled
                is_rescaled_data = np.where(
                    rescaled_weights_data != weights_orig, True, False
                )
                is_rescaled.append(weights_slice.copy(data=is_rescaled_data))

        weights = rescaled_weights.merge_cube()
        rescaled = is_rescaled.merge_cube()

        return weights, rescaled

    def _rescale_unmasked_weights(self, weights, is_rescaled):
        """Increase weights of unmasked slices at locations where masked slices
        have been smoothed, so that the sum of weights over self.blend_coord is
        re-normalised (sums to 1) at each point and the relative weightings of
        multiple unmasked slices are preserved.  Modifies weights cube in place.

        Args:
            weights (iris.cube.Cube):
                Cube of weights to which fuzzy smoothing has been applied to any
                masked slices
            is_rescaled (iris.cube.Cube):
                Cube matching weights.shape, with value of 1 where masked weights
                have been rescaled, and 0 where they are unchanged.
        """
        rescaled_data = np.multiply(weights.data, is_rescaled.data)
        unscaled_data = np.multiply(weights.data, ~is_rescaled.data)
        unscaled_sum = np.sum(unscaled_data, axis=self.blend_axis)
        required_sum = 1.0 - np.sum(rescaled_data, axis=self.blend_axis)
        normalisation_factor = np.where(
            unscaled_sum > 0, np.divide(required_sum, unscaled_sum), 0
        )
        normalised_weights = (
            np.multiply(unscaled_data, normalisation_factor) + rescaled_data
        )
        weights.data = normalised_weights.astype(FLOAT_DTYPE)

    def process(self, cube_to_collapse, one_dimensional_weights_cube):
        """
        Create fuzzy spatial weights based on missing data in the cube we
        are going to collapse and combine these with 1D weights along the
        blend_coord.  The method is as follows:

        1. Broadcast 1D weights to the 3D shape of the input cube
        2. Set masked weights to zero.  If there is no mask, return original
           unnormalised 1D weights with a warning message.
        3. Normalise 3D weights along the blend axis.  This is needed so that a
           smooth spatial transition between layers can be achieved near mask boundaries.
        4. Reduce weights of masked layers near the mask boundary.
        5. Increase weights of unmasked layers near the mask boundary.

        Args:
            cube_to_collapse (iris.cube.Cube):
                The cube that will be collapsed along self.blend_coord
                using the spatial weights generated using this plugin. Must
                be masked where there is invalid data. The mask may only
                vary along the blend and spatial coordinates, and not along
                any other dimensions on the cube.
            one_dimensional_weights_cube (iris.cube.Cube):
                A cube containing a single dimension coordinate with the same
                name given blend_coord. This cube contains 1D weights
                that will be applied along the blend_coord but need
                adjusting spatially based on missing data.

        Returns:
            iris.cube.Cube:
                A cube containing normalised 3D spatial weights based on the
                cube_to_collapse mask and the one_dimensional weights supplied.
                Has dimensions: self.blend_coord, y, x.
        """
        template_cube = self._create_template_slice(cube_to_collapse)
        (self.blend_axis,) = template_cube.coord_dims(self.blend_coord)

        weights_data = iris.util.broadcast_to_shape(
            one_dimensional_weights_cube.data, template_cube.shape, (self.blend_axis,)
        )
        weights = template_cube.copy(data=weights_data)

        if np.ma.is_masked(template_cube.data):
            # Set masked weights to zero
            weights.data = np.where(template_cube.data.mask, 0, weights.data)
        else:
            message = "Expected masked input to SpatiallyVaryingWeightsFromMask"
            warnings.warn(message)
            return weights

        self._normalise_initial_weights(weights)
        weights, rescaled = self._rescale_masked_weights(weights)
        self._rescale_unmasked_weights(weights, rescaled)

        return weights
